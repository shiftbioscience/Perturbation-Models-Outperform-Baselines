"""
scGPT model wrapper for CellSimBench integration.
Handles training and prediction with internal checkpointing.
"""

import logging
from math import e
import pickle
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time
import copy
import warnings
from pathlib import Path
from typing import Dict, List
from gears import PertData
import os
from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager
from torch_geometric.data import Data

# scGPT imports
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id
from scgpt.utils.util import load_pretrained
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from anndata import AnnData
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli
from torch.nn import functional as F
from typing import Mapping, Optional

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class GeneEncoder(nn.Module):
    """
    Gene/token encoder with built-in normalization.
    This is recommended by scGPT authors for better training stability.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)  # Apply LayerNorm for stability
        return x


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, 
    weights: torch.Tensor = None, reduction: str = 'sum'
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    
    This loss function is used in scGPT training to compute gene expression prediction loss.
    
    Args:
        input: Predicted values [batch_size, n_features]
        target: Target values [batch_size, n_features] 
        mask: Mask indicating which elements to include [batch_size, n_features]
        weights: Not used, kept for compatibility
        reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
                   'mean': the sum of the output will be divided by the number of 
                           elements in the mask
                   'sum': the output will be summed
    
    Returns:
        Masked MSE loss
    """
    mask = mask.float()
    
    # Compute squared differences
    squared_diff = (input - target) ** 2
    
    # Apply mask
    masked_diff = squared_diff * mask
    
    # Sum over features and batch
    loss = masked_diff.sum()
    
    if reduction == 'mean':
        # Normalize by sum of mask
        normalization = mask.sum()
        if normalization > 0:
            return loss / normalization
        else:
            return torch.tensor(0.0, device=input.device)
    elif reduction == 'sum':
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean' or 'sum'.")


class TransformerGeneratorWithLayerNorm(TransformerGenerator):
    """TransformerGenerator with LayerNorm on perturbation embeddings."""
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Remove n_covariates if it exists in kwargs
        kwargs.pop('n_covariates', None)
        super().__init__(*args, **kwargs)
        
        # Override the perturbation encoder to use GeneEncoder with normalization
        # This follows the recommendation from scGPT authors for better training stability
        # The pert_pad_id is already set in parent class as self.pert_pad_id
        self.pert_encoder = GeneEncoder(3, self.d_model, padding_idx=self.pert_pad_id)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        """Encode without covariates, using LayerNorm on perturbation embeddings."""
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize) - now with LayerNorm
        
        total_embs = src + values + perts

        # total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        if self.explicit_zero_prob and not do_sample and not self.training:
            do_sample = True
            log.warning("Auto set do_sample to True when model is in eval mode.")

        # binning input gene values
        if self.n_input_bins > 0:
            from scgpt.preprocess import binning

            processed_values = torch.stack(
                [binning(row, n_bins=self.n_input_bins) for row in values], dim=0
            ).to(values.device)
        else:
            processed_values = values

        transformer_output = self._encode(
            src, processed_values, input_pert_flags, src_key_padding_mask
        )
        output = {}
        mlm_output = self.decoder(transformer_output, values)
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )  # (batch, seq_len)
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        return output
    
    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


class SCGPTPertData(PertData):
    """Subclass of GEARS PertData that adds support for custom splits for scGPT."""

    def __init__(self, data_path):
        super().__init__(data_path)
        




    def prepare_split(
        self,
        split="simulation",
        seed=1,
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        combo_single_split_test_set_fraction=0.1,
        test_perts=None,
        only_test_set_perts=False,
        test_pert_genes=None,
        split_dict_path=None,
    ):
        """
        Extended prepare_split that adds support for custom splits via split_dict_path
        """
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.train_gene_set_size = train_gene_set_size

        if split == "custom":
            try:
                with open(split_dict_path, "rb") as f:
                    self.set2conditions = pickle.load(f)
                log.info("Loaded custom split from " + split_dict_path)
                return
            except:
                raise ValueError(
                    "Please provide valid split_dict_path for custom split"
                )

        # Call parent class method for all other split types
        super().prepare_split(
            split=split,
            seed=seed,
            train_gene_set_size=train_gene_set_size,
            combo_seen2_train_frac=combo_seen2_train_frac,
            combo_single_split_test_set_fraction=combo_single_split_test_set_fraction,
            test_perts=test_perts,
            only_test_set_perts=only_test_set_perts,
            test_pert_genes=test_pert_genes,
        )

    def load(self, data_path = None):

        if os.path.exists(data_path):
            print("Loading data from " + data_path + "...")
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.adata.var.index.name = None
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            

        else:
            raise ValueError("data must be a path to an h5ad file")
        
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        
        if os.path.isfile(dataset_fname):
            print("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))     
            self.gene_names = self.adata.var.gene_name
            print("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            print("Creating pyg object for each cell in the data...")
            self.dataset_processed = self.create_dataset_file()
            print("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print("Done!")

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        pert_feats = np.zeros(len(X[0]))
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = 1
        pert_feats = np.expand_dims(pert_feats, 0)
        feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T
        return Data(x=feature_mat, edge_index=None, edge_attr=None,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """

        num_de_genes = 20
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        de_genes = adata_.uns['rank_genes_groups_cov_all']
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category, adata_)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]

            for i, cell_z in enumerate(adata_.X):
                # Use samples from control as basal expression
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for i, cell_z in enumerate(adata_.X):
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs

    def create_dataset_file(self):
        dl = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            cell_graph_dataset = self.create_cell_graph_dataset(self.adata, p, num_samples=1)
            dl[p] = cell_graph_dataset
        return dl
    


    def create_cell_graph_for_prediction(self, X, pert_idx, pert_gene):
        pert_feats = np.zeros(len(X))
        for p in pert_idx:
            pert_feats[int(np.abs(p))] = np.sign(p)
        feature_mat = torch.Tensor(np.vstack([X, pert_feats])).T
        
        return Data(x=feature_mat, pert=pert_gene)

    def create_cell_graph_dataset_for_prediction(self, pert_gene, ctrl_adata, gene_names, device, num_samples = 300):
        # Get the indices (and signs) of applied perturbation
        pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

        # Create graphs for prediction
        Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
        cell_graphs_list = [self.create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs]
        return cell_graphs_list


class SCGPTWrapper:
    """scGPT model wrapper handling its own checkpointing and data conversion."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.pert_data = None
        self.gene_ids = None
        self.n_genes = None
        self.vocab = None
        self.best_model = None
        self.data_manager = DataManager(self.config)
        
    def train(self):
        """Train scGPT model with internal checkpointing."""
        
        log.info("Starting scGPT training process...")
        
        # 1. Load and preprocess data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")
        
        # 2. Convert to scGPT format
        log.info("Converting to scGPT format...")
        self.pert_data = self._convert_to_scgpt_format(adata)
        
        # 3. Prepare splits
        log.info("Preparing data splits...")
        self._prepare_scgpt_splits()
        
        # 4. Get dataloaders
        log.info("Creating data loaders...")
        self.pert_data.get_dataloader(
            batch_size=self.config['hyperparameters']['batch_size'],
            test_batch_size=self.config['hyperparameters']['batch_size']
        )
        
        # 5. Initialize scGPT model
        log.info("Initializing scGPT model...")
        self._initialize_scgpt_model()
        
        # 6. Check for existing checkpoints
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 7. Train with scGPT's training loop
        log.info("Starting model training...")
        self._train_scgpt()
        
        # 8. Save final model
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving model to {output_dir}")
        self._save_model(output_dir)
        self._save_metadata(output_dir)
        
        log.info("Training completed successfully")
        
    def predict(self):
        """Generate predictions using trained scGPT model."""
        
        log.info("Starting scGPT prediction process...")
        
        # 1. Load trained model
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        # 2. Load and preprocess data for prediction
        adata = self.data_manager.load_dataset()
        self.pert_data = self._convert_to_scgpt_format_for_prediction(adata)
        
        # 3. Initialize model and load trained weights
        log.info("Initializing scGPT model...")
        self._initialize_scgpt_model()
        
        # 4. Generate predictions
        test_conditions = self.config['test_conditions']
        log.info(f"Generating predictions for {len(test_conditions)} conditions")
        
        predictions_adata = self._generate_predictions(test_conditions)
        
        # 5. Save predictions
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)
        
        log.info("Prediction completed successfully")
        
    def _convert_to_scgpt_format(self, adata: sc.AnnData) -> SCGPTPertData:
        """Convert CellSimBench data to scGPT PertData format."""
        
        log.info("Converting CellSimBench data to scGPT format...")
        
        # Create a copy for scGPT processing
        adata_scgpt = adata.copy()

        # TODO: We should be passing the control value as a parameter
        CTRL_VALUES = ['ctrl_iegfp', 'control', 'non-targeting']

        # TODO: Maybe we should do this another way. Seems a bit hacky.
        if "symbol_scgpt" in adata_scgpt.var.columns:
            # Remove all the NaNs from the symbol_scgpt column
            adata_scgpt = adata_scgpt[:, adata_scgpt.var.symbol_scgpt.notna().values]
            # Remove all the perturbations that are not in the symbol_scgpt column
            all_perts = adata_scgpt.obs['condition'].unique()
            # Select the perts that are in the symbol_scgpt column or contain 'ctrl'
            perts_to_keep = []
            for p in all_perts:
                if p in adata_scgpt.var.symbol_scgpt.values or 'ctrl' in p or any(ctrl_value in p for ctrl_value in CTRL_VALUES):
                    perts_to_keep.append(p)
            adata_scgpt = adata_scgpt[adata_scgpt.obs['condition'].isin(perts_to_keep)]
        else:
            raise ValueError("symbol_scgpt column is missing")


        
        # Remove all the rows where the condition contains 'ctrl' but is not "control" or CTRL_VALUE
        adata_scgpt = adata_scgpt[~adata_scgpt.obs['condition'].str.contains('ctrl') | adata_scgpt.obs['condition'].isin(CTRL_VALUES)]
        
        # Preprocess for scGPT
        adata_scgpt = self._prep_adata_for_scgpt(adata_scgpt)

        # # Get all unique perturbations
        # unique_perturbations = adata_scgpt.obs['condition'].unique()
        # # Strip ctrl+ from the perturbations
        # unique_perturbations = [p.replace('ctrl+', '') for p in unique_perturbations]
        # # Find the ones that are missing from adata.var.symbol_scgpt
        # missing_perturbations = []
        # for p in unique_perturbations:
        #     if p not in adata_scgpt.var.symbol_scgpt.values:
        #         missing_perturbations.append(p)
        # # Add the missing perturbations to adata.var.symbol_scgpt
        # adata_scgpt.var.loc[missing_perturbations, 'symbol_scgpt'] = missing_perturbations
        
        # Use output directory for persistent storage
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving processed scGPT data to: {processed_data_dir}")
        
        # Create SCGPTPertData object
        if os.path.exists(processed_data_dir / 'cellsimbench_scgpt' / 'perturb_processed.h5ad'):
            log.warning(f"Existing {processed_data_dir / 'cellsimbench_scgpt' / 'perturb_processed.h5ad'} file found. This means that data will not be reprocessed!! Delete file to reprocess.")
        pert_data = SCGPTPertData(str(processed_data_dir))
        processed_dataset_path = processed_data_dir / 'cellsimbench_scgpt'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        # TODO: Remove this when we have covariates
        adata_scgpt.obs['cell_type'] = "NOTHING"
        if not cell_graphs_file.exists():
            log.info("Creating new processed scGPT dataset...")
            pert_data.new_data_process(dataset_name='cellsimbench_scgpt', adata=adata_scgpt)
        else:
            log.info("Loading existing processed scGPT dataset...")
            pert_data.load(data_path=str(processed_dataset_path))
        
        return pert_data
        
    def _convert_to_scgpt_format_for_prediction(self, adata: sc.AnnData) -> SCGPTPertData:
        """Convert CellSimBench data to scGPT PertData format for prediction only."""
        
        # Use the same output directory as training for loading processed data
        if 'model_path' in self.config:
            model_dir = Path(self.config['model_path'])
            processed_data_dir = model_dir / 'processed_data'
        else:
            raise FileNotFoundError(f"Processed scGPT data not found. Please run training first.")
        
        processed_dataset_path = processed_data_dir / 'cellsimbench_scgpt'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        split_dict_file = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Loading processed scGPT data from: {processed_dataset_path}")
        
        # Create SCGPTPertData object and load existing processed data
        pert_data = SCGPTPertData(str(processed_data_dir))
        
        if cell_graphs_file.exists():
            log.info("Loading existing processed scGPT data for prediction...")
            pert_data.load(data_path=str(processed_dataset_path))
        else:
            raise FileNotFoundError(
                f"Processed scGPT data not found at '{cell_graphs_file}'. "
                "Please run training first to create the required data structures."
            )
        
        # Load and prepare splits
        if split_dict_file.exists():
            log.info("Loading existing split dictionary...")
            pert_data.prepare_split(split='custom', seed=42, split_dict_path=str(split_dict_file))
        else:
            raise FileNotFoundError(
                f"Split dictionary not found at '{split_dict_file}'. "
                "Please run training first to create the required data structures."
            )
        
        return pert_data
        
    def _prep_adata_for_scgpt(self, adata: AnnData) -> AnnData:
        """Prepare AnnData object for scGPT training/evaluation."""
        
        log.info("Preprocessing AnnData for scGPT...")
        
        if "symbol_scgpt" in adata.var.columns:
            adata.var["gene_name"] = adata.var.symbol_scgpt
        else:
            raise ValueError("symbol_scgpt column is missing")
            
        # TODO: We should be passing the control value as a parameter
        CTRL_VALUE = 'ctrl_iegfp'
        
        # Fix condition format for scGPT
        def fix_condition(condition):
            if condition == 'control' or condition == CTRL_VALUE:
                return 'ctrl'
            elif "ctrl" not in condition and "+" not in condition:
                return "ctrl+" + condition
            else:
                return condition
        
        adata.obs["condition"] = adata.obs["condition"].apply(fix_condition)

        # # Get all unique perturbations
        # unique_perturbations = adata.obs['condition'].unique()
        # # Strip ctrl+ from the perturbations
        # unique_perturbations = [p.replace('ctrl+', '') for p in unique_perturbations]
        # # Find the ones that are missing from adata.var.symbol_scgpt
        # missing_perturbations = []
        # for p in unique_perturbations:
        #     if p not in adata.var.symbol_scgpt.values:
        #         missing_perturbations.append(p)
        # # Add the missing perturbations to adata.var.symbol_scgpt
        # adata.var.loc[missing_perturbations, 'symbol_scgpt'] = missing_perturbations
        
        # Data validation checks
        assert "condition" in adata.obs.columns, "Condition column is missing"
        assert "ctrl" in adata.obs.condition.values, "Control condition is missing or is not named 'ctrl'"
        assert any("+" in cond for cond in adata.obs.condition.values), "Perturbation combinations are missing or are not delimited with a '+' sign"
        
        # Ensure log1p transformation
        if self.config['hyperparameters']['dolog1p']:
            log.info("Applying log1p transformation...")
            sc.pp.log1p(adata)
        
        # Mark HVGs in uns
        if 'hvg' not in adata.uns.keys():
            adata.uns['hvg'] = {'indices': np.arange(adata.n_vars)}
        
        assert "gene_name" in adata.var.columns, "gene_name column is missing"
        
        log.info(f"scGPT preprocessing completed. Final shape: {adata.shape}")
        return adata
        
    def _prepare_scgpt_splits(self):
        """Prepare train/val/test splits for scGPT."""
        
        # Get conditions from config
        train_conditions = self.config['train_conditions']
        val_conditions = self.config['val_conditions']
        test_conditions = self.config['test_conditions']
        CTRL_VALUES = ['ctrl_iegfp', 'control', 'non-targeting']
        
        # Convert to scGPT format
        def convert_conditions(conditions, keys):
            scgpt_conditions = []
            for cond in conditions:
                if cond in CTRL_VALUES:
                    scgpt_conditions.append('ctrl')
                    continue
                elif '+' not in cond:
                    result = f"ctrl+{cond}"
                else:
                    result = cond
                if not result in keys:
                    # CASE: for some reason, the cell graphs didn't include this one
                    continue
                scgpt_conditions.append(result)
            return scgpt_conditions
        
        split_dict = {
            'train': convert_conditions(train_conditions, self.pert_data.dataset_processed.keys()),
            'val': convert_conditions(val_conditions, self.pert_data.dataset_processed.keys()),
            'test': convert_conditions(test_conditions, self.pert_data.dataset_processed.keys())
        }
        
        log.info(f"Prepared splits: train={len(split_dict['train'])}, val={len(split_dict['val'])}, test={len(split_dict['test'])}")
        
        # Save split dictionary to persistent location
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        split_path = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Saving split dictionary to: {split_path}")
        with open(split_path, 'wb') as f:
            pickle.dump(split_dict, f)
        
        # Prepare split in PertData
        self.pert_data.prepare_split(split='custom', seed=42, split_dict_path=str(split_path))
        
    def _initialize_scgpt_model(self):
        """Initialize scGPT model from pretrained checkpoint."""
        
        log.info("Initializing scGPT model...")
        
        # Load model configuration and pretrained weights
        model_dir = Path(self.config['hyperparameters']['model_loc'])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        
        # Check if files exist
        if not model_config_file.exists():
            raise FileNotFoundError(f"Model config not found at {model_config_file}")
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found at {model_file}")
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary not found at {vocab_file}")
        
        # Load vocabulary
        self.vocab = GeneVocab.from_file(vocab_file)
        for s in self.config['hyperparameters']['special_tokens']:
            if s not in self.vocab:
                self.vocab.append_token(s)
        
        # Map genes to vocabulary
        self.pert_data.adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in self.pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(self.pert_data.adata.var["id_in_vocab"])
        log.info(f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(self.vocab)}")
        
        genes = self.pert_data.adata.var["gene_name"].tolist()
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.gene_ids = np.array(
            [self.vocab[gene] if gene in self.vocab else self.vocab["<pad>"] for gene in genes], 
            dtype=int
        )
        self.n_genes = len(genes)
        
        # Load model configuration
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        
        log.info(f"Loading model config from {model_config_file}")
        
        # Override with config hyperparameters
        hyperparams = self.config['hyperparameters']
        embsize = model_configs.get("embsize", hyperparams['embsize'])
        nhead = model_configs.get("nheads", hyperparams['nheads'])
        d_hid = model_configs.get("d_hid", hyperparams['d_hid'])
        nlayers = model_configs.get("nlayers", hyperparams['nlayers'])
        n_layers_cls = model_configs.get("n_layers_cls", hyperparams['n_layers_cls'])


        ntokens = len(self.vocab)
        
        # Initialize model
        self.model = TransformerGeneratorWithLayerNorm(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=1,
            vocab=self.vocab,
            dropout=hyperparams['dropout'],
            pad_token=hyperparams['pad_token'],
            pad_value=hyperparams['pad_value'],
            pert_pad_id=hyperparams['pert_pad_id'],
            do_mvc=hyperparams['MVC'],
            cell_emb_style=hyperparams['cell_emb_style'],
            mvc_decoder_style=hyperparams['mvc_decoder_style'],
            use_fast_transformer=hyperparams['use_fast_transformer'],
        )
        
        # Load pretrained weights
        log.info(f"Loading pretrained weights from {model_file}")
        pretrained_dict = torch.load(model_file, map_location='cpu')
        self.model = load_pretrained(
            model=self.model, 
            pretrained_params=pretrained_dict, 
            strict=False, 
            verbose=True
        )
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        log.info(f"Model initialized on device: {device}")
        
    def _train_scgpt(self):
        """Execute scGPT training with CellSimBench data."""
        
        log.info("Starting scGPT training loop...")
        
        # Training configuration
        hyperparams = self.config['hyperparameters']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize training components
        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
        scheduler = StepLR(optimizer, hyperparams['schedule_interval'], gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=hyperparams['amp'])
        
        # Get data loaders
        train_loader = self.pert_data.dataloader["train_loader"]
        val_loader = self.pert_data.dataloader["val_loader"]
        
        # Training loop
        best_val_loss = float('inf')
        self.best_model = None
        
        for epoch in tqdm(range(1, hyperparams['max_epochs'] + 1), desc="Training Epochs"):
            epoch_start_time = time.time()
            
            # Train epoch
            self._train_epoch(
                epoch=epoch,
                train_loader=train_loader,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                criterion=criterion,
                hyperparams=hyperparams
            )
            
            # Validation
            val_loss = self._validate_epoch(val_loader, device, criterion, hyperparams)
            
            elapsed = time.time() - epoch_start_time
            log.info(f"Epoch {epoch:3d} | Time: {elapsed:5.2f}s | Val Loss: {val_loss:5.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                log.info(f"New best model with validation loss: {val_loss:5.4f}")
            
            scheduler.step()
        
        log.info("Training completed!")
        
    def _train_epoch(self, epoch, train_loader, device, optimizer, scheduler, scaler, criterion, hyperparams):
        """Train the model for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        running_loss = 0.0
        running_loss_steps = 0
        
        num_batches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Training Epoch {epoch}")
        
        for batch, batch_data in pbar:
            batch_size = len(batch_data.y)
            batch_data.to(device)
            
            x = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
            pert_flags = x[:, 1].long().view(batch_size, self.n_genes)
            target_gene_values = batch_data.y
            


            # Prepare input
            if hyperparams['include_zero_gene'] == "all":
                input_gene_ids = torch.arange(self.n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            
            # Sample input_gene_id if too long
            if len(input_gene_ids) > hyperparams['max_seq_len']:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:hyperparams['max_seq_len']]
            
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]
            
            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
            
            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=hyperparams['amp']):
                output_dict = self.model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=hyperparams['CLS'],
                    CCE=hyperparams['CCE'],
                    MVC=hyperparams['MVC'],
                    ECS=hyperparams['ECS'],
                )
                output_values = output_dict["mlm_output"]
                
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                
                # Compute loss without weights
                loss = criterion(output_values, target_values, masked_positions)
            
            # Backward pass
            self.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            running_loss += loss.item()
            running_loss_steps += 1
            
            # Update progress bar every 5 steps
            if batch % 5 == 0:
                avg_loss = running_loss / running_loss_steps if running_loss_steps > 0 else 0
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log progress
            if batch % 50 == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / 50
                cur_loss = total_loss / 50
                log.info(f"Epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                        f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f}")
                total_loss = 0
                start_time = time.time()
                running_loss = 0
                running_loss_steps = 0
                
    def _validate_epoch(self, val_loader, device, criterion, hyperparams):
        """Validate the model for one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_size = len(batch_data.y)
                batch_data.to(device)
                
                x = batch_data.x
                ori_gene_values = x[:, 0].view(batch_size, self.n_genes)
                pert_flags = x[:, 1].long().view(batch_size, self.n_genes)
                target_gene_values = batch_data.y
                

                
                # Prepare input (same as training)
                if hyperparams['include_zero_gene'] == "all":
                    input_gene_ids = torch.arange(self.n_genes, device=device, dtype=torch.long)
                else:
                    input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                
                if len(input_gene_ids) > hyperparams['max_seq_len']:
                    input_gene_ids = input_gene_ids[:hyperparams['max_seq_len']]
                
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                
                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
                
                src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=hyperparams['amp']):
                    output_dict = self.model(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=hyperparams['CLS'],
                        CCE=hyperparams['CCE'],
                        MVC=hyperparams['MVC'],
                        ECS=hyperparams['ECS'],
                    )
                    output_values = output_dict["mlm_output"]
                    
                    masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                    
                    # Compute loss without weights
                    loss = criterion(output_values, target_values, masked_positions)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    


        
    def _generate_predictions(self, test_conditions: List[str]) -> AnnData:
        """Generate scGPT predictions for test conditions."""
        
        log.info("Generating scGPT predictions...")
        gene_list = self.pert_data.gene_names.values.tolist()
        CTRL_VALUES = ['ctrl_iegfp', 'control', 'non-targeting']
        
        # Convert conditions to prediction format
        prediction_conditions = []
        for cond in test_conditions:
            if cond not in CTRL_VALUES:
                if 'ctrl+' not in cond and '+' not in cond:
                    if cond not in gene_list:
                        log.warning(f"Condition {cond} not found in gene list. Skipping.")
                        continue
                    prediction_conditions.append([cond])
                else:
                    # Remove ctrl+ prefix and split if combo
                    clean_cond = cond.replace('ctrl+', '')
                    if '+' in clean_cond:
                        split_cond = clean_cond.split('+')
                        valid_genes = []
                        for gene in split_cond:
                            if gene not in gene_list:
                                log.warning(f"Gene {gene} not found in gene list. Skipping from combo condition.")
                                continue
                            valid_genes.append(gene)
                        if valid_genes:
                            prediction_conditions.append(valid_genes)
                    else:
                        if clean_cond not in gene_list:
                            log.warning(f"Condition {clean_cond} not found in gene list. Skipping.")
                            continue
                        prediction_conditions.append([clean_cond])
        
        log.info(f"Generating predictions for {len(prediction_conditions)} conditions")
        
        # Get control cells for prediction
        ctrl_adata = self.pert_data.adata[self.pert_data.adata.obs['condition'] == 'ctrl']
        
        # Generate predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(device)
        
        results_pred = {}
        
        with torch.no_grad():
            for pert in tqdm(prediction_conditions, desc="Predicting perturbations"):
                # Create cell graphs for this perturbation
                cell_graphs = self.pert_data.create_cell_graph_dataset_for_prediction(
                    pert_gene=pert,
                    ctrl_adata=ctrl_adata,
                    gene_names=gene_list,   
                    device=device,
                    num_samples=self.config['hyperparameters']['pool_size'],
                )
                
                loader = DataLoader(
                    cell_graphs, 
                    batch_size=self.config['hyperparameters']['batch_size'], 
                    shuffle=False
                )
                
                preds = []
                for batch_data in tqdm(loader, desc=f"Predicting {pert}"):
                    pred = self.model.pred_perturb(
                        batch_data,
                        include_zero_gene=self.config['hyperparameters']['include_zero_gene'],
                        gene_ids=self.gene_ids,
                    )
                    preds.append(pred)
                
                preds = torch.cat(preds, dim=0)
                results_pred["+".join(pert)] = preds.cpu().numpy()
        
        # Convert to AnnData format
        return self._format_predictions_as_anndata(results_pred, ctrl_adata)
        
    def _format_predictions_as_anndata(self, predictions_dict: Dict, ctrl_adata: AnnData) -> AnnData:
        """Convert dictionary of predictions to AnnData format compatible with CellSimBench."""
        
        test_conditions = self.config['test_conditions']
        
        # Build prediction matrix and metadata
        prediction_list = []
        condition_list = []
        
        for condition in test_conditions:
            if condition != 'control':
                # Find prediction key for this condition
                pred_key = None
                for key in predictions_dict.keys():
                    if condition == key or condition.replace('ctrl+', '') == key:
                        pred_key = key
                        break
                
                if pred_key is not None:
                    predictions = predictions_dict[pred_key]
                    # Use mean prediction across all samples for this condition
                    prediction_list.append(np.mean(predictions, axis=0))
                    condition_list.append(condition)
        
        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")
        
        # Stack predictions
        prediction_matrix = np.vstack(prediction_list)
        
        # Create obs dataframe  
        # Note: We still need to create dummy covariate and pair_key columns for compatibility
        # with the evaluation framework, even though the model doesn't use covariates
        obs_df = pd.DataFrame({
            'covariate': ['none'] * len(condition_list),  # Dummy value
            'condition': condition_list,
            'pair_key': [f"none_{cond}" for cond in condition_list]  # Dummy value
        })
        
        # Create AnnData object
        adata_pred = AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = self.pert_data.adata.var_names
        
        return adata_pred
        
        
    def _save_model(self, output_dir: Path):
        """Save the trained model."""
        
        if self.best_model is not None:
            model_to_save = self.best_model
        else:
            model_to_save = self.model
        
        # Save model weights
        model_file = output_dir / 'best_model.pt'
        torch.save(model_to_save.state_dict(), model_file)
        
        # Save vocabulary
        vocab_file = output_dir / 'vocab.json'
        self.vocab.save_json(vocab_file)
        
        # Save model config
        config_file = output_dir / 'args.json'
        model_config = {
            'embsize': self.config['hyperparameters']['embsize'],
            'nheads': self.config['hyperparameters']['nheads'],
            'd_hid': self.config['hyperparameters']['d_hid'],
            'nlayers': self.config['hyperparameters']['nlayers'],
            'n_layers_cls': self.config['hyperparameters']['n_layers_cls'],
            'dropout': self.config['hyperparameters']['dropout'],
        }
        
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        log.info(f"Model saved to {output_dir}")
        
    def _save_metadata(self, output_dir: Path):
        """Save training metadata."""
        
        metadata = {
            'model_type': 'scGPT',
            'config': self.config,
            'data_shape': self.pert_data.adata.shape if self.pert_data else None,
            'n_genes': self.n_genes,
            'vocab_size': len(self.vocab) if self.vocab else None,
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder) 