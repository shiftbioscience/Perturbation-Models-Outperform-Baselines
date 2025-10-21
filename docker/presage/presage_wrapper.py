# SPDX-License-Identifier: LicenseRef-Genentech-NonCommercial-1.0
# Copyright (c) 2025 Genentech, Inc.
#
# Licensed under the Genentech Non-Commercial Software License Version 1.0 (September 2022).
# You may not use this file except in compliance with the License.
# A copy of the License is included in this directory as "docker/presage/LICENSE" and in built images at "/LICENSE".
#
# NOTICE OF MODIFICATIONS:
# This file was created or modified by the CellSimBench team to integrate PRESAGE with CellSimBench.
# See docker/presage/MODIFICATIONS.md for a summary of changes.
"""
PRESAGE model wrapper for CellSimBench integration.
Handles training and prediction with pre-computed knowledge embeddings.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tempfile
import shutil

# Add PRESAGE source to path
sys.path.insert(0, '/presage_src' if os.path.exists('/presage_src') else 'ref/PRESAGE/src')

from presage_datamodule import PRESAGEDataModule
from model_harness import ModelHarness
from presage import PRESAGE
from train import set_seed, parse_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import seed_everything

from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager
from utils import convert_perturbation_names, validate_presage_data
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from presage import GeneEmbeddingTransformation, PrepareInputs, ItemNet, Pool

# Import our custom datamodule (handle both Docker and local paths)
if os.path.exists('/cellsimbench_datamodule.py'):
    # Docker environment
    from cellsimbench_datamodule import CellSimBenchDataModule
else:
    # Local environment - add current dir to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cellsimbench_datamodule import CellSimBenchDataModule

log = logging.getLogger(__name__)

class ComboPRESAGE(nn.Module):
    """PRESAGE model variant with support for combination perturbations.
    
    This class extends the standard PRESAGE architecture to handle combination
    perturbations by aggregating embeddings for multiple genes in combo conditions.
    Uses standard MSE loss without any weight modulation.
    """
    def __init__(self, config, datamodule, input_dimension, output_dimension):
        super(ComboPRESAGE, self).__init__()

        # prepare variables
        self.batch_size = datamodule.batch_size
        self.config = config
        self.validation_step_outputs = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = datamodule
        self.genes = datamodule.train_dataset.adata.var.index.to_numpy().ravel()
        self.n_nmf_embeddings = config["n_nmf_embedding"]

        self.num_genes = output_dimension
        self.pca_dim = config["pca_dim"]


        # get gene embeddings
        self.gene_embeddings = PrepareInputs(datamodule, config)._prep_inputs()

        norms = np.linalg.norm(self.gene_embeddings, axis=1, keepdims=True)

        norms = np.median(norms, axis=(0, 1), keepdims=True)

        # fixing areas with norm of 0
        norms[norms == 0] = 1

        self.gene_embeddings = self.gene_embeddings / (norms)

        self.gene_embeddings = (
            torch.tensor(self.gene_embeddings).type(torch.float32).to(self._device)
        )

        self.mask = torch.sum(self.gene_embeddings, dim=1, keepdims=True) != 0

        if config["learnable_gene_embedding"]:
            self.learnable_embedding = nn.Embedding(
                self.gene_embeddings.shape[0], config["item_hidden_size"]
            )

        ngenes, _, n_pathways = self.gene_embeddings.shape

        self.activation = nn.LeakyReLU()
        self.temperature = config["softmax_temperature"]

        # Map from raw gene embeddings to aligned gene embeddings
        # pathway encoder
        config["num_heads"] = 4  # hidden_dim must be divisible by num_heads
        pathway_encoder_function = "MLP"  # MLP MOE MHA

        self.pathway_encoder = GeneEmbeddingTransformation(
            self.gene_embeddings, config, pathway_encoder_function
        )

        # Pool type to perturbation latent space
        config["num_genes"] = self.pca_dim or self.num_genes  # self.num_genes
        self.pool = Pool(n_pathways, config)
        # self.pool.KG_weights = None

        # map from latent to output (logFC)
        self.item_net = ItemNet("MLP", self.pca_dim or self.num_genes, config)

    def forward_to_emb(self, locs_gene, locs_combos):
        """Override to handle combo perturbations by summing embeddings."""
        # Get the gene embeddings for the perturbed genes
        emb = self.gene_embeddings[locs_gene, :, :]
        mask = self.mask[locs_gene, :, :]
        
        # Map KG specific embeddings to KG shared latent space
        emb_h = self.pathway_encoder(emb)
        emb_h = self.activation(emb_h)
        
        # Handle combo perturbations by aggregating embeddings for the same sample
        # Get unique sample indices
        unique_combos, inverse_indices = torch.unique(locs_combos, return_inverse=True)
        
        # Aggregate embeddings for each unique sample
        aggregated_emb_h = []
        aggregated_mask = []
        
        for i, combo_idx in enumerate(unique_combos):
            # Find all embeddings for this sample
            sample_mask = (locs_combos == combo_idx)
            sample_embeddings = emb_h[sample_mask]
            sample_masks = mask[sample_mask]
            
            # Sum embeddings for combo perturbations
            # This models the combined effect as additive in embedding space
            summed_embedding = sample_embeddings.sum(dim=0, keepdim=True)
            summed_mask = sample_masks.sum(dim=0, keepdim=True)
            
            aggregated_emb_h.append(summed_embedding)
            aggregated_mask.append(summed_mask)
        
        # Concatenate all aggregated embeddings
        emb_h_aggregated = torch.cat(aggregated_emb_h, dim=0)
        mask_aggregated = torch.cat(aggregated_mask, dim=0)
        
        # Normalize mask to 0 or 1
        mask_aggregated = (mask_aggregated > 0).float()
        
        # Store for potential visualization/debugging
        self.emb_h = emb_h_aggregated
        
        # Create new locs_combos that matches the aggregated embeddings
        # This ensures the pool function gets the right indices
        new_locs_combos = torch.arange(len(unique_combos), device=locs_combos.device)
        
        # Pool to latent space
        emb_h_final = self.pool(emb_h_aggregated, new_locs_combos, mask_aggregated)
        
        # Handle pool attributes
        if hasattr(self.pool.pool, "p_weight_vec"):
            self.pathway_weight_vector = self.pool.pool.p_weight_vec
        if hasattr(self.pool.pool, "attention_weights"):
            self.attention_weights = self.pool.pool.attention_weights
        
        return emb_h_aggregated, emb_h_final


    def emb_to_out(self, emb_h, locs_gene, locs_combos):

        # transformation to output dimensions uses ItemNet class
        out = self.item_net(emb_h)
        
        if self.pca_dim is not None:
            out = self.pca.inverse_transform(out)


        return out
    
    def forward(self, pert_inds, cov=None, update_node_embeddings=True):
        """Override forward to handle combo perturbations properly."""
        self.pert_inds = pert_inds
        locs = torch.nonzero(pert_inds)
        
        # indices of perturbed genes
        locs_gene = locs[:, 1]
        # indices of perturbations within a batch
        locs_combos = locs[:, 0]
        
        self.locs_gene = locs_gene
        self.locs_combos = locs_combos
        
        # Get embeddings with combo handling
        emb_h_temp, emb_h = self.forward_to_emb(locs_gene, locs_combos)
        
        # For emb_to_out, we need to pass the aggregated indices
        # Create dummy locs that match the aggregated embeddings
        unique_combos = torch.unique(locs_combos)
        dummy_locs_gene = torch.zeros_like(unique_combos)
        dummy_locs_combos = torch.arange(len(unique_combos), device=locs_combos.device)
        
        # Latent space to output dimension
        out = self.emb_to_out(emb_h, dummy_locs_gene, dummy_locs_combos)
        
        return out, emb_h_temp, "None"
    



    def compute_loss(self, pred, expr, tensor, pred_clust=None, expr_clust=None):
        """Compute standard MSE loss."""
        residual2 = (pred - expr) ** 2
        loss = residual2.mean()
        return loss

class CustomProgressBar(TQDMProgressBar):
    """Enhanced progress bar with better formatting."""
    
    def get_metrics(self, trainer, model):
        # Get the default metrics
        items = super().get_metrics(trainer, model)
        
        # Format losses with more decimal places and compact display
        if 'train_loss' in items:
            items['train_loss'] = f"{items['train_loss']:.6f}"
        if 'val_loss' in items:
            items['val_loss'] = f"{items['val_loss']:.6f}"
        
        return items

class CustomEarlyStopping(EarlyStopping):
    """Custom early stopping that displays losses with more precision."""
    
    def _improvement_message(self, current: float) -> str:
        """Generate an improvement message with more decimal places."""
        if self.best_score is None:
            return f"Metric {self.monitor} improved. New best score: {current:.6f}"
        else:
            improvement = abs(self.best_score - current)
            return (f"Metric {self.monitor} improved by {improvement:.6f} >= "
                   f"min_delta = {self.min_delta:.6f}. New best score: {current:.6f}")

class CellSimBenchModelHarness(ModelHarness):
    """Custom ModelHarness that skips evaluator initialization for CellSimBench."""
    
    def __init__(self, module, datamodule, config, encoder=None, decoder=None):
        # Initialize base Lightning module
        pl.LightningModule.__init__(self)
        
        # Copy necessary initialization from ModelHarness
        self.module = module
        self.module.current_batch = 0
        self.var_names = datamodule.var_names
        self.degs = datamodule.degs
        self.config = config
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_step_outputs = []
        self.test_set_keys = getattr(datamodule, "test_set_keys", [""])
        self.encoder = encoder
        self.decoder = decoder
        
        datamodule.encoder = encoder
        self.do_test_eval = False  # Disable test evaluation
        
        # Skip evaluator initialization - not needed for CellSimBench
        self.evaluator = None
        self.second_evaluator = None
        
        # Always set train_perturb_labels to None to avoid classification paths
        self.train_perturb_labels = None
        
        # Initialize model tracking attributes
        self.all_embh = []
        self.all_coef = []
        self.attention_weights = []
        self.transformed_embs = []
        self.all_locs_gene = []
        self.all_locs_ind = []
    
    def _step(self, batch, batch_idx):
        """Standard training step."""
        src, cov, tgt = self.unpack_batch(batch)
        
        pred, tensor, pred_clust = self(src, cov)
        loss = self.module.compute_loss(pred, tgt, tensor)
        return loss, None, None, src, pred
    
    def training_step(self, batch, batch_idx):
        """Custom training step with better logging."""
        self.module.current_batch += 1
        loss, ce_loss, accuracy, src, pred = self._step(batch, batch_idx)
        
        # Log with progress bar and on_step for real-time updates
        self.log("train_loss", loss, 
                prog_bar=True,   # Show in progress bar
                on_step=True,    # Log at each step
                on_epoch=True)   # Also log epoch average
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Custom validation step with better logging."""
        loss, ce_loss, accuracy, src, pred = self._step(batch, batch_idx)
        
        # Log validation loss with sync_dist for multi-GPU
        self.log("val_loss", loss,
                prog_bar=True,   # Show in progress bar
                on_step=False,   # Don't log each step
                on_epoch=True,   # Log epoch average
                sync_dist=True)  # Sync across devices if needed
        
        self.validation_step_outputs.append(
            {"src": src, "tgt": batch["expr"], "pred": pred}
        )
        return loss
    
    def on_validation_epoch_end(self):
        """Override to skip evaluator calls."""
        self.module.current_batch = 0
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Override to skip evaluator calls."""  
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Override to handle predictions."""
        src, cov, tgt = self.unpack_batch(batch)
        preds, tensor, pred_clust = self(src, cov)
        
        # Store embeddings and other info for visualization
        if hasattr(self.module, "emb_h"):
            self.all_embh.append(self.module.emb_h.detach().cpu().numpy())
        self.all_locs_gene.append(self.module.locs_gene.detach().cpu().numpy())
        self.all_locs_ind.append(self.module.locs_combos.detach().cpu().numpy())
        
        if hasattr(self.module, "pathway_weight_vector"):
            self.pathway_weight_vector = self.module.pathway_weight_vector
        if hasattr(self.module, "attention_weights"):
            self.attention_weights.append(self.module.attention_weights)
        
        self.transformed_embs.append(self.module.emb_h.cpu().numpy())
        
        if preds is not None:
            if self.decoder is not None:
                preds = self.decoder(preds.cpu())
            
            # Get keys from the batch - REQUIRED
            if 'pert_key' not in batch:
                raise KeyError("Batch MUST contain 'pert_key' with perturbation format")
            
            # Extract the perturbation keys from the batch
            keys = batch['pert_key']
            # Convert to numpy array if it's a list
            if isinstance(keys, list):
                keys = np.array(keys)
            
            return keys, preds

class PRESAGEWrapper:
    """Main wrapper class for PRESAGE model integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Set up paths
        self.data_path = config['data_path']
        # TODO: Implement model path for continuing training in the future
        self.model_path = config.get('model_path', None)
        
        # Parse hyperparameters
        self.hyperparams = config['hyperparameters']
        
        # Create DataManager instance like GEARS does
        self.data_manager = DataManager(self.config)
        self.data_manager.load_dataset()
        
        # Get seed from config or use default
        self.seed = self.hyperparams['seed']
        
        # Set random seed for reproducibility using both methods
        set_seed(self.seed)  # PRESAGE's comprehensive seed setting
        seed_everything(self.seed, workers=True)  # PyTorch Lightning's seed setting
        
        log.info(f"Initialized PRESAGE wrapper with seed={self.seed}")
    
    def train(self):
        """Train PRESAGE model using PyTorch Lightning."""
        log.info("Starting PRESAGE training...")
        
        # Re-set seeds at the start of training for reproducibility
        set_seed(self.seed)
        seed_everything(self.seed, workers=True)
        log.info(f"Training with seed={self.seed}")
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Training output directory: {self.output_dir}")
        

        
        # 1. Load and convert data
        adata = self._load_and_convert_data()
        
        # 2. Create PRESAGE data module
        datamodule = self._create_datamodule(adata)
        
        # 3. Build model configuration
        model_config = self._build_model_config(datamodule)
        
        # 4. Initialize model with combo support
        model = ComboPRESAGE(model_config, datamodule,
                                input_dimension=len(datamodule.train_dataset.adata.var),
                                output_dimension=len(datamodule.train_dataset.adata.var))
        
        log.info("Using ComboPRESAGE with combo support")

        
        # 5. Create model harness for training
        harness = CellSimBenchModelHarness(model, datamodule, model_config)
        
        # 6. Set up trainer
        trainer = self._create_trainer()
        
        # 7. Train model
        trainer.fit(harness, datamodule)
        
        # 8. Save model and metadata
        self._save_training_artifacts(harness, model_config)
        
        log.info("Training completed successfully")
    
    def predict(self):
        """Generate predictions using trained PRESAGE model."""
        log.info("Starting PRESAGE prediction...")
        
        # 1. Load trained model
        harness, datamodule, control_mean, perts_as_delta_ref = self._load_trained_model()
        
        # 2. Get test conditions
        test_conditions = self.config['test_conditions']
        
        # 3. Generate predictions (Note: _generate_predictions loads data internally)
        predictions_adata = self._generate_predictions(harness, datamodule, control_mean, 
                                                      perts_as_delta_ref, test_conditions)
        
        if predictions_adata is None:
            raise RuntimeError("Failed to generate predictions")
        
        # 4. Save predictions
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)
        
        log.info("Prediction completed successfully")
    
    def _load_and_convert_data(self):
        """Load and convert CellSimBench data to PRESAGE format."""
        log.info(f"Loading data from {self.data_path}")
        adata = sc.read_h5ad(self.data_path)
        
        # Convert to PRESAGE format
        adata = self._convert_to_presage_format(adata)
        
        log.info(f"Loaded and converted data: {adata.shape}")
        return adata
    
    def _convert_to_presage_format(self, adata_raw):
        """Convert CellSimBench format to PRESAGE format."""
        adata = adata_raw.copy()

        # If 'perturbation' is already a column, drop it
        if 'perturbation' in adata.obs.columns and 'condition' in adata.obs.columns:
            adata.obs.drop(columns=['perturbation'], inplace=True)
        
        # Rename condition column if needed
        if 'condition' in adata.obs.columns:
            adata.obs['perturbation'] = adata.obs['condition'].copy()
        
        # Convert perturbation names: "GENE1+GENE2" -> "GENE1_GENE2" 
        adata.obs["perturbation"] = adata.obs["perturbation"].apply(
            lambda x: x.replace("+", "_").replace("ctrl_iegfp", "control").replace("ctrl", "control")
        )
        
        # Add nperts column
        # Calculate number of perturbations per cell using vectorized operations
        perturbations = adata.obs["perturbation"].astype(str)
        
        # Create boolean mask for control conditions
        is_control = perturbations.isin(["control", "ctrl", "ctrl_iegfp", "nan"])
        
        # Count underscores for non-control conditions
        underscore_counts = perturbations.str.count("_")
        
        # Assign nperts: 0 for controls, 1 + underscore_count for others
        adata.obs["nperts"] = np.where(is_control, 0, 1 + underscore_counts)
        
        # Ensure gene names are set correctly
        if 'gene_name' not in adata.var.columns:
            adata.var['gene_name'] = adata.var.index.values
        
        # Make var names unique and reset index
        adata.var.index.name = None
        adata.var = adata.var.reset_index().set_index("gene_name")
        adata.var_names_make_unique()
        
        # Ensure X is dense array
        if hasattr(adata.X, 'toarray'):
            adata.X = adata.X.toarray()
        
        adata.X = adata.X.astype(np.float32)
        
        log.info("Converted data to PRESAGE format")
        return adata
    
    def _create_datamodule(self, adata):
        """Create custom CellSimBench data module for PRESAGE."""
        # Create output directory structure
        data_dir = self.output_dir / "presage_data"
        data_dir.mkdir(exist_ok=True)
        
        # Save processed data 
        processed_path = data_dir / "cellsimbench_processed.h5ad"
        print(f"Saving processed data to {processed_path}...")
        adata.write_h5ad(processed_path)
        
        # Create splits from config
        splits = {'train': [], 'val': [], 'test': []}
        
        for split_name, conditions in [
            ('train', self.config['train_conditions']),
            ('val', self.config['val_conditions']),
            ('test', self.config['test_conditions'])
        ]:
            for cond in conditions:
                # Convert condition name to PRESAGE format
                presage_cond = cond.replace("+", "_").replace("ctrl", "control")
                splits[split_name].append(presage_cond)
        
        # Save splits
        splits_path = data_dir / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f)
        

        
        # Create datamodule config
        datamodule_config = {
            'processed_adata_path': str(processed_path),
            'splits_path': str(splits_path),
            'dataset': 'cellsimbench',
            'data_dir': str(data_dir),
            'batch_size': self.hyperparams['batch_size'],
            'use_pseudobulk': True,
            'preprocessing_zscore': False,
            'noisy_pseudobulk': False,
            'perturb_field': 'perturbation',
            'control_key': 'control',
            'split_name': self.config['split_name'],
            'seed': self.seed,  # Pass seed for reproducibility
            'perts_as_delta_ref': self.hyperparams['perts_as_delta_ref']  # Required flag for delta reference
        }
        
        # Create our custom datamodule
        datamodule = CellSimBenchDataModule.from_config(datamodule_config)
        
        # Prepare and setup data
        datamodule.prepare_data()
        datamodule.setup("fit")
        
        log.info(f"Created CellSimBench datamodule with batch_size={self.hyperparams['batch_size']}")
        return datamodule
    
    def _build_model_config(self, datamodule):
        """Build model configuration from hyperparameters."""
        # PRESAGE hardcodes cache to ./cache/pathway_embeddings/
        # We need to ensure the cache exists there
        cache_path = "./cache"
        
        # Get path to PRESAGE sample files
        import os
        if os.path.exists('/presage_src/sample_files'):
            # Docker environment
            sample_files_path = '/presage_src'
        else:
            # Local environment
            sample_files_path = str(Path(__file__).parent / 'ref' / 'PRESAGE')
        
        # Create a temporary pathway file with corrected absolute paths
        original_pathway_file = f'{sample_files_path}/sample_files/prior_files/sample.knowledge_experimental.txt'
        temp_pathway_file = self.output_dir / 'pathway_files_absolute.txt'
        
        with open(original_pathway_file, 'r') as f_in:
            lines = f_in.readlines()
        
        # Replace relative paths with absolute paths to our cache
        # PRESAGE expects cache at ./cache/ relative to where it runs
        corrected_lines = []
        for line in lines:
            line = line.strip()
            if line and line.startswith('../cache/'):
                # Replace ../cache/ with ./cache/ since PRESAGE uses ./cache/
                corrected_path = line.replace('../cache/', './cache/')
                corrected_lines.append(corrected_path + '\n')
            else:
                corrected_lines.append(line + '\n' if line else '\n')
        
        with open(temp_pathway_file, 'w') as f_out:
            f_out.writelines(corrected_lines)
        
        log.info(f"Created temporary pathway file with corrected paths: {temp_pathway_file}")
        
        config = {
            # Architecture parameters
            'item_hidden_size': self.hyperparams['item_hidden_size'],
            'item_nlayers': self.hyperparams['item_nlayers'],
            'pathway_item_hidden_size': self.hyperparams['pathway_item_hidden_size'],
            'pathway_item_nlayers': self.hyperparams['pathway_item_nlayers'],
            
            # Pooling configuration
            'pathway_pool_type': self.hyperparams['pathway_pool_type'],
            'pathway_weight_type': self.hyperparams['pathway_weight_type'],
            'pool_nlayers': self.hyperparams['pool_nlayers'],
            'softmax_temperature': self.hyperparams['softmax_temperature'],
            'gat_weight': self.hyperparams['gat_weight'],
            
            # Embeddings
            'n_nmf_embedding': self.hyperparams['n_nmf_embedding'],
            
            # Knowledge source files (use temp file with absolute paths)
            'pathway_files': str(temp_pathway_file),
            'embedding_files': 'None',  # No embedding file provided
            
            # Pre-computed embeddings from cache - PRESAGE expects them at ./cache/pathway_embeddings/
            'pathway_embedding_dimension': 5,
            'pathway_embedding_files': './cache/pathway_embeddings/',
            
            # Node2Vec parameters
            'node2vec_walk_length': self.hyperparams['node2vec_walk_length'],
            'node2vec_context_size': self.hyperparams['node2vec_context_size'],
            'node2vec_walks_per_node': self.hyperparams['node2vec_walks_per_node'],
            'node2vec_num_negative_samples': self.hyperparams['node2vec_num_negative_samples'],
            'node2vec_p': self.hyperparams['node2vec_p'],
            'node2vec_q': self.hyperparams['node2vec_q'],
            'node2vec_batchsize': self.hyperparams['batch_size'],
            
            # Loss scales
            'mse_loss_scale': self.hyperparams['mse_loss_scale'],
            'cosine_loss_scale': self.hyperparams['cosine_loss_scale'], 
            'vector_norm_loss_scale': self.hyperparams['vector_norm_loss_scale'],
            
            # Other required parameters
            'batch_norm': True,
            'learnable_gene_embedding': False,
            'pca_dim': None,
            'input_preparation': 'prep_gene_embeddings',
            'dim_red_alg': 'Node2Vec',
            'n_neigh_prune': '5',
            'dataset': 'cellsimbench',
            'source': 'cellsimbench',
            'num_heads': 4,  # For attention mechanism
            
            # Optimizer parameters (required by ModelHarness)
            'lr': 2.4e-3,
            'weight_decay': 2.17e-13,
            'momentum': 0.9,
            'optimizer': 'Adam'
        }
        
        log.info("Built model configuration with pre-computed knowledge embeddings")
        return config
    
    def _create_trainer(self):
        """Create PyTorch Lightning trainer."""
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename='presage-{epoch:02d}-{val_loss:.6f}',  # More decimal places
            monitor='val_loss',
            save_top_k=1,
            mode='min'
        )
        
        early_stop_callback = CustomEarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=True,
            mode='min'
        )
        
        progress_bar = CustomProgressBar(refresh_rate=10)  # Use custom progress bar
        
        trainer = pl.Trainer(
            max_epochs=self.hyperparams['max_epochs'],
            callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,  # No logger
            enable_checkpointing=True,
            deterministic=True,  # Ensures reproducible results
            benchmark=False  # Disable cudnn.benchmark for reproducibility
        )
        
        log.info(f"Created trainer with max_epochs={self.hyperparams['max_epochs']}")
        return trainer
    
    def _save_training_artifacts(self, harness, model_config):
        """Save trained model and metadata."""
        # Save model state
        model_path = self.output_dir / "trained_model.ckpt"
        torch.save(harness.state_dict(), model_path)
        
        # Save model config
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save the temporary pathway file (needed for prediction)
        pathway_file_path = self.output_dir / "pathway_files_absolute.txt"
        if pathway_file_path.exists():
            log.info(f"Saved pathway file: {pathway_file_path}")
        
        # Copy the cache that PRESAGE created at ./cache/ to model output directory
        # PRESAGE always creates embeddings at ./cache/pathway_embeddings/
        cache_source = Path("./cache")
        cache_dest = self.output_dir / "cache"
        
        if cache_source.exists():
            log.info(f"Copying PRESAGE cache from {cache_source} to {cache_dest}...")
            cache_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cache_source, cache_dest, dirs_exist_ok=True)
            log.info("PRESAGE cache copied successfully")
        else:
            log.warning(f"PRESAGE cache not found at {cache_source}")
        
        # Save control mean (needed to convert deltas back to absolute expression)
        if not hasattr(harness.module.datamodule.train_dataset, 'control_mean'):
            raise RuntimeError("Training dataset missing control_mean")
            
        control_mean = harness.module.datamodule.train_dataset.control_mean
        if control_mean is None:
            raise RuntimeError("control_mean is None")
            
        control_mean_path = self.output_dir / "control_mean.csv"
        control_mean.to_csv(control_mean_path)
        log.info("Saved control mean")
        
        # Save perturbation mean if using perts_as_delta_ref
        perturbation_mean_path = None
        if self.hyperparams['perts_as_delta_ref']:
            if not hasattr(harness.module.datamodule.train_dataset, 'perturbation_mean'):
                raise RuntimeError("Training dataset missing perturbation_mean")
                
            perturbation_mean = harness.module.datamodule.train_dataset.perturbation_mean
            if perturbation_mean is None:
                raise RuntimeError("perturbation_mean is None")
                
            perturbation_mean_path = self.output_dir / "perturbation_mean.csv"
            perturbation_mean.to_csv(perturbation_mean_path)
            log.info("Saved perturbation mean")
        
        # Save the processed data path and splits for prediction
        processed_data_path = self.output_dir / "presage_data" / "cellsimbench_processed.h5ad"
        splits_path = self.output_dir / "presage_data" / "splits.json"
        
        # Save training checkpoint metadata with all necessary paths
        checkpoint_path = self.output_dir / "presage_training_hparams.json"
        checkpoint_data = {
            'model_config': model_config,
            'hyperparameters': self.hyperparams,
            'data_path': str(self.data_path),
            'split_name': self.config['split_name'],
            'processed_data_path': str(processed_data_path),
            'splits_path': str(splits_path),
            'pathway_file': str(pathway_file_path),
            'cache_dir': str(cache_dest),  # Where we saved the cache
            'control_mean_path': str(control_mean_path),  # Control mean file
            'perts_as_delta_ref': self.hyperparams['perts_as_delta_ref'],  # Flag for delta reference type
            'training_completed': True
        }
        
        # Add perturbation mean path if using perts_as_delta_ref
        if self.hyperparams['perts_as_delta_ref'] and perturbation_mean_path:
            checkpoint_data['perturbation_mean_path'] = str(perturbation_mean_path)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, cls=PathEncoder)
        
        log.info(f"Saved training artifacts to {self.output_dir}")
    
    def _load_trained_model(self):
        """Load trained model for prediction."""
        if not self.model_path:
            raise ValueError("model_path must be provided for prediction mode")
        
        model_path = Path(self.model_path)
        
        # Load training checkpoint with all metadata
        checkpoint_path = model_path / "presage_training_hparams.json"
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load model config
        config_path = model_path / "model_config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Restore cache to ./cache/ where PRESAGE expects it
        # PRESAGE is hardcoded to use ./cache/pathway_embeddings/
        cache_source = Path(checkpoint_data['cache_dir'].replace("/model_output/", "/pretrained_model/"))
        cache_dest = Path("./cache")
        
        if cache_source.exists():
            log.info(f"Restoring PRESAGE cache from {cache_source} to {cache_dest}...")
            cache_dest.parent.mkdir(parents=True, exist_ok=True)
            # Remove existing cache if it exists to avoid conflicts
            if cache_dest.exists():
                shutil.rmtree(cache_dest)
            shutil.copytree(cache_source, cache_dest)
            log.info("PRESAGE cache restored to ./cache/ successfully")
        else:
            raise FileNotFoundError(f"Cache not found at {cache_source}")
        
        # Load control mean (needed to convert deltas back to absolute expression)
        if 'control_mean_path' not in checkpoint_data:
            raise KeyError("checkpoint_data missing control_mean_path")
        if 'perts_as_delta_ref' not in checkpoint_data:
            raise KeyError("checkpoint_data missing perts_as_delta_ref")
            
        control_mean_path = Path(checkpoint_data['control_mean_path'].replace("/model_output/", "/pretrained_model/"))
        if not control_mean_path.exists():
            raise FileNotFoundError(f"Control mean file not found: {control_mean_path}")
            
        control_mean = pd.read_csv(control_mean_path, index_col=0)
        log.info("Loaded control mean")
        
        # Load perturbation mean if using perts_as_delta_ref
        perturbation_mean = None
        if checkpoint_data['perts_as_delta_ref']:
            if 'perturbation_mean_path' not in checkpoint_data:
                raise KeyError("checkpoint_data missing perturbation_mean_path when perts_as_delta_ref=True")
                
            perturbation_mean_path = Path(checkpoint_data['perturbation_mean_path'].replace("/model_output/", "/pretrained_model/"))
            if not perturbation_mean_path.exists():
                raise FileNotFoundError(f"Perturbation mean file not found: {perturbation_mean_path}")
                
            perturbation_mean = pd.read_csv(perturbation_mean_path, index_col=0)
            log.info("Loaded perturbation mean")
        
        # Load the processed data from training
        processed_data_path = Path(checkpoint_data['processed_data_path'].replace("/model_output/", "/pretrained_model/"))
        log.info(f"Loading processed data from {processed_data_path}")
        assert processed_data_path.exists()
        
        # Recreate the datamodule with the same configuration as training
        splits_path = Path(checkpoint_data['splits_path'].replace("/model_output/", "/pretrained_model/"))
        
        # Update model_config to use the saved pathway file
        model_config['pathway_files'] = checkpoint_data['pathway_file'].replace("/model_output/", "/pretrained_model/")
        
        # Create datamodule config matching training
        datamodule_config = {
            'processed_adata_path': str(processed_data_path),
            'splits_path': str(splits_path),
            'dataset': 'cellsimbench',
            'data_dir': str(model_path / "presage_data"),
            'batch_size': self.hyperparams['batch_size'],
            'use_pseudobulk': True,
            'preprocessing_zscore': False,
            'noisy_pseudobulk': False,
            'perturb_field': 'perturbation',
            'control_key': 'control',
            'split_name': checkpoint_data['split_name'],
            'seed': self.seed,
            'perts_as_delta_ref': checkpoint_data['perts_as_delta_ref']  # Required flag for delta reference
        }
        
        # Create datamodule
        datamodule = CellSimBenchDataModule.from_config(datamodule_config)
        
        # Prepare and setup data (important for initializing datasets)
        datamodule.prepare_data()
        datamodule.setup("fit")  # Setup for fit first to initialize train dataset
        
        # Initialize ComboPRESAGE model with the same configuration
        # ComboPRESAGE is needed for combo perturbation support
        model = ComboPRESAGE(
            model_config, 
            datamodule,
            input_dimension=len(datamodule.train_dataset.adata.var),
            output_dimension=len(datamodule.train_dataset.adata.var)
        )
        
        log.info("Initialized model for loading")
        
        # Create model harness
        harness = CellSimBenchModelHarness(model, datamodule, model_config)
        
        # Load the saved model weights
        model_weights_path = model_path / "trained_model.ckpt"
        log.info(f"Loading model weights from {model_weights_path}")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        # Load state dict with appropriate device mapping
        state_dict = torch.load(model_weights_path, map_location=device)
        harness.load_state_dict(state_dict)
        
        # Move model to device
        harness.to(device)
        
        # Set to evaluation mode
        harness.eval()
        
        log.info("Model loaded successfully for prediction")
        return harness, datamodule, control_mean, checkpoint_data['perts_as_delta_ref']
    
    def _generate_predictions(self, harness, datamodule, control_mean, 
                             perts_as_delta_ref, test_conditions):
        """Generate predictions for test conditions."""
        import torch
        from torch.utils.data import DataLoader
        
        # Load the full test data (we need it for creating proper test dataset)
        test_data_path = self.config['data_path']
        full_adata = sc.read_h5ad(test_data_path)
        
        # Convert to PRESAGE format first
        full_adata = self._convert_to_presage_format(full_adata)
        
        # Convert test conditions to PRESAGE format
        presage_test_conditions = []
        for cond in test_conditions:
            presage_cond = cond.replace("+", "_").replace("ctrl", "control")
            presage_test_conditions.append(presage_cond)
        
        log.info(f"Generating predictions for {len(presage_test_conditions)} test conditions")
        
        # Update the splits to include only our test conditions
        with open(datamodule.splits_json_path, 'r') as f:
            original_splits = json.load(f)
        
        # Create new splits with our test conditions
        test_splits = {
            'train': original_splits['train'],  # Keep original train
            'val': original_splits['val'],      # Keep original val
            'test': presage_test_conditions     # Use our test conditions
        }
        
        # Save temporary test splits
        output_dir = Path(self.config['output_path']).parent
        temp_splits_path = output_dir / "test_splits.json"
        with open(temp_splits_path, 'w') as f:
            json.dump(test_splits, f)
        
        # Update datamodule's split path for test
        datamodule.split_path = str(temp_splits_path)
        datamodule.splits = test_splits
        
        # Setup datamodule for test stage
        datamodule._data_setup = False  # Force re-setup
        datamodule.setup("test")
        
        # Get the test dataloader
        test_loader = datamodule.test_dataloader()
        
        # Run predictions
        log.info("Running model inference...")
        harness.eval()
        
        all_keys = []
        all_predictions = []
        
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device if CUDA is available
                device = next(harness.parameters()).device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Run predict_step (following ModelHarness.predict_step logic)
                keys, preds = harness.predict_step(batch, batch_idx)
                
                # Collect results
                all_keys.extend(keys)
                if preds is not None:
                    all_predictions.append(preds.cpu().numpy())
        
        # Concatenate all predictions
        if all_predictions:
            all_predictions = np.concatenate(all_predictions, axis=0)
        else:
            log.error("No predictions generated!")
            raise ValueError("No predictions generated!")
        
        # Create DataFrame with predictions (these are still deltas)
        pred_df = pd.DataFrame(
            data=all_predictions,
            index=pd.Index(all_keys, name="perturbation"),
            columns=datamodule.var_names
        )
        
        # Each row is a unique perturbation
        log.info(f"Generated predictions for {len(pred_df)} perturbations")
        
        # Filter to only measured genes (remove the fake genes added for missing perturbations)
        if hasattr(datamodule.train_dataset.adata.var, 'measured_gene'):
            measured_genes = datamodule.train_dataset.adata.var.measured_gene
            pred_df = pred_df.loc[:, measured_genes]
        
        # IMPORTANT: Convert predictions from deltas to absolute expression
        # PRESAGE predicts deltas, so we need to add back the reference mean
        if perts_as_delta_ref:
            # This would use perturbation mean as reference, but for now we don't have it
            # TODO: Need to implement perturbation mean logic
            log.warning("perts_as_delta_ref=True but perturbation mean logic not implemented yet")
            reference_mean = control_mean
        else:
            log.info("Converting predictions using control mean")
            reference_mean = control_mean

        # Apply reference mean to all predictions
        common_genes = pred_df.columns.intersection(reference_mean.columns)
        if len(common_genes) != len(pred_df.columns):
            raise ValueError(f"Gene mismatch: prediction has {len(pred_df.columns)} genes, reference mean has {len(common_genes)} common genes")
            
        reference_mean_values = reference_mean[common_genes].values.flatten()
        pred_df.loc[:, common_genes] = pred_df.loc[:, common_genes] + np.float32(reference_mean_values)
        
        # Create output AnnData with predictions
        # Get the test samples from the original data
        test_perturbations = presage_test_conditions.copy()
        
        # Add control to the list and make unique
        test_perturbations.append('control')
        test_perturbations = list(set(test_perturbations))  # Remove duplicates
        
        # Create mask using just the perturbation names
        test_mask = full_adata.obs['perturbation'].isin(test_perturbations)
        test_adata = full_adata[test_mask].copy()
                
        # Create pseudobulk version for output
        pred_df.index.name = None
        output_adata = sc.AnnData(
            X=pred_df.values,
            obs=pd.DataFrame(index=pred_df.index),
            var=test_adata.var[test_adata.var.index.isin(pred_df.columns)].copy()
        )
        
        # Create perturbation and condition columns from the index
        conditions = []
        for key in output_adata.obs.index:
            # Convert perturbation back to CellSimBench format
            condition = key.replace("_", "+").replace("control", "ctrl")
            conditions.append(condition)
        
        output_adata.obs['condition'] = conditions
        output_adata.obs['perturbation'] = output_adata.obs['condition']  # Keep for compatibility
        
        # Add any additional metadata
        output_adata.obs['model'] = 'presage'
        output_adata.obs['is_control'] = output_adata.obs['condition'] == 'ctrl'
        
        log.info(f"Created output AnnData with shape {output_adata.shape}")
        
        return output_adata
