"""
GEARS model wrapper for CellSimBench integration.
Handles training and prediction with internal checkpointing.
"""

import logging
import pickle
import json
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List
from gears import PertData
import os
from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager

log = logging.getLogger(__name__)


# ==========================================================================================================================================================
# =================== BASIC GEARS IMPLEMENTATION ===================
# ==========================================================================================================================================================

from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.seed import seed_everything

from gears.model import GEARS_Model
from gears.inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis
from gears.utils import uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params
from torch_geometric.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
seed_everything(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',
                 loss_weights_dict = None,
                 use_mse_loss = False):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'
        loss_weights: dict
            Dictionary of loss weights for each loss function. Default: None
        use_mse_loss: bool
            Whether to use MSE loss. Default: False

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.gene2go = pert_data.gene2go
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        self.loss_weights_dict = loss_weights_dict
        self.use_mse_loss = use_mse_loss

        self.ctrl_expression = torch.tensor(
            np.mean(self.adata[self.adata.obs.condition == 'ctrl'].X,
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               gene2go=self.gene2go,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               gene2go=self.gene2go,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on GEARS
        predictions

        """

        ## if uncertainty mode is on, also return uncertainty score.
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file = None):
        """
        Plot the perturbation graph

        Parameters
        ----------
        query: str
            condition to be queried
        save_file: str
            path to save the plot

        Returns
        -------
        None

        """

        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        
        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    
    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    print(pred.shape, logvar.shape, y.shape)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'],
                                  loss_weights_dict = self.loss_weights_dict,
                                  use_mse_loss = self.use_mse_loss)
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
                             
            # Print Pearson correlation metrics for overall
            log = "Train Overall Pearson: {:.4f} " \
                  "Validation Overall Pearson: {:.4f}. "
            print_sys(log.format(train_metrics['pearson'],
                             val_metrics['pearson']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
                             
            # Print Pearson correlation metrics for DE genes
            log = "Train Top 20 DE Pearson: {:.4f} " \
                  "Validation Top 20 DE Pearson: {:.4f}. "
            print_sys(log.format(train_metrics['pearson_de'],
                             val_metrics['pearson_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        
        print_sys('Done!')


def loss_fct(pred, y, perts, ctrl = None, direction_lambda = 1e-3, dict_filter = None, loss_weights_dict = None, use_mse_loss = False):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions
        loss_weights_dict (dict): dictionary of loss weights for each perturbation
        use_mse_loss (bool): whether to use MSE loss

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        if loss_weights_dict is not None:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            weights = loss_weights_dict[p]
            weights = torch.tensor(weights).to(pred.device)
            weights = weights[:pred_p.shape[1]]
        else:
            # during training, we remove the all zero genes into calculation of loss.
            # this gives a cleaner direction loss. empirically, the performance stays the same.
            if p!= 'ctrl':
                retain_idx = dict_filter[p]
                pred_p = pred[pert_idx][:, retain_idx]
                y_p = y[pert_idx][:, retain_idx]
            else:
                pred_p = pred[pert_idx]
                y_p = y[pert_idx]
            weights = torch.ones(pred_p.shape[1]).to(pred.device)
        
        if not use_mse_loss:
            losses = losses + torch.sum(weights * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(weights * (pred_p - y_p)**2)/pred_p.shape[0]/pred_p.shape[1]

        ## direction loss
        if not use_mse_loss:
            if (p!= 'ctrl'):
                if loss_weights_dict is not None:
                    losses = losses + torch.sum(weights * direction_lambda *
                                        (torch.sign(y_p - ctrl) -
                                         torch.sign(pred_p - ctrl))**2)/\
                                         pred_p.shape[0]/pred_p.shape[1]
                else:
                    losses = losses + torch.sum(weights * direction_lambda *
                                        (torch.sign(y_p - ctrl[retain_idx]) -
                                         torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                         pred_p.shape[0]/pred_p.shape[1]
            else:
                losses = losses + torch.sum(weights * direction_lambda * (torch.sign(y_p - ctrl) -
                                                torch.sign(pred_p - ctrl))**2)/\
                                                pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))


class GEARSWrapper:
    """GEARS model wrapper handling its own checkpointing and data conversion."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.pert_data = None
        self.gene2go = self._load_gene2go()
        self.data_manager = DataManager(self.config)
        
    def _load_gene2go(self) -> Dict:
        """Load gene2go dictionary from pickle file."""
        gene2go_path = self.config['gene2go_path'] if 'gene2go_path' in self.config else '/app/gene2go_all.pkl'
        try:
            with open(gene2go_path, 'rb') as f:
                gene2go = pickle.load(f)
            log.info(f"Loaded gene2go dictionary with {len(gene2go)} entries")
            return gene2go
        except Exception as e:
            log.error(f"Failed to load gene2go from {gene2go_path}: {e}")
            raise
        
    def train(self):
        """Train GEARS model with internal checkpointing."""
        
        log.info("Starting GEARS training process...")
        
        # 1. Load and preprocess data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")

        # 2. Convert to GEARS format
        log.info("Converting to GEARS format...")
        self.pert_data = self._convert_to_gears_format(adata)
        
        # 3. Prepare splits
        log.info("Preparing data splits...")
        self._prepare_gears_splits()
        
        # 4. Get dataloaders
        log.info("Creating data loaders...")
        self.pert_data.get_dataloader(
            batch_size=self.config['hyperparameters']['batch_size'],
            test_batch_size=self.config['hyperparameters']['test_batch_size']
        )
        
        # 5. Initialize GEARS model
        log.info("Initializing GEARS model...")
        
        # Handle weights - remove covariate prefixes if they exist
        loss_weights_dict = None

        self.model = GEARS(
            self.pert_data, 
            device='cuda',
            weight_bias_track=False,
            loss_weights_dict=loss_weights_dict,
            use_mse_loss=self.config['hyperparameters']['use_mse_loss'],
            proj_name='cellsimbench',
            exp_name='gears_training'
        )
        
        # Initialize model with hyperparameters
        model_params = self._get_model_params()
        self.model.model_initialize(**model_params)
        
        # 6. Check for existing checkpoints
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self._has_checkpoint(checkpoint_dir):
            log.info("Found existing checkpoint, resuming training...")
            self.model.load_pretrained(str(checkpoint_dir))
        
        # 7. Train with GEARS' own checkpointing
        log.info("Starting model training...")
        epochs = self.config['hyperparameters']['epochs']
        lr = self.config['hyperparameters']['lr']
        weight_decay = self.config['hyperparameters']['weight_decay']
        
        self.model.train(epochs=epochs, lr=lr, weight_decay=weight_decay)
        
        # 8. Save final model
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving model to {output_dir}")
        self.model.save_model(str(output_dir))
        self._save_metadata(output_dir)
        
        log.info("Training completed successfully")
        
    def predict(self):
        """Generate predictions using trained GEARS model."""
        
        log.info("Starting GEARS prediction process...")
        
        # 1. Load trained model
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        # 2. Load and preprocess data for prediction
        adata = self.data_manager.load_dataset()
        self.pert_data = self._convert_to_gears_format_for_prediction(adata)
        
        # 3. Get dataloaders
        log.info("Creating data loaders...")
        self.pert_data.get_dataloader(
            batch_size=self.config['hyperparameters']['batch_size'],
            test_batch_size=self.config['hyperparameters']['test_batch_size']
        )
        
        # 4. Initialize GEARS model
        log.info("Initializing GEARS model...")
        
        # Handle weights - remove covariate prefixes if they exist
        loss_weights_dict = None

        
        self.model = GEARS(
            self.pert_data, 
            device='cuda',
            weight_bias_track=False,
            loss_weights_dict=loss_weights_dict,
            use_mse_loss=self.config['hyperparameters']['use_mse_loss'],
            proj_name='cellsimbench',
            exp_name='gears_prediction'
        )
            
        # Initialize model with hyperparameters
        model_params = self._get_model_params()
        self.model.model_initialize(**model_params)
        
        # 5. Load the pre-trained model
        log.info("Loading pre-trained weights...")
        self.model.load_pretrained(str(model_path))
        
        # 6. Generate predictions
        test_conditions = self.config['test_conditions']
        log.info(f"Generating predictions for {len(test_conditions)} conditions")
        
        predictions = self._generate_predictions(test_conditions)
        
        # 7. Convert to CellSimBench format
        log.info("Converting predictions to CellSimBench format...")
        predictions_adata = self._convert_to_cellsimbench_format(predictions, adata)
        
        # 8. Save predictions
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)
        
        log.info("Prediction completed successfully")
        
    def _convert_to_gears_format(self, adata: sc.AnnData) -> PertData:
        """Convert CellSimBench data to GEARS PertData format."""
        
        # Create a copy for GEARS processing
        adata_gears = adata.copy()

        # TODO: We should be passing the control value as a parameter
        CTRL_VALUE = 'ctrl_iegfp'
        
        # Remove all the rows where the condition contains 'ctrl' but is not "control" or CTRL_VALUE
        adata_gears = adata_gears[~adata_gears.obs['condition'].str.contains('ctrl') | adata_gears.obs['condition'].isin(['control', CTRL_VALUE])]
                
        # Process condition labels for GEARS format
        # GEARS expects: 'ctrl', 'GENE1+ctrl', 'GENE1+GENE2'
        def process_condition(cond):
            # TODO: We should be passing the control value as a parameter
            if cond == 'control' or cond == 'ctrl' or cond == CTRL_VALUE:
                return 'ctrl'
            elif '+' not in cond:
                # Single perturbation - add +ctrl
                return f"{cond}+ctrl"
            else:
                # Already in correct format for combo perturbations
                return cond

        adata_gears.obs['condition'] = adata_gears.obs['condition'].astype(str)
        adata_gears.obs['condition'] = adata_gears.obs['condition'].apply(process_condition)
        # TODO: Needs to be removed when doing covariates
        adata_gears.obs['cell_type'] = "NOTHING"
        
        # Use output directory for persistent storage
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving processed GEARS data to: {processed_data_dir}")
        
        # Create standard PertData object
        pert_data = PertData(str(processed_data_dir), default_pert_graph=False, gene2go=self.gene2go)
        
        processed_dataset_path = processed_data_dir / 'cellsimbench_gears'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        if not cell_graphs_file.exists():
            log.info("Creating new processed GEARS dataset...")
            pert_data.new_data_process(dataset_name='cellsimbench_gears', adata=adata_gears)
        else:
            log.info("Loading existing processed GEARS dataset...")
            pert_data.load(data_path=str(processed_dataset_path))
        
        return pert_data
        
    def _convert_to_gears_format_for_prediction(self, adata: sc.AnnData) -> PertData:
        """Convert CellSimBench data to GEARS PertData format for prediction only."""
        
        # Use the same output directory as training for loading processed data
        if 'model_path' in self.config:
            # For prediction, model_path points to the trained model directory
            model_dir = Path(self.config['model_path'])
            processed_data_dir = model_dir / 'processed_data'
        else:
            raise FileNotFoundError(f"Processed GEARS data not found. Please run training first.")
        
        processed_dataset_path = processed_data_dir / 'cellsimbench_gears'
        cell_graphs_file = processed_dataset_path / 'data_pyg' / 'cell_graphs.pkl'
        split_dict_file = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Loading processed GEARS data from: {processed_dataset_path}")
        
        # Create standard PertData object and load existing processed data
        pert_data = PertData(str(processed_data_dir), default_pert_graph=False, gene2go=self.gene2go)
        
        # Always load existing processed data
        if cell_graphs_file.exists():
            log.info("Loading existing processed GEARS data for prediction...")
            pert_data.load(data_path=str(processed_dataset_path))
        else:
            raise FileNotFoundError(
                f"Processed GEARS data not found at '{cell_graphs_file}'. "
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
        
    def _prepare_gears_splits(self):
        """Prepare train/val/test splits for GEARS."""
        
        # Get conditions from config
        train_conditions = self.config['train_conditions']
        val_conditions = self.config['val_conditions']
        test_conditions = self.config['test_conditions']
        
        # Convert to GEARS format
        def convert_conditions(conditions):
            gears_conditions = []
            for cond in conditions:
                if cond == 'control':
                    gears_conditions.append('ctrl')
                elif '+' not in cond:
                    gears_conditions.append(f"{cond}+ctrl")
                else:
                    gears_conditions.append(cond)
            return gears_conditions
        
        split_dict = {
            'train': convert_conditions(train_conditions),
            'val': convert_conditions(val_conditions),
            'test': convert_conditions(test_conditions)
        }
        
        # Filter out genes not in gene2go
        for split in ['train', 'val', 'test']:
            original_count = len(split_dict[split])
            filtered_conditions = []
            
            for cond in split_dict[split]:
                if cond == 'ctrl':
                    filtered_conditions.append(cond)
                else:
                    # Parse genes from condition
                    genes = self._parse_perturbation(cond)
                    if all(gene in self.pert_data.gene2go.keys() for gene in genes):
                        filtered_conditions.append(cond)
            
            split_dict[split] = filtered_conditions
            filtered_count = len(split_dict[split])
            log.info(f"{split} split: {filtered_count}/{original_count} conditions kept")
        
        # Save split dictionary to persistent location
        output_dir = Path(self.config['output_dir'])
        processed_data_dir = output_dir / 'processed_data'
        split_path = processed_data_dir / 'cellsimbench_split_dict.pkl'
        
        log.info(f"Saving split dictionary to: {split_path}")
        with open(split_path, 'wb') as f:
            pickle.dump(split_dict, f)
        
        # Prepare split in PertData
        self.pert_data.prepare_split(split='custom', seed=42, split_dict_path=str(split_path))
        
    def _parse_perturbation(self, pert: str) -> List[str]:
        """Parse perturbation string into list of genes."""
        if pert == 'ctrl':
            return []
        elif '+ctrl' in pert:
            return [pert.replace('+ctrl', '')]
        elif '+' in pert:
            return pert.split('+')
        else:
            return [pert]
            
    def _get_model_params(self) -> Dict:
        """Get model initialization parameters from config."""
        hyperparams = self.config['hyperparameters']
        
        params = {
            'hidden_size': hyperparams['hidden_size'],
            'num_go_gnn_layers': hyperparams['num_go_gnn_layers'],
            'num_gene_gnn_layers': hyperparams['num_gene_gnn_layers'],
            'decoder_hidden_size': hyperparams['decoder_hidden_size'],
            'num_similar_genes_go_graph': hyperparams['num_similar_genes_go_graph'],
            'num_similar_genes_co_express_graph': hyperparams['num_similar_genes_co_express_graph'],
            'coexpress_threshold': hyperparams['coexpress_threshold'],
            'uncertainty': hyperparams['uncertainty'],
            'uncertainty_reg': hyperparams['uncertainty_reg'],
            'direction_lambda': hyperparams['direction_lambda']
        }
        
        return params
        
    def _has_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Check if GEARS checkpoint exists."""
        return (checkpoint_dir / 'model.pt').exists() and (checkpoint_dir / 'config.pkl').exists()
        
    def _generate_predictions(self, test_conditions: List[str]) -> Dict:
        """Generate GEARS predictions for test conditions."""
        
        # Convert conditions to GEARS format and parse into gene lists
        gears_conditions = []
        for cond in test_conditions:
            if cond != 'control':  # Skip control condition
                # Convert to GEARS format first
                if '+' not in cond:
                    # Single perturbation - add +ctrl for parsing
                    gears_cond = f"{cond}+ctrl"
                else:
                    gears_cond = cond
                    
                # Parse into gene list
                genes = self._parse_perturbation(gears_cond)
                # Check if all genes are in gene2go dictionary
                if genes and all(gene in self.pert_data.gene2go.keys() for gene in genes):
                    gears_conditions.append(genes)
                    log.info(f"Adding condition for prediction: {cond} -> {genes}")
                else:
                    log.warning(f"Skipping condition {cond}: genes {genes} not found in gene2go")
        
        log.info(f"Generating predictions for {len(gears_conditions)} valid conditions")
        
        if not gears_conditions:
            log.warning("No valid conditions found for prediction")
            return {}
        
        # Get predictions from standard GEARS
        predictions = self.model.predict(gears_conditions)
        
        return predictions
        
    def _convert_to_cellsimbench_format(self, predictions: Dict, original_adata: sc.AnnData) -> sc.AnnData:
        """Convert GEARS predictions to CellSimBench format."""
        
        # Handle empty predictions
        if not predictions:
            log.warning("No predictions to convert - returning empty AnnData")
            obs_df = pd.DataFrame(columns=['condition', 'pair_key'])
            adata_pred = sc.AnnData(X=np.empty((0, original_adata.n_vars)), obs=obs_df)
            adata_pred.var_names = original_adata.var_names
            return adata_pred
        
        test_conditions = self.config['test_conditions']
        
        # Build prediction matrix and metadata
        prediction_list = []
        condition_list = []
        pair_key_list = []
        
        for condition in test_conditions:
            if condition != 'control':
                # Find prediction key for this condition
                pred_key = None
                
                # Look for predictions matching this condition
                for key in predictions.keys():
                    # Convert GEARS key back to condition format
                    genes = key.split('_')
                    if len(genes) == 1:
                        # Single perturbation
                        if condition == genes[0]:
                            pred_key = key
                            break
                    else:
                        # Combo perturbation
                        if condition == '+'.join(genes):
                            pred_key = key
                            break
                
                if pred_key is not None:
                    prediction_list.append(predictions[pred_key])
                    condition_list.append(condition)
                    pair_key_list.append(condition)
                else:
                    log.warning(f"No prediction found for {condition}")
        
        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")
        
        # Stack predictions
        prediction_matrix = np.vstack(prediction_list)
        
        # Create obs dataframe
        obs_df = pd.DataFrame({
            'condition': condition_list,
            'pair_key': pair_key_list
        })
        
        # Create AnnData object
        adata_pred = sc.AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = original_adata.var_names
        
        return adata_pred
        
    def _save_metadata(self, output_dir: Path):
        """Save training metadata."""
        metadata = {
            'model_type': 'GEARS',
            'config': self.config,
            'data_shape': self.pert_data.adata.shape if self.pert_data else None,
            'n_genes': len(self.pert_data.gene_names) if self.pert_data else None,
            'n_perturbations': len(self.pert_data.pert_names) if self.pert_data else None
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder)