"""
fMLP model wrapper for CellSimBench integration.
Simplified VAE model using pre-computed foundation model embeddings.
"""

import logging
import pickle
import json
import time
import os
import copy
import warnings
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# =============================================================================
# Neural Network Components
# =============================================================================

class Encoder(nn.Module):
    """Simple MLP encoder network for gene embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.leaky_relu(self.fc_input(x))
        h = self.leaky_relu(self.fc_input2(h))
        return self.fc_output(h)


class Decoder(nn.Module):
    """Decoder network for expression prediction."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leaky_relu(self.fc_hidden(x))
        h = self.leaky_relu(self.fc_hidden2(h))
        out = self.fc_output(h)
        return out


class FMLPNet(nn.Module):
    """Simplified fMLP network with MLP encoder-decoder structure."""
    
    def __init__(self, 
                 x_dim: int,
                 gene_emb_dim: int,
                 latent_dim: int = 128,
                 hidden_dim: int = 512):
        super(FMLPNet, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            raise ValueError("fMLP requires GPU for training. Please use a GPU.")
        
        # Simple gene embedding encoder
        self.encoder_g = Encoder(
            input_dim=gene_emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # Simple decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=x_dim
        )
    
    def forward(self, gene_emb):
        # For combination perturbations, gene_emb might be a list
        if isinstance(gene_emb, list):
            # Encode each gene separately and sum the latents
            z_g = None
            for emb in gene_emb:
                z = self.encoder_g(emb)
                if z_g is None:
                    z_g = z
                else:
                    z_g += z
        else:
            # Single gene perturbation
            z_g = self.encoder_g(gene_emb)
        
        # Decode to expression
        expr_x = self.decoder(z_g)
        
        return expr_x


class PertDataset(Dataset):
    """Simplified dataset class for fMLP training (no covariates)."""
    
    def __init__(self, gene_emb: torch.Tensor, expr_x: torch.Tensor, 
                 conditions: np.ndarray):
        self.gene_emb = gene_emb
        self.expr_x = expr_x  # Absolute expression values
        self.conditions = conditions
    
    def __len__(self):
        return len(self.gene_emb)
    
    def __getitem__(self, idx):
        return (self.gene_emb[idx], self.expr_x[idx], 
                self.conditions[idx])


# =============================================================================
# Main fMLP Model Class
# =============================================================================

class FMLPModel:
    """Main fMLP model class with CellSimBench integration."""
    
    def __init__(self,
                 adata: sc.AnnData,
                 foundation_model: str,
                 embedding_key: str,
                 split_name: str = 'split',
                 latent_dim: int = 128,
                 hidden_dim: int = 512,
                 training_epochs: int = 200,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 seed: int = 42,
                 model_path: str = "models",
                 val_every: int = 10):
        
        # Set device and random seeds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set all random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Store parameters
        self.adata = adata.copy()
        self.foundation_model = foundation_model
        self.embedding_key = embedding_key
        
        # Model dimensions
        self.x_dim = adata.shape[1]
        # Get embedding dimension from uns
        if embedding_key not in adata.uns:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.uns")
        self.gene_emb_dim = adata.uns[embedding_key].shape[1]
        
        # Model parameters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_path = model_path
        self.split_name = split_name
        self.val_every = val_every
        
        # Initialize simple storage
        self.pert_mean = None  # Single global perturbation mean
        self.pert_x = None     # Sample of perturbation cells
        
        # Load gene embeddings from adata.uns
        self.gene_embeddings = self._load_gene_embeddings()
    
    def _load_gene_embeddings(self) -> Dict[str, np.ndarray]:
        """Load gene embeddings from adata.uns."""
        log.info(f"Loading {self.foundation_model} embeddings from adata.uns['{self.embedding_key}']")
        
        embeddings = {}
        emb_df = self.adata.uns[self.embedding_key]  # This is a DataFrame
        
        for gene in emb_df.index:
            # Skip genes with NaN embeddings
            gene_embedding = emb_df.loc[gene].values
            if not np.isnan(gene_embedding).any():
                embeddings[gene] = gene_embedding.astype(np.float32)
        
        # Add control embedding (zero vector)
        embeddings['ctrl'] = np.zeros(self.gene_emb_dim, dtype=np.float32)
        
        log.info(f"Loaded embeddings for {len(embeddings)-1} genes (+ ctrl)")
        return embeddings
    
    def _setup(self):
        """Setup the simplified fMLP model."""
        log.info("Setting up simplified fMLP model...")
        
        # Remove control cells FIRST - they're not needed for training
        log.info("Removing control cells from dataset...")
        n_before = self.adata.shape[0]
        self.adata = self.adata[self.adata.obs['condition'] != 'ctrl'].copy()
        n_after = self.adata.shape[0]
        log.info(f"Removed {n_before - n_after} control cells from dataset")
        
        # Filter out conditions without embeddings
        log.info("Filtering conditions without gene embeddings...")
        removed_conditions = self._filter_conditions_without_embeddings()
        
        # Compute perturbation embeddings for all cells
        log.info(f"Computing {self.gene_emb_dim}-dimensional gene embeddings for {self.adata.shape[0]} cells...")
        self._compute_gene_embeddings()
        
        # Process data and compute simple pert mean
        self._process_perturbation_data()
        
        # Split datasets
        log.info("Splitting data...")
        self._prepare_data_splits(self.split_name)
    
    def _filter_conditions_without_embeddings(self):
        """Remove cells with conditions that lack gene embeddings."""
        conditions_to_remove = []
        
        for condition in self.adata.obs['condition'].unique():
            genes = condition.split('+')
            
            # Check if all genes in the condition have embeddings
            has_all_embeddings = True
            for gene in genes:
                if gene != 'ctrl' and gene not in self.gene_embeddings:
                    has_all_embeddings = False
                    break
            
            if not has_all_embeddings:
                conditions_to_remove.append(condition)
                log.warning(f"Removing condition '{condition}' - missing embeddings for one or more genes")
        
        if conditions_to_remove:
            # Filter out cells with these conditions
            mask = ~self.adata.obs['condition'].isin(conditions_to_remove)
            n_before = self.adata.shape[0]
            self.adata = self.adata[mask].copy()
            n_after = self.adata.shape[0]
            log.info(f"Filtered out {n_before - n_after} cells with missing gene embeddings")
            log.info(f"Removed conditions: {conditions_to_remove}")
        
        return conditions_to_remove
    
    def _compute_gene_embeddings(self):
        """Compute gene embeddings for all cells based on their conditions."""
        self.gene_emb_cells = np.zeros((self.adata.shape[0], self.gene_emb_dim), dtype=np.float32)
        self.gene_emb_by_condition = {}
        
        for condition in tqdm(self.adata.obs['condition'].unique(), desc="Computing gene embeddings"):
            genes = condition.split('+')
            
            if len(genes) > 1:
                # Combination perturbation: store list of embeddings (will sum in latent space)
                gene_emb_list = []
                for gene in genes:
                    gene_emb_list.append(self.gene_embeddings.get(gene, np.zeros(self.gene_emb_dim)))
                self.gene_emb_by_condition[condition] = gene_emb_list
                # For cells, just use the first embedding as placeholder
                gene_emb = gene_emb_list[0]
            else:
                # Single perturbation
                gene = genes[0]
                gene_emb = self.gene_embeddings.get(gene, np.zeros(self.gene_emb_dim))
                self.gene_emb_by_condition[condition] = gene_emb
            
            # Assign to cells with this condition (placeholder for combos)
            mask = self.adata.obs['condition'] == condition
            self.gene_emb_cells[mask] = gene_emb.reshape(1, -1)
        
        self.adata.obsm['gene_emb'] = self.gene_emb_cells
    
    def _process_perturbation_data(self):
        """Process perturbation cells and compute single global pert mean."""
        # All cells are perturbation cells now (controls removed)
        pert_cells = self.adata
        
        log.info(f"Processing {len(pert_cells)} perturbation cells...")
        
        # Compute balanced perturbation mean (equal weight per perturbation)
        pert_means_list = []
        for condition in self.adata.obs['condition'].unique():
            # Get cells for this condition
            cond_cells = self.adata[self.adata.obs['condition'] == condition]
            
            if len(cond_cells) > 0:
                cond_x = cond_cells.X
                if hasattr(cond_x, 'toarray'):
                    cond_x = cond_x.toarray()
                
                # Compute mean for this condition
                cond_mean = np.mean(cond_x, axis=0)
                pert_means_list.append(cond_mean)
        
        if len(pert_means_list) == 0:
            raise ValueError("No perturbations found in dataset")
        
        # Balanced mean: each perturbation contributes equally
        self.pert_mean = np.mean(np.array(pert_means_list), axis=0).astype(np.float32)
        
        # Store sample of perturbed cells for prediction
        pert_x = pert_cells.X
        if hasattr(pert_x, 'toarray'):
            pert_x = pert_x.toarray()
        
        # Sample up to 1024 cells
        n_cells = min(1024, pert_x.shape[0])
        sample_idx = np.random.choice(pert_x.shape[0], size=n_cells, replace=False)
        self.pert_x = torch.from_numpy(pert_x[sample_idx]).float().to(self.device)
        
        # Convert to array if needed but DO NOT center
        if hasattr(self.adata.X, 'toarray'):
            self.adata.X = self.adata.X.toarray()
        
        log.info("Keeping absolute expression values (no centering applied)")
    
    def _prepare_data_splits(self, split_name: str):
        """Prepare training and validation splits."""
        if split_name not in self.adata.obs.columns:
            raise ValueError(f"Split column '{split_name}' not found in adata.obs")
        
        self.adata_train = self.adata[self.adata.obs[split_name] == 'train']
        self.adata_val = self.adata[self.adata.obs[split_name] == 'val']
        self.pert_val = self.adata_val.obs['condition'].unique()
        
        # Create training dataset (simplified - no covariates)
        train_gene_emb = torch.from_numpy(self.adata_train.obsm['gene_emb']).float().to(self.device)
        train_expr_x = torch.from_numpy(self.adata_train.X).float().to(self.device)  # Absolute expression
        
        self.train_data = PertDataset(
            train_gene_emb,
            train_expr_x,  # Pass absolute expression
            self.adata_train.obs['condition'].values
        )
        
        # Create dataloader with fixed seed
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        def worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)
        
        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
            num_workers=0  # Use 0 for full determinism
        )
    

    
    def loss_function(self, expr_x_true, expr_x_pred):
        """Simple MSE reconstruction loss."""
        return F.mse_loss(expr_x_pred, expr_x_true)

    
    def _compute_validation_loss(self):
        """Compute validation loss on validation set."""
        self.net.eval()
        val_losses = []
        
        # Create validation dataloader if not exists
        if not hasattr(self, 'val_dataloader'):
            val_gene_emb = torch.from_numpy(self.adata_val.obsm['gene_emb']).float().to(self.device)
            val_expr_x = torch.from_numpy(self.adata_val.X).float().to(self.device)  # Absolute expression
            
            val_data = PertDataset(
                val_gene_emb,
                val_expr_x,  # Pass absolute expression
                self.adata_val.obs['condition'].values
            )
            
            self.val_dataloader = DataLoader(
                val_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        with torch.no_grad():
            for batch_data in self.val_dataloader:
                gene_emb, expr_x, batch_conditions = batch_data
                
                # Forward pass
                expr_x_pred = self.net(gene_emb)
                
                # Compute loss
                loss = self.loss_function(expr_x, expr_x_pred)
                val_losses.append(loss.item())
        
        return np.mean(val_losses) if val_losses else 0.0
    
    def _validate(self):
        """Simple validation using Pearson correlation and loss."""
        self.net.eval()
        
        # Compute validation loss
        val_loss = self._compute_validation_loss()
        
        all_corrs = []
        
        for condition in self.pert_val:
            # Get gene embedding(s) for this condition
            if condition not in self.gene_emb_by_condition:
                log.debug(f"Skipping validation for condition '{condition}' - no embedding")
                continue
            gene_emb = self.gene_emb_by_condition[condition]
            
            # Create batch of gene embeddings using pert_x
            if isinstance(gene_emb, list):
                # Combination perturbation: create list of batches
                val_gene_emb = []
                for emb in gene_emb:
                    val_gene_emb.append(torch.from_numpy(
                        np.tile(emb, (self.pert_x.shape[0], 1))
                    ).float().to(self.device))
            else:
                # Single perturbation
                val_gene_emb = torch.from_numpy(
                    np.tile(gene_emb, (self.pert_x.shape[0], 1))
                ).float().to(self.device)
            
            # Get ground truth for this condition
            condition_cells = self.adata_val[self.adata_val.obs['condition'] == condition]
            if len(condition_cells) == 0:
                continue
                
            actual_mean = np.mean(condition_cells.X, axis=0)
            
            with torch.no_grad():
                expr_x_pred = self.net(val_gene_emb)
                # Average prediction across cells
                expr_x_pred_mean = np.mean(expr_x_pred.detach().cpu().numpy(), axis=0)
            
            # Convert predictions and ground truth to deltas (relative to pert mean)
            delta_x_pred_mean = expr_x_pred_mean - self.pert_mean
            actual_delta = actual_mean - self.pert_mean
            
            # Full correlation on deltas
            corr = np.corrcoef(delta_x_pred_mean, actual_delta)[0, 1]
            if not np.isnan(corr):
                all_corrs.append(corr)
        
        # Calculate mean correlation
        mean_corr = np.mean(all_corrs) if all_corrs else 0.0
        
        # Log results
        if all_corrs:
            log.info(f"Validation loss: {val_loss:.5f} | Δcorr: {mean_corr:.5f} (n={len(all_corrs)})")
        else:
            log.info(f"Validation loss: {val_loss:.5f} | No correlations computed")
        
        return mean_corr, val_loss
    
    def train(self):
        """Train the simplified fMLP model."""
        log.info("Initializing simplified fMLP network...")
        
        self.net = FMLPNet(
            x_dim=self.x_dim,
            gene_emb_dim=self.gene_emb_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Set up optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        
        best_val_loss = float('inf')
        best_corr = 0
        self.model_best = copy.deepcopy(self.net)
        self.net.train()
        
        log.info("Starting training...")
        epoch_pbar = tqdm(range(self.training_epochs), desc="Training")
        
        # Track validation metrics for display
        last_val_loss = 0.0
        last_corr = 0.0
        
        for epoch in epoch_pbar:
            epoch_losses = []
            
            # Training loop
            self.net.train()
            for batch_data in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
                gene_emb, expr_x, batch_conditions = batch_data
                
                # Forward pass  
                expr_x_pred = self.net(gene_emb)
                
                # Compute loss
                loss = self.loss_function(expr_x, expr_x_pred)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Calculate average training loss
            avg_train_loss = np.mean(epoch_losses)
            
            # Validation every val_every epochs
            if (epoch + 1) % self.val_every == 0:
                if len(self.pert_val) > 0:
                    corr, val_loss = self._validate()
                    last_val_loss = val_loss
                    last_corr = corr
                    
                    # Model selection based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.model_best = copy.deepcopy(self.net)
                        log.info(f"New best model saved (val_loss: {val_loss:.5f})")
                    
                    # Track best correlation
                    if corr > best_corr:
                        best_corr = corr
                    
                    # Update progress bar
                    epoch_pbar.set_description(
                        f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val: {last_val_loss:.4f} | ΔCorr: {last_corr:.4f}"
                    )
                    
                    log.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Δcorr: {corr:.5f}")
                else:
                    # No validation data
                    epoch_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f}")
            else:
                # Update with training loss and last known validation metrics
                if last_val_loss > 0:
                    epoch_pbar.set_description(
                        f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val: {last_val_loss:.4f} | ΔCorr: {last_corr:.4f}"
                    )
                else:
                    epoch_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f}")
            
            scheduler.step()
            
            # Always update best model at the end of training
            if epoch == (self.training_epochs - 1):
                # Final validation
                if len(self.pert_val) > 0:
                    corr, val_loss = self._validate()
                    if val_loss < best_val_loss:
                        self.model_best = copy.deepcopy(self.net)
                        log.info(f"Final model is best (val_loss: {val_loss:.5f})")
                else:
                    self.model_best = copy.deepcopy(self.net)
        
        log.info(f"Training completed! Best val_loss: {best_val_loss:.5f} | Best Δcorr: {best_corr:.5f}")
        self.net = self.model_best
        
        # Save model
        self._save_model()
    
    def _save_model(self):
        """Save the trained model."""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model state
        state = {'net': self.net.state_dict()}
        torch.save(state, os.path.join(self.model_path, "model.pth"))
        
        # Save model configuration (simplified)
        config = {
            'x_dim': self.x_dim,
            'gene_emb_dim': self.gene_emb_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'foundation_model': self.foundation_model,
            'embedding_key': self.embedding_key,
            'pert_mean': self.pert_mean.tolist()
        }
        with open(os.path.join(self.model_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        log.info(f"Model saved to {self.model_path}")
    
    def load_pretrained(self, model_path: str):
        """Load a pre-trained model."""
        self.model_path = model_path
        
        # Load configuration
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        self.x_dim = config['x_dim']
        self.gene_emb_dim = config['gene_emb_dim']
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']
        self.foundation_model = config['foundation_model']
        self.embedding_key = config['embedding_key']
        self.pert_mean = np.array(config['pert_mean'], dtype=np.float32)
        
        # Initialize and load model
        self.net = FMLPNet(
            x_dim=self.x_dim,
            gene_emb_dim=self.gene_emb_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        state = torch.load(os.path.join(model_path, "model.pth"), map_location=self.device)
        self.net.load_state_dict(state['net'])
        self.net.eval()
        
        log.info(f"Model loaded from {model_path}")
    
    def predict(self, pert_test: Union[str, List[str]],
                return_type: str = 'mean') -> Dict:
        """Predict perturbation effects (simplified - no covariates)."""
        self.net.eval()
        results = {}
        
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        
        for condition in pert_test:
            # Check if all genes in condition have embeddings
            genes = condition.split('+')
            missing_genes = []
            for gene in genes:
                if gene != 'ctrl' and gene not in self.gene_embeddings:
                    missing_genes.append(gene)
            
            if missing_genes:
                log.warning(f"Skipping condition '{condition}' - missing embeddings for: {missing_genes}")
                continue
            
            # Get gene embedding for this condition
            if condition not in self.gene_emb_by_condition:
                # Compute embedding on-the-fly if not already computed
                if len(genes) > 1:
                    # Combination perturbation: list of embeddings (will sum in latent space)
                    gene_emb_list = []
                    for gene in genes:
                        gene_emb_list.append(self.gene_embeddings.get(gene, np.zeros(self.gene_emb_dim)))
                    gene_emb = gene_emb_list
                else:
                    # Single perturbation
                    gene = genes[0]
                    gene_emb = self.gene_embeddings.get(gene, np.zeros(self.gene_emb_dim))
                
                self.gene_emb_by_condition[condition] = gene_emb
                log.info(f"Computed embedding for condition '{condition}' on-the-fly")
            
            gene_emb = self.gene_emb_by_condition[condition]
            
            # Create batch of gene embeddings using pert_x as baseline
            if isinstance(gene_emb, list):
                # Combination perturbation: create list of batches
                val_gene_emb = []
                for emb in gene_emb:
                    val_gene_emb.append(torch.from_numpy(
                        np.tile(emb, (self.pert_x.shape[0], 1))
                    ).float().to(self.device))
            else:
                # Single perturbation
                val_gene_emb = torch.from_numpy(
                    np.tile(gene_emb, (self.pert_x.shape[0], 1))
                ).float().to(self.device)
            
            with torch.no_grad():
                expr_x_pred = self.net(val_gene_emb)
                
                if return_type == 'cells':
                    # Return individual predicted cells (absolute expression)
                    results[condition] = expr_x_pred.detach().cpu().numpy()
                elif return_type == 'mean':
                    # Return mean prediction (absolute expression)
                    expr_x_pred_mean = np.mean(expr_x_pred.detach().cpu().numpy(), axis=0)
                    results[condition] = expr_x_pred_mean
                else:
                    raise ValueError("return_type must be 'mean' or 'cells'")
        
        return results
    



# =============================================================================
# CellSimBench Wrapper
# =============================================================================

class FMLPWrapper:
    """fMLP model wrapper for CellSimBench integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.data_manager = DataManager(self.config)
        self.mode = self.config['mode']
        
        # Get foundation model and embedding key from config
        hyperparams = self.config['hyperparameters']
        self.foundation_model = hyperparams['foundation_model']
        self.embedding_key = hyperparams['embedding_key']
    
    def train(self):
        """Train fMLP model."""
        log.info("Starting fMLP training process...")
        
        # Load data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")
        
        # Check that embeddings exist
        if self.embedding_key not in adata.uns:
            raise ValueError(f"Embedding key '{self.embedding_key}' not found in adata.uns. "
                           f"Available keys: {list(adata.uns.keys())}")
        
        # No DEG weights in simplified version
        
        # Convert to fMLP format (same as scLambda format)
        log.info("Converting to fMLP format...")
        adata_fmlp = self._convert_to_fmlp_format(adata)
        
        # Initialize and train model
        log.info("Initializing fMLP model...")
        hyperparams = self.config['hyperparameters']
        
        self.model = FMLPModel(
            adata=adata_fmlp,
            foundation_model=self.foundation_model,
            embedding_key=self.embedding_key,
            split_name=self.config['split_name'],
            latent_dim=hyperparams['latent_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            training_epochs=hyperparams['training_epochs'],
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            seed=hyperparams['seed'],
            model_path=self.config['output_dir'],
            val_every=hyperparams['val_every'],

        )
        
        # Setup the model
        self.model._setup()
        
        # Train the model
        self.model.train()
        
        # Save metadata
        self._save_metadata()
        
        log.info("Training completed successfully")
    
    def predict(self):
        """Generate predictions using trained fMLP model."""
        log.info("Starting fMLP prediction process...")
        
        # Load trained model
        model_path = self.config['model_path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        # Load data for prediction
        adata = self.data_manager.load_dataset()
        adata_fmlp = self._convert_to_fmlp_format(adata)
        
        # Initialize model and load pretrained weights
        hyperparams = self.config['hyperparameters']
        self.model = FMLPModel(
            adata=adata_fmlp,
            foundation_model=self.foundation_model,
            embedding_key=self.embedding_key,
            split_name=self.config['split_name'],
            latent_dim=hyperparams['latent_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            training_epochs=hyperparams['training_epochs'],
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            seed=hyperparams['seed'],
            val_every=hyperparams['val_every'],

        )
        
        # Load embeddings from adata
        self.model.gene_embeddings = self.model._load_gene_embeddings()
        
        # Load pretrained model
        self.model.load_pretrained(model_path)
        
        # Setup the model
        self.model._setup()
        
        # Generate predictions
        test_conditions = self.config['test_conditions']
        # Convert test conditions to fMLP format (same as scLambda)
        test_conditions = [condition if '+' in condition else f"{condition}+ctrl" for condition in test_conditions]
        log.info(f"Generating predictions for {len(test_conditions)} conditions")
        
        predictions = self.model.predict(test_conditions, return_type='mean')
        
        # Convert to CellSimBench format
        log.info("Converting predictions to CellSimBench format...")
        predictions_adata = self._convert_to_cellsimbench_format(predictions, adata)
        
        # Save predictions
        output_path = self.config['output_path']
        log.info(f"Saving predictions to {output_path}")
        predictions_adata.write_h5ad(output_path)
        
        log.info("Prediction completed successfully")
    
    def _convert_to_fmlp_format(self, adata: sc.AnnData) -> sc.AnnData:
        """Convert CellSimBench data to fMLP format (same as scLambda)."""
        adata_fmlp = adata.copy()
        
        # Define control condition values
        CTRL_VALUE = 'ctrl_iegfp'
        
        # Process condition labels
        def process_condition(cond):
            if cond == 'control' or cond == 'ctrl' or cond == CTRL_VALUE:
                return 'ctrl'
            elif '+' not in cond:
                # Single perturbation
                return f"{cond}+ctrl"
            else:
                # Already in correct format for combo perturbations
                return cond
        
        adata_fmlp.obs['condition'] = adata_fmlp.obs['condition'].astype(str)
        adata_fmlp.obs['condition'] = adata_fmlp.obs['condition'].apply(process_condition)
        
        return adata_fmlp
    
    def _convert_to_cellsimbench_format(self, predictions: Dict, original_adata: sc.AnnData) -> sc.AnnData:
        """Convert fMLP predictions to CellSimBench format (simplified - no covariates)."""
        # Handle empty predictions
        if not predictions:
            log.warning("No predictions to convert - returning empty AnnData")
            obs_df = pd.DataFrame(columns=['condition'])
            adata_pred = sc.AnnData(X=np.empty((0, original_adata.n_vars)), obs=obs_df)
            adata_pred.var_names = original_adata.var_names
            return adata_pred
        
        test_conditions = self.config['test_conditions']
        
        # Build prediction matrix and metadata
        prediction_list = []
        condition_list = []
        
        for condition in test_conditions:
            if condition != 'control':
                # Convert condition to fMLP format for lookup
                fmlp_condition = condition if '+' in condition else f"{condition}+ctrl"
                
                if fmlp_condition in predictions:
                    # Get single prediction (no covariates)
                    pred_vector = predictions[fmlp_condition]
                    prediction_list.append(pred_vector)
                    condition_list.append(condition)
        
        if not prediction_list:
            raise ValueError("No valid predictions found for test conditions")
        
        # Stack predictions
        prediction_matrix = np.vstack(prediction_list)
        
        # Create obs dataframe
        obs_df = pd.DataFrame({
            'condition': condition_list
        })
        
        # Create AnnData object
        adata_pred = sc.AnnData(X=prediction_matrix, obs=obs_df)
        adata_pred.var_names = original_adata.var_names
        
        return adata_pred
    
    def _save_metadata(self):
        """Save training metadata."""
        metadata = {
            'model_type': 'Simplified fMLP (MLP encoder-decoder)',
            'foundation_model': self.foundation_model,
            'embedding_key': self.embedding_key,
            'config': self.config,
            'data_shape': self.model.adata.shape if self.model else None,
            'embedding_dimension': self.model.gene_emb_dim if self.model else None,
            'latent_dimension': self.model.latent_dim if self.model else None
        }
        
        output_dir = Path(self.config['output_dir'])
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder)
