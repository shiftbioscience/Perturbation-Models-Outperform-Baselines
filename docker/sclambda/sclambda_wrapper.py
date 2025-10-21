"""
scLambda model wrapper for CellSimBench integration.
Handles training and prediction with gene description embeddings and internal checkpointing.
"""

# This file is part of a derivative work of the scLambda project.
# Copyright (c) 2025 Shift Bioscience and scLambda contributors.
# Distributed under the GNU General Public License v3.0 (GPL-3.0).
# See docker/sclambda/LICENSE for details.

import logging
import pickle
import json
import hashlib
import time
import os
import copy
import warnings
import re
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
import requests
from pybiomart import Dataset as BioMartDataset
import openai
from cellsimbench.utils.utils import PathEncoder
from cellsimbench.core.data_manager import DataManager

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# =============================================================================
# Gene Embedding Generation Pipeline
# =============================================================================

class GeneDescriptionFetcher:
    """Fetches gene descriptions from Ensembl using pybiomart with caching and error handling."""
    
    def __init__(self, cache_dir: str = "./gene_descriptions_cache", gene_list: Optional[List[str]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Path for NCBI gene summary file
        self.gene_summary_path = self.cache_dir / "gene_summary.gz"
        
        # Initialize Ensembl dataset connection
        try:
            self.dataset = BioMartDataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
            log.info("Connected to Ensembl BioMart successfully")
        except Exception as e:
            log.error(f"Failed to connect to Ensembl BioMart: {e}")
            raise e
        
        # Ensure gene summary file is available
        self._ensure_gene_summary_file()
        
        # Cache for bulk descriptions
        self._description_cache = {}
        self._cache_loaded = False
        
        # If gene list provided, fetch all descriptions upfront
        if gene_list:
            log.info(f"Fetching descriptions for {len(gene_list)} genes upfront...")
            self.get_gene_descriptions_batch(gene_list)
    
    def _ensure_gene_summary_file(self):
        """Download NCBI gene summary file if it doesn't exist."""
        if self.gene_summary_path.exists():
            log.info(f"NCBI gene summary file found at {self.gene_summary_path}")
            return
        
        log.info("Downloading NCBI gene summary file...")
        import urllib.request
        
        try:
            ncbi_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_summary.gz"
            log.info(f"Downloading from {ncbi_url}")
            
            with urllib.request.urlopen(ncbi_url) as response:
                with open(self.gene_summary_path, 'wb') as f:
                    f.write(response.read())
            
            log.info(f"Successfully downloaded gene summary to {self.gene_summary_path}")
            
        except Exception as e:
            log.error(f"Failed to download NCBI gene summary: {e}")
            raise RuntimeError(f"Could not download NCBI gene summary file: {e}")
    
    def _load_bulk_cache(self):
        """Load the bulk cache file if it exists."""
        if self._cache_loaded:
            return
            
        cache_file = self.cache_dir / "bulk_descriptions_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self._description_cache = json.load(f)
                log.info(f"Loaded {len(self._description_cache)} cached descriptions")
            except Exception as e:
                log.warning(f"Failed to load bulk cache: {e}")
                self._description_cache = {}
        
        self._cache_loaded = True
    
    def _save_bulk_cache(self):
        """Save the bulk cache to file."""
        cache_file = self.cache_dir / "bulk_descriptions_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._description_cache, f, indent=2)
            log.info(f"Saved {len(self._description_cache)} descriptions to cache")
        except Exception as e:
            log.error(f"Failed to save bulk cache: {e}")
            raise e
    
    def get_gene_descriptions_batch(self, gene_symbols: List[str]) -> Dict[str, str]:
        """Get descriptions for multiple genes using bulk BioMart query."""
        log.info(f"Fetching descriptions for {len(gene_symbols)} genes from Ensembl...")
        
        # Load existing cache
        self._load_bulk_cache()
        
        # Find genes that need to be fetched
        missing_genes = [gene for gene in gene_symbols if gene not in self._description_cache]
        
        if missing_genes:
            log.info(f"Fetching {len(missing_genes)} new gene descriptions from BioMart...")
            
            try:
                # Query BioMart for all missing genes at once
                result = self.dataset.query(
                    attributes=['external_gene_name', 'entrezgene_id'],
                )
                # Rename the column 'NCBI gene (formerly Entrezgene) ID' to 'GeneID'
                result = result.rename(columns={'NCBI gene (formerly Entrezgene) ID': 'GeneID'})
                
                # Load NCBI gene summaries from downloaded file
                log.info(f"Loading NCBI gene summaries from {self.gene_summary_path}")
                entrez_summaries = pd.read_table(self.gene_summary_path)
                entrez_ids = result['GeneID'].unique()
                entrez_summaries = entrez_summaries[entrez_summaries['GeneID'].isin(entrez_ids)]
                
                # Merge the two dataframes on the 'GeneID' column
                result = pd.merge(result, entrez_summaries, on='GeneID', how='inner')
                result = result.rename(columns={'GeneID': 'entrezgene_id'})
                
                log.info(f"BioMart query returned {len(result)} results")
                
                # Process entire result at once
                if len(result) > 0:
                    # Clean descriptions vectorized
                    result['cleaned_summary'] = result['Summary'].fillna('').apply(self._clean_description)
                    
                    # Create dictionary from entire result
                    gene_descriptions = dict(zip(result['Gene name'], result['cleaned_summary']))
                    
                    # Update cache with all results at once
                    self._description_cache.update(gene_descriptions)
                    log.info(f"Added {len(gene_descriptions)} gene descriptions to cache")
                
                # Add empty descriptions for genes not found in BioMart
                missing_after_query = [gene for gene in missing_genes if gene not in self._description_cache]
                if missing_after_query:
                    empty_descriptions = {gene: "" for gene in missing_after_query}
                    self._description_cache.update(empty_descriptions)
                    log.warning(f"No descriptions found for {len(missing_after_query)} genes: {missing_after_query[:5]}...")
                
                # Save updated cache
                self._save_bulk_cache()
                
            except Exception as e:
                log.error(f"Failed to fetch descriptions from BioMart: {e}")
                # Add empty descriptions for all missing genes as fallback
                for gene in missing_genes:
                    if gene not in self._description_cache:
                        self._description_cache[gene] = ""
        
        # Return descriptions for requested genes
        descriptions = {}
        for gene in gene_symbols:
            descriptions[gene] = self._description_cache.get(gene, "")
        
        log.info(f"Retrieved descriptions for {len([d for d in descriptions.values() if d])} genes")
        return descriptions
    
    def get_gene_description(self, gene_symbol: str) -> str:
        """Get description for a single gene (cache lookup only)."""
        self._load_bulk_cache()
        return self._description_cache.get(gene_symbol, "")
    
    def _clean_description(self, description: str) -> str:
        """Clean and normalize gene description text."""
        if not description or pd.isna(description):
            return ""
        
        # Convert to string and clean
        description = str(description)
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            "[Source:HGNC Symbol;Acc:HGNC:",
            "[Source:Uniprot/SWISSPROT;Acc:",
            "[Source:",
        ]
        
        for prefix in prefixes_to_remove:
            if prefix in description:
                description = description.split(prefix)[0].strip()
        
        # Remove trailing brackets and cleanup
        description = re.sub(r'\s*\[.*?\]\s*$', '', description)
        description = description.replace('\n', ' ').replace('\r', ' ')
        description = ' '.join(description.split())  # Normalize whitespace
        
        return description.strip()


class GeneEmbeddingGenerator:
    """Generate gene embeddings using OpenAI's latest embedding models."""
    
    def __init__(self, 
                 model: str = "text-embedding-3-large",
                 dimensions: int = 3072,
                 cache_dir: str = "./gene_embeddings_cache",
                 api_key_env: str = "OPENAI_API_KEY",
                 gene_list: Optional[List[str]] = None):
        self.model = model
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load .env file if it exists (for OpenAI API key)
        env_file = Path('/app/.env')
        if env_file.exists():
            log.info("Loading .env file for OpenAI API key...")
            try:
                env_vars_loaded = 0
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value
                                env_vars_loaded += 1
                log.info(f"✓ Loaded {env_vars_loaded} environment variables from .env file")
            except Exception as e:
                raise RuntimeError(f"Failed to load .env file for scLambda: {e}")
        
        # Set up OpenAI client - fail fast if key is missing
        api_key = os.getenv(api_key_env)
        if not api_key:
            error_msg = f"scLambda requires OpenAI API key but {api_key_env} environment variable not found."
            if env_file.exists():
                error_msg += f" The .env file was loaded from {env_file} but did not contain {api_key_env}."
            else:
                error_msg += f" No .env file found at {env_file}."
            raise ValueError(error_msg)
        
        # Validate API key format
        if not api_key.startswith('sk-'):
            raise ValueError(f"Invalid OpenAI API key format. Key should start with 'sk-' but got: {api_key[:10]}...")
        
        log.info(f"✓ OpenAI API key found and validated (key: {api_key[:10]}...)")
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize description fetcher with gene list for bulk loading
        self.description_fetcher = GeneDescriptionFetcher(
            cache_dir=str(self.cache_dir / "descriptions"),
            gene_list=gene_list
        )
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        content = f"{self.model}_{self.dimensions}_{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        if not text.strip():
            log.warning("Empty text provided, returning zero embedding")
            return np.zeros(self.dimensions)
        
        try:
            # Clean text
            cleaned_text = text.replace("\n", " ").strip()
            
            # Generate embedding
            response = self.client.embeddings.create(
                input=[cleaned_text],
                model=self.model,
                dimensions=self.dimensions
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            log.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.dimensions, dtype=np.float32)
    
    def get_gene_embedding(self, gene_symbol: str) -> np.ndarray:
        """Get embedding for a single gene with caching."""
        # Get gene description
        description = self.description_fetcher.get_gene_description(gene_symbol)
        
        # Generate embedding
        log.info(f"Generating embedding for gene: {gene_symbol}")
        embedding = self._get_openai_embedding(description)
        
        return embedding
    
    def get_gene_embeddings_batch(self, gene_symbols: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple genes with rate limiting."""
        log.info(f"Generating embeddings for {len(gene_symbols)} genes...")
        
        # All descriptions should already be cached from initialization
        embeddings = {}
        
        for gene_symbol in tqdm(gene_symbols, desc="Generating gene embeddings"):
            embeddings[gene_symbol] = self.get_gene_embedding(gene_symbol)
            time.sleep(0.05)  # Rate limiting for API calls
        
        # Add control embedding (zero vector)
        embeddings['ctrl'] = np.zeros(self.dimensions, dtype=np.float32)
        
        return embeddings 


# =============================================================================
# scLambda Neural Network Architecture
# =============================================================================




class Encoder(nn.Module):
    """Encoder network for scLambda."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, VAE: bool = True):
        super(Encoder, self).__init__()
        self.VAE = VAE
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        if self.VAE:
            self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.leaky_relu(self.fc_input(x))
        h = self.leaky_relu(self.fc_input2(h))
        mean = self.fc_mean(h)
        if self.VAE:
            log_var = self.fc_var(h)
            return mean, log_var
        else:
            return mean


class Decoder(nn.Module):
    """Decoder network for scLambda."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, covariate_dim: int = 0):
        super(Decoder, self).__init__()
        # Input dimension is latent_dim + covariate_dim if covariate-aware
        input_size = latent_dim + covariate_dim
        self.fc_hidden = nn.Linear(input_size, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leaky_relu(self.fc_hidden(x))
        h = self.leaky_relu(self.fc_hidden2(h))
        out = self.fc_output(h)
        return out


class MINE(nn.Module):
    """Mutual Information Neural Estimation network."""
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        super(MINE, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, z, s):
        h = self.leaky_relu(self.fc_hidden(torch.cat((z, s), 1)))
        h = self.leaky_relu(self.fc_hidden2(h))
        T = self.fc_output(h)
        return torch.clamp(T, min=-50.0, max=50.0)


class SCLambdaNet(nn.Module):
    """Complete scLambda network architecture."""
    
    def __init__(self, x_dim: int, p_dim: int, latent_dim: int = 30, hidden_dim: int = 512):
        super(SCLambdaNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == 'cpu':
            raise ValueError("scLambda does not support CPU training. Please use a GPU.")
        
        # Standard encoders
        self.encoder_x = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, VAE=True)
        self.encoder_p = Encoder(input_dim=p_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, VAE=False)
        
        # Decoders
        self.decoder_x = Decoder(
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            output_dim=x_dim,
            covariate_dim=0
        )
        self.decoder_p = Decoder(
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            output_dim=p_dim,
            covariate_dim=0
        )
        
        self.mine = MINE(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x, p):
        # Encode expression and perturbation
        mean_z, log_var_z = self.encoder_x(x)
        z = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z))
        s = self.encoder_p(p)
        
        # Decode
        x_hat = self.decoder_x(z + s)
        p_hat = self.decoder_p(s)
        
        return x_hat, p_hat, mean_z, log_var_z, s


class PertDataset(Dataset):
    """Dataset class for scLambda training."""
    
    def __init__(self, x: torch.Tensor, p: torch.Tensor, conditions: np.ndarray):
        self.x = x
        self.p = p
        self.conditions = conditions  # Array of condition strings
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.conditions[idx]


# =============================================================================
# Main scLambda Model Class
# =============================================================================

class SCLambdaModel:
    """Main scLambda model class with CellSimBench integration."""
    
    def __init__(self, 
                 adata: sc.AnnData,
                 gene_embeddings: Dict[str, np.ndarray],
                 split_name: str = 'split',
                 latent_dim: int = 30,
                 hidden_dim: int = 512,
                 training_epochs: int = 200,
                 batch_size: int = 500,
                 lambda_MI: float = 200,
                 eps: float = 0.001,
                 seed: int = 1234,
                 model_path: str = "models",
                 val_every: int = 10,
                 multi_gene: bool = True,
                 use_delta: bool = False,
                 delta_type: str = "ctrl_mean"):
        
        # Set device and random seeds for full determinism
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set all random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set environment variable for hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA ops

        self.gene_embeddings = gene_embeddings
        self.use_delta = use_delta
        self.delta_type = delta_type
        
        # Store parameters
        print(f"Copying adata with shape: {adata.shape}")
        self.adata = adata.copy()

        self.x_dim = adata.shape[1]
        self.p_dim = gene_embeddings[list(gene_embeddings.keys())[0]].shape[0]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lambda_MI = lambda_MI
        self.eps = eps
        self.seed = seed  # Store seed as instance attribute
        self.model_path = model_path
        self.multi_gene = multi_gene
        self.split_name = split_name
        self.val_every = val_every
        
        # Initialize data storage
        self.ctrl_mean = None
        self.pert_mean = None
        self.ctrl_x = None
        self.pert_deltas = {}

    def _setup(self):
        """Setup the scLambda model."""

        # Ensure control embedding exists
        if 'ctrl' not in self.gene_embeddings:
            self.gene_embeddings['ctrl'] = np.zeros(self.p_dim, dtype=np.float32)

        # Compute perturbation embeddings for all cells
        log.info(f"Computing {self.p_dim}-dimensional perturbation embeddings for {self.adata.shape[0]} cells...")
        self._compute_perturbation_embeddings()

        # Process control cells and center data
        self._process_control_data()

        # Split datasets
        log.info("Splitting data...")
        self._prepare_data_splits(self.split_name)

        # Compute perturbation deltas
        self._compute_perturbation_deltas()

    def _compute_perturbation_embeddings(self):
        """Compute perturbation embeddings for all cells."""
        self.pert_emb_cells = np.zeros((self.adata.shape[0], self.p_dim), dtype=np.float32)
        self.pert_emb = {}
        for condition in tqdm(self.adata.obs['condition'].unique(), desc="Computing perturbation embeddings"):
            genes = condition.split('+')
            
            if len(genes) > 1 and self.multi_gene:
                # Combination perturbation: add embeddings
                pert_emb_p = sum(self.gene_embeddings.get(gene, np.zeros(self.p_dim)) for gene in genes)
            else:
                # Single perturbation
                pert_emb_p = self.gene_embeddings.get(genes[0], np.zeros(self.p_dim))
            
            # Assign to cells with this condition
            mask = self.adata.obs['condition'] == condition
            self.pert_emb_cells[mask] = pert_emb_p.reshape(1, -1)
            self.pert_emb[condition] = pert_emb_p.astype(np.float32)
        self.adata.obsm['pert_emb'] = self.pert_emb_cells

    def _process_control_data(self):
        """Process control cells and center expression data."""
        ctrl_cells = self.adata[self.adata.obs['condition'] == 'ctrl']
        pert_cells = self.adata[self.adata.obs['condition'] != 'ctrl']
        if len(ctrl_cells) == 0:
            raise ValueError("No control cells found. Ensure 'ctrl' condition exists in data.")
        
        log.info(f"Processing {len(ctrl_cells)} control cells")
        
        # Compute control mean
        ctrl_x = ctrl_cells.X
        if hasattr(ctrl_x, 'toarray'):
            ctrl_x = ctrl_x.toarray()
        self.ctrl_mean = np.mean(ctrl_x, axis=0).astype(np.float32)
        
        # Compute balanced perturbation mean (equal weight per perturbation)
        pert_means_list = []
        for condition in self.adata.obs['condition'].unique():
            if condition == 'ctrl':
                continue
                
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
            raise ValueError("No perturbations found")
        
        # Balanced mean: each perturbation contributes equally
        self.pert_mean = np.mean(np.array(pert_means_list), axis=0).astype(np.float32)
        
        # Store centered control cells
        self.ctrl_x = torch.from_numpy(
            ctrl_x - self.ctrl_mean.reshape(1, -1)
        ).float().to(self.device)
        
        # Center ALL data using the single baseline
        if hasattr(self.adata.X, 'toarray'):
            self.adata.X = self.adata.X.toarray()
        
        log.info("Centering all expression data...")
        if self.use_delta:
            if self.delta_type == "ctrl_mean":
                baseline_mean = self.ctrl_mean
            elif self.delta_type == "pert_mean":
                baseline_mean = self.pert_mean
            else:
                raise ValueError(f"Invalid delta type: {self.delta_type}. Must be one of 'ctrl_mean' or 'pert_mean'.")
            self.adata.X = self.adata.X - baseline_mean.reshape(1, -1)
        else:
            raise ValueError("use_delta must be True")
        
        log.info(f"Centered {self.adata.shape[0]} cells")

    def _prepare_data_splits(self, split_name: str):
        """Prepare training and validation splits."""
        if split_name not in self.adata.obs.columns:
            raise ValueError(f"Split column '{split_name}' not found in adata.obs")
        
        # Examine self.pert_emb_cells and remove perturbations where all the cells are 0
        all_zero_pert = np.all(self.pert_emb_cells == 0, axis=1)
        self.adata = self.adata[~all_zero_pert]
        # TODO: Examine whether we should do something else here.
        
        self.adata_train = self.adata[self.adata.obs[split_name] == 'train']
        self.adata_val = self.adata[self.adata.obs[split_name] == 'val']
        self.pert_val = self.adata_val.obs['condition'].unique()
        
        # Create training dataset
        train_x = torch.from_numpy(self.adata_train.X).float().to(self.device)
        train_p = torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device)
        self.train_data = PertDataset(
            train_x, 
            train_p, 
            self.adata_train.obs['condition'].values
        )
        
        # Create a generator with fixed seed for DataLoader
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        # Define worker init function for deterministic behavior
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

    def _compute_perturbation_deltas(self):
        """Compute perturbation effects."""
        log.info("Computing perturbation deltas...")
        
        for condition in self.adata.obs['condition'].unique():
            if condition == 'ctrl':
                continue
                
            # Get cells for this condition
            condition_cells = self.adata[self.adata.obs['condition'] == condition]
            
            if len(condition_cells) > 0:
                # Delta is mean of centered expression (already centered by baseline)
                delta = np.mean(condition_cells.X, axis=0).astype(np.float32)
                self.pert_deltas[condition] = delta
                log.info(f"Computed delta for {condition} ({len(condition_cells)} cells)")
            else:
                # No data for this condition
                self.pert_deltas[condition] = None
                log.warning(f"No cells found for condition {condition}")



    def loss_function(self, x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T):
        """Complete loss function with reconstruction, KLD, and MI terms."""
        
        # Gene reconstruction loss (unweighted)
        gene_reconstruction = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1))
        
        # Perturbation reconstruction loss
        pert_reconstruction = 0.5 * torch.mean(torch.sum((p_hat - p)**2, axis=1))
        
        reconstruction_loss = gene_reconstruction + pert_reconstruction
        
        KLD_z = -0.5 * torch.mean(torch.sum(1 + log_var_z - mean_z**2 - log_var_z.exp(), axis=1))
        
        MI_latent = torch.mean(T(mean_z, s.detach())) - \
                   torch.log(torch.mean(torch.exp(T(mean_z, s_marginal.detach()))))
        
        total_loss = reconstruction_loss + KLD_z + self.lambda_MI * MI_latent
        
        return total_loss, reconstruction_loss, KLD_z, MI_latent

    def loss_recon(self, x, x_hat):
        """Reconstruction loss only."""
        return 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1))

    def loss_MINE(self, mean_z, s, s_marginal, T):
        """MINE loss for mutual information estimation."""
        MI_latent = torch.mean(T(mean_z, s)) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal))))
        return -MI_latent

    def _validate(self):
        """Validation method."""
        self.net.eval()
        all_corrs = []
        shuffled_corrs = []
        
        for condition in tqdm(self.pert_val, desc="Validating via pearson correlation"):
            # Skip if no ground truth for this condition
            if condition not in self.pert_deltas or self.pert_deltas[condition] is None:
                continue
            
            genes = condition.split('+')
            if self.multi_gene and len(genes) > 1:
                pert_emb_p = sum(self.gene_embeddings.get(gene, np.zeros(self.p_dim)) for gene in genes)
            else:
                pert_emb_p = self.gene_embeddings.get(genes[0], np.zeros(self.p_dim))
            
            # Use control cells for validation
            val_p = torch.from_numpy(
                np.tile(pert_emb_p, (self.ctrl_x.shape[0], 1))
            ).float().to(self.device)

            # Generate white noise baseline with same mean and std as ctrl_x
            x_mean = self.ctrl_x.mean()
            x_std = self.ctrl_x.std()
            x_noise = torch.randn_like(self.ctrl_x) * x_std + x_mean
            
            with torch.no_grad():
                x_hat, _, _, _, _ = self.net(self.ctrl_x, val_p)
                x_hat_noise, _, _, _, _ = self.net(x_noise, val_p)
                # Average prediction across control cells
                x_hat_mean = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                x_hat_noise_mean = np.mean(x_hat_noise.detach().cpu().numpy(), axis=0)
            
            # Compare predicted delta to actual delta
            actual_delta = self.pert_deltas[condition]
            
            # Full correlation
            corr = np.corrcoef(x_hat_mean, actual_delta)[0, 1]
            corr_noise = np.corrcoef(x_hat_noise_mean, actual_delta)[0, 1]
            if not np.isnan(corr):
                all_corrs.append(corr)
                shuffled_corrs.append(corr_noise)
        
        if all_corrs:
            mean_corr = np.mean(all_corrs)
            log.info(f"Validation correlation: {mean_corr:.5f} (n={len(all_corrs)} comparisons)")
            print(f"Validation correlation: {mean_corr:.5f} (n={len(all_corrs)} comparisons)")
            
            if shuffled_corrs:
                mean_noise_corr = np.mean(shuffled_corrs)
                log.info(f"Noise baseline correlation: {mean_noise_corr:.5f} (n={len(shuffled_corrs)} comparisons)")
                print(f"Noise baseline correlation: {mean_noise_corr:.5f} (n={len(shuffled_corrs)} comparisons)")
            
            return mean_corr
        
        return 0.0

    def train(self):
        """Train the scLambda model."""
        log.info("Initializing scLambda network...")
        self.net = SCLambdaNet(
            x_dim=self.x_dim, 
            p_dim=self.p_dim, 
            latent_dim=self.latent_dim, 
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Set up optimizers
        params = (list(self.net.encoder_x.parameters()) + 
                 list(self.net.encoder_p.parameters()) + 
                 list(self.net.decoder_x.parameters()) + 
                 list(self.net.decoder_p.parameters()))
        
        optimizer = optim.Adam(params, lr=0.0005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        optimizer_MINE = optim.Adam(self.net.mine.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler_MINE = StepLR(optimizer_MINE, step_size=30, gamma=0.2)

        corr_val_best = 0
        # Initialize best model to current model to avoid AttributeError
        self.model_best = copy.deepcopy(self.net)
        self.net.train()
        
        log.info("Starting training...")
        epoch_pbar = tqdm(range(self.training_epochs), desc="Training")
        
        for epoch in epoch_pbar:
            # Track losses for this epoch
            epoch_total_losses = []
            epoch_recon_losses = []
            epoch_kld_losses = []
            epoch_mi_losses = []
            epoch_mine_losses = []
            for batch_data in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
                # Unpack batch data
                x, p, batch_conditions = batch_data
                
                # Adversarial training on perturbation embeddings
                p.requires_grad = True
                self.net.eval()
                
                with torch.enable_grad():
                    x_hat, _, _, _, _ = self.net(x, p)
                    recon_loss = self.loss_recon(x, x_hat)
                    grads = torch.autograd.grad(recon_loss, p)[0]
                    p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data)

                self.net.train()
                x_hat, p_hat, mean_z, log_var_z, s = self.net(x, p_ae)

                # MINE training
                index_marginal = np.random.choice(len(self.train_data), size=x_hat.shape[0])
                p_marginal = self.train_data.p[index_marginal]
                s_marginal = self.net.encoder_p(p_marginal)
                
                for _ in range(1):
                    optimizer_MINE.zero_grad()
                    mine_loss = self.loss_MINE(mean_z, s, s_marginal, T=self.net.mine)
                    mine_loss.backward(retain_graph=True)
                    optimizer_MINE.step()
                    epoch_mine_losses.append(mine_loss.item())

                # Main training step
                optimizer.zero_grad()
                total_loss, recon_loss, kld_loss, mi_loss = self.loss_function(
                    x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, 
                    T=self.net.mine
                )
                total_loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_total_losses.append(total_loss.item())
                epoch_recon_losses.append(recon_loss.item())
                epoch_kld_losses.append(kld_loss.item())
                epoch_mi_losses.append(mi_loss.item())
            
            # Update progress bar with average losses for this epoch
            avg_total = np.mean(epoch_total_losses)
            avg_recon = np.mean(epoch_recon_losses)
            avg_kld = np.mean(epoch_kld_losses)
            avg_mi = np.mean(epoch_mi_losses)
            avg_mine = np.mean(epoch_mine_losses)
            
            epoch_pbar.set_description(
                f"Epoch {epoch+1} | Total: {avg_total:.3f} | Recon: {avg_recon:.3f} | KLD: {avg_kld:.3f} | MI: {avg_mi:.3f} | MINE: {avg_mine:.3f}"
            )
            
            scheduler.step()
            scheduler_MINE.step()
            
            # Validation
            if (epoch + 1) % self.val_every == 0:
                log.info(f"Epoch {epoch + 1} complete! Avg Total Loss: {avg_total:.5f}")
                
                if len(self.pert_val) > 0:
                    corr_val = self._validate()
                    
                    if corr_val > corr_val_best:
                        corr_val_best = corr_val
                        self.model_best = copy.deepcopy(self.net)
                    
                    self.net.train()
            
            # Always update best model at the end of training
            if epoch == (self.training_epochs - 1):
                self.model_best = copy.deepcopy(self.net)

        log.info("Training completed!")
        
        self.net = self.model_best
        
        # Save model
        self._save_model()

    def _save_model(self):
        """Save the trained model and embeddings."""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model state
        state = {'net': self.net.state_dict()}
        torch.save(state, os.path.join(self.model_path, "model.pth"))
        
        # Save gene embeddings
        with open(os.path.join(self.model_path, "gene_embeddings.pkl"), 'wb') as f:
            pickle.dump(self.gene_embeddings, f)
        
        # Save model configuration
        config = {
            'x_dim': self.x_dim,
            'p_dim': self.p_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'multi_gene': self.multi_gene,
            'ctrl_mean': self.ctrl_mean.tolist()
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
        self.p_dim = config['p_dim']
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']
        self.multi_gene = config['multi_gene']
        self.ctrl_mean = np.array(config['ctrl_mean'], dtype=np.float32)
        
        # Initialize and load model
        self.net = SCLambdaNet(
            x_dim=self.x_dim,
            p_dim=self.p_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        state = torch.load(os.path.join(model_path, "model.pth"), map_location=self.device)
        self.net.load_state_dict(state['net'])
        self.net.eval()
        
        log.info(f"Model loaded from {model_path}")

    def predict(self, pert_test: Union[str, List[str]], 
                return_type: str = 'mean') -> Dict:
        """Predict perturbation effects."""
        self.net.eval()
        results = {}
        
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        
        for condition in pert_test:
            # Skip control conditions - they don't have perturbation embeddings
            if condition in ['control', 'ctrl']:
                continue
                
            genes = condition.split('+')
            # Filter out control/ctrl from gene list
            genes = [gene for gene in genes if gene not in ['control', 'ctrl']]
            
            if not genes:
                continue  # Skip if no actual genes left after filtering
            
            if self.multi_gene and len(genes) > 1:
                pert_emb_p = sum(self.gene_embeddings[gene] for gene in genes)
            else:
                pert_emb_p = self.gene_embeddings[genes[0]]
            
            # Use control cells for prediction
            if self.use_delta:
                if self.delta_type == 'ctrl_mean':
                    delta_reference = self.ctrl_mean
                elif self.delta_type == 'pert_mean':
                    delta_reference = self.pert_mean
                else:
                    raise ValueError(f"Invalid delta_type: {self.delta_type}")
            else:
                raise ValueError("use_delta must be True")
            
            # Create perturbation embedding matrix
            val_p = torch.from_numpy(
                np.tile(pert_emb_p, (self.ctrl_x.shape[0], 1))
            ).float().to(self.device)
            
            with torch.no_grad():
                x_hat, _, _, _, _ = self.net(self.ctrl_x, val_p)
                    
                if return_type == 'cells':
                    # Return individual predicted cells
                    # Add back baseline
                    results[condition] = x_hat.detach().cpu().numpy() + delta_reference
                elif return_type == 'mean':
                    # Return mean prediction
                    # Add back baseline
                    x_hat_mean = np.mean(x_hat.detach().cpu().numpy(), axis=0) + delta_reference
                    results[condition] = x_hat_mean
                else:
                    raise ValueError("return_type must be 'mean' or 'cells'")
        
        return results 

 


# =============================================================================
# CellSimBench Wrapper
# =============================================================================

class SCLambdaWrapper:
    """scLambda model wrapper for CellSimBench integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.gene_embeddings = {}
        self.data_manager = DataManager(self.config)
        self.mode = self.config['mode']
        
        # Initialize embedding generator (defer until we have gene list)
        # Handle gene_embedding_config in different locations (for training vs prediction)
        if 'gene_embedding_config' in self.config:
            self.embedding_config = self.config['gene_embedding_config']
        elif 'hyperparameters' in self.config and 'gene_embedding_config' in self.config['hyperparameters']:
            self.embedding_config = self.config['hyperparameters']['gene_embedding_config']
        else:
            raise KeyError("gene_embedding_config not found in config or config['hyperparameters']")
        self.embedding_generator = None
        if self.mode == 'predict':
            # For prediction, embeddings are loaded from the pretrained model
            if 'gene_embeddings_path' in self.config:
                self.gene_embeddings_path = self.config['gene_embeddings_path']
            else:
                # Default to model_path/gene_embeddings.pkl for benchmark prediction
                self.gene_embeddings_path = str(Path(self.config['model_path']) / 'gene_embeddings.pkl')
            
            with open(self.gene_embeddings_path, 'rb') as f:
                self.gene_embeddings = pickle.load(f)
        else:
            self.gene_embeddings_path = None
        
    def train(self):
        """Train scLambda model with internal checkpointing."""
        log.info("Starting scLambda training process...")
        
        # Load and preprocess data
        log.info("Loading CellSimBench data...")
        adata = self.data_manager.load_dataset()
        log.info(f"Loaded data with shape: {adata.shape}")

        # Convert to scLambda format
        log.info("Converting to scLambda format...")
        adata_sclambda = self._convert_to_sclambda_format(adata)
        
        # Get unique genes and initialize embedding generator with gene list
        unique_genes = self._get_unique_genes(adata_sclambda)
        log.info(f"Found {len(unique_genes)} unique genes")
        
        hyperparams = self.config['hyperparameters']
        self.embedding_generator = GeneEmbeddingGenerator(
            model=hyperparams['embedding_model'],
            dimensions=hyperparams['embedding_dimensions'],
            cache_dir=self.embedding_config['cache_dir'],
            api_key_env=self.embedding_config['openai_api_key_env'],
            gene_list=unique_genes  # Pass gene list for bulk loading
        )
        
        # Generate gene embeddings (descriptions already fetched during init)
        log.info("Generating gene embeddings...")
        # Check if gene embeddings already saved to file
        if not (Path(self.embedding_config['cache_dir']) / 'gene_embeddings.pkl').exists():
            self.gene_embeddings = self.embedding_generator.get_gene_embeddings_batch(unique_genes)
            with open(Path(self.embedding_config['cache_dir']) / 'gene_embeddings.pkl', 'wb') as f:
                pickle.dump(self.gene_embeddings, f)
        else:
            log.info("Loading gene embeddings from file...")
            with open(Path(self.embedding_config['cache_dir']) / 'gene_embeddings.pkl', 'rb') as f:
                self.gene_embeddings = pickle.load(f)
        
        # Initialize and train model
        log.info("Initializing scLambda model...")
        hyperparams = self.config['hyperparameters']
        if not hyperparams['use_delta']:
            raise ValueError("use_delta must be True")
        
        self.model = SCLambdaModel(
            adata=adata_sclambda,
            gene_embeddings=self.gene_embeddings,
            split_name=self.config['split_name'],
            latent_dim=hyperparams['latent_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            training_epochs=hyperparams['training_epochs'],
            batch_size=hyperparams['batch_size'],
            lambda_MI=hyperparams['lambda_MI'],
            eps=hyperparams['eps'],
            seed=hyperparams['seed'],  # Add seed from hyperparameters
            model_path=self.config['output_dir'],
            multi_gene=hyperparams['multi_gene'],
            val_every=hyperparams['val_every'],
            use_delta=hyperparams['use_delta'],
            delta_type=hyperparams['delta_type']
        )
        # Setup the model
        self.model._setup()
        # Train the model
        self.model.train()
        
        # Save metadata
        self._save_metadata()
        
        log.info("Training completed successfully")

    def predict(self):
        """Generate predictions using trained scLambda model."""
        log.info("Starting scLambda prediction process...")
        
        # Load trained model
        model_path = self.config['model_path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        # Load data for prediction
        adata = self.data_manager.load_dataset()
        adata_sclambda = self._convert_to_sclambda_format(adata)

        # Initialize model and load pretrained weights
        hyperparams = self.config['hyperparameters']
        self.model = SCLambdaModel(
            adata=adata_sclambda,
            gene_embeddings=self.gene_embeddings,  # Will be loaded from checkpoint
            split_name=self.config['split_name'],
            latent_dim=hyperparams['latent_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            training_epochs=hyperparams['training_epochs'],
            batch_size=hyperparams['batch_size'],
            lambda_MI=hyperparams['lambda_MI'],
            eps=hyperparams['eps'],
            seed=hyperparams['seed'],  # Add seed from hyperparameters
            multi_gene=hyperparams['multi_gene'],
            val_every=hyperparams['val_every'],
            use_delta=hyperparams['use_delta'],
            delta_type=hyperparams['delta_type']
        )
        self.model.load_pretrained(model_path)

        # Setup the model
        self.model._setup()
        
        # Generate predictions
        test_conditions = self.config['test_conditions']
        # Convert test conditions to scLambda format
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

    def _convert_to_sclambda_format(self, adata: sc.AnnData) -> sc.AnnData:
        """Convert CellSimBench data to scLambda format."""
        adata_sclambda = adata.copy()
        
        # Define control condition values (same as GEARS and scGPT)
        CTRL_VALUE = 'ctrl_iegfp'
        
        # Process condition labels for scLambda format
        def process_condition(cond):
            if cond == 'control' or cond == 'ctrl' or cond == CTRL_VALUE:
                return 'ctrl'
            elif '+' not in cond:
                # Single perturbation
                return f"{cond}+ctrl"
            else:
                # Already in correct format for combo perturbations
                return cond

        adata_sclambda.obs['condition'] = adata_sclambda.obs['condition'].astype(str)
        adata_sclambda.obs['condition'] = adata_sclambda.obs['condition'].apply(process_condition)
        
        return adata_sclambda

    def _get_unique_genes(self, adata: sc.AnnData) -> List[str]:
        """Extract unique genes from perturbation conditions."""
        all_genes = set()
        
        for condition in adata.obs['condition'].unique():
            if condition != 'ctrl':
                genes = condition.replace('+ctrl', '').split('+')
                all_genes.update(genes)
        
        return sorted(list(all_genes))

    def _convert_to_cellsimbench_format(self, predictions: Dict, original_adata: sc.AnnData) -> sc.AnnData:
        """Convert scLambda predictions to CellSimBench format."""
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
                # Convert condition to scLambda format for lookup
                sclambda_condition = condition if '+' in condition else f"{condition}+ctrl"
                
                if sclambda_condition in predictions:
                    # Get prediction (now just a single vector per condition)
                    pred_vector = predictions[sclambda_condition]
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
            'model_type': 'scLambda',
            'config': self.config,
            'data_shape': self.model.adata.shape if self.model else None,
            'embedding_dimensions': self.model.p_dim if self.model else None,
            'n_unique_genes': len(self.gene_embeddings) - 1 if self.gene_embeddings else None  # -1 for ctrl
        }

        output_dir = Path(self.config['output_dir'])
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=PathEncoder) 