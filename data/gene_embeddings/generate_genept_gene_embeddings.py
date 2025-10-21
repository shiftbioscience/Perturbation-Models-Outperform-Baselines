import requests
import gzip
import os
import pandas as pd
from pathlib import Path
import argparse
import scanpy as sc
import numpy as np
from tqdm import tqdm
import json
import hashlib
import time
import re
import urllib.request
from pybiomart import Dataset as BioMartDataset
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set CUDA device if available
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class GeneDescriptionFetcher:
    """Fetches gene descriptions from Ensembl using pybiomart with caching and error handling."""
    
    def __init__(self, cache_dir: str = "./gene_descriptions_cache", gene_list=None):
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
    
    def get_gene_descriptions_batch(self, gene_symbols):
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
                 gene_list=None,
                 data_source: str = "ncbi"):
        self.model = model
        self.dimensions = dimensions
        self.data_source = data_source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hash for model and data source configuration
        config_string = f"{model}_{dimensions}_{data_source}"
        self.config_hash = hashlib.md5(config_string.encode()).hexdigest()[:8]
        
        # Create embeddings directory with hash
        self.embeddings_dir = self.cache_dir / self.config_hash
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Using embedding cache directory: {self.embeddings_dir}")
        log.info(f"Configuration hash: {self.config_hash} (model={model}, dims={dimensions}, source={data_source})")
        
        # Load .env file if it exists (for OpenAI API key)
        env_file = Path('.env')
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
                raise RuntimeError(f"Failed to load .env file for GenePT: {e}")
        
        # Set up OpenAI client - fail fast if key is missing
        api_key = os.getenv(api_key_env)
        if not api_key:
            error_msg = f"GenePT requires OpenAI API key but {api_key_env} environment variable not found."
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
    
    def _get_embedding_file_path(self, gene_name: str) -> Path:
        """Get the file path for a gene's embedding."""
        # Sanitize gene name for filename
        safe_gene_name = re.sub(r'[^\w\-_.]', '_', gene_name)
        return self.embeddings_dir / f"{safe_gene_name}.npy"
    
    def _load_cached_embedding(self, gene_name: str) -> np.ndarray:
        """Load a cached embedding for a gene."""
        embedding_file = self._get_embedding_file_path(gene_name)
        if embedding_file.exists():
            try:
                embedding = np.load(embedding_file)
                return embedding.astype(np.float32)
            except Exception as e:
                log.warning(f"Failed to load cached embedding for {gene_name}: {e}")
                return None
        return None
    
    def _save_embedding(self, gene_name: str, embedding: np.ndarray):
        """Save an embedding for a gene."""
        embedding_file = self._get_embedding_file_path(gene_name)
        try:
            np.save(embedding_file, embedding.astype(np.float32))
            log.debug(f"Saved embedding for {gene_name} to {embedding_file}")
        except Exception as e:
            log.error(f"Failed to save embedding for {gene_name}: {e}")
    
    def _count_cached_embeddings(self) -> int:
        """Count the number of cached embeddings."""
        return len(list(self.embeddings_dir.glob("*.npy")))
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        if not text.strip():
            log.warning("Empty text provided, returning zero embedding")
            return np.zeros(self.dimensions, dtype=np.float32)
        
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
        # Check if embedding is already cached
        cached_embedding = self._load_cached_embedding(gene_symbol)
        if cached_embedding is not None:
            log.debug(f"Loaded cached embedding for {gene_symbol}")
            return cached_embedding
        
        # Get gene description
        description = self.description_fetcher.get_gene_description(gene_symbol)
        
        # Generate embedding
        log.info(f"Generating embedding for gene: {gene_symbol}")
        embedding = self._get_openai_embedding(description)
        
        # Save the embedding to cache
        self._save_embedding(gene_symbol, embedding)
        
        return embedding
    
    def get_gene_embeddings_batch(self, gene_symbols) -> dict:
        """Get embeddings for multiple genes with rate limiting."""
        log.info(f"Processing embeddings for {len(gene_symbols)} genes...")
        
        # Count existing cached embeddings
        initial_cached_count = self._count_cached_embeddings()
        log.info(f"Found {initial_cached_count} existing cached embeddings")
        
        embeddings = {}
        new_embeddings_count = 0
        cached_count = 0
        
        for gene_symbol in tqdm(gene_symbols, desc="Processing gene embeddings"):
            # Check if embedding is already cached
            cached_embedding = self._load_cached_embedding(gene_symbol)
            if cached_embedding is not None:
                embeddings[gene_symbol] = cached_embedding
                cached_count += 1
            else:
                # Generate new embedding
                embedding = self.get_gene_embedding(gene_symbol)
                embeddings[gene_symbol] = embedding
                new_embeddings_count += 1
                time.sleep(0.05)  # Rate limiting for API calls
        
        log.info(f"Embedding generation complete:")
        log.info(f"  - Loaded from cache: {cached_count}")
        log.info(f"  - Generated new: {new_embeddings_count}")
        log.info(f"  - Total processed: {len(embeddings)}")
        
        return embeddings


def get_all_human_gene_symbols(include_only_protein_coding: bool = True) -> list[str]:
    """Get human gene symbols from Ensembl using BioMart with specific filters.
    
    Args:
        include_only_protein_coding: If True, only include protein-coding genes.
                                   If False, include all gene biotypes.
    
    Returns:
        List of unique human gene symbols from Ensembl.
        
    Raises:
        RuntimeError: If unable to connect to or query the BioMart database.
        ValueError: If no gene symbols are retrieved.
    """
    log.info("Connecting to Ensembl BioMart to retrieve human gene symbols...")
    
    try:
        # Initialize Ensembl dataset connection
        dataset = BioMartDataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
        log.info("Successfully connected to Ensembl BioMart")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ensembl BioMart: {e}")
    
    try:
        # Query for human gene symbols with biotype information
        log.info("Querying BioMart for human gene symbols with biotype filtering...")
        
        # Get gene symbols along with biotype and status for filtering
        attributes = ['external_gene_name', 'gene_biotype', 'chromosome_name']
        result = dataset.query(attributes=attributes)
        
        if result.empty:
            raise ValueError("BioMart query returned no results")
        
        log.info(f"Initial query returned {len(result)} gene records")
        
        # Filter for standard chromosomes (1-22, X, Y, MT) to exclude scaffolds
        standard_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        result = result[result['Chromosome/scaffold name'].isin(standard_chromosomes)]
        log.info(f"After chromosome filtering: {len(result)} records")
        
        # Apply biotype filtering if requested
        if include_only_protein_coding:
            result = result[result['Gene type'] == 'protein_coding']
            log.info(f"After protein-coding filter: {len(result)} records")
        
        # Extract unique gene symbols and remove empty/null entries
        gene_symbols = result['Gene name'].dropna().unique().tolist()
        gene_symbols = [symbol for symbol in gene_symbols if symbol.strip()]
        
        if not gene_symbols:
            raise ValueError("No valid gene symbols found after filtering")
        
        filter_desc = "protein-coding" if include_only_protein_coding else "all biotypes"
        log.info(f"Retrieved {len(gene_symbols)} unique human gene symbols ({filter_desc}) from Ensembl")
        
        return gene_symbols
        
    except Exception as e:
        raise RuntimeError(f"Failed to query BioMart for gene symbols: {e}")


def get_genept_embeddings(
    adata_path: str,
    output_path: str,
    cache_dir: str = "data/gene_embeddings/genept", 
    model: str = "text-embedding-3-large",
    dimensions: int = 3072,
    data_source: str = "ncbi",
):
    """Generate GenePT (OpenAI text-embedding) gene embeddings for an AnnData object.

    Args:
        adata_path: Path to input h5ad file
        output_path: Path to save output h5ad file with embeddings
        cache_dir: Directory to cache embeddings and descriptions
        model: OpenAI embedding model to use
        dimensions: Embedding dimensions
        data_source: Data source for gene descriptions (e.g., ncbi, ensembl, uniprot)
    """
    print("Loading GenePT embedding generation...")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Get unique gene names from all human genes
    gene_names = get_all_human_gene_symbols(include_only_protein_coding=True)
    log.info(f"Found {len(gene_names)} genes for embedding generation")
    
    # Initialize embedding generator
    embedding_generator = GeneEmbeddingGenerator(
        model=model,
        dimensions=dimensions,
        cache_dir=str(cache_path),
        gene_list=gene_names,
        data_source=data_source
    )
    
    # Generate embeddings for all genes
    embeddings_dict = embedding_generator.get_gene_embeddings_batch(gene_names)
    
    # Convert to DataFrame format
    embedding_matrix = np.zeros((len(gene_names), dimensions), dtype=np.float32)
    
    for i, gene_name in enumerate(gene_names):
        if gene_name in embeddings_dict:
            embedding_matrix[i] = embeddings_dict[gene_name]
        else:
            log.warning(f"No embedding found for gene: {gene_name}")
            embedding_matrix[i] = np.zeros(dimensions, dtype=np.float32)
    
    # Create DataFrame with proper column names
    gene_embeddings = pd.DataFrame(
        embedding_matrix, 
        index=gene_names,
        columns=[str(i) for i in range(dimensions)]
    )
    
    # Sort by index and manual modifications
    gene_embeddings.sort_index(inplace=True)
    gene_embeddings.columns = gene_embeddings.columns.astype(str)

    # If any duplicates, remove and warn
    if gene_embeddings.index.duplicated().any():
        print("Warning: Duplicate genes found in embeddings index.\nRemoving duplicates.")
        gene_embeddings = gene_embeddings[~gene_embeddings.index.duplicated(keep="first")]

    # Read adata and add embeddings
    adata = sc.read_h5ad(adata_path)
    adata.uns['embeddings_genept'] = gene_embeddings
    
    # Save result
    adata.write_h5ad(output_path)

    # Print important information
    print('='*100)
    print("GenePT embeddings")
    print('='*100)
    print(f"Input data: {adata_path}")
    print(f"Output data: {output_path}")
    print(f"Number of genes: {len(gene_embeddings)}")
    print(f"Embedding dimension: {gene_embeddings.shape[1]}")
    print(f"Added key: embeddings_genept")
    print('='*100)


def main():
    parser = argparse.ArgumentParser(
        description='Generate GenePT (OpenAI text-embedding) embeddings for an AnnData object',
        epilog='Embeddings are cached as individual .npy files in <cache-dir>/<config-hash>/<gene-name>.npy where config-hash includes model, dimensions, and data-source'
    )
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to save output h5ad file')
    parser.add_argument('--cache_dir', default='data/gene_embeddings/genept', help='Directory to cache embeddings and descriptions')
    parser.add_argument('--model', default='text-embedding-3-large', help='OpenAI embedding model to use')
    parser.add_argument('--dimensions', type=int, default=3072, help='Embedding dimensions')
    parser.add_argument('--data_source', default='ncbi', help='Data source for gene descriptions (e.g., ncbi, ensembl, uniprot)')
    
    args = parser.parse_args()

    get_genept_embeddings(
        adata_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model=args.model,
        dimensions=args.dimensions,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()