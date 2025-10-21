#! /usr/bin/env python3
import argparse
import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import scanpy as sc
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# ===============================================================================
#                              Helper Functions
# ===============================================================================
def download_and_extract(url, output_folder):
    """Download and extract a gzipped file from Ensembl."""
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    filename = url.split('/')[-1]
    output_path = os.path.join(output_folder, filename)
    extracted_path = output_path[:-3]  # Remove .gz extension
    
    # If the file already exists, return the path
    if os.path.exists(extracted_path):
        return extracted_path

    # Download file
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save compressed file
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract file
    print(f"Extracting {filename}...")
    with gzip.open(output_path, 'rb') as f_in:
        with open(extracted_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    return extracted_path


# ===============================================================================
#                              Caching Functions
# ===============================================================================
def get_cache_path(
    cache_dir: str, 
    model_name: str, 
    embedding_type: str, 
    gene_name: str
) -> Path:
    """Generate cache file path for a specific gene embedding.
    
    Args:
        cache_dir: Base cache directory
        model_name: Model identifier (e.g., 'facebook/esm2_t30_150M_UR50D')
        embedding_type: Type of embedding (bos, mean, sum, max)
        gene_name: Gene symbol
        
    Returns:
        Path to cache file
    """
    # Clean model name for filesystem
    clean_model = model_name.replace('/', '_').replace('-', '_')
    cache_path = Path(cache_dir) / clean_model / embedding_type
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"{gene_name}.pkl"


def load_from_cache(
    cache_dir: str,
    model_name: str, 
    embedding_type: str,
    gene_name: str
) -> Optional[np.ndarray]:
    """Load embedding from cache if it exists.
    
    Args:
        cache_dir: Base cache directory
        model_name: Model identifier
        embedding_type: Type of embedding (bos, mean, sum, max)
        gene_name: Gene symbol
        
    Returns:
        Cached embedding array or None if not found
    """
    cache_path = get_cache_path(cache_dir, model_name, embedding_type, gene_name)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache for {gene_name}: {e}")
    return None


def save_to_cache(
    cache_dir: str,
    model_name: str,
    embedding_type: str,
    gene_name: str,
    embedding: np.ndarray
) -> None:
    """Save embedding to cache.
    
    Args:
        cache_dir: Base cache directory
        model_name: Model identifier
        embedding_type: Type of embedding (bos, mean, sum, max)
        gene_name: Gene symbol
        embedding: Embedding array to save
    """
    cache_path = get_cache_path(cache_dir, model_name, embedding_type, gene_name)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    except Exception as e:
        print(f"Warning: Failed to save cache for {gene_name}: {e}")


def get_sequence_df(canonical_path, pep_path):
    """
    Create aligned DataFrame of canonical sequences.
    
    Args:
        canonical_path (str): Path to canonical TSV file
        pep_path (str): Path to peptide FASTA file
        
    Returns:
        pd.DataFrame: Aligned DataFrame containing only Ensembl Canonical
            sequences
    """
    # Read and analyze canonical TSV file
    print("Analyzing canonical TSV file...")
    df_canonical = pd.read_csv(
        canonical_path, 
        sep='\t', 
        names=[
            'ensembl_gene_id', 
            'ensembl_transcript_id', 
            'transcript_is_canonical'
        ]
    )
    
    # Filter for Ensembl Canonical genes
    df_canonical = df_canonical[
        df_canonical.iloc[:, 2] == "Ensembl Canonical"
    ]
    
    # Read and analyze peptide FASTA file
    print("Analyzing peptide FASTA file...")
    # Convert FASTA to DataFrame
    records = []
    for record in SeqIO.parse(pep_path, "fasta"):
        # Start with basic record info
        record_dict = {
            'id': record.id,
            'description': record.description,
            'sequence': str(record.seq)
        }
        
        # Parse description into key-value pairs
        # Skip the first part which is the ID
        desc_parts = record.description.split(' ')[1:]
        for part in desc_parts:
            if ':' in part:
                key, value = part.split(':', 1)
                # Extract transcript ID and gene symbol/name
                if key in ["transcript", "gene_symbol"]:
                    record_dict[key] = value
        
        records.append(record_dict)
    
    df_fasta = pd.DataFrame(records)
    
    # Align dataframes using transcript IDs
    merged_df = pd.merge(
        df_canonical,
        df_fasta,
        left_on='ensembl_transcript_id',
        right_on='transcript',
        how='inner'
    ).drop('transcript', axis=1)
    
    return merged_df


def get_one_seq_embedding(seq, model, tokenizer):
    """Get ESM2 embeddings for a single sequence."""
    # Tokenize sequences    
    curr_tokens = tokenizer(seq, return_tensors="pt").to(model.device)
    # Generate embeddings
    if len(seq) > 15_000:
        print(f"Sequence too long: {len(seq)}")
        return None
    
    try:
        with torch.inference_mode():
            outputs = model(**curr_tokens)
            embeddings = outputs.last_hidden_state
    except Exception as e:
        print(f"Error generating embeddings for {seq}: {e}")
        return None

    return_dict = {
        'embeddings_esm2_bos': embeddings[0, 0, :].cpu().numpy(),
        'embeddings_esm2_mean': embeddings.mean(dim=1).flatten().cpu().numpy(),
        'embeddings_esm2_sum': embeddings.sum(dim=1).flatten().cpu().numpy(),
        'embeddings_esm2_max': embeddings.max(dim=1).values.flatten().cpu().numpy()
    }
    return return_dict


# ===============================================================================
#                         Main ESM2 Embeddings Function
# ===============================================================================
def get_esm2_embeddings(
    adata_path: str,
    output_path: str,
    cache_dir: str = "data/gene_embeddings/esm2/cache",
    model_name: str = "facebook/esm2_t30_150M_UR50D",
):
    """Generate ESM2 gene embeddings for an AnnData object.

    Args:
        adata_path: Path to input h5ad file
        output_path: Path to save output h5ad file with embeddings
        cache_dir: Directory to store/load cached embeddings
        model_name: ESM2 model to use (150M, 3B, or 15B variant)
    """
    # URLs
    canonical_url = (
        "https://ftp.ensembl.org/pub/release-113/tsv/homo_sapiens/"
        "Homo_sapiens.GRCh38.113.canonical.tsv.gz"
    )
    pep_url = (
        "https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/pep/"
        "Homo_sapiens.GRCh38.pep.all.fa.gz"
    )
    
    # Output folder
    download_folder = os.path.join("data", "gene_embeddings", "esm2")

    # Download and extract files
    canonical_path = download_and_extract(canonical_url, download_folder)
    pep_path = download_and_extract(pep_url, download_folder)
    
    # Get aligned sequence DataFrame
    sequence_df = get_sequence_df(canonical_path, pep_path)

    # Load ESM2 model
    print(f"Loading ESM2 model: {model_name}")
    model = EsmModel.from_pretrained(model_name, device_map="auto").half()
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    # Prepare sequence data for processing
    sequence_df['len_seq'] = sequence_df['sequence'].apply(len)
    sequence_df = sequence_df.sort_values(by='len_seq', ascending=False)
    
    # Use gene_symbol as index for embeddings
    sequence_df = sequence_df.set_index('gene_symbol', drop=False)

    # Remove entries without a gene symbol and print number removed
    n_before = len(sequence_df)
    sequence_df = sequence_df[sequence_df['gene_symbol'].notna()]
    n_after = len(sequence_df)
    print(f"Removed {n_before - n_after} entries without a gene symbol")
    # Remove duplicate gene symbols and print number removed
    n_before = len(sequence_df)
    sequence_df = sequence_df.drop_duplicates(subset='gene_symbol')
    n_after = len(sequence_df)
    print(f"Removed {n_before - n_after} duplicate gene symbol entries")

    # TESTING: Using 100 shortest sequences for testing #########################
    # sequence_df = sequence_df.tail(100)
    ###########################################################################

    
    # Initialize embeddings dictionaries
    embeddings_dict = {
        'embeddings_esm2_bos': {},
        'embeddings_esm2_mean': {},
        'embeddings_esm2_sum': {},
        'embeddings_esm2_max': {}
    }

    total_len = len(sequence_df)
    total_skipped = 0
    cache_hits = 0
    cache_misses = 0
    
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate embeddings for all sequences
    with tqdm(total=total_len, desc="Generating ESM2 embeddings") as pbar:
        for gene_symbol, row in sequence_df.iterrows():
            curr_seq = row['sequence']
            
            # Check cache first for all embedding types
            cached_embeddings = {}
            all_cached = True
            for emb_type in embeddings_dict.keys():
                # Remove 'embeddings_esm2_' prefix for cache key
                cache_emb_type = emb_type.replace('embeddings_esm2_', '')
                cached_emb = load_from_cache(cache_dir, model_name, cache_emb_type, gene_symbol)
                if cached_emb is not None:
                    cached_embeddings[emb_type] = cached_emb
                else:
                    all_cached = False
                    break
            
            if all_cached:
                # Use cached embeddings
                for emb_type, emb_val in cached_embeddings.items():
                    embeddings_dict[emb_type][gene_symbol] = emb_val
                cache_hits += 1
                pbar.set_postfix({'cache_hits': cache_hits, 'cache_misses': cache_misses})
                pbar.update(1)
                continue
            
            # Generate new embeddings if not cached
            cache_misses += 1
            curr_emb_dict = get_one_seq_embedding(curr_seq, model, tokenizer)
            if curr_emb_dict is None:
                print(f"Skipping {gene_symbol} because sequence is too long")
                total_skipped += 1
                pbar.update(1)
                continue
            
            # Add embeddings to dict and save to cache
            for emb_type, emb_val in curr_emb_dict.items():
                embeddings_dict[emb_type][gene_symbol] = emb_val
                # Save to cache -- Cached with model name for future flexibility [Coding Agent]
                cache_emb_type = emb_type.replace('embeddings_esm2_', '')
                save_to_cache(cache_dir, model_name, cache_emb_type, gene_symbol, emb_val)
            
            pbar.set_postfix({'cache_hits': cache_hits, 'cache_misses': cache_misses})
            pbar.update(1)
            
    print(f"Total skipped: {total_skipped}")
    print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
    
    dataframes_dict = {k: pd.DataFrame.from_dict(v, orient='index') for k, v in embeddings_dict.items()}

    print('='*100)
    print(f"ESM2 embeddings")
    print('='*100)
    print(f"Input data: {adata_path}")
    print(f"Output data: {output_path}")
    print('='*100)
    
    
    # Read adata
    adata = sc.read(adata_path)
    

    # Add embeddings to adata
    for emb_key, emb_df in dataframes_dict.items():
        
        # Sort by index and manual modifications
        emb_df.sort_index(inplace=True)
        emb_df.columns = emb_df.columns.astype(str)

        # If any duplicates, remove and warn
        if emb_df.index.duplicated().any():
            print("Warning: Duplicate genes found in embeddings index.\nRemoving duplicates.")
            emb_df = emb_df[~emb_df.index.duplicated(keep="first")]

        adata.uns[emb_key] = emb_df

        # Print important information
        print('='*100)
        print(f"{emb_key} embeddings")
        print('='*100)
        print(f"Number of genes: {len(emb_df)}")
        print(f"Embedding dimension: {emb_df.shape[1]}")
        print(f"Added key: {emb_key}")
        print('='*100)

    adata.write_h5ad(output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate ESM2 embeddings for an AnnData object')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to save output h5ad file')
    parser.add_argument('--cache_dir', default='data/gene_embeddings/esm2/cache', 
                       help='Directory to store/load cached embeddings (default: data/gene_embeddings/esm2/cache)')
    parser.add_argument('--model', 
                       choices=['facebook/esm2_t30_150M_UR50D', 
                               'facebook/esm2_t36_3B_UR50D',
                               'facebook/esm2_t48_15B_UR50D'],
                       default='facebook/esm2_t30_150M_UR50D',
                       help='ESM2 model variant to use (default: 150M)')
    
    args = parser.parse_args()

    get_esm2_embeddings(
        adata_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()