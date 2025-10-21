#! /usr/bin/env python3
import pickle
import scanpy as sc
import numpy as np
import argparse
import pandas as pd
import os
import glob

def get_presage_embeddings(
    adata_path: str,
    output_path: str,
    embeddings_dir: str = "data/gene_embeddings/PRESAGE/cache",
):
    """Generate PRESAGE gene embeddings for an AnnData object.

    Args:
        adata_path: Path to input h5ad file
        output_path: Path to save output h5ad file with embeddings
        model_dir: Directory containing pretrained Geneformer model
        tokenizer_dir: Directory containing Geneformer tokenizer
        gene_name_dict_dir: Directory containing Geneformer gene name dictionary
    """
    print('='*100)
    print("Getting all PRESAGE embeddings")
    print(f"Input data: {adata_path}")
    print(f"Output data: {output_path}")
    print('='*100)

    # Read adata
    adata = sc.read_h5ad(adata_path)
    
    # Get all paths to embeddings
    embedding_paths = glob.glob(os.path.join(embeddings_dir, "**", "*.pkl"), recursive=True)
    
    for embedding_path in embedding_paths:
        emb_name = os.path.basename(embedding_path).replace(".pkl", "")
        with open(embedding_path, "rb") as f:
            emb = pickle.load(f)
        
        if isinstance(emb, pd.DataFrame):
            emb.columns = emb.columns.astype(str)
            adata.uns[f"embeddings_{emb_name}"] = emb

        print('='*100)
        print(f"{emb_name} embeddings")
        print('='*100)
        print(f"Number of genes: {len(emb)}")
        print(f"Embedding dimension: {emb.shape[1]}")
        print(f"Added key: embeddings_{emb_name}")
        print('='*100)
    
    # Save adata
    adata.write_h5ad(output_path)

    
    
def main():
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings for an AnnData object')
    parser.add_argument('--input', required=False, default="data/norman19/norman19_processed_complete.h5ad", help='Path to input h5ad file')
    parser.add_argument('--output', required=False, default="data/norman19/test.h5ad", help='Path to save output h5ad file')
    parser.add_argument('--embeddings_dir', default="data/gene_embeddings/PRESAGE/cache", help='Directory containing PRESAGE embeddings')
    
    args = parser.parse_args()

    get_presage_embeddings(
        adata_path=args.input,
        output_path=args.output,
        embeddings_dir=args.embeddings_dir,
    )

if __name__ == '__main__':
    main()

