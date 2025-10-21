#! /usr/bin/env python3
import pickle
import scanpy as sc
from geneformer.perturber_utils import load_model
import torch
import numpy as np
import argparse
import pandas as pd

def load_geneformer_model(model_dir: str, tokenizer_dir: str, gene_name_dict_dir: str):
    """Load Geneformer model from directory."""

    # Load model
    model = load_model(model_type="Pretrained", num_classes=None, model_directory=model_dir, mode="eval")

    # Load tokenizer
    with open(tokenizer_dir, 'rb') as f:
        gene2idx = pickle.load(f)

    # Load gene name dictionary
    with open(gene_name_dict_dir, 'rb') as f:
        gene_name_dict = pickle.load(f)

    return model, gene2idx, gene_name_dict

def get_geneformer_embeddings(
    adata_path: str,
    output_path: str,
    model_dir: str = "data/gene_embeddings/Geneformer/Geneformer-V2-104M",
    tokenizer_dir: str = "data/gene_embeddings/Geneformer/geneformer/token_dictionary_gc104M.pkl",
    gene_name_dict_dir: str = "data/gene_embeddings/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl",
):
    """Generate Geneformer gene embeddings for an AnnData object.

    Args:
        adata_path: Path to input h5ad file
        output_path: Path to save output h5ad file with embeddings
        model_dir: Directory containing pretrained Geneformer model
        tokenizer_dir: Directory containing Geneformer tokenizer
        gene_name_dict_dir: Directory containing Geneformer gene name dictionary
    """
    print("Loading model...")
    model, gene2idx, gene_name_dict = load_geneformer_model(model_dir, tokenizer_dir, gene_name_dict_dir)

    # Retrieve the data-independent gene embeddings from Geneformer
    gene_ids = pd.Series(gene2idx)
    gene_ids_list = gene_ids.tolist()
    embedding_layer = model.bert.embeddings.word_embeddings
    gene_embeddings = embedding_layer(torch.tensor(gene_ids_list, dtype=torch.long, device=model.device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()
    gene_embeddings = pd.DataFrame(gene_embeddings, index=gene_ids.index)

    # Mapping from ensemple ids to gene names
    ensg_2_gene_name = {v:k for k,v in gene_name_dict.items()}
    unmapped_genes = set(gene_embeddings.index) - set(ensg_2_gene_name.keys())
    gene_embeddings = gene_embeddings.drop(unmapped_genes)
    gene_embeddings.index = [ensg_2_gene_name[ensg] for ensg in gene_embeddings.index]
    
    # Sort by index and manual modifications
    gene_embeddings.sort_index(inplace=True)
    gene_embeddings.columns = gene_embeddings.columns.astype(str)

    # If any duplicates, remove and warn
    if gene_embeddings.index.duplicated().any():
        print("Warning: Duplicate genes found in embeddings index.\nRemoving duplicates.")
        gene_embeddings = gene_embeddings[~gene_embeddings.index.duplicated(keep="first")]

    # Read adata and add embeddings
    adata = sc.read_h5ad(adata_path)
    adata.uns['embeddings_geneformer'] = gene_embeddings
    
    # Save result
    adata.write_h5ad(output_path)

    # Print important information
    print('='*100)
    print("Geneformer embeddings")
    print('='*100)
    print(f"Input data: {adata_path}")
    print(f"Output data: {output_path}")
    print(f"Number of genes: {len(gene_embeddings)}")
    print(f"Embedding dimension: {gene_embeddings.shape[1]}")
    print(f"Added key: embeddings_geneformer")
    print('='*100)
    
def main():
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings for an AnnData object')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--output', required=True, help='Path to save output h5ad file')
    parser.add_argument('--model_dir', required=True, default="data/gene_embeddings/Geneformer/Geneformer-V2-104M", help='Directory containing pretrained Geneformer model')
    parser.add_argument('--tokenizer_dir', required=True, default="data/gene_embeddings/Geneformer/geneformer/token_dictionary_gc104M.pkl", help='Directory containing Geneformer tokenizer')
    parser.add_argument('--gene_name_dict_dir', required=True, default="data/gene_embeddings/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl", help='Directory containing Geneformer gene name dictionary')
    
    args = parser.parse_args()

    get_geneformer_embeddings(
        adata_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        gene_name_dict_dir=args.gene_name_dict_dir,
    )

if __name__ == '__main__':
    main()

