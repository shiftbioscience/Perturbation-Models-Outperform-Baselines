#! /usr/bin/env python3
import scanpy as sc
import scgpt as scg
import argparse
import torch
import json
import os
from pathlib import Path
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
import numpy as np
import subprocess
import shutil
import pandas as pd

def download_scgpt_model(model_dir: str, gdown_path: str):
    """Download scGPT model if it doesn't exist.
    
    Args:
        model_dir: Directory where model should be stored
    """
    
    model_path = os.path.join(model_dir, "best_model.pt")
    
    # Check if model already exists
    if os.path.exists(model_path):
        print("scGPT model already exists, skipping download")
        return
    
    print("Downloading scGPT model...")
    
    # Create temp directory for download
    temp_dir = os.path.join(model_dir, "scGPT_CP")
    
    # Download using gdown
    subprocess.run([gdown_path, "--folder", "1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB", "--output", temp_dir], check=True)
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Move files from temp directory to model directory
    for file in os.listdir(temp_dir):
        src = os.path.join(temp_dir, file)
        dst = os.path.join(model_dir, file)
        if not os.path.exists(dst):  # Only copy if file doesn't exist
            shutil.copy2(src, dst)
    
    # Remove temp directory
    shutil.rmtree(temp_dir)
    
    print("Model downloaded successfully")

def load_scgpt_model(model_dir: str):
    """Load scGPT model from directory."""

    # Define special constant values
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = -2
    n_bins = 51
    n_input_bins = n_bins
    ###


    # Specify model path; here we load the pre-trained scGPT blood model
    model_dir = Path(model_dir)
    model_config_file = os.path.join(model_dir, "args.json")
    model_file = os.path.join(model_dir, "best_model.pt")
    vocab_file = os.path.join(model_dir, "vocab.json")

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    gene2idx = vocab.get_stoi()

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    
    model.eval()

    return model, gene2idx

def get_scgpt_embeddings(adata_path: str, model_dir: str, output_path: str):
    """Generate scGPT cell and gene embeddings for an AnnData object.
    For gene embeddings we follow the tutorial at:
    https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_GRN.ipynb
    
    Args:
        adata_path: Path to input h5ad file
        model_dir: Directory containing pretrained scGPT model
        output_path: Path to save output h5ad file with embeddings
    """
    
    print("Loading model...")
    model, gene2idx = load_scgpt_model(model_dir)

    # Retrieve the data-independent gene embeddings from scGPT
    gene_ids = pd.Series(gene2idx)
    gene_ids_list = gene_ids.tolist()
    gene_embeddings = model.encoder(torch.tensor(gene_ids_list, dtype=torch.long))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()
    gene_embeddings = pd.DataFrame(gene_embeddings, index=gene_ids.index)

    # Sort by index and manual modifications
    gene_embeddings.sort_index(inplace=True)
    gene_embeddings.columns = gene_embeddings.columns.astype(str)

    # If any duplicates, remove and warn
    if gene_embeddings.index.duplicated().any():
        print("Warning: Duplicate genes found in embeddings index.\nRemoving duplicates.")
        gene_embeddings = gene_embeddings[~gene_embeddings.index.duplicated(keep="first")]

    # Read adata and add embeddings
    adata = sc.read_h5ad(adata_path)
    adata.uns['embeddings_scgpt'] = gene_embeddings
    
    # Save result
    adata.write_h5ad(output_path)

    # Print important information
    print('='*100)
    print("scGPT embeddings")
    print('='*100)
    print(f"Input data: {adata_path}")
    print(f"Output data: {output_path}")
    print(f"Number of genes: {len(gene_embeddings)}")
    print(f"Embedding dimension: {gene_embeddings.shape[1]}")
    print(f"Added key: embeddings_scgpt")
    print('='*100)


def main():
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings for an AnnData object')
    parser.add_argument('--input', required=True, help='Path to input h5ad file')
    parser.add_argument('--model_dir', required=True, help='Directory containing pretrained scGPT model')
    parser.add_argument('--output', required=True, help='Path to save output h5ad file')
    parser.add_argument('--gdown_path', required=True, help='Path to gdown executable')
    args = parser.parse_args()
    
    # Optionally download scGPT model
    download_scgpt_model(args.model_dir, args.gdown_path)
    get_scgpt_embeddings(args.input, args.model_dir, args.output)

if __name__ == '__main__':
    main()

