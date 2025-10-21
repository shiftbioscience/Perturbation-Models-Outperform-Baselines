#!/usr/bin/env python3
import argparse
import scanpy as sc
from pathlib import Path


def transfer_gene_embeddings(
    input_path: str,
    reference_adata_path: str,
    output_path: str
):
    """Transfer gene embeddings from reference AnnData to target AnnData.
    
    Args:
        input_path: Path to input h5ad file (target)
        reference_adata_path: Path to reference h5ad file containing embeddings
        output_path: Path to save output h5ad file with transferred embeddings
    """
    print('='*100)
    print("Transferring gene embeddings from reference")
    print('='*100)
    print(f"Input data: {input_path}")
    print(f"Reference data: {reference_adata_path}")
    print(f"Output data: {output_path}")
    print('='*100)
    
    # Load input and reference data
    print("Loading input AnnData...")
    adata = sc.read_h5ad(input_path)
    
    print("Loading reference AnnData...")
    ref_adata = sc.read_h5ad(reference_adata_path)
    
    # Find all embedding keys in reference data
    embedding_keys = [key for key in ref_adata.uns.keys() if key.startswith('embeddings_')]
    
    if not embedding_keys:
        print("Warning: No embedding keys found in reference data")
        print("Available keys in reference uns:", list(ref_adata.uns.keys()))
        # Just save the input as output if no embeddings found
        adata.write_h5ad(output_path)
        return
    
    print(f"Found {len(embedding_keys)} embedding keys in reference data:")
    for key in embedding_keys:
        print(f"  - {key}")
    
    # Transfer all embedding keys from reference to target
    embeddings_transferred = 0
    for key in embedding_keys:
        try:
            adata.uns[key] = ref_adata.uns[key].copy()
            embeddings_transferred += 1
            
            # Print info about transferred embedding
            emb_data = ref_adata.uns[key]
            print(f"Transferred {key}: {emb_data.shape[0]} genes x {emb_data.shape[1]} dimensions")
            
        except Exception as e:
            print(f"Warning: Failed to transfer {key}: {e}")
    
    # Save result
    adata.write_h5ad(output_path)
    
    print('='*100)
    print(f"Transfer complete: {embeddings_transferred}/{len(embedding_keys)} embeddings transferred")
    print('='*100)


def main():
    parser = argparse.ArgumentParser(
        description='Transfer gene embeddings from reference AnnData to target AnnData',
        epilog='All embedding keys (starting with "embeddings_") will be transferred from reference to target'
    )
    parser.add_argument('--input', required=True, help='Path to input h5ad file (target)')
    parser.add_argument('--reference_adata', required=True, help='Path to reference h5ad file containing embeddings')
    parser.add_argument('--output', required=True, help='Path to save output h5ad file')
    
    args = parser.parse_args()

    transfer_gene_embeddings(
        input_path=args.input,
        reference_adata_path=args.reference_adata,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()