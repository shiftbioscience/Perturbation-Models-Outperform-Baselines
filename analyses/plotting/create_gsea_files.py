import scanpy as sc
import os
import pandas as pd

adata = sc.read_h5ad("../../data/replogle22k562gwps/replogle22k562gwps_processed_complete.h5ad")

# Read in CHEA gene sets
with open("ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X.txt") as f:
    chea_sets = {}
    for line in f:
        if line.strip():
            items = [s.replace(" CHEA", "").replace(" ENCODE", "") for s in line.strip().split('\t')]
            key = items[0]
            chea_sets[key] = [x for x in items[1:] if x != ""]


# Create directory for output
os.makedirs('gsea_input_for_R', exist_ok=True)

# Find overlapping genes between chea_sets and adata
chea_genes = set(chea_sets.keys())
adata_perturbations = set([key.split('_')[-1] for key in adata.uns['names_df_dict_gt'].keys()])
overlap_genes = chea_genes & adata_perturbations

print(f"Found {len(overlap_genes)} genes in both chea_sets and adata")
print(f"Genes: {sorted(overlap_genes)}")

# Save each gene's ranked list with scores
saved_files = []
for gene in sorted(overlap_genes):
    key = f'replogle22k562gwps_{gene}'
    
    if key in adata.uns['names_df_dict_gt'] and key in adata.uns['scores_df_dict_gt']:
        gene_names = adata.uns['names_df_dict_gt'][key]
        gene_scores = adata.uns['scores_df_dict_gt'][key]
        
        # Create DataFrame with gene names and scores
        ranked_df = pd.DataFrame({
            'gene': gene_names,
            'score': gene_scores
        })
        
        # Save to CSV
        output_file = f'gsea_input_for_R/{gene}_ranked_genes.csv'
        ranked_df.to_csv(output_file, index=False)
        saved_files.append({'gene': gene, 'file': output_file, 'n_genes': len(ranked_df)})
        
print(f"\nSaved {len(saved_files)} ranked gene lists")

# Create a summary file
summary_df = pd.DataFrame(saved_files)
summary_df.to_csv('gsea_input_for_R/summary.csv', index=False)
print(f"Summary saved to gsea_input_for_R/summary.csv")
