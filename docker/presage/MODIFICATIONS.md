Summary of modifications relative to the upstream PRESAGE project (no dates):

- presage_wrapper.py
  - Added a PRESAGE wrapper tailored for CellSimBench, including:
    - WeightedPRESAGE module supporting DEG-weighted loss and combo perturbations.
    - CellSimBenchModelHarness derived from PRESAGE training harness with simplified evaluator behavior.
    - Training, checkpointing, and artifact export that preserve PRESAGE pathway cache and control means.
    - Prediction flow restoring cache and converting deltas back to absolute expression using covariate-specific control means.

- cellsimbench_datamodule.py
  - Added a custom data module for CellSimBench:
    - Pseudobulk creation per covariate::perturbation.
    - Covariate-specific control mean centering and keys of the form covariate::perturbation.
    - Dataset class emitting inds, expr, optional gene_weights, and pert_key with strict key validation.
    - Split handling that properly restricts controls to the corresponding split to avoid leakage.

- Dockerfile
  - Added Docker build for a non-commercial image, bundling wrapper code and upstream PRESAGE sources.
  - Included LICENSE, NOTICE, and MODIFICATIONS.md in the image and added OCI license/author labels.

Note: These files are newly authored for the CellSimBench integration and were not present in the upstream PRESAGE repository.
