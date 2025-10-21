# SPDX-License-Identifier: LicenseRef-Genentech-NonCommercial-1.0
# Copyright (c) 2025 Genentech, Inc.
# Licensed under the Genentech Non-Commercial Software License Version 1.0 (September 2022).
# See the "LICENSE" file included in this image and repository.
# NOTICE: This Dockerfile was modified by the CellSimBench team to build a non-commercial image.
FROM condaforge/mambaforge:latest

# OCI image labels
LABEL org.opencontainers.image.licenses="LicenseRef-Genentech-NonCommercial-1.0" \
      org.opencontainers.image.authors="CellSimBench team"

# Install system dependencies
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

# Copy and create conda environment
COPY docker/presage/ref/PRESAGE/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && mamba clean -afy

# Download PRESAGE embeddings cache
WORKDIR /
RUN wget -c -O cache.tar.gz https://zenodo.org/records/15587986/files/cache.tar.gz?download=1 && \
    tar -xzf cache.tar.gz && rm cache.tar.gz

# Install CellSimBench
COPY pyproject.toml /tmp/pyproject.toml
COPY cellsimbench /tmp/cellsimbench
RUN mamba run -n presage pip install -e /tmp/

# Clone PRESAGE repository to /presage_src https://github.com/Genentech/PRESAGE.git
# Then checkout 2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb
RUN git clone https://github.com/Genentech/PRESAGE.git /presage_repo 
RUN cd /presage_repo && git checkout 2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb
RUN cp -r /presage_repo/PRESAGE/src/ /presage_src/
RUN cp -r /presage_repo/PRESAGE/sample_files/ /presage_src/sample_files/

# Copy our wrapper files
COPY docker/presage/presage_wrapper.py docker/presage/run_model.py docker/presage/utils.py docker/presage/cellsimbench_datamodule.py /

# Include license and notice files in the image (copied from docker/presage)
COPY docker/presage/LICENSE docker/presage/NOTICE docker/presage/MODIFICATIONS.md /

# Force environment to numpy<2
RUN mamba run -n presage pip install "numpy<2" "anndata>=0.12.0"

WORKDIR /
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "presage", "python", "run_model.py"]
