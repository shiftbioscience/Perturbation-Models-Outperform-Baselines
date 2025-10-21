FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch-geometric \
    scanpy \
    "anndata>=0.12.0" \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    omegaconf \
    hydra-core

# Install cellsimbench
# Copy cellsimbench source code from build context root
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/millerh1/GEARS.git

COPY pyproject.toml /tmp/pyproject.toml


# Install cellsimbench in development mode
RUN pip install -e /tmp/
COPY cellsimbench /tmp/cellsimbench
RUN pip install -e /tmp/

COPY docker/gears/gene2go_all.pkl /app/gene2go_all.pkl


# Install ipdb
RUN pip install ipdb

# Create app directory
WORKDIR /app

# Create data directory that GEARS expects for intermediate files
RUN mkdir -p /app/data

# Copy model wrapper and utilities
COPY docker/gears/run_model.py /app/run_model.py
COPY docker/gears/gears_wrapper.py /app/gears_wrapper.py
COPY docker/gears/utils.py /app/utils.py

# Make run_model.py executable
RUN chmod +x /app/run_model.py

# Set up PYTHONBREAKPOINT="ipdb.set_trace"
ENV PYTHONBREAKPOINT="ipdb.set_trace"
RUN echo "export PYTHONBREAKPOINT='ipdb.set_trace'" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["python", "/app/run_model.py"] 