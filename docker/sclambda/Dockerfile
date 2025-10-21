FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CellSimBench in development mode
COPY pyproject.toml /tmp/pyproject.toml
COPY cellsimbench /tmp/cellsimbench
RUN pip install -e /tmp/

# Install scLambda dependencies
COPY docker/sclambda/requirements_sclambda.txt /tmp/requirements_sclambda.txt
RUN pip install --no-cache-dir -r /tmp/requirements_sclambda.txt

# Create app directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/data /app/models
RUN mkdir -p /data /model_output /output /model_output/embeddings_cache

# Copy scLambda-specific files
COPY docker/sclambda/run_model.py /app/run_model.py
COPY docker/sclambda/sclambda_wrapper.py /app/sclambda_wrapper.py
COPY docker/sclambda/utils.py /app/utils.py

# Make run_model.py executable
RUN chmod +x /app/run_model.py

# Set up PYTHONBREAKPOINT for debugging
ENV PYTHONBREAKPOINT="ipdb.set_trace"
RUN echo "export PYTHONBREAKPOINT='ipdb.set_trace'" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["python", "/app/run_model.py"] 