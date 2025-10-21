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

# Install fMLP dependencies
COPY docker/fmlp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create app directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/data /app/models
RUN mkdir -p /data /model_output /output

# Copy fMLP-specific files
COPY docker/fmlp/run_model.py /app/run_model.py
COPY docker/fmlp/fmlp_wrapper.py /app/fmlp_wrapper.py
COPY docker/fmlp/utils.py /app/utils.py

# Make run_model.py executable
RUN chmod +x /app/run_model.py

# Set up PYTHONBREAKPOINT for debugging
ENV PYTHONBREAKPOINT="ipdb.set_trace"
RUN echo "export PYTHONBREAKPOINT='ipdb.set_trace'" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["python", "/app/run_model.py"]
