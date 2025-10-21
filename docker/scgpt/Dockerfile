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

# Install scGPT and its dependencies
COPY docker/scgpt/requirements_scgpt.txt /tmp/requirements_scgpt.txt
RUN pip install --no-cache-dir -r /tmp/requirements_scgpt.txt

# Create app directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/models/scgpt
RUN mkdir -p /app/data

# Copy scGPT-specific files
COPY docker/scgpt/run_model.py /app/run_model.py
COPY docker/scgpt/scgpt_wrapper.py /app/scgpt_wrapper.py
COPY docker/scgpt/utils.py /app/utils.py

# Make run_model.py executable
RUN chmod +x /app/run_model.py

# Set up PYTHONBREAKPOINT for debugging
ENV PYTHONBREAKPOINT="ipdb.set_trace"
RUN echo "export PYTHONBREAKPOINT='ipdb.set_trace'" >> ~/.bashrc

# Set the entrypoint
ENTRYPOINT ["python", "/app/run_model.py"] 