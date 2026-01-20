"""
Shared Docker container execution for CellSimBench framework.

Provides unified Docker container management for model training and inference.
"""

import docker
import logging
from pathlib import Path
from typing import Dict, List, Deque, Optional, Any
from collections import deque

log = logging.getLogger(__name__)

# Constants for log buffering
MAX_LOG_BUFFER_SIZE = 100  # Keep last 100 lines in memory for error reporting
ERROR_LOG_TAIL_SIZE = 50   # Show last 50 lines in error messages


class DockerRunner:
    """Handles Docker container execution with common patterns.
    
    Provides unified interface for running Docker containers with proper
    resource management, logging, and error handling.
    
    Attributes:
        docker_client: Docker client instance or None if Docker unavailable.
        
    Example:
        >>> runner = DockerRunner()
        >>> runner.run_container(image, command, volumes, config)
    """
    
    def __init__(self) -> None:
        """Initialize DockerRunner and check Docker availability."""
        try:
            self.docker_client: Optional[docker.DockerClient] = docker.from_env()
        except docker.errors.DockerException:
            log.warning("Docker not available.")
            self.docker_client = None
    
    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        docker_config: Dict[str, Any],
        container_name: str = "cellsimbench",
        environment: Optional[Dict[str, str]] = None,
        gpu_id: Optional[int] = None
    ) -> None:
        """Run a Docker container with standard configuration.
        
        Handles resource limits, GPU support, volume mounting, and streaming logs.
        
        Args:
            image: Docker image name (e.g., 'cellsimbench/sclambda:latest').
            command: Command to run in container (e.g., ['train', '/config.json']).
            volumes: Volume mount configuration mapping host paths to container paths.
            docker_config: Docker settings including memory, cpus, gpu support.
            container_name: Container name for logging purposes.
            environment: Optional environment variables to pass to container.
            
        Raises:
            RuntimeError: If Docker is not available or container fails.
        """
        if self.docker_client is None:
            raise RuntimeError("Docker is not available")
        
        # Build container arguments
        container_args: Dict[str, Any] = {
            'image': image,
            'command': command,
            'volumes': volumes,
            'detach': True,
            'remove': False  # Don't auto-remove so we can get logs on failure
        }
        
        # Add environment variables if provided
        if environment:
            container_args['environment'] = environment
            log.info(f"Setting environment variables: {list(environment.keys())}")
        
        # Add memory limit if not "max"
        memory_config = docker_config.get('memory', 'max')
        if memory_config != 'max':
            container_args['mem_limit'] = memory_config
            log.info(f"Using memory limit: {memory_config}")
        else:
            log.info("Using maximum available memory (no limit)")
        
        # Add CPU limit if not "max"
        cpus_config = docker_config.get('cpus', 'max')
        if cpus_config != 'max':
            if isinstance(cpus_config, int):
                container_args['cpuset_cpus'] = f"0-{cpus_config-1}"
                log.info(f"Using CPU cores: 0-{cpus_config-1}")
            else:
                container_args['cpuset_cpus'] = str(cpus_config)
                log.info(f"Using CPU cores: {cpus_config}")
        else:
            log.info("Using all available CPU cores (no limit)")
        
        # Add GPU support if enabled
        if docker_config.get('gpu', True):
            if gpu_id is not None:
                # Assign specific GPU to this container
                container_args['device_requests'] = [
                    docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[['gpu']])
                ]
                log.info(f"Assigning GPU {gpu_id} to container")
            else:
                # Default: give access to all GPUs
                container_args['device_requests'] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ]
                log.info("Assigning all available GPUs to container")
        
        # Add shared memory configuration for PyTorch DataLoaders
        shm_size = docker_config.get('shm_size', '2g')  # Default 2GB for ML workloads
        container_args['shm_size'] = shm_size
        log.info(f"Using shared memory size: {shm_size}")
        
        log.info(f"Volume mounts:")
        for host_path, mount_config in volumes.items():
            log.info(f"  {host_path} -> {mount_config['bind']}")
        
        log.info(f"Starting {container_name} with Docker image: {image}")
        
        # Run container
        container = self.docker_client.containers.run(**container_args)
        
        # Buffer to capture recent logs for error reporting
        log_buffer: Deque[str] = deque(maxlen=MAX_LOG_BUFFER_SIZE)
        
        # Stream logs while capturing for error reporting
        for line in container.logs(stream=True, follow=True):
            decoded_line = line.decode().strip()
            
            # Log in real-time
            log.info(f"[DOCKER] {decoded_line}")
            
            # Buffer for error reporting
            log_buffer.append(decoded_line)
        
        # Wait for completion
        result = container.wait()
        
        # Clean up container
        try:
            container.remove()
        except Exception:
            pass  # Container might already be removed
        
        if result['StatusCode'] != 0:
            # Build error message using buffered logs
            error_msg = f"{container_name} failed with exit code: {result['StatusCode']}\n"
            error_msg += f"Command: {' '.join(command)}\n"
            error_msg += f"Image: {image}\n\n"
            
            # Use buffered logs for error context
            if log_buffer:
                buffered_logs = '\n'.join(log_buffer)
                error_msg += f"Container output (last {len(log_buffer)} lines):\n"
                error_msg += f"{buffered_logs}\n\n"
            
            # Add debugging hints
            error_msg += "Debugging hints:\n"
            error_msg += "- Check if all required files are mounted correctly\n"
            error_msg += "- Verify the Docker image contains all dependencies\n"
            error_msg += "- Check configuration file format and values\n"
            error_msg += "- Ensure sufficient memory/disk space is available"
            
            raise RuntimeError(error_msg)
        
        log.info(f"{container_name} completed successfully") 