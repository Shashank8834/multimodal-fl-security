"""
Experiment Runner

Main entry point for running FL experiments with configurable attacks and defenses.
"""

import argparse
import yaml
import os
import sys
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, Any
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_server(config: Dict[str, Any]):
    """Run the FL server in a subprocess."""
    server_config = config.get('server', {})
    logging_config = config.get('logging', {})
    
    cmd = [
        sys.executable,
        "src/server/fl_server.py",
        "--server-address", server_config.get('address', '0.0.0.0:8080'),
        "--num-rounds", str(server_config.get('num_rounds', 10)),
        "--min-clients", str(server_config.get('min_clients', 2)),
        "--log-dir", logging_config.get('log_dir', './experiments/logs')
    ]
    
    logger.info(f"Starting server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def run_client(client_id: int, config: Dict[str, Any]):
    """Run a single FL client in a subprocess."""
    server_config = config.get('server', {})
    client_config = config.get('client', {})
    data_config = config.get('data', {})
    
    # Get server address for clients (use localhost instead of 0.0.0.0)
    server_address = server_config.get('address', '0.0.0.0:8080')
    if server_address.startswith('0.0.0.0'):
        server_address = server_address.replace('0.0.0.0', '127.0.0.1')
    
    cmd = [
        sys.executable,
        "src/client/fl_client.py",
        "--client-id", str(client_id),
        "--server-address", server_address,
        "--num-clients", str(client_config.get('num_clients', 3)),
        "--partition", data_config.get('partition', 'iid'),
        "--local-epochs", str(client_config.get('local_epochs', 1)),
        "--batch-size", str(client_config.get('batch_size', 32)),
        "--learning-rate", str(client_config.get('learning_rate', 0.01)),
        "--data-dir", data_config.get('data_dir', './data')
    ]
    
    logger.info(f"Starting client {client_id}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def run_experiment(config: Dict[str, Any]):
    """
    Run a complete FL experiment.
    
    Starts the server and all clients, waits for completion.
    """
    experiment_name = config.get('experiment', {}).get('name', 'unnamed')
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Create output directories
    log_dir = config.get('logging', {}).get('log_dir', './experiments/logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = os.path.join(log_dir, f"config_{timestamp}.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {config_save_path}")
    
    # Start server
    server_process = run_server(config)
    
    # Wait for server to start
    time.sleep(3)
    
    # Start clients
    num_clients = config.get('client', {}).get('num_clients', 3)
    client_processes = []
    
    for client_id in range(num_clients):
        client_process = run_client(client_id, config)
        client_processes.append(client_process)
        time.sleep(0.5)  # Stagger client starts
    
    # Wait for all processes to complete
    logger.info("Waiting for training to complete...")
    
    try:
        server_process.wait()
        for cp in client_processes:
            cp.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted! Terminating processes...")
        server_process.terminate()
        for cp in client_processes:
            cp.terminate()
    
    logger.info("Experiment completed!")


def main():
    parser = argparse.ArgumentParser(description="FL Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run only the server (for manual client start)"
    )
    parser.add_argument(
        "--client",
        type=int,
        default=None,
        help="Run only a specific client"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    if args.server_only:
        # Run server only
        server_process = run_server(config)
        try:
            server_process.wait()
        except KeyboardInterrupt:
            server_process.terminate()
    elif args.client is not None:
        # Run specific client only
        client_process = run_client(args.client, config)
        try:
            client_process.wait()
        except KeyboardInterrupt:
            client_process.terminate()
    else:
        # Run full experiment
        run_experiment(config)


if __name__ == "__main__":
    main()
