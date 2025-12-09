"""
Federated Learning Server

Implements the FL server using Flower framework with FedAvg aggregation.
Supports pluggable aggregation strategies and defense mechanisms.
"""

import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import argparse
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Aggregated metrics dictionary
    """
    # Weighted average of accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Extended FedAvg strategy with logging and hooks for defenses.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn=weighted_average,
        log_dir: str = "./experiments/logs",
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs
        )
        
        self.log_dir = log_dir
        self.round_accuracies = []
        self.round_losses = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """
        Aggregate model updates from clients.
        
        This is where defense mechanisms can be inserted.
        """
        if not results:
            return None, {}
        
        # Log client participation
        logger.info(f"Round {server_round}: Received updates from {len(results)} clients")
        
        for client_proxy, fit_res in results:
            logger.debug(f"  Client {client_proxy.cid}: {fit_res.num_examples} examples")
        
        # Call parent FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, any]]:
        """
        Aggregate evaluation results from clients.
        """
        if not results:
            return None, {}
        
        # Aggregate metrics
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            self.round_accuracies.append(accuracy)
            logger.info(f"Round {server_round}: Global accuracy = {accuracy:.4f}")
        
        if loss is not None:
            self.round_losses.append(loss)
            logger.info(f"Round {server_round}: Global loss = {loss:.4f}")
        
        return loss, metrics
    
    def save_results(self, filename: str = "training_results.npz"):
        """Save training results to file."""
        filepath = os.path.join(self.log_dir, filename)
        np.savez(
            filepath,
            accuracies=np.array(self.round_accuracies),
            losses=np.array(self.round_losses)
        )
        logger.info(f"Results saved to {filepath}")


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    min_clients: int = 2,
    log_dir: str = "./experiments/logs"
):
    """
    Start the Flower FL server.
    
    Args:
        server_address: Address to start server on
        num_rounds: Number of FL rounds
        min_clients: Minimum number of clients required
        log_dir: Directory for logs
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    logger.info(f"Starting FL Server")
    logger.info(f"  Address: {server_address}")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  Min clients: {min_clients}")
    logger.info(f"  Log dir: {run_log_dir}")
    
    # Create strategy
    strategy = FedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        log_dir=run_log_dir
    )
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    # Save results after training
    strategy.save_results()
    
    logger.info("Training completed!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of FL rounds (default: 10)"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of clients (default: 2)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./experiments/logs",
        help="Log directory (default: ./experiments/logs)"
    )
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        log_dir=args.log_dir
    )
