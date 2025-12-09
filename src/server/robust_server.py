"""
Robust FL Server

FL server with pluggable defense strategies for Byzantine-robust aggregation.
"""

import flwr as fl
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from typing import List, Tuple, Dict, Optional, Union, Callable
import numpy as np
import torch
import argparse
import logging
import os
from datetime import datetime

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(sys_path)

from defenses import get_defense

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class RobustFedAvg(fl.server.strategy.FedAvg):
    """
    Robust FedAvg Strategy with Defense Mechanisms.
    
    Extends FedAvg to support various Byzantine-robust aggregation methods.
    """
    
    def __init__(
        self,
        defense_type: str = "none",
        defense_config: Dict = None,
        log_dir: str = "./experiments/logs",
        **kwargs
    ):
        super().__init__(
            evaluate_metrics_aggregation_fn=weighted_average,
            **kwargs
        )
        
        self.defense_type = defense_type
        self.defense_config = defense_config or {}
        self.defense = get_defense(defense_type, self.defense_config)
        
        self.log_dir = log_dir
        self.round_accuracies = []
        self.round_losses = []
        self.defense_metrics = []
        
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"Initialized RobustFedAvg with defense: {self.defense}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, any]]:
        """Aggregate with defense mechanism."""
        if not results:
            return None, {}
        
        logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")
        
        # Check for attacks in results
        for client_proxy, fit_res in results:
            if fit_res.metrics and 'attack_type' in fit_res.metrics:
                logger.warning(
                    f"  Client {client_proxy.cid}: ATTACK DETECTED - "
                    f"{fit_res.metrics.get('attack_type')}"
                )
        
        # Use standard FedAvg if no special defense
        if self.defense_type in ['none', 'fedavg']:
            return super().aggregate_fit(server_round, results, failures)
        
        # Extract parameters and convert to tensors
        client_updates = []
        num_examples = []
        
        for client_proxy, fit_res in results:
            parameters = fit_res.parameters
            ndarrays = parameters_to_ndarrays(parameters)
            tensors = [torch.tensor(arr) for arr in ndarrays]
            client_updates.append(tensors)
            num_examples.append(fit_res.num_examples)
        
        # Apply defense aggregation
        try:
            aggregated_tensors = self.defense.aggregate(client_updates, num_examples)
            
            # Log defense metrics
            metrics = self.defense.get_metrics()
            self.defense_metrics.append(metrics)
            logger.info(f"  Defense metrics: {metrics}")
            
            # Detect malicious clients
            detected = self.defense.detect_malicious(client_updates, num_examples)
            if detected:
                logger.warning(f"  Detected potentially malicious clients: {detected}")
            
        except Exception as e:
            logger.error(f"Defense aggregation failed: {e}, falling back to FedAvg")
            return super().aggregate_fit(server_round, results, failures)
        
        # Convert back to Parameters
        aggregated_ndarrays = [t.numpy() for t in aggregated_tensors]
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        return aggregated_parameters, {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, any]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            self.round_accuracies.append(metrics["accuracy"])
            logger.info(f"Round {server_round}: Global accuracy = {metrics['accuracy']:.4f}")
        
        if loss is not None:
            self.round_losses.append(loss)
        
        return loss, metrics
    
    def save_results(self, filename: str = "training_results.npz"):
        """Save training results and defense metrics."""
        filepath = os.path.join(self.log_dir, filename)
        np.savez(
            filepath,
            accuracies=np.array(self.round_accuracies),
            losses=np.array(self.round_losses),
            defense_type=self.defense_type
        )
        logger.info(f"Results saved to {filepath}")


def start_robust_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    min_clients: int = 2,
    defense_type: str = "none",
    defense_config: Dict = None,
    log_dir: str = "./experiments/logs"
):
    """Start the robust FL server."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    logger.info(f"Starting Robust FL Server")
    logger.info(f"  Defense: {defense_type}")
    logger.info(f"  Config: {defense_config}")
    
    strategy = RobustFedAvg(
        defense_type=defense_type,
        defense_config=defense_config,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        log_dir=run_log_dir
    )
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    strategy.save_results()
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust FL Server")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "krum", "multi_krum", "trimmed_mean", "median"])
    parser.add_argument("--num-malicious", type=int, default=1)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    defense_config = {
        'num_malicious': args.num_malicious,
        'multi_k': 3,
        'trim_ratio': args.trim_ratio
    }
    
    start_robust_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_clients=args.min_clients,
        defense_type=args.defense,
        defense_config=defense_config
    )
