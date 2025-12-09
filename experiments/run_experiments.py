"""
Comprehensive Experiment Runner

Runs FL experiments with configurable attacks and defenses,
collects metrics, and generates visualizations.
"""

import yaml
import os
import sys
import subprocess
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_mnist, get_client_data
from src.utils.metrics import evaluate_model, compute_attack_success_rate
from src.models.simple_cnn import create_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    dataset: str = "mnist"
    num_clients: int = 10
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    partition: str = "iid"
    
    # Attack configuration
    attack_enabled: bool = False
    attack_type: str = "none"
    malicious_clients: List[int] = None
    poison_ratio: float = 0.1
    target_class: int = 0
    
    # Defense configuration
    defense_enabled: bool = False
    defense_type: str = "none"
    num_malicious_assumed: int = 1
    trim_ratio: float = 0.1
    
    def __post_init__(self):
        if self.malicious_clients is None:
            self.malicious_clients = []


@dataclass
class ExperimentResults:
    """Results from an experiment."""
    config: Dict
    round_accuracies: List[float]
    round_losses: List[float]
    final_accuracy: float
    final_loss: float
    attack_success_rate: Optional[float] = None
    training_time_seconds: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ExperimentRunner:
    """
    Runs FL experiments and collects results.
    
    Supports:
    - Clean baseline experiments
    - Attack-only experiments
    - Defense-only experiments
    - Attack vs Defense experiments
    """
    
    def __init__(self, results_dir: str = "./experiments/results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def run_simulation(self, config: ExperimentConfig) -> ExperimentResults:
        """
        Run a simulated FL experiment (without distributed processes).
        
        This is faster for testing and produces the same results.
        """
        import torch
        from torch.utils.data import DataLoader
        
        logger.info(f"Running experiment: {config.name}")
        start_time = time.time()
        
        # Load data based on dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if config.dataset == "mnist":
            train_data, test_data = load_mnist("./data")
            global_model = create_model(device=device)
            get_client_data_fn = get_client_data
        elif config.dataset == "cub200":
            from src.utils.cub200_loader import load_cub200, get_cub200_client_data
            from src.models.cub200_cnn import create_cub200_model
            train_data, test_data = load_cub200("./data")
            global_model = create_cub200_model(device=device)
            get_client_data_fn = get_cub200_client_data
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")
        
        # Setup attack if enabled
        attack = None
        if config.attack_enabled and config.attack_type != "none":
            from src.attacks import get_attack
            attack_cfg = {
                'source_class': 7,
                'target_class': config.target_class,
                'poison_ratio': config.poison_ratio,
                'trigger_size': 3
            }
            attack = get_attack(config.attack_type, attack_cfg)
        
        # Setup defense if enabled
        defense = None
        if config.defense_enabled and config.defense_type != "none":
            from src.defenses import get_defense
            defense_cfg = {
                'num_malicious': config.num_malicious_assumed,
                'trim_ratio': config.trim_ratio,
                'multi_k': max(1, config.num_clients // 2)
            }
            defense = get_defense(config.defense_type, defense_cfg)
        
        # Create client data loaders
        client_loaders = []
        for client_id in range(config.num_clients):
            client_data = get_client_data_fn(
                train_data, client_id, config.num_clients, config.partition
            )
            
            # Apply attack to malicious clients
            if attack and client_id in config.malicious_clients:
                client_data = attack.poison_data(client_data)
                logger.info(f"Client {client_id} POISONED")
            
            loader = DataLoader(client_data, batch_size=config.batch_size, shuffle=True)
            client_loaders.append(loader)
        
        test_loader = DataLoader(test_data, batch_size=config.batch_size)
        
        # Training loop
        round_accuracies = []
        round_losses = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for round_num in range(1, config.num_rounds + 1):
            # Simulate local training for each client
            client_updates = []
            num_examples = []
            
            for client_id, loader in enumerate(client_loaders):
                # Clone global model for local training
                local_model = create_model(device=device)
                local_model.load_state_dict(global_model.state_dict())
                
                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9
                )
                
                # Local training
                local_model.train()
                for _ in range(config.local_epochs):
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = local_model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Get update
                update = [p.data.clone() for p in local_model.parameters()]
                client_updates.append(update)
                num_examples.append(len(loader.dataset))
            
            # Aggregate updates
            if defense:
                aggregated = defense.aggregate(client_updates, num_examples)
            else:
                # FedAvg
                total_examples = sum(num_examples)
                aggregated = []
                for param_idx in range(len(client_updates[0])):
                    weighted_sum = sum(
                        num_examples[i] * client_updates[i][param_idx]
                        for i in range(len(client_updates))
                    )
                    aggregated.append(weighted_sum / total_examples)
            
            # Update global model
            with torch.no_grad():
                for param, new_val in zip(global_model.parameters(), aggregated):
                    param.copy_(new_val)
            
            # Evaluate
            metrics = evaluate_model(global_model, test_loader, device)
            round_accuracies.append(metrics['accuracy'])
            round_losses.append(metrics['loss'])
            
            logger.info(f"Round {round_num}: Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}")
        
        # Compute attack success rate if applicable
        asr = None
        if attack and hasattr(attack, 'create_poisoned_testset'):
            from src.attacks.backdoor import BackdoorAttack
            if isinstance(attack, BackdoorAttack):
                triggered_test = attack.create_poisoned_testset(test_data)
                triggered_loader = DataLoader(triggered_test, batch_size=config.batch_size)
                asr = compute_attack_success_rate(
                    global_model, triggered_loader, config.target_class, device
                )
                logger.info(f"Attack Success Rate: {asr:.4f}")
        
        training_time = time.time() - start_time
        
        results = ExperimentResults(
            config=asdict(config),
            round_accuracies=round_accuracies,
            round_losses=round_losses,
            final_accuracy=round_accuracies[-1],
            final_loss=round_losses[-1],
            attack_success_rate=asr,
            training_time_seconds=training_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        results_file = os.path.join(
            self.results_dir,
            f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results.save(results_file)
        logger.info(f"Results saved to {results_file}")
        
        return results


def run_baseline_experiments():
    """Run baseline experiments (clean training)."""
    runner = ExperimentRunner()
    
    # Baseline: Clean training with different numbers of clients
    for num_clients in [3, 5, 10]:
        config = ExperimentConfig(
            name=f"baseline_clean_{num_clients}clients",
            num_clients=num_clients,
            num_rounds=5,
            partition="iid"
        )
        runner.run_simulation(config)


def run_attack_experiments():
    """Run attack experiments (attack only, no defense)."""
    runner = ExperimentRunner()
    
    # Label flip attack
    config = ExperimentConfig(
        name="attack_labelflip_no_defense",
        num_clients=5,
        num_rounds=5,
        attack_enabled=True,
        attack_type="label_flip",
        malicious_clients=[0, 1],  # 40% malicious
        poison_ratio=0.5
    )
    runner.run_simulation(config)
    
    # Backdoor attack
    config = ExperimentConfig(
        name="attack_backdoor_no_defense",
        num_clients=5,
        num_rounds=5,
        attack_enabled=True,
        attack_type="backdoor",
        malicious_clients=[0],  # 20% malicious
        poison_ratio=0.1,
        target_class=0
    )
    runner.run_simulation(config)


def run_defense_experiments():
    """Run attack + defense experiments."""
    runner = ExperimentRunner()
    
    # Backdoor attack + Krum defense
    config = ExperimentConfig(
        name="attack_backdoor_defense_krum",
        num_clients=5,
        num_rounds=5,
        attack_enabled=True,
        attack_type="backdoor",
        malicious_clients=[0],
        poison_ratio=0.1,
        defense_enabled=True,
        defense_type="krum",
        num_malicious_assumed=1
    )
    runner.run_simulation(config)
    
    # Backdoor + Trimmed Mean
    config = ExperimentConfig(
        name="attack_backdoor_defense_trimmed_mean",
        num_clients=5,
        num_rounds=5,
        attack_enabled=True,
        attack_type="backdoor",
        malicious_clients=[0],
        poison_ratio=0.1,
        defense_enabled=True,
        defense_type="trimmed_mean",
        trim_ratio=0.2
    )
    runner.run_simulation(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FL Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "baseline", "attack", "defense"])
    args = parser.parse_args()
    
    if args.experiment in ["all", "baseline"]:
        print("\n" + "="*50)
        print("Running Baseline Experiments")
        print("="*50)
        run_baseline_experiments()
    
    if args.experiment in ["all", "attack"]:
        print("\n" + "="*50)
        print("Running Attack Experiments")
        print("="*50)
        run_attack_experiments()
    
    if args.experiment in ["all", "defense"]:
        print("\n" + "="*50)
        print("Running Defense Experiments")
        print("="*50)
        run_defense_experiments()
    
    print("\n[OK] All experiments completed!")
