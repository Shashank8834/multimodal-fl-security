"""
Comprehensive Experiment Matrix Runner

Runs systematic experiments across attack-defense combinations.
Generates results for paper.
"""

import os
import sys
import yaml
import json
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentRunner, ExperimentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentMatrix:
    """Defines the full experiment matrix."""
    attacks: List[str] = field(default_factory=lambda: [
        'none', 'label_flip', 'backdoor', 'model_replacement'
    ])
    defenses: List[str] = field(default_factory=lambda: [
        'none', 'krum', 'trimmed_mean', 'median', 'dp_sgd'
    ])
    datasets: List[str] = field(default_factory=lambda: ['mnist', 'cub200'])
    partitions: List[str] = field(default_factory=lambda: ['iid', 'dirichlet'])
    num_clients_list: List[int] = field(default_factory=lambda: [5, 10])
    malicious_ratios: List[float] = field(default_factory=lambda: [0.2])
    
    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        return (len(self.attacks) * len(self.defenses) * 
                len(self.datasets) * len(self.partitions) * 
                len(self.num_clients_list) * len(self.malicious_ratios))
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        configs = []
        
        for attack, defense, dataset, partition, num_clients, mal_ratio in itertools.product(
            self.attacks, self.defenses, self.datasets, 
            self.partitions, self.num_clients_list, self.malicious_ratios
        ):
            # Skip invalid combinations
            if attack == 'none' and defense != 'none':
                # Defense-only experiments less interesting but include some
                if defense not in ['none', 'krum']:
                    continue
            
            # Calculate malicious clients
            num_malicious = max(1, int(num_clients * mal_ratio))
            malicious_clients = list(range(num_malicious))
            
            name = f"{attack}_{defense}_{dataset}_{partition}_{num_clients}c"
            
            config = ExperimentConfig(
                name=name,
                dataset=dataset,
                num_clients=num_clients,
                num_rounds=5,  # Quick experiments
                local_epochs=1,
                batch_size=32,
                learning_rate=0.01,
                partition=partition,
                attack_enabled=(attack != 'none'),
                attack_type=attack if attack != 'none' else 'none',
                malicious_clients=malicious_clients if attack != 'none' else [],
                poison_ratio=0.1,
                target_class=0,
                defense_enabled=(defense != 'none'),
                defense_type=defense if defense != 'none' else 'none',
                num_malicious_assumed=num_malicious,
                trim_ratio=0.1
            )
            
            configs.append(config)
        
        return configs


class BatchExperimentRunner:
    """Runs experiments in batches with aggregated results."""
    
    def __init__(self, results_dir: str = "./experiments/matrix_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.runner = ExperimentRunner(results_dir)
        self.all_results = []
        
    def run_matrix(
        self,
        matrix: ExperimentMatrix,
        skip_existing: bool = True,
        max_experiments: Optional[int] = None
    ) -> List[Dict]:
        """
        Run all experiments in the matrix.
        
        Args:
            matrix: Experiment matrix configuration
            skip_existing: Skip already completed experiments
            max_experiments: Limit number of experiments (for testing)
        """
        configs = matrix.generate_configs()
        
        if max_experiments:
            configs = configs[:max_experiments]
        
        logger.info(f"Running {len(configs)} experiments...")
        
        for i, config in enumerate(configs):
            logger.info(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
            
            # Check if already exists
            if skip_existing and self._result_exists(config.name):
                logger.info(f"  Skipping (already exists)")
                continue
            
            try:
                result = self.runner.run_simulation(config)
                self.all_results.append(result.to_dict())
            except Exception as e:
                logger.error(f"  Failed: {e}")
                self.all_results.append({
                    'config': asdict(config),
                    'error': str(e),
                    'final_accuracy': 0.0
                })
        
        # Save aggregated results
        self._save_aggregated_results()
        
        return self.all_results
    
    def _result_exists(self, name: str) -> bool:
        """Check if result already exists."""
        import glob
        pattern = os.path.join(self.results_dir, f"{name}_*.json")
        return len(glob.glob(pattern)) > 0
    
    def _save_aggregated_results(self):
        """Save all results to a single file."""
        filepath = os.path.join(
            self.results_dir,
            f"matrix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        logger.info(f"Aggregated results saved to {filepath}")
        
        # Also save summary table
        self._save_summary_table()
    
    def _save_summary_table(self):
        """Generate summary table."""
        summary_path = os.path.join(self.results_dir, "summary_table.md")
        
        lines = [
            "# Experiment Matrix Results",
            "",
            "| Attack | Defense | Dataset | Partition | Clients | Accuracy | ASR |",
            "|--------|---------|---------|-----------|---------|----------|-----|"
        ]
        
        for result in self.all_results:
            if 'error' in result:
                continue
            
            cfg = result.get('config', {})
            acc = result.get('final_accuracy', 0)
            asr = result.get('attack_success_rate')
            asr_str = f"{asr:.2%}" if asr is not None else "-"
            
            lines.append(
                f"| {cfg.get('attack_type', 'none')} | "
                f"{cfg.get('defense_type', 'none')} | "
                f"{cfg.get('dataset', '-')} | "
                f"{cfg.get('partition', '-')} | "
                f"{cfg.get('num_clients', '-')} | "
                f"{acc:.2%} | {asr_str} |"
            )
        
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Summary saved to {summary_path}")


def run_full_matrix():
    """Run the complete experiment matrix (all combinations for research)."""
    matrix = ExperimentMatrix()
    total = matrix.get_total_experiments()
    configs = matrix.generate_configs()
    
    logger.info(f"Full matrix: {total} combinations, {len(configs)} after filtering")
    logger.info(f"Estimated time: ~{len(configs) * 2.5:.0f} minutes (~{len(configs) * 2.5 / 60:.1f} hours)")
    
    runner = BatchExperimentRunner("./experiments/full_results")
    results = runner.run_matrix(matrix, max_experiments=None)  # No limit - run all
    
    return results


def run_quick_comparison():
    """Run a quick comparison of key attack-defense combinations."""
    matrix = ExperimentMatrix(
        attacks=['none', 'backdoor'],
        defenses=['none', 'krum', 'trimmed_mean'],
        datasets=['mnist'],
        partitions=['iid'],
        num_clients_list=[5],
        malicious_ratios=[0.2]
    )
    
    logger.info(f"Quick comparison: {matrix.get_total_experiments()} experiments")
    
    runner = BatchExperimentRunner("./experiments/quick_results")
    results = runner.run_matrix(matrix)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment Matrix")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["full", "quick"])
    parser.add_argument("--max-experiments", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.mode == "full":
        results = run_full_matrix()
    else:
        results = run_quick_comparison()
    
    print(f"\n[OK] Completed {len(results)} experiments")
