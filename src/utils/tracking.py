"""
Experiment Tracking

TensorBoard and logging utilities for experiment monitoring.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class ExperimentTracker:
    """
    Unified experiment tracking with TensorBoard and file logging.
    
    Tracks:
    - Training metrics (accuracy, loss per round)
    - Attack metrics (ASR, poison ratio)
    - Defense metrics (rejected clients, trust scores)
    - System metrics (time, memory)
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./experiments/logs",
        use_tensorboard: bool = True,
        config: Optional[Dict] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        self.writer = None
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir)
        
        # Store metrics
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'attack_success_rate': [],
            'defense_metrics': []
        }
        
        self.config = config or {}
        self.start_time = datetime.now()
        
        # Save config
        if config:
            self._save_config(config)
    
    def _save_config(self, config: Dict):
        """Save experiment configuration."""
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        attack_success_rate: Optional[float] = None,
        defense_metrics: Optional[Dict] = None,
        **kwargs
    ):
        """Log metrics for a training round."""
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
        
        if attack_success_rate is not None:
            self.metrics['attack_success_rate'].append(attack_success_rate)
        
        if defense_metrics:
            self.metrics['defense_metrics'].append(defense_metrics)
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Training/Accuracy', accuracy, round_num)
            self.writer.add_scalar('Training/Loss', loss, round_num)
            
            if attack_success_rate is not None:
                self.writer.add_scalar('Attack/SuccessRate', attack_success_rate, round_num)
            
            if defense_metrics:
                for key, value in defense_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Defense/{key}', value, round_num)
            
            # Log any extra kwargs
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Other/{key}', value, round_num)
    
    def log_client_update(
        self,
        round_num: int,
        client_id: int,
        loss: float,
        num_samples: int,
        is_malicious: bool = False
    ):
        """Log per-client update information."""
        if self.writer:
            tag = f"Client_{client_id}/Loss"
            self.writer.add_scalar(tag, loss, round_num)
            
            if is_malicious:
                self.writer.add_scalar(
                    f"Client_{client_id}/Malicious", 1, round_num
                )
    
    def log_model_weights(
        self,
        round_num: int,
        model_params: Dict[str, Any],
        prefix: str = "Model"
    ):
        """Log model weight statistics."""
        if not self.writer:
            return
        
        for name, param in model_params.items():
            import torch
            if isinstance(param, torch.Tensor):
                self.writer.add_histogram(f"{prefix}/{name}", param, round_num)
                self.writer.add_scalar(
                    f"{prefix}/{name}_norm",
                    torch.norm(param).item(),
                    round_num
                )
    
    def log_aggregation(
        self,
        round_num: int,
        selected_clients: List[int],
        rejected_clients: List[int],
        client_scores: Optional[List[float]] = None
    ):
        """Log aggregation decisions."""
        if self.writer:
            self.writer.add_scalar(
                'Aggregation/NumSelected',
                len(selected_clients),
                round_num
            )
            self.writer.add_scalar(
                'Aggregation/NumRejected',
                len(rejected_clients),
                round_num
            )
    
    def finish(self, final_metrics: Optional[Dict] = None):
        """Finalize experiment and save results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'metrics': self.metrics,
            'duration_seconds': duration,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        if final_metrics:
            results['final_metrics'] = final_metrics
        
        # Save results
        results_path = os.path.join(self.log_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        logger.info(f"Experiment finished. Duration: {duration:.1f}s")
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_experiment_tracker(
    experiment_name: str,
    config: Dict,
    log_dir: str = "./experiments/logs"
) -> ExperimentTracker:
    """Create an experiment tracker with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{experiment_name}_{timestamp}"
    return ExperimentTracker(name, log_dir, config=config)


if __name__ == "__main__":
    # Test tracking
    config = {
        'num_rounds': 5,
        'num_clients': 3,
        'attack': 'backdoor',
        'defense': 'krum'
    }
    
    with ExperimentTracker("test_experiment", config=config) as tracker:
        for round_num in range(1, 6):
            accuracy = 0.9 + round_num * 0.01
            loss = 0.5 - round_num * 0.05
            asr = 0.1 + round_num * 0.05
            
            tracker.log_round(
                round_num,
                accuracy=accuracy,
                loss=loss,
                attack_success_rate=asr,
                defense_metrics={'rejected': 1}
            )
    
    print("Tracking test completed!")
