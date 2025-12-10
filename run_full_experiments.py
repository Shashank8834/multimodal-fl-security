#!/usr/bin/env python
"""
GPU-Optimized Full Experiment Runner

Run all experiments (MNIST, CUB-200) on GPU for research paper.
Estimated time: ~15 hours on high-end GPU

Usage:
    python run_full_experiments.py --all           # All datasets
    python run_full_experiments.py --mnist         # MNIST only
    python run_full_experiments.py --cub200        # CUB-200 only
    python run_full_experiments.py --continue      # Resume from checkpoint
"""

import os
import sys
import argparse
import torch
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.experiment_matrix import ExperimentMatrix, BatchExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and print info."""
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("CUDA Available: No - Using CPU (will be slow)")
    
    print("="*60)
    return torch.cuda.is_available()


def run_dataset_experiments(dataset: str, results_dir: str):
    """Run experiments for a single dataset."""
    matrix = ExperimentMatrix(
        datasets=[dataset],
        attacks=['none', 'label_flip', 'backdoor', 'model_replacement'],
        defenses=['none', 'krum', 'trimmed_mean', 'median', 'dp_sgd'],
        partitions=['iid', 'noniid'],
        num_clients_list=[5, 10],
        malicious_ratios=[0.2]
    )
    
    configs = matrix.generate_configs()
    logger.info(f"Dataset: {dataset.upper()}")
    logger.info(f"Experiments: {len(configs)}")
    
    runner = BatchExperimentRunner(results_dir)
    results = runner.run_matrix(matrix, skip_existing=True, max_experiments=None)
    
    return results


def run_all_experiments(datasets, results_base_dir):
    """Run experiments for selected datasets."""
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING {dataset.upper()} EXPERIMENTS")
        logger.info(f"{'='*60}\n")
        
        results_dir = os.path.join(results_base_dir, dataset)
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            results = run_dataset_experiments(dataset, results_dir)
            all_results[dataset] = results
            logger.info(f"Completed {dataset}: {len(results)} experiments")
        except Exception as e:
            logger.error(f"Failed {dataset}: {e}")
            all_results[dataset] = {'error': str(e)}
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run FL Security Experiments")
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--mnist', action='store_true', help='Run MNIST experiments')
    parser.add_argument('--cub200', action='store_true', help='Run CUB-200 experiments')
    parser.add_argument('--continue', dest='resume', action='store_true', 
                       help='Skip existing experiments')
    parser.add_argument('--output', type=str, default='./experiments/full_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Determine datasets
    datasets = []
    if args.all:
        datasets = ['mnist', 'cub200']
    else:
        if args.mnist:
            datasets.append('mnist')
        if args.cub200:
            datasets.append('cub200')
    
    if not datasets:
        print("\nNo dataset specified. Use --all or --mnist/--cub200")
        print("\nExamples:")
        print("  python run_full_experiments.py --all")
        print("  python run_full_experiments.py --mnist --cub200")
        return
    
    # Estimate time
    estimates = {'mnist': 3, 'cub200': 12}
    total_hours = sum(estimates.get(d, 5) for d in datasets)
    
    print(f"\nSelected datasets: {datasets}")
    print(f"Estimated time: ~{total_hours} hours" + (" (with GPU)" if has_gpu else " (CPU - much slower!)"))
    print(f"Results will be saved to: {args.output}")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to start...")
    
    try:
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Run experiments
    results = run_all_experiments(datasets, args.output)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for dataset, result in results.items():
        if isinstance(result, dict) and 'error' in result:
            print(f"  {dataset}: FAILED - {result['error']}")
        else:
            print(f"  {dataset}: {len(result)} experiments completed")
    print("="*60)


if __name__ == "__main__":
    main()
