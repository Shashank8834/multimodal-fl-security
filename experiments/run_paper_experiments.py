#!/usr/bin/env python
"""
Research Paper Experiment Runner

Runs comprehensive experiments for the paper:
"Security Analysis of Federated Learning: Evaluating Attacks and 
Defenses Under IID and Non-IID Data Distributions"

Outputs results in LaTeX table format ready for publication.

Usage:
    python experiments/run_paper_experiments.py
    python experiments/run_paper_experiments.py --quick  # Fast test run
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiments import ExperimentConfig, ExperimentRunner

# ============================================================================
# CONFIGURATION
# ============================================================================

# MNIST Research settings
MNIST_CONFIG = {
    "dataset": "mnist",
    "num_clients": 10,
    "num_rounds": 10,       # MNIST converges fast
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.01,
    "seeds": [42, 123, 456, 789, 1024],  # 5 seeds
}

# CUB-200 Research settings (reduced for feasibility)
CUB200_CONFIG = {
    "dataset": "cub200",
    "num_clients": 3,       # Fewer clients (user request)
    "num_rounds": 75,       # 75 rounds (user request)
    "local_epochs": 1,
    "batch_size": 16,       # Smaller batch for 8GB VRAM
    "learning_rate": 0.001, # Lower LR for transfer learning
    "seeds": [42, 123, 456],  # 3 seeds for CUB (time constraint)
}

# Quick test settings
QUICK_CONFIG = {
    "dataset": "mnist",
    "num_clients": 5,
    "num_rounds": 3,
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.01,
    "seeds": [42, 123],  # 2 seeds for quick testing
}

# Data distributions to test
DISTRIBUTIONS = ["iid", "noniid"]

# Non-IID alpha values (lower = more heterogeneous)
NONIID_ALPHAS = [0.5, 0.1]  # moderate and extreme heterogeneity

# Attack configurations
ATTACKS = {
    "none": {"enabled": False},
    "label_flip": {
        "enabled": True,
        "type": "label_flip",
        "malicious_clients": [0, 1],  # 20% malicious
        "poison_ratio": 0.3,
        "source_class": 0,
        "target_class": 8,
    },
    "backdoor": {
        "enabled": True,
        "type": "backdoor",
        "malicious_clients": [0, 1],
        "poison_ratio": 0.1,
        "target_class": 0,
    },
    "model_replacement": {
        "enabled": True,
        "type": "model_replacement",
        "malicious_clients": [0],  # 10% malicious
        "scale_factor": 10.0,
    },
}

# Defense configurations
DEFENSES = {
    "none": {"enabled": False},
    "krum": {
        "enabled": True,
        "type": "krum",
        "num_malicious": 2,
        "multi_k": 1,
    },
    "trimmed_mean": {
        "enabled": True,
        "type": "trimmed_mean",
        "trim_ratio": 0.2,
    },
    "fltrust": {
        "enabled": True,
        "type": "fltrust",
        "root_size": 100,
    },
    "dp_sgd": {
        "enabled": True,
        "type": "dp_sgd",
        "clip_norm": 1.0,
        "noise_multiplier": 0.1,
    },
}


@dataclass
class ExperimentResult:
    """Single experiment result."""
    attack: str
    defense: str
    distribution: str
    alpha: float
    seed: int
    accuracy: float
    loss: float
    asr: float  # Attack Success Rate


@dataclass
class AggregatedResult:
    """Aggregated results across seeds."""
    attack: str
    defense: str
    distribution: str
    alpha: float
    mean_accuracy: float
    std_accuracy: float
    mean_asr: float
    std_asr: float
    num_seeds: int


def run_single_experiment(
    attack_name: str,
    defense_name: str,
    distribution: str,
    alpha: float,
    seed: int,
    config: dict,
    runner: ExperimentRunner
) -> ExperimentResult:
    """Run a single experiment configuration."""
    
    attack_cfg = ATTACKS[attack_name]
    defense_cfg = DEFENSES[defense_name]
    
    exp_config = ExperimentConfig(
        name=f"{config.get('dataset', 'mnist')}_{attack_name}_{defense_name}_{distribution}_a{alpha}_s{seed}",
        dataset=config.get("dataset", "mnist"),
        num_clients=config["num_clients"],
        num_rounds=config["num_rounds"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        partition=distribution,
        seed=seed,
        # Attack settings
        attack_enabled=attack_cfg.get("enabled", False),
        attack_type=attack_cfg.get("type", "none"),
        malicious_clients=attack_cfg.get("malicious_clients", []),
        poison_ratio=attack_cfg.get("poison_ratio", 0.1),
        target_class=attack_cfg.get("target_class", 0),
        # Defense settings
        defense_enabled=defense_cfg.get("enabled", False),
        defense_type=defense_cfg.get("type", "none"),
        num_malicious_assumed=defense_cfg.get("num_malicious", 2),
        trim_ratio=defense_cfg.get("trim_ratio", 0.1),
    )
    
    result = runner.run_simulation(exp_config)
    
    return ExperimentResult(
        attack=attack_name,
        defense=defense_name,
        distribution=distribution,
        alpha=alpha,
        seed=seed,
        accuracy=result.final_accuracy,
        loss=result.final_loss,
        asr=result.attack_success_rate if result.attack_success_rate else 0.0
    )


def aggregate_results(results: List[ExperimentResult]) -> AggregatedResult:
    """Aggregate results across seeds."""
    accuracies = [r.accuracy for r in results]
    asrs = [r.asr for r in results]
    
    return AggregatedResult(
        attack=results[0].attack,
        defense=results[0].defense,
        distribution=results[0].distribution,
        alpha=results[0].alpha,
        mean_accuracy=np.mean(accuracies),
        std_accuracy=np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
        mean_asr=np.mean(asrs),
        std_asr=np.std(asrs, ddof=1) if len(asrs) > 1 else 0.0,
        num_seeds=len(results)
    )


def generate_latex_table(
    aggregated_results: List[AggregatedResult],
    distribution: str,
    alpha: float = None
) -> str:
    """Generate LaTeX table for paper."""
    
    # Filter results
    filtered = [r for r in aggregated_results if r.distribution == distribution]
    if alpha is not None:
        filtered = [r for r in filtered if r.alpha == alpha]
    
    # Build table
    dist_label = "IID" if distribution == "iid" else f"Non-IID (α={alpha})"
    
    lines = [
        f"% Table: {dist_label} Results",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Attack and Defense Performance under {dist_label} Distribution}}",
        f"\\label{{tab:{distribution.replace('.', '')}_results}}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Attack & Defense & Accuracy (\\%) & ASR (\\%) \\\\",
        "\\midrule",
    ]
    
    for r in filtered:
        acc_str = f"{r.mean_accuracy*100:.2f} $\\pm$ {r.std_accuracy*100:.2f}"
        asr_str = f"{r.mean_asr*100:.2f} $\\pm$ {r.std_asr*100:.2f}" if r.attack != "none" else "-"
        
        attack_display = r.attack.replace("_", " ").title()
        defense_display = r.defense.replace("_", " ").title()
        
        lines.append(f"{attack_display} & {defense_display} & {acc_str} & {asr_str} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def generate_markdown_table(aggregated_results: List[AggregatedResult]) -> str:
    """Generate Markdown table for README/preview."""
    
    lines = [
        "| Distribution | Attack | Defense | Accuracy (%) | ASR (%) |",
        "|--------------|--------|---------|--------------|---------|",
    ]
    
    for r in aggregated_results:
        dist_label = "IID" if r.distribution == "iid" else f"Non-IID (alpha={r.alpha})"
        acc_str = f"{r.mean_accuracy*100:.2f} +/- {r.std_accuracy*100:.2f}"
        asr_str = f"{r.mean_asr*100:.2f} +/- {r.std_asr*100:.2f}" if r.attack != "none" else "-"
        
        lines.append(f"| {dist_label} | {r.attack} | {r.defense} | {acc_str} | {asr_str} |")
    
    return "\n".join(lines)


def run_all_experiments(config: dict, results_dir: str) -> List[AggregatedResult]:
    """Run all experiment combinations."""
    
    runner = ExperimentRunner(results_dir)
    all_results = []
    aggregated = []
    
    # Define experiment matrix
    experiments = []
    
    # Baseline (no attack, no defense)
    for dist in DISTRIBUTIONS:
        alphas = [1.0] if dist == "iid" else NONIID_ALPHAS
        for alpha in alphas:
            experiments.append(("none", "none", dist, alpha))
    
    # Full attack x defense matrix (all combinations)
    attacks = ["label_flip", "backdoor", "model_replacement"]
    
    defenses = ["none", "krum", "trimmed_mean", "fltrust", "dp_sgd"]
    
    for attack in attacks:
        for defense in defenses:
            for dist in DISTRIBUTIONS:
                alphas = [1.0] if dist == "iid" else NONIID_ALPHAS
                for alpha in alphas:
                    experiments.append((attack, defense, dist, alpha))
    
    total = len(experiments) * len(config["seeds"])
    print(f"\n{'='*60}")
    print(f"RESEARCH PAPER EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Seeds per experiment: {len(config['seeds'])}")
    print(f"Total runs: {total}")
    print(f"Estimated time: {total * 0.5:.0f} - {total * 2:.0f} minutes")
    print(f"{'='*60}\n")
    
    completed = 0
    start_time = time.time()
    
    for attack, defense, dist, alpha in experiments:
        seed_results = []
        
        for seed in config["seeds"]:
            completed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            
            print(f"[{completed}/{total}] {attack}+{defense} | {dist}(a={alpha}) | seed={seed} "
                  f"| ETA: {eta/60:.1f}min")
            
            try:
                result = run_single_experiment(
                    attack, defense, dist, alpha, seed, config, runner
                )
                seed_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        if seed_results:
            agg = aggregate_results(seed_results)
            aggregated.append(agg)
            print(f"  → Accuracy: {agg.mean_accuracy*100:.2f}% ± {agg.std_accuracy*100:.2f}%")
    
    return aggregated


def save_results(
    aggregated: List[AggregatedResult],
    results_dir: str
):
    """Save results in multiple formats."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON (raw data)
    json_path = os.path.join(results_dir, f"paper_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in aggregated], f, indent=2)
    print(f"\nJSON results: {json_path}")
    
    # Markdown table
    md_path = os.path.join(results_dir, f"paper_results_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write(generate_markdown_table(aggregated))
    print(f"Markdown table: {md_path}")
    
    # LaTeX tables
    tex_path = os.path.join(results_dir, f"paper_tables_{timestamp}.tex")
    with open(tex_path, 'w') as f:
        f.write("% Auto-generated LaTeX tables for paper\n\n")
        
        # IID table
        f.write(generate_latex_table(aggregated, "iid"))
        f.write("\n\n")
        
        # Non-IID tables
        for alpha in NONIID_ALPHAS:
            f.write(generate_latex_table(aggregated, "noniid", alpha))
            f.write("\n\n")
    
    print(f"LaTeX tables: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 seed, 3 rounds)
  python run_paper_experiments.py --quick
  
  # MNIST only, no seeds (fast exploration)
  python run_paper_experiments.py --dataset mnist --seeds 1
  
  # MNIST with full seeds (paper quality)
  python run_paper_experiments.py --dataset mnist --seeds 5
  
  # Full run (both datasets, all seeds)
  python run_paper_experiments.py --dataset all
        """
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cub200", "all"],
                       help="Dataset: mnist, cub200, or all")
    parser.add_argument("--seeds", type=int, default=None,
                       help="Number of seeds (1=fast, 5=paper). Default: 5 for MNIST, 3 for CUB200")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick test (1 seed, 3 rounds, MNIST only)")
    parser.add_argument("--output", type=str, default="./experiments/paper_results",
                       help="Output directory")
    args = parser.parse_args()
    
    results_dir = args.output
    os.makedirs(results_dir, exist_ok=True)
    
    all_aggregated = []
    
    # Determine which datasets to run
    if args.quick:
        datasets_to_run = ["mnist"]
        configs = {"mnist": QUICK_CONFIG.copy()}
        configs["mnist"]["seeds"] = [42]  # Single seed for quick
    elif args.dataset == "all":
        datasets_to_run = ["mnist", "cub200"]
        configs = {"mnist": MNIST_CONFIG.copy(), "cub200": CUB200_CONFIG.copy()}
    else:
        datasets_to_run = [args.dataset]
        if args.dataset == "mnist":
            configs = {"mnist": MNIST_CONFIG.copy()}
        else:
            configs = {"cub200": CUB200_CONFIG.copy()}
    
    # Override seeds if specified
    if args.seeds is not None:
        all_seeds = [42, 123, 456, 789, 1024]
        for dataset in configs:
            configs[dataset]["seeds"] = all_seeds[:args.seeds]
    
    for dataset in datasets_to_run:
        config = configs[dataset]
        
        print(f"\n{'='*60}")
        print(f"RUNNING {dataset.upper()} EXPERIMENTS")
        print(f"{'='*60}")
        print(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, "
              f"Seeds: {len(config['seeds'])}")
        
        # Run experiments
        aggregated = run_all_experiments(config, results_dir)
        all_aggregated.extend(aggregated)
        
        # Save intermediate results
        save_results(aggregated, results_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(generate_markdown_table(all_aggregated))
    print("\n[DONE] Results saved to:", results_dir)


if __name__ == "__main__":
    main()


