"""
Visualization Utilities

Generate plots and tables for experiment results.
"""

import os
import json
import glob
from typing import List, Dict, Optional
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


def load_results(results_dir: str = "./experiments/results") -> List[Dict]:
    """Load all experiment results from directory."""
    results = []
    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    return results


def plot_accuracy_comparison(
    results: List[Dict],
    output_path: str = "./experiments/results/accuracy_comparison.png",
    title: str = "Accuracy Comparison"
):
    """Plot accuracy vs round for multiple experiments."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot - matplotlib not available")
        return
    
    plt.figure(figsize=(10, 6))
    
    for result in results:
        name = result['config']['name']
        accuracies = result['round_accuracies']
        rounds = list(range(1, len(accuracies) + 1))
        plt.plot(rounds, accuracies, marker='o', label=name)
    
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_attack_defense_matrix(
    results: List[Dict],
    output_path: str = "./experiments/results/attack_defense_matrix.png"
):
    """Plot heatmap of accuracy for attack vs defense combinations."""
    if not HAS_MATPLOTLIB:
        return
    
    # Extract attack and defense types
    attacks = set()
    defenses = set()
    data = {}
    
    for result in results:
        cfg = result['config']
        attack = cfg.get('attack_type', 'none') if cfg.get('attack_enabled') else 'none'
        defense = cfg.get('defense_type', 'none') if cfg.get('defense_enabled') else 'none'
        
        attacks.add(attack)
        defenses.add(defense)
        data[(attack, defense)] = result['final_accuracy']
    
    attacks = sorted(attacks)
    defenses = sorted(defenses)
    
    # Create matrix
    matrix = np.zeros((len(attacks), len(defenses)))
    for i, attack in enumerate(attacks):
        for j, defense in enumerate(defenses):
            matrix[i, j] = data.get((attack, defense), 0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    plt.xticks(range(len(defenses)), defenses, rotation=45)
    plt.yticks(range(len(attacks)), attacks)
    plt.xlabel('Defense')
    plt.ylabel('Attack')
    plt.title('Final Accuracy: Attack vs Defense')
    plt.colorbar(label='Accuracy')
    
    # Add text annotations
    for i in range(len(attacks)):
        for j in range(len(defenses)):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_table(
    results: List[Dict],
    output_path: str = "./experiments/results/results_table.md"
):
    """Generate markdown table of results."""
    
    lines = [
        "# Experiment Results",
        "",
        "| Experiment | Clients | Attack | Defense | Final Acc | ASR | Time (s) |",
        "|------------|---------|--------|---------|-----------|-----|----------|"
    ]
    
    for result in results:
        cfg = result['config']
        name = cfg.get('name', 'unknown')[:20]
        clients = cfg.get('num_clients', '-')
        attack = cfg.get('attack_type', 'none') if cfg.get('attack_enabled') else 'none'
        defense = cfg.get('defense_type', 'none') if cfg.get('defense_enabled') else 'none'
        acc = result.get('final_accuracy', 0)
        asr = result.get('attack_success_rate')
        asr_str = f"{asr:.2%}" if asr is not None else "-"
        time_s = result.get('training_time_seconds', 0)
        
        lines.append(f"| {name} | {clients} | {attack} | {defense} | {acc:.2%} | {asr_str} | {time_s:.1f} |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved: {output_path}")


def generate_all_visualizations(results_dir: str = "./experiments/results"):
    """Generate all visualizations from experiment results."""
    results = load_results(results_dir)
    
    if not results:
        print("No results found in", results_dir)
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Generate plots
    plot_accuracy_comparison(results)
    plot_attack_defense_matrix(results)
    generate_results_table(results)
    
    print("\n✅ All visualizations generated!")


if __name__ == "__main__":
    generate_all_visualizations()
