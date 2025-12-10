"""
Statistical Tests for FL Security Experiments

Provides statistical analysis utilities including t-tests, confidence intervals,
and significance testing for comparing attack/defense scenarios.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import os


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data) if n > 1 else 0
    
    if n > 1:
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        h = 0
    
    return mean, mean - h, mean + h


def paired_t_test(
    data1: List[float],
    data2: List[float],
    alternative: str = 'two-sided'
) -> Dict:
    """
    Perform paired t-test between two conditions.
    
    Args:
        data1: First condition measurements
        data2: Second condition measurements
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length")
    
    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
    
    # Effect size (Cohen's d)
    diff = np.array(data1) - np.array(data2)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


def independent_t_test(
    data1: List[float],
    data2: List[float],
    alternative: str = 'two-sided'
) -> Dict:
    """
    Perform independent samples t-test.
    
    Args:
        data1: Group 1 measurements
        data2: Group 2 measurements
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with test results
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, alternative=alternative)
    
    # Effect size (Cohen's d for independent samples)
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


def compare_attack_defense_scenarios(
    baseline_results: List[float],
    attack_only_results: List[float],
    attack_defense_results: List[float]
) -> Dict:
    """
    Compare three scenarios: baseline, attack-only, attack+defense.
    
    Returns:
        Dictionary with all comparisons and significance tests
    """
    comparisons = {}
    
    # Baseline vs Attack (should show degradation)
    comparisons['baseline_vs_attack'] = {
        'baseline_mean': np.mean(baseline_results),
        'attack_mean': np.mean(attack_only_results),
        'difference': np.mean(baseline_results) - np.mean(attack_only_results),
        **independent_t_test(baseline_results, attack_only_results)
    }
    
    # Attack-only vs Attack+Defense (defense effectiveness)
    comparisons['attack_vs_defense'] = {
        'attack_mean': np.mean(attack_only_results),
        'defense_mean': np.mean(attack_defense_results),
        'improvement': np.mean(attack_defense_results) - np.mean(attack_only_results),
        **independent_t_test(attack_defense_results, attack_only_results, alternative='greater')
    }
    
    # Baseline vs Attack+Defense (recovery)
    comparisons['baseline_vs_defense'] = {
        'baseline_mean': np.mean(baseline_results),
        'defense_mean': np.mean(attack_defense_results),
        'gap': np.mean(baseline_results) - np.mean(attack_defense_results),
        **independent_t_test(baseline_results, attack_defense_results)
    }
    
    return comparisons


def analyze_experiment_results(results_dir: str) -> Dict:
    """
    Analyze all experiment results with statistical tests.
    
    Args:
        results_dir: Directory containing experiment JSON files
        
    Returns:
        Comprehensive statistical analysis
    """
    # Load all results
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and not filename.startswith('matrix'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Extract key info
                name = data.get('config', {}).get('name', filename)
                results[name] = data
    
    # Group by attack/defense
    grouped = {}
    for name, data in results.items():
        config = data.get('config', {})
        attack = config.get('attack_type', 'none')
        defense = config.get('defense_type', 'none')
        key = (attack, defense)
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(data.get('final_accuracy', 0))
    
    # Compute statistics per group
    analysis = {}
    for (attack, defense), accuracies in grouped.items():
        mean, lower, upper = compute_confidence_interval(accuracies)
        analysis[f"{attack}_{defense}"] = {
            'attack': attack,
            'defense': defense,
            'mean_accuracy': mean,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': np.std(accuracies),
            'n': len(accuracies)
        }
    
    return analysis


def generate_statistical_report(
    analysis: Dict,
    output_path: str
):
    """Generate markdown statistical report."""
    lines = [
        "# Statistical Analysis Report",
        "",
        "## Summary Statistics",
        "",
        "| Scenario | Mean Acc | 95% CI | Std | N |",
        "|----------|----------|--------|-----|---|"
    ]
    
    for name, stats in sorted(analysis.items()):
        lines.append(
            f"| {name} | {stats['mean_accuracy']:.4f} | "
            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | "
            f"{stats['std']:.4f} | {stats['n']} |"
        )
    
    lines.extend([
        "",
        "## Significance Tests",
        "",
        "Statistical significance determined at p < 0.05",
        ""
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    print("Statistical Tests Demo")
    print("="*50)
    
    # Example data
    baseline = [0.95, 0.96, 0.94, 0.95, 0.97]
    attacked = [0.75, 0.78, 0.72, 0.76, 0.74]
    defended = [0.92, 0.91, 0.93, 0.90, 0.94]
    
    # Run comparisons
    comparisons = compare_attack_defense_scenarios(baseline, attacked, defended)
    
    print("\nBaseline vs Attack:")
    print(f"  Baseline: {np.mean(baseline):.3f}")
    print(f"  Attack: {np.mean(attacked):.3f}")
    print(f"  p-value: {comparisons['baseline_vs_attack']['p_value']:.4f}")
    print(f"  Significant: {comparisons['baseline_vs_attack']['significant_005']}")
    
    print("\nAttack vs Defense:")
    print(f"  Attack only: {np.mean(attacked):.3f}")
    print(f"  With defense: {np.mean(defended):.3f}")
    print(f"  p-value: {comparisons['attack_vs_defense']['p_value']:.4f}")
    print(f"  Significant: {comparisons['attack_vs_defense']['significant_005']}")
    
    print("\nBaseline vs Defense:")
    print(f"  Baseline: {np.mean(baseline):.3f}")
    print(f"  With defense: {np.mean(defended):.3f}")
    print(f"  Gap: {comparisons['baseline_vs_defense']['gap']:.3f}")
