# Defense Mechanisms Module

from .base_defense import BaseDefense, NoDefense
from .krum import KrumDefense, MultiKrumDefense
from .trimmed_mean import TrimmedMeanDefense, MedianDefense, GeometricMedianDefense
from .fltrust import FLTrustDefense
from .differential_privacy import DPSGDDefense, GradientClippingDefense, NormBoundingDefense

__all__ = [
    # Base
    'BaseDefense',
    'NoDefense',
    # Byzantine-robust aggregation
    'KrumDefense',
    'MultiKrumDefense',
    'TrimmedMeanDefense',
    'MedianDefense',
    'GeometricMedianDefense',
    # Trust-based
    'FLTrustDefense',
    # Privacy-preserving
    'DPSGDDefense',
    'GradientClippingDefense',
    'NormBoundingDefense',
]


def get_defense(defense_type: str, defense_config: dict):
    """
    Factory function to get defense by name.
    
    Args:
        defense_type: Defense type name
        defense_config: Defense configuration dictionary
        
    Returns:
        Defense instance
    """
    defenses = {
        'none': NoDefense,
        'fedavg': NoDefense,
        # Byzantine-robust
        'krum': KrumDefense,
        'multi_krum': MultiKrumDefense,
        'trimmed_mean': TrimmedMeanDefense,
        'median': MedianDefense,
        'geometric_median': GeometricMedianDefense,
        # Trust-based
        'fltrust': FLTrustDefense,
        # Privacy-preserving
        'dp_sgd': DPSGDDefense,
        'gradient_clipping': GradientClippingDefense,
        'norm_bounding': NormBoundingDefense,
    }
    
    if defense_type not in defenses:
        raise ValueError(f"Unknown defense type: {defense_type}. Available: {list(defenses.keys())}")
    
    return defenses[defense_type](defense_config)
