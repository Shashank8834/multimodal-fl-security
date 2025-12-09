# Defense Mechanisms Module

from .base_defense import BaseDefense, NoDefense
from .krum import KrumDefense, MultiKrumDefense
from .trimmed_mean import TrimmedMeanDefense, MedianDefense, GeometricMedianDefense
from .fltrust import FLTrustDefense

__all__ = [
    'BaseDefense',
    'NoDefense',
    'KrumDefense',
    'MultiKrumDefense',
    'TrimmedMeanDefense',
    'MedianDefense',
    'GeometricMedianDefense',
    'FLTrustDefense',
]


def get_defense(defense_type: str, defense_config: dict):
    """
    Factory function to get defense by name.
    
    Args:
        defense_type: Defense type ('none', 'krum', 'trimmed_mean', etc.)
        defense_config: Defense configuration dictionary
        
    Returns:
        Defense instance
    """
    defenses = {
        'none': NoDefense,
        'fedavg': NoDefense,
        'krum': KrumDefense,
        'multi_krum': MultiKrumDefense,
        'trimmed_mean': TrimmedMeanDefense,
        'median': MedianDefense,
        'geometric_median': GeometricMedianDefense,
        'fltrust': FLTrustDefense,
    }
    
    if defense_type not in defenses:
        raise ValueError(f"Unknown defense type: {defense_type}. Available: {list(defenses.keys())}")
    
    return defenses[defense_type](defense_config)
