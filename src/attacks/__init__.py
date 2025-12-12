# Attack Implementations Module

from .base_attack import BaseAttack, NoAttack
from .label_flip import LabelFlipAttack, AllToOneAttack
from .backdoor import BackdoorAttack, DistributedBackdoorAttack
from .model_poisoning import (
    ModelReplacementAttack,
    AdaptiveKrumAttack,
    ScalingAttack,
    InnerProductManipulationAttack
)
# Cross-modal attacks moved to future_paper2_crossmodal/ for Paper 2

__all__ = [
    # Base
    'BaseAttack',
    'NoAttack',
    # Data poisoning
    'LabelFlipAttack',
    'AllToOneAttack',
    'BackdoorAttack',
    'DistributedBackdoorAttack',
    # Model poisoning
    'ModelReplacementAttack',
    'AdaptiveKrumAttack',
    'ScalingAttack',
    'InnerProductManipulationAttack',
]


def get_attack(attack_type: str, attack_config: dict):
    """
    Factory function to get attack by name.
    
    Args:
        attack_type: Attack type name
        attack_config: Attack configuration dictionary
        
    Returns:
        Attack instance
    """
    attacks = {
        'none': NoAttack,
        # Data poisoning
        'label_flip': LabelFlipAttack,
        'all_to_one': AllToOneAttack,
        'backdoor': BackdoorAttack,
        'distributed_backdoor': DistributedBackdoorAttack,
        # Model poisoning
        'model_replacement': ModelReplacementAttack,
        'adaptive_krum': AdaptiveKrumAttack,
        'scaling': ScalingAttack,
        'ipm': InnerProductManipulationAttack,
    }
    
    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(attacks.keys())}")
    
    return attacks[attack_type](attack_config)

