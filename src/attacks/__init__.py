# Attack Implementations Module

from .base_attack import BaseAttack, NoAttack
from .label_flip import LabelFlipAttack, AllToOneAttack
from .backdoor import BackdoorAttack, DistributedBackdoorAttack

__all__ = [
    'BaseAttack',
    'NoAttack',
    'LabelFlipAttack',
    'AllToOneAttack',
    'BackdoorAttack',
    'DistributedBackdoorAttack',
]


def get_attack(attack_type: str, attack_config: dict):
    """
    Factory function to get attack by name.
    
    Args:
        attack_type: Attack type ('none', 'label_flip', 'backdoor', etc.)
        attack_config: Attack configuration dictionary
        
    Returns:
        Attack instance
    """
    attacks = {
        'none': NoAttack,
        'label_flip': LabelFlipAttack,
        'all_to_one': AllToOneAttack,
        'backdoor': BackdoorAttack,
        'distributed_backdoor': DistributedBackdoorAttack,
    }
    
    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(attacks.keys())}")
    
    return attacks[attack_type](attack_config)
