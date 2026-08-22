"""defenses module. See GEMINI.md §4 for the responsibility boundary of this package."""
from admarl.defenses.base import BaseCriticRegularizer
from admarl.defenses.factory import get_regularizer, get_training_defense
from admarl.defenses.grad_penalty import GradientPenaltyRegularizer
from admarl.defenses.no_defense import NoRegularizer
from admarl.defenses.sa_ppo import SAPPOAdversarialDefense
from admarl.defenses.training_defense import BaseTrainingDefense, NoActorDefense

__all__ = [
    "BaseCriticRegularizer",
    "BaseTrainingDefense",
    "GradientPenaltyRegularizer",
    "NoActorDefense",
    "NoRegularizer",
    "SAPPOAdversarialDefense",
    "get_regularizer",
    "get_training_defense",
]
