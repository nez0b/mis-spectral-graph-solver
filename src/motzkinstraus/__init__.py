"""
Motzkin-Straus Maximum Independent Set Solver

This package implements a Maximum Independent Set (MIS) solver using the 
Motzkin-Straus theorem to transform the discrete optimization problem into
a continuous quadratic programming problem.
"""

from .algorithms import find_mis_with_oracle, find_mis_brute_force
from .oracles import get_available_oracles

__version__ = "0.1.0"
__all__ = ["find_mis_with_oracle", "find_mis_brute_force", "get_available_oracles"]