"""
Oracle implementations for solving the Motzkin-Straus quadratic program.
"""

from .base import Oracle
from typing import List, Type

def get_available_oracles() -> List[Type[Oracle]]:
    """
    Returns a list of available oracle implementations.
    
    Returns:
        List of available Oracle classes that can be instantiated.
    """
    available = []
    
    try:
        from .gurobi import GurobiOracle
        available.append(GurobiOracle)
    except ImportError:
        pass
    
    try:
        from .dirac import DiracOracle
        available.append(DiracOracle)
    except ImportError:
        pass
    
    try:
        from .jax_pgd import ProjectedGradientDescentOracle
        available.append(ProjectedGradientDescentOracle)
    except ImportError:
        pass
    
    try:
        from .jax_mirror import MirrorDescentOracle
        available.append(MirrorDescentOracle)
    except ImportError:
        pass
    
    return available

__all__ = ["Oracle", "get_available_oracles"]