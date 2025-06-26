"""
Direct solvers for Maximum Independent Set and Maximum Clique problems.

This module contains direct combinatorial optimization solvers that don't use
the Motzkin-Straus oracle pattern. These are separate from the oracle-based
approaches and provide exact solutions using MILP formulations.
"""

from .milp import solve_max_clique_milp, solve_mis_milp, GUROBI_AVAILABLE

__all__ = ["solve_max_clique_milp", "solve_mis_milp", "GUROBI_AVAILABLE"]