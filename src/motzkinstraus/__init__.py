"""
Motzkin-Straus Maximum Independent Set and Maximum Clique Solver

This package implements solvers for Maximum Independent Set (MIS) and Maximum Clique 
problems using both:
1. The Motzkin-Straus theorem to transform discrete optimization into continuous QP
2. Direct MILP formulations for exact solutions

The package provides both oracle-based algorithms and direct MILP solvers.
"""

from .algorithms import (
    find_mis_with_oracle, 
    find_mis_brute_force,
    find_max_clique_with_oracle,
    find_max_clique_brute_force,
    verify_independent_set,
    verify_clique
)
from .oracles import get_available_oracles

# Import Gurobi MILP solvers with graceful handling of missing dependencies
try:
    from .solvers.milp import (
        solve_max_clique_milp as solve_max_clique_gurobi,
        solve_mis_milp as solve_mis_gurobi,
        get_clique_number_milp as get_clique_number_gurobi,
        get_independence_number_milp as get_independence_number_gurobi,
        GUROBI_AVAILABLE
    )
except ImportError:
    GUROBI_AVAILABLE = False
    solve_max_clique_gurobi = None
    solve_mis_gurobi = None
    get_clique_number_gurobi = None
    get_independence_number_gurobi = None

# Import SciPy MILP solvers with graceful handling of missing dependencies
try:
    from .solvers.scipy_milp import (
        solve_max_clique_scipy,
        solve_mis_scipy,
        get_clique_number_scipy,
        get_independence_number_scipy,
        SCIPY_MILP_AVAILABLE
    )
except ImportError:
    SCIPY_MILP_AVAILABLE = False
    solve_max_clique_scipy = None
    solve_mis_scipy = None
    get_clique_number_scipy = None
    get_independence_number_scipy = None

# For backward compatibility, provide generic MILP functions
# Prefer Gurobi if available (better performance), otherwise use SciPy
MILP_AVAILABLE = GUROBI_AVAILABLE or SCIPY_MILP_AVAILABLE

if GUROBI_AVAILABLE:
    solve_max_clique_milp = solve_max_clique_gurobi
    solve_mis_milp = solve_mis_gurobi
    get_clique_number_milp = get_clique_number_gurobi
    get_independence_number_milp = get_independence_number_gurobi
elif SCIPY_MILP_AVAILABLE:
    solve_max_clique_milp = solve_max_clique_scipy
    solve_mis_milp = solve_mis_scipy
    get_clique_number_milp = get_clique_number_scipy
    get_independence_number_milp = get_independence_number_scipy
else:
    solve_max_clique_milp = None
    solve_mis_milp = None
    get_clique_number_milp = None
    get_independence_number_milp = None

__version__ = "0.2.0"
__all__ = [
    # Oracle-based algorithms
    "find_mis_with_oracle", 
    "find_mis_brute_force",
    "find_max_clique_with_oracle",
    "find_max_clique_brute_force",
    # Verification functions
    "verify_independent_set",
    "verify_clique",
    # Oracle utilities
    "get_available_oracles",
    # Generic MILP solvers (backward compatibility)
    "solve_max_clique_milp",
    "solve_mis_milp", 
    "get_clique_number_milp",
    "get_independence_number_milp",
    # Specific MILP solvers
    "solve_max_clique_gurobi",
    "solve_mis_gurobi",
    "get_clique_number_gurobi", 
    "get_independence_number_gurobi",
    "solve_max_clique_scipy",
    "solve_mis_scipy",
    "get_clique_number_scipy",
    "get_independence_number_scipy",
    # Availability flags
    "MILP_AVAILABLE",
    "GUROBI_AVAILABLE",
    "SCIPY_MILP_AVAILABLE"
]