"""
MILP solvers for Maximum Independent Set and Maximum Clique using SciPy.

These solvers use scipy.optimize.milp, which relies on the HiGHS backend.
This provides a dependency-light, open-source alternative to Gurobi.

Mathematical Formulations:
- MIS: maximize Σx_i subject to x_i + x_j ≤ 1 for each edge (i,j)
- Max Clique: maximize Σx_i subject to x_i + x_j ≤ 1 for each non-edge (i,j)
- Variables: x_i ∈ {0,1} for each vertex i

Note: scipy.milp minimizes, so we minimize -Σx_i to maximize Σx_i.
"""

import networkx as nx
import numpy as np
from typing import Set

try:
    from scipy.optimize import milp, OptimizeResult, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    SCIPY_MILP_AVAILABLE = True
except ImportError:
    SCIPY_MILP_AVAILABLE = False
    milp = None
    OptimizeResult = None
    Bounds = None
    LinearConstraint = None
    lil_matrix = None


def solve_mis_scipy(graph: nx.Graph, suppress_output: bool = True) -> Set[int]:
    """
    Find the maximum independent set of a graph using scipy.optimize.milp.
    
    Args:
        graph: Input graph
        suppress_output: Whether to suppress solver output
        
    Returns:
        Set of nodes forming the maximum independent set
        
    Raises:
        ImportError: If scipy.optimize.milp is not available
        RuntimeError: If optimization fails
    """
    if not SCIPY_MILP_AVAILABLE:
        raise ImportError("scipy.optimize.milp is not available. Please upgrade SciPy to version 1.9.0 or later.")

    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Handle edge cases
    if n == 0:
        return set()
    
    if graph.number_of_edges() == 0:
        # No edges means all nodes are independent
        return set(nodes)

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Objective: maximize Σx_i  --->  minimize -Σx_i
    c = -np.ones(n)

    # Constraints: x_i + x_j <= 1 for each edge (i, j)
    num_constraints = graph.number_of_edges()
    A_ub = lil_matrix((num_constraints, n), dtype=np.float64)
    
    for i, (u, v) in enumerate(graph.edges()):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        A_ub[i, u_idx] = 1
        A_ub[i, v_idx] = 1
    
    # Convert to efficient format for matrix operations
    A_ub = A_ub.tocsr()
    b_ub = np.ones(num_constraints)

    # All variables are binary (integer, bounds 0 to 1)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)  # Lower and upper bound for all variables
    
    # Create linear constraint
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)

    # Solve the MILP
    options = {'disp': not suppress_output}
    res: OptimizeResult = milp(
        c=c, 
        constraints=constraints, 
        integrality=integrality, 
        bounds=bounds, 
        options=options
    )

    if not res.success:
        raise RuntimeError(f"SciPy MILP optimization failed: {res.message}")

    # Extract solution (values > 0.5 are considered selected)
    mis_nodes = {nodes[i] for i, val in enumerate(res.x) if val > 0.5}
    return mis_nodes


def solve_max_clique_scipy(graph: nx.Graph, suppress_output: bool = True) -> Set[int]:
    """
    Find the maximum clique of a graph using scipy.optimize.milp.
    
    Args:
        graph: Input graph
        suppress_output: Whether to suppress solver output
        
    Returns:
        Set of nodes forming the maximum clique
        
    Raises:
        ImportError: If scipy.optimize.milp is not available
        RuntimeError: If optimization fails
    """
    if not SCIPY_MILP_AVAILABLE:
        raise ImportError("scipy.optimize.milp is not available. Please upgrade SciPy to version 1.9.0 or later.")

    nodes = list(graph.nodes())
    n = len(nodes)
    
    # Handle edge cases
    if n == 0:
        return set()
    
    # Get non-edges (missing edges in the graph)
    non_edges = list(nx.non_edges(graph))
    
    if len(non_edges) == 0:
        # Graph is complete, so all nodes form a clique
        return set(nodes)

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Objective: maximize Σx_i  --->  minimize -Σx_i
    c = -np.ones(n)

    # Constraints: x_i + x_j <= 1 for each NON-edge (i, j)
    num_constraints = len(non_edges)
    A_ub = lil_matrix((num_constraints, n), dtype=np.float64)
    
    for i, (u, v) in enumerate(non_edges):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        A_ub[i, u_idx] = 1
        A_ub[i, v_idx] = 1
    
    # Convert to efficient format
    A_ub = A_ub.tocsr()
    b_ub = np.ones(num_constraints)

    # All variables are binary
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    # Create linear constraint
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)

    # Solve the MILP
    options = {'disp': not suppress_output}
    res: OptimizeResult = milp(
        c=c, 
        constraints=constraints, 
        integrality=integrality, 
        bounds=bounds, 
        options=options
    )

    if not res.success:
        raise RuntimeError(f"SciPy MILP optimization failed: {res.message}")

    # Extract solution
    clique_nodes = {nodes[i] for i, val in enumerate(res.x) if val > 0.5}
    return clique_nodes


def get_independence_number_scipy(graph: nx.Graph, suppress_output: bool = True) -> int:
    """
    Get the independence number (size of maximum independent set) using SciPy.
    
    Args:
        graph: Input graph
        suppress_output: Whether to suppress solver output
        
    Returns:
        Size of the maximum independent set
    """
    mis = solve_mis_scipy(graph, suppress_output)
    return len(mis)


def get_clique_number_scipy(graph: nx.Graph, suppress_output: bool = True) -> int:
    """
    Get the clique number (size of maximum clique) using SciPy.
    
    Args:
        graph: Input graph
        suppress_output: Whether to suppress solver output
        
    Returns:
        Size of the maximum clique
    """
    clique = solve_max_clique_scipy(graph, suppress_output)
    return len(clique)


# Aliases for consistency with Gurobi naming
solve_mis_milp_scipy = solve_mis_scipy
solve_max_clique_milp_scipy = solve_max_clique_scipy
get_independence_number_milp_scipy = get_independence_number_scipy
get_clique_number_milp_scipy = get_clique_number_scipy