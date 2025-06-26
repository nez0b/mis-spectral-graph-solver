"""
MILP (Mixed Integer Linear Programming) solvers for Maximum Independent Set and Maximum Clique problems.

These solvers use direct combinatorial optimization formulations rather than
the continuous Motzkin-Straus approach. They provide exact solutions using
Gurobi's MILP solver.
"""

import networkx as nx
import numpy as np
from typing import Set

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None


def solve_max_clique_milp(graph: nx.Graph, suppress_output: bool = True) -> Set[int]:
    """
    Find the maximum clique of a graph using a Gurobi MILP model.
    
    Formulation:
    - Variables: x_i ∈ {0, 1} for each vertex i
    - Objective: maximize Σx_i
    - Constraints: x_i + x_j ≤ 1 for all pairs (i, j) where there is NO edge
    
    Args:
        graph: The input networkx graph.
        suppress_output: Whether to suppress Gurobi's console output.
        
    Returns:
        A set of node IDs representing the maximum clique.
        
    Raises:
        ImportError: If Gurobi is not available.
        RuntimeError: If the optimization fails.
    """
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is not available for MILP solver. Please install gurobipy.")
    
    nodes = list(graph.nodes())
    n = len(nodes)
    
    if n == 0:
        return set()
    
    # Handle trivial cases
    if graph.number_of_edges() == 0:
        # No edges means maximum clique size is 1
        return {nodes[0]} if nodes else set()
    
    # Map node IDs to indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    try:
        # Create Gurobi environment and model
        with gp.Env(empty=True) as env:
            if suppress_output:
                env.setParam('OutputFlag', 0)
            env.start()
            
            with gp.Model("max_clique", env=env) as model:
                # Variables: x_i = 1 if node i is in the clique
                x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")
                
                # Objective: maximize the size of the clique
                model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
                
                # Constraints: for non-adjacent nodes (u, v), x_u + x_v ≤ 1
                # This ensures only vertices that are all connected can be selected
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        if i < j and not graph.has_edge(u, v):
                            model.addConstr(x[i] + x[j] <= 1, f"non_edge_{u}_{v}")
                
                # Optimize
                model.optimize()
                
                # Extract solution
                clique_nodes = set()
                if model.status == GRB.OPTIMAL:
                    solution = x.X
                    for i, is_in_clique in enumerate(solution):
                        if is_in_clique > 0.5:  # Binary variable
                            clique_nodes.add(nodes[i])
                elif model.status == GRB.INFEASIBLE:
                    # Should not happen for clique problems
                    raise RuntimeError("MILP model is infeasible")
                else:
                    raise RuntimeError(f"Optimization failed with status: {model.status}")
                
                return clique_nodes
                
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in MILP solver: {e}")


def solve_mis_milp(graph: nx.Graph, suppress_output: bool = True) -> Set[int]:
    """
    Find the maximum independent set of a graph using a Gurobi MILP model.
    
    Formulation:
    - Variables: x_i ∈ {0, 1} for each vertex i
    - Objective: maximize Σx_i
    - Constraints: x_i + x_j ≤ 1 for all pairs (i, j) where there IS an edge
    
    Args:
        graph: The input networkx graph.
        suppress_output: Whether to suppress Gurobi's console output.
        
    Returns:
        A set of node IDs representing the maximum independent set.
        
    Raises:
        ImportError: If Gurobi is not available.
        RuntimeError: If the optimization fails.
    """
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is not available for MILP solver. Please install gurobipy.")
    
    nodes = list(graph.nodes())
    n = len(nodes)
    
    if n == 0:
        return set()
    
    # Handle trivial cases
    if graph.number_of_edges() == 0:
        # No edges means all vertices are independent
        return set(nodes)
    
    # Map node IDs to indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    try:
        # Create Gurobi environment and model
        with gp.Env(empty=True) as env:
            if suppress_output:
                env.setParam('OutputFlag', 0)
            env.start()
            
            with gp.Model("max_independent_set", env=env) as model:
                # Variables: x_i = 1 if node i is in the independent set
                x = model.addMVar(shape=n, vtype=GRB.BINARY, name="x")
                
                # Objective: maximize the size of the independent set
                model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
                
                # Constraints: for adjacent nodes (u, v), x_u + x_v ≤ 1
                # This ensures no two adjacent vertices can both be selected
                for u, v in graph.edges():
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    model.addConstr(x[u_idx] + x[v_idx] <= 1, f"edge_{u}_{v}")
                
                # Optimize
                model.optimize()
                
                # Extract solution
                mis_nodes = set()
                if model.status == GRB.OPTIMAL:
                    solution = x.X
                    for i, is_in_mis in enumerate(solution):
                        if is_in_mis > 0.5:  # Binary variable
                            mis_nodes.add(nodes[i])
                elif model.status == GRB.INFEASIBLE:
                    # Should not happen for MIS problems
                    raise RuntimeError("MILP model is infeasible")
                else:
                    raise RuntimeError(f"Optimization failed with status: {model.status}")
                
                return mis_nodes
                
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in MILP solver: {e}")


def get_clique_number_milp(graph: nx.Graph, suppress_output: bool = True) -> int:
    """
    Get the clique number (size of maximum clique) using MILP.
    
    Args:
        graph: The input networkx graph.
        suppress_output: Whether to suppress Gurobi's console output.
        
    Returns:
        The size of the maximum clique.
    """
    clique = solve_max_clique_milp(graph, suppress_output)
    return len(clique)


def get_independence_number_milp(graph: nx.Graph, suppress_output: bool = True) -> int:
    """
    Get the independence number (size of maximum independent set) using MILP.
    
    Args:
        graph: The input networkx graph.
        suppress_output: Whether to suppress Gurobi's console output.
        
    Returns:
        The size of the maximum independent set.
    """
    mis = solve_mis_milp(graph, suppress_output)
    return len(mis)