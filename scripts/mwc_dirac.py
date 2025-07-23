#!/usr/bin/env python3
"""
Maximum Weight Clique solver using Dirac-3 quantum annealing with Gibbons' weighted Motzkin-Straus theorem.

This implementation uses the script-level Dirac oracle adapter instead of JAX-PGD optimization:
- For weighted clique number ω(w,G), we have 1/ω(w,G) = min{x^T B x | e^T x = 1, x ≥ 0}
- Matrix B construction: B[i,i] = 1/w[i], B[i,j] = 0 for adjacent vertices
- Uses QCI's Dirac-3 continuous cloud solver for quantum annealing optimization
"""

import numpy as np
import networkx as nx
import sys
import os
from typing import Dict, List, Set, Tuple, Any, Optional
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Dirac oracle adapter
try:
    from oracles.dirac_adapter import DiracAdapter, DiracConfig
    from oracles.base import OracleAdapter
    DIRAC_AVAILABLE = True
    print("Dirac oracle adapter available")
except ImportError as e:
    print(f"Dirac oracle adapter not available: {e}")
    DIRAC_AVAILABLE = False
    DiracAdapter = None
    DiracConfig = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available for plotting")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available")

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
    print("SciPy MILP solver available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy MILP solver not available")


def construct_gibbons_matrix(graph: nx.Graph, weights: Dict[int, float]) -> Tuple[np.ndarray, List[int]]:
    """
    Construct the matrix B according to Gibbons' Theorem 5.
    
    From the paper: B ∈ M(w,G) where:
    - B[i,i] = 1/w[i] for all vertices i  
    - B[i,j] = 0 for all adjacent vertices (i,j) ∈ E
    - B[i,j] + B[j,i] ≥ (1/w[i] + 1/w[j]) for all non-adjacent vertices (i,j) ∈ Ē
    
    Args:
        graph: The input graph
        weights: Vertex weights dictionary
        
    Returns:
        Tuple of (B_matrix, node_list)
    """
    node_list = list(graph.nodes())
    n = len(node_list)
    
    # Initialize B matrix
    B = np.zeros((n, n))
    
    # Set diagonal entries: B[i,i] = 1/w[i]
    for i, node in enumerate(node_list):
        B[i, i] = 1.0 / weights.get(node, 1.0)
    
    # Set off-diagonal entries
    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if i != j:
                if graph.has_edge(node_i, node_j):
                    # Adjacent vertices: B[i,j] = 0
                    B[i, j] = 0.0
                else:
                    # Non-adjacent vertices: B[i,j] + B[j,i] ≥ (1/w[i] + 1/w[j])
                    # To satisfy the constraint with equality, set:
                    # B[i,j] = B[j,i] = (1/w[i] + 1/w[j]) / 2
                    w_i = weights.get(node_i, 1.0)
                    w_j = weights.get(node_j, 1.0) 
                    constraint_value = (1.0 / w_i + 1.0 / w_j) / 2.0
                    B[i, j] = constraint_value
    
    return B, node_list


def validate_clique(graph: nx.Graph, vertices: Set[int]) -> Tuple[bool, str]:
    """
    Validate that a set of vertices forms a valid clique.
    
    Args:
        graph: The input graph
        vertices: Set of vertices to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(vertices) <= 1:
        return True, "Single vertex or empty set is always a clique"
    
    # Check all pairs are connected
    vertices_list = list(vertices)
    for i in range(len(vertices_list)):
        for j in range(i + 1, len(vertices_list)):
            v1, v2 = vertices_list[i], vertices_list[j]
            if not graph.has_edge(v1, v2):
                return False, f"Vertices {v1} and {v2} are not connected"
    
    return True, f"Valid clique of size {len(vertices)}"


def plot_graph_with_clique(
    graph: nx.Graph, 
    weights: Dict[int, float], 
    clique_vertices: Set[int],
    title: str,
    save_path: str = None
) -> bool:
    """
    Plot graph with maximum weight clique highlighted.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        clique_vertices: Vertices in the maximum weight clique
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        True if plot was created successfully
    """
    if not MATPLOTLIB_AVAILABLE:
        return False
    
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for nice visualization
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Draw all edges first (in gray)
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, alpha=0.6)
    
    # Highlight clique edges (in red)
    clique_edges = []
    for v1 in clique_vertices:
        for v2 in clique_vertices:
            if v1 < v2 and graph.has_edge(v1, v2):
                clique_edges.append((v1, v2))
    
    if clique_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=clique_edges, 
                               edge_color='red', width=3, alpha=0.8)
    
    # Draw non-clique vertices
    non_clique_vertices = set(graph.nodes()) - clique_vertices
    if non_clique_vertices:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(non_clique_vertices),
                               node_color='lightblue', node_size=300, alpha=0.7)
    
    # Draw clique vertices (highlighted)
    if clique_vertices:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(clique_vertices),
                               node_color='red', node_size=500, alpha=0.9)
    
    # Add labels with weights
    labels = {}
    for node in graph.nodes():
        weight = weights.get(node, 1.0)
        labels[node] = f"{node}\n({weight:.2f})"
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')
    
    # Add title and info
    clique_weight = sum(weights.get(v, 1.0) for v in clique_vertices)
    plt.title(f"{title}\nMax Weight Clique: {clique_vertices} (weight: {clique_weight:.3f})", 
              fontsize=12, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Clique vertices'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=8, label='Other vertices'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Clique edges'),
        plt.Line2D([0], [0], color='lightgray', linewidth=1, label='Other edges')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return True


def plot_graph_with_dual_solutions(
    graph: nx.Graph, 
    weights: Dict[int, float], 
    milp_clique: Set[int],
    dirac_clique: Set[int],
    title: str,
    save_path: str = None
) -> bool:
    """
    Plot both MILP and Dirac solutions with different colors.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        milp_clique: Vertices in MILP solution
        dirac_clique: Vertices in Dirac solution
        title: Plot title
        save_path: Path to save the plot (optional)
        
    Returns:
        True if plot was created successfully
    """
    if not MATPLOTLIB_AVAILABLE:
        return False
    
    plt.figure(figsize=(14, 10))
    
    # Layout
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Draw all edges (light gray)
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, alpha=0.5)
    
    # MILP clique edges (red)
    milp_edges = []
    for v1 in milp_clique:
        for v2 in milp_clique:
            if v1 < v2 and graph.has_edge(v1, v2):
                milp_edges.append((v1, v2))
    
    # Dirac clique edges (blue)
    dirac_edges = []
    for v1 in dirac_clique:
        for v2 in dirac_clique:
            if v1 < v2 and graph.has_edge(v1, v2):
                dirac_edges.append((v1, v2))
    
    # Find overlapping and unique edges
    milp_edges_set = set(milp_edges)
    dirac_edges_set = set(dirac_edges)
    common_edges = list(milp_edges_set & dirac_edges_set)
    milp_only_edges = list(milp_edges_set - dirac_edges_set)
    dirac_only_edges = list(dirac_edges_set - milp_edges_set)
    
    # Draw edges
    if common_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=common_edges, 
                               edge_color='purple', width=4, alpha=0.9)
    if milp_only_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=milp_only_edges, 
                               edge_color='red', width=3, alpha=0.8)
    if dirac_only_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=dirac_only_edges, 
                               edge_color='blue', width=3, alpha=0.8)
    
    # Categorize vertices
    milp_only = milp_clique - dirac_clique
    dirac_only = dirac_clique - milp_clique
    common_vertices = milp_clique & dirac_clique
    other_vertices = set(graph.nodes()) - milp_clique - dirac_clique
    
    # Draw vertices
    if other_vertices:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(other_vertices),
                               node_color='lightgray', node_size=300, alpha=0.6)
    if milp_only:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(milp_only),
                               node_color='red', node_size=500, alpha=0.9, 
                               edgecolors='darkred', linewidths=2)
    if dirac_only:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(dirac_only),
                               node_color='blue', node_size=500, alpha=0.9,
                               edgecolors='darkblue', linewidths=2)
    if common_vertices:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(common_vertices),
                               node_color='purple', node_size=600, alpha=0.9,
                               edgecolors='darkmagenta', linewidths=3)
    
    # Labels
    labels = {node: f"{node}\\n({weights.get(node, 1.0):.2f})" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')
    
    # Calculate weights
    milp_weight = sum(weights.get(v, 1.0) for v in milp_clique)
    dirac_weight = sum(weights.get(v, 1.0) for v in dirac_clique)
    
    # Title
    solutions_match = milp_clique == dirac_clique
    match_text = "✓ MATCH" if solutions_match else "✗ DIFFER"
    
    plt.title(f"{title}\\n" + 
              f"MILP: {milp_clique} (weight: {milp_weight:.3f})\\n" + 
              f"Dirac: {dirac_clique} (weight: {dirac_weight:.3f}) - {match_text}", 
              fontsize=11, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                   markersize=12, markeredgecolor='darkmagenta', markeredgewidth=2,
                   label='Both MILP & Dirac'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='darkred', markeredgewidth=2,
                   label='MILP only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='darkblue', markeredgewidth=2,
                   label='Dirac only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=8, label='Other vertices'),
        plt.Line2D([0], [0], color='purple', linewidth=4, label='Common edges'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='MILP edges'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Dirac edges')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual solution plot saved to {save_path}")
    
    plt.close()
    return True


def solve_maximum_weight_clique_theory(
    graph: nx.Graph, 
    weights: Dict[int, float], 
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve maximum weight clique using theoretical analysis for simple cases.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        verbose: Print debug information
        
    Returns:
        Dictionary with solution information
    """
    if verbose:
        print(f"Theoretical analysis for graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Find all maximal cliques and their weights
    maximal_cliques = list(nx.find_cliques(graph))
    clique_weights = []
    
    for clique in maximal_cliques:
        clique_set = set(clique)
        weight = sum(weights.get(v, 1.0) for v in clique_set)
        clique_weights.append((clique_set, weight))
        if verbose:
            print(f"  Clique {clique_set}: weight = {weight}")
    
    # Find maximum weight clique
    if clique_weights:
        best_clique, max_weight = max(clique_weights, key=lambda x: x[1])
    else:
        best_clique, max_weight = set(), 0.0
    
    theoretical_objective = 1.0 / max_weight if max_weight > 0 else float('inf')
    
    if verbose:
        print(f"Maximum weight clique: {best_clique} with weight {max_weight}")
        print(f"Theoretical objective 1/ω = {theoretical_objective:.6f}")
    
    return {
        'max_clique': best_clique,
        'max_weight': max_weight,
        'theoretical_objective': theoretical_objective,
        'all_cliques': clique_weights
    }


def solve_maximum_weight_clique_dirac(
    graph: nx.Graph,
    weights: Dict[int, float],
    config: Optional[DiracConfig] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve maximum weight clique using Dirac-3 quantum annealing.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        config: Dirac configuration parameters
        verbose: Print debug information
        
    Returns:
        Dictionary with solution information
    """
    if not DIRAC_AVAILABLE:
        raise ImportError("Dirac oracle adapter is required for quantum annealing")
    
    # Default configuration
    if config is None:
        config = DiracConfig(
            num_samples=100,
            relaxation_schedule=2,
            solution_precision=None,  # Use highest precision
            sum_constraint=1,
            save_raw_data=False,
            job_timeout=300
        )
    
    if verbose:
        print(f"Dirac solver for graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Configuration: {config.to_dict()}")
    
    # Initialize Dirac adapter
    dirac_adapter = DiracAdapter(config, verbose=verbose, enable_refinement=False)
    
    if verbose:
        print(f"Using oracle: {dirac_adapter.name}")
        print(f"Oracle available: {dirac_adapter.is_available}")
    
    # Run Dirac optimization
    start_time = time.time()
    
    try:
        # Find maximal cliques using Dirac oracle
        # Note: We pass the graph directly (not complement) since we're solving max weight clique
        maximal_cliques = dirac_adapter.find_maximal_cliques(
            graph, 
            support_threshold=1e-5, 
            weights=weights
        )
        
        runtime = time.time() - start_time
        
        # Find the best clique by weight
        best_clique = set()
        best_weight = 0.0
        
        for clique in maximal_cliques:
            clique_weight = sum(weights.get(v, 1.0) for v in clique)
            if clique_weight > best_weight:
                best_clique = clique
                best_weight = clique_weight
        
        # Get optimization details
        opt_details = dirac_adapter.get_optimization_details()
        
        # Calculate derived objective
        derived_objective = 1.0 / best_weight if best_weight > 0 else float('inf')
        
        # Verify it's actually a clique
        is_clique = True
        if best_clique:
            for i in best_clique:
                for j in best_clique:
                    if i != j and not graph.has_edge(i, j):
                        is_clique = False
                        break
                if not is_clique:
                    break
        
        if verbose:
            print(f"Dirac found {len(maximal_cliques)} maximal cliques")
            print(f"Best clique: {best_clique}")
            print(f"Best clique weight: {best_weight:.6f}")
            print(f"Derived objective: {derived_objective:.8f}")
            print(f"Is valid clique: {is_clique}")
            print(f"Runtime: {runtime:.4f} seconds")
            print(f"Theoretical ω(w,G) = {best_weight:.6f}")
        
        return {
            'clique_nodes': best_clique,
            'clique_weight': best_weight,
            'derived_objective': derived_objective,
            'is_valid_clique': is_clique,
            'runtime': runtime,
            'all_cliques': maximal_cliques,
            'num_cliques_found': len(maximal_cliques),
            'optimization_details': opt_details
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        if verbose:
            print(f"Dirac solver failed: {e}")
            print(f"Runtime: {runtime:.4f} seconds")
        
        return {
            'clique_nodes': set(),
            'clique_weight': 0.0,
            'derived_objective': float('inf'),
            'is_valid_clique': False,
            'runtime': runtime,
            'all_cliques': [],
            'num_cliques_found': 0,
            'error': str(e)
        }


def solve_maximum_weight_clique_milp(
    graph: nx.Graph,
    weights: Dict[int, float],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve maximum weight clique using SciPy MILP solver as ground truth.
    
    Formulation:
    maximize: sum(w[i] * x[i]) for all vertices i
    subject to: x[i] + x[j] <= 1 for all non-adjacent vertices (i,j)
                x[i] ∈ {0,1} for all vertices i
    
    Args:
        graph: Input graph
        weights: Vertex weights
        verbose: Print debug information
        
    Returns:
        Dictionary with solution information
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for MILP solving")
    
    node_list = list(graph.nodes())
    n = len(node_list)
    
    if verbose:
        print(f"Setting up MILP for graph with {n} vertices, {graph.number_of_edges()} edges")
    
    # Objective: maximize sum(w[i] * x[i])
    c = np.array([-weights.get(node, 1.0) for node in node_list])  # Negative for maximization
    
    # Constraints: x[i] + x[j] <= 1 for all non-adjacent vertices (i,j)
    constraints = []
    
    # Add clique constraints
    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if i < j and not graph.has_edge(node_i, node_j):
                # Non-adjacent vertices cannot both be selected
                A_row = np.zeros(n)
                A_row[i] = 1
                A_row[j] = 1
                constraints.append((A_row, 1.0))  # x[i] + x[j] <= 1
    
    if constraints:
        A_ub = np.vstack([row for row, _ in constraints])
        b_ub = np.array([bound for _, bound in constraints])
        constraint = LinearConstraint(A_ub, -np.inf, b_ub)
    else:
        # No constraints (complete graph case)
        constraint = None
    
    # Variable bounds: x[i] ∈ {0,1}
    bounds = Bounds(0, 1)
    
    # Integer constraints
    integrality = np.ones(n)  # All variables are integers
    
    if verbose:
        print(f"MILP has {len(constraints)} constraints")
    
    # Solve
    start_time = time.time()
    
    if constraint is not None:
        result = milp(c, integrality=integrality, bounds=bounds, constraints=constraint)
    else:
        result = milp(c, integrality=integrality, bounds=bounds)
    
    runtime = time.time() - start_time
    
    if not result.success:
        if verbose:
            print(f"MILP solver failed: {result.message}")
        return {
            'success': False,
            'message': result.message,
            'runtime': runtime
        }
    
    # Extract solution
    solution = result.x
    objective_value = -result.fun  # Convert back from minimization
    
    # Find selected vertices
    threshold = 0.5
    selected_indices = np.where(solution > threshold)[0]
    selected_vertices = {node_list[i] for i in selected_indices}
    
    # Verify it's a valid clique
    is_clique = True
    for i in selected_vertices:
        for j in selected_vertices:
            if i != j and not graph.has_edge(i, j):
                is_clique = False
                break
        if not is_clique:
            break
    
    clique_weight = sum(weights.get(v, 1.0) for v in selected_vertices)
    
    if verbose:
        print(f"MILP solution: {solution}")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Selected vertices: {selected_vertices}")
        print(f"Is valid clique: {is_clique}")
        print(f"Clique weight: {clique_weight:.6f}")
        print(f"Runtime: {runtime:.4f} seconds")
    
    return {
        'success': True,
        'solution': solution,
        'objective_value': objective_value,
        'selected_vertices': selected_vertices,
        'clique_weight': clique_weight,
        'is_valid_clique': is_clique,
        'runtime': runtime,
        'node_list': node_list
    }


def test_known_solutions():
    """Test the implementation on graphs with known maximum weight cliques."""
    print("=" * 60)
    print("TESTING MAXIMUM WEIGHT CLIQUE WITH KNOWN SOLUTIONS")
    print("=" * 60)
    print()
    
    # Test 1: Single vertex
    print("Test 1: Single vertex with weight 5")
    print("-" * 35)
    G1 = nx.Graph()
    G1.add_node(0)
    weights1 = {0: 5.0}
    
    theory1 = solve_maximum_weight_clique_theory(G1, weights1, verbose=True)
    
    if DIRAC_AVAILABLE:
        dirac1 = solve_maximum_weight_clique_dirac(G1, weights1, verbose=True)
        print(f"Theory vs Dirac: {theory1['max_weight']:.6f} vs {dirac1['clique_weight']:.6f}")
        print(f"Match: {abs(theory1['max_weight'] - dirac1['clique_weight']) < 0.01}")
    
    print()
    
    # Test 2: Complete triangle with weights [1, 2, 3]
    print("Test 2: Complete triangle K3 with weights [1, 2, 3]")
    print("-" * 50)
    G2 = nx.complete_graph(3)
    weights2 = {0: 1.0, 1: 2.0, 2: 3.0}
    
    theory2 = solve_maximum_weight_clique_theory(G2, weights2, verbose=True)
    
    if DIRAC_AVAILABLE:
        dirac2 = solve_maximum_weight_clique_dirac(G2, weights2, verbose=True)
        print(f"Theory vs Dirac: {theory2['max_weight']:.6f} vs {dirac2['clique_weight']:.6f}")
        print(f"Match: {abs(theory2['max_weight'] - dirac2['clique_weight']) < 0.1}")
    
    print()


def generate_random_graphs(num_graphs: int = 5) -> List[Tuple[str, nx.Graph, Dict[int, float]]]:
    """Generate random graphs for testing."""
    test_cases = []
    
    # Test case 1: Small Erdős–Rényi graphs
    for i, (n, p) in enumerate([(8, 0.3), (10, 0.4), (12, 0.5)]):
        G = nx.erdos_renyi_graph(n, p, seed=42 + i)
        # Random weights between 1 and 10
        np.random.seed(100 + i)
        weights = {node: np.random.uniform(1, 10) for node in G.nodes()}
        test_cases.append((f"Erdős-Rényi G({n},{p})", G, weights))
    
    # Test case 2: Scale-free graphs 
    for i, n in enumerate([8, 10]):
        G = nx.barabasi_albert_graph(n, 3, seed=50 + i)
        np.random.seed(200 + i)
        weights = {node: np.random.uniform(1, 5) for node in G.nodes()}
        test_cases.append((f"Barabási-Albert G({n},3)", G, weights))
    
    return test_cases


def test_dirac_vs_milp():
    """Test Dirac solver against MILP ground truth on various graphs."""
    print("=" * 70)
    print("TESTING DIRAC SOLVER VS MILP GROUND TRUTH")
    print("=" * 70)
    print()
    
    test_cases = generate_random_graphs()
    
    results = []
    plot_counter = 1
    
    for name, graph, weights in test_cases:
        print(f"Testing: {name}")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print(f"  Density: {nx.density(graph):.3f}")
        print()
        
        # MILP baseline
        milp_result = None
        if SCIPY_AVAILABLE:
            try:
                print("  MILP Solver:")
                milp_result = solve_maximum_weight_clique_milp(graph, weights, verbose=True)
                print()
            except Exception as e:
                print(f"  MILP failed: {e}")
                print()
        
        # Dirac solver
        dirac_result = None
        if DIRAC_AVAILABLE:
            try:
                print("  Dirac Solver:")
                dirac_result = solve_maximum_weight_clique_dirac(graph, weights, verbose=True)
                print()
            except Exception as e:
                print(f"  Dirac failed: {e}")
                print()
        
        # Determine which result to use for visualization
        best_result = None
        best_clique = set()
        
        if milp_result and milp_result['success']:
            best_result = milp_result
            best_clique = milp_result['selected_vertices']
            solver_name = "MILP"
        elif dirac_result and 'error' not in dirac_result:
            best_result = dirac_result
            best_clique = dirac_result['clique_nodes']
            solver_name = "Dirac"
        
        # Validate the clique
        if best_clique:
            is_valid, validation_msg = validate_clique(graph, best_clique)
            print(f"  Clique validation: {validation_msg}")
            print()
        
        # Compare results if both available
        if (milp_result and milp_result['success'] and 
            dirac_result and 'error' not in dirac_result):
            
            milp_weight = milp_result['clique_weight']
            dirac_weight = dirac_result['clique_weight']
            
            print("  Comparison:")
            print(f"    MILP weight:      {milp_weight:.6f}")
            print(f"    Dirac weight:     {dirac_weight:.6f}")
            print(f"    Difference:       {abs(milp_weight - dirac_weight):.6f}")
            print(f"    Match (±0.1):     {abs(milp_weight - dirac_weight) < 0.1}")
            print(f"    MILP runtime:     {milp_result['runtime']:.4f}s")
            print(f"    Dirac runtime:    {dirac_result['runtime']:.4f}s")
            
            # Check if cliques are valid
            print(f"    MILP valid:       {milp_result['is_valid_clique']}")
            print(f"    Dirac valid:      {dirac_result['is_valid_clique']}")
            print(f"    Dirac # cliques:  {dirac_result['num_cliques_found']}")
            
            # Create dual comparison plot
            if MATPLOTLIB_AVAILABLE:
                safe_name = name.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_")
                plot_filename = f"{plot_counter:02d}_{safe_name}_dirac_vs_milp.png"
                plot_path = os.path.join("plots", plot_filename)
                
                plot_title = f"{name} - Dirac vs MILP Comparison"
                success = plot_graph_with_dual_solutions(
                    graph, weights, 
                    milp_result['selected_vertices'], 
                    dirac_result['clique_nodes'],
                    plot_title, plot_path
                )
                
                if success:
                    print(f"    Dual plot saved:  {plot_filename}")
                else:
                    print(f"    Dual plot failed: {name}")
            
            results.append({
                'name': name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'milp_weight': milp_weight,
                'dirac_weight': dirac_weight,
                'match': abs(milp_weight - dirac_weight) < 0.1,
                'milp_time': milp_result['runtime'],
                'dirac_time': dirac_result['runtime'],
                'dirac_cliques': dirac_result['num_cliques_found']
            })
        
        plot_counter += 1
        print("-" * 50)
        print()
    
    # Summary
    if results:
        print("SUMMARY:")
        print("=" * 70)
        matches = sum(1 for r in results if r['match'])
        print(f"Test cases: {len(results)}")
        print(f"Matches: {matches}/{len(results)} ({100*matches/len(results):.1f}%)")
        print()
        
        print("Detailed Results:")
        print(f"{'Graph':<20} {'Nodes':<5} {'MILP':<8} {'Dirac':<8} {'Match':<5} {'MILP(s)':<8} {'Dirac(s)':<8} {'#Cliq':<6}")
        print("-" * 80)
        for r in results:
            match_str = "✓" if r['match'] else "✗"
            print(f"{r['name']:<20} {r['nodes']:<5} {r['milp_weight']:<8.3f} {r['dirac_weight']:<8.3f} "
                  f"{match_str:<5} {r['milp_time']:<8.4f} {r['dirac_time']:<8.4f} {r['dirac_cliques']:<6}")
    
    print(f"\nDirac vs MILP comparison plots saved to /plots directory")


def demo_dirac_solver():
    """Demonstrate the Dirac solver on example graphs."""
    print("=" * 70)
    print("DEMONSTRATION: DIRAC SOLVER ON EXAMPLE GRAPHS")
    print("=" * 70)
    print()
    
    if not DIRAC_AVAILABLE:
        print("Dirac solver not available - cannot run demonstration")
        return
    
    examples = []
    
    # Example 1: Triangle (complete graph)
    print("Example 1: Triangle K3 with weights [1, 2, 3]")
    G1 = nx.complete_graph(3)
    weights1 = {0: 1.0, 1: 2.0, 2: 3.0}
    examples.append(("Triangle K3", G1, weights1))
    
    # Example 2: Path graph
    print("Example 2: Path P4 with weights [2, 1, 3, 4]")
    G2 = nx.path_graph(4)
    weights2 = {0: 2.0, 1: 1.0, 2: 3.0, 3: 4.0}
    examples.append(("Path P4", G2, weights2))
    
    # Example 3: Diamond graph (K4 minus one edge)
    print("Example 3: Diamond graph with weights [2, 3, 1, 4]")
    G3 = nx.complete_graph(4)
    G3.remove_edge(0, 2)  # Remove edge to make diamond
    weights3 = {0: 2.0, 1: 3.0, 2: 1.0, 3: 4.0}
    examples.append(("Diamond graph", G3, weights3))
    
    print()
    print("Testing Dirac solver on each example...")
    print()
    
    for i, (name, graph, weights) in enumerate(examples, 1):
        print(f"Processing {name}...")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        # Solve using Dirac
        try:
            config = DiracConfig(num_samples=50, relaxation_schedule=2, job_timeout=300)
            result = solve_maximum_weight_clique_dirac(graph, weights, config=config, verbose=True)
            
            if 'error' not in result:
                clique = result['clique_nodes']
                clique_weight = result['clique_weight']
                
                # Validate clique
                is_valid, msg = validate_clique(graph, clique)
                print(f"  Solution: {clique} (weight: {clique_weight:.3f})")
                print(f"  Validation: {msg}")
                print(f"  Found {result['num_cliques_found']} total maximal cliques")
                
                # Create plot
                if MATPLOTLIB_AVAILABLE:
                    plot_filename = f"dirac_demo_{i:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}.png"
                    plot_path = os.path.join("plots", plot_filename)
                    plot_title = f"{name} - Maximum Weight Clique (Dirac)"
                    
                    success = plot_graph_with_clique(graph, weights, clique, plot_title, plot_path)
                    if success:
                        print(f"  Plot saved: {plot_filename}")
                    else:
                        print(f"  Plot failed")
            else:
                print(f"  Dirac solver failed: {result['error']}")
                
        except Exception as e:
            print(f"  Error solving {name}: {e}")
            
        print()
    
    print("All Dirac demonstration plots saved to /plots directory")
    print()


if __name__ == "__main__":
    print("Maximum Weight Clique Solver with Dirac-3 Quantum Annealing")
    print("Using Gibbons' weighted Motzkin-Straus theorem and QCI's Dirac-3 solver")
    print()
    
    # Check dependencies
    print("Dependency Check:")
    print(f"  Dirac oracle available: {DIRAC_AVAILABLE}")
    print(f"  MILP solver available:  {SCIPY_AVAILABLE}")
    print(f"  Matplotlib available:   {MATPLOTLIB_AVAILABLE}")
    print()
    
    if not DIRAC_AVAILABLE:
        print("WARNING: Dirac oracle not available - some tests will be skipped")
        print()
    
    # Run demonstrations
    if DIRAC_AVAILABLE:
        demo_dirac_solver()
    
    # Run basic theoretical validation
    test_known_solutions()
    
    # Test Dirac vs MILP comparison
    if DIRAC_AVAILABLE and SCIPY_AVAILABLE:
        print("Running Dirac vs MILP comparison tests...")
        test_dirac_vs_milp()
    else:
        print("Skipping Dirac vs MILP tests (missing dependencies)")
    
    print("\n" + "=" * 60)
    print("ALL DIRAC SOLVER TESTS AND DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\nCheck the /plots directory for all generated visualizations!")