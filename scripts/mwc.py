#!/usr/bin/env python3
"""
Maximum Weight Clique solver using Gibbons' weighted Motzkin-Straus theorem.

This implementation follows Gibbons' Theorem 5 exactly:
- For weighted clique number ω(w,G), we have 1/ω(w,G) = min{x^T B x | e^T x = 1, x ≥ 0}
- Matrix B construction: B[i,i] = 1/w[i], B[i,j] = 0 for adjacent vertices
"""

import numpy as np
import networkx as nx
import sys
import os
from typing import Dict, List, Set, Tuple, Any, Optional
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from motzkinstraus.jax_optimizers import (
        matrix_to_polynomial,
        run_multi_restart_optimization,
        JAXOptimizerConfig
    )
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print("JAX optimization available")
except ImportError as e:
    print(f"JAX optimization not available: {e}")
    JAX_AVAILABLE = False

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


def solve_maximum_weight_clique_jax(
    graph: nx.Graph,
    weights: Dict[int, float],
    config: Optional[JAXOptimizerConfig] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve maximum weight clique using JAX optimization.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        config: JAX optimization configuration
        verbose: Print debug information
        
    Returns:
        Dictionary with solution information
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for optimization")
    
    # Construct Gibbons matrix
    B, node_list = construct_gibbons_matrix(graph, weights)
    n = len(node_list)
    
    if verbose:
        print(f"Constructed Gibbons matrix B ({n}x{n}):")
        print(B)
        print()
    
    # Negate for maximization (we want to minimize x^T B x)
    B_opt = -B
    
    # Convert to polynomial format
    poly_indices, poly_coeffs = matrix_to_polynomial(B_opt, scale_factor=1.0)
    
    # Default configuration
    if config is None:
        config = JAXOptimizerConfig(
            learning_rate=0.1,
            max_iterations=2000,
            num_restarts=20,
            tolerance=1e-8,
            verbose=verbose
        )
    
    if verbose:
        print(f"Running JAX optimization with {config.num_restarts} restarts...")
    
    # Run optimization
    start_time = time.time()
    best_x, best_energy, all_histories, all_energies = run_multi_restart_optimization(
        poly_indices, poly_coeffs, n, config, algorithm="pgd"
    )
    runtime = time.time() - start_time
    
    # Convert results
    objective_value = -best_energy  # Convert back from maximization
    derived_omega = 1.0 / objective_value if objective_value > 0 else float('inf')
    
    # Extract clique from solution
    support_threshold = 1e-5
    support_indices = np.where(best_x > support_threshold)[0]
    clique_nodes = {node_list[i] for i in support_indices}
    
    # Verify it's actually a clique
    is_clique = True
    for i in clique_nodes:
        for j in clique_nodes:
            if i != j and not graph.has_edge(i, j):
                is_clique = False
                break
        if not is_clique:
            break
    
    clique_weight = sum(weights.get(v, 1.0) for v in clique_nodes)
    
    if verbose:
        print(f"JAX solution: x = {best_x}")
        print(f"Objective value: {objective_value:.8f}")
        print(f"Derived ω(w,G) = {derived_omega:.6f}")
        print(f"Support nodes: {clique_nodes}")
        print(f"Is valid clique: {is_clique}")
        print(f"Clique weight: {clique_weight:.6f}")
        print(f"Runtime: {runtime:.4f} seconds")
    
    return {
        'solution_vector': best_x,
        'objective_value': objective_value,
        'derived_omega': derived_omega,
        'clique_nodes': clique_nodes,
        'clique_weight': clique_weight,
        'is_valid_clique': is_clique,
        'runtime': runtime,
        'node_list': node_list,
        'matrix_B': B
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
    
    if JAX_AVAILABLE:
        jax1 = solve_maximum_weight_clique_jax(G1, weights1, verbose=True)
        print(f"Theory vs JAX: {theory1['max_weight']:.6f} vs {jax1['derived_omega']:.6f}")
        print(f"Match: {abs(theory1['max_weight'] - jax1['derived_omega']) < 0.01}")
    
    print()
    
    # Test 2: Complete triangle with weights [1, 2, 3]
    print("Test 2: Complete triangle K3 with weights [1, 2, 3]")
    print("-" * 50)
    G2 = nx.complete_graph(3)
    weights2 = {0: 1.0, 1: 2.0, 2: 3.0}
    
    theory2 = solve_maximum_weight_clique_theory(G2, weights2, verbose=True)
    
    if JAX_AVAILABLE:
        jax2 = solve_maximum_weight_clique_jax(G2, weights2, verbose=True)
        print(f"Theory vs JAX: {theory2['max_weight']:.6f} vs {jax2['derived_omega']:.6f}")
        print(f"Match: {abs(theory2['max_weight'] - jax2['derived_omega']) < 0.1}")
    
    print()
    
    # Test 3: Path graph P3 with weights [2, 1, 3]
    print("Test 3: Path P3 with weights [2, 1, 3]")
    print("-" * 40)
    G3 = nx.path_graph(3)
    weights3 = {0: 2.0, 1: 1.0, 2: 3.0}
    
    theory3 = solve_maximum_weight_clique_theory(G3, weights3, verbose=True)
    
    if JAX_AVAILABLE:
        jax3 = solve_maximum_weight_clique_jax(G3, weights3, verbose=True)
        print(f"Theory vs JAX: {theory3['max_weight']:.6f} vs {jax3['derived_omega']:.6f}")
        print(f"Match: {abs(theory3['max_weight'] - jax3['derived_omega']) < 0.1}")
    
    print()
    
    # Test 4: Two isolated vertices
    print("Test 4: Two isolated vertices with weights [3, 7]")
    print("-" * 45)
    G4 = nx.Graph()
    G4.add_nodes_from([0, 1])  # No edges
    weights4 = {0: 3.0, 1: 7.0}
    
    theory4 = solve_maximum_weight_clique_theory(G4, weights4, verbose=True)
    
    if JAX_AVAILABLE:
        jax4 = solve_maximum_weight_clique_jax(G4, weights4, verbose=True)
        print(f"Theory vs JAX: {theory4['max_weight']:.6f} vs {jax4['derived_omega']:.6f}")
        print(f"Match: {abs(theory4['max_weight'] - jax4['derived_omega']) < 0.1}")
    
    print()
    
    # Test 5: 4-clique with weights [1, 2, 3, 4]
    print("Test 5: Complete 4-clique K4 with weights [1, 2, 3, 4]")
    print("-" * 50)
    G5 = nx.complete_graph(4)
    weights5 = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    
    theory5 = solve_maximum_weight_clique_theory(G5, weights5, verbose=True)
    
    if JAX_AVAILABLE:
        jax5 = solve_maximum_weight_clique_jax(G5, weights5, verbose=True)
        print(f"Theory vs JAX: {theory5['max_weight']:.6f} vs {jax5['derived_omega']:.6f}")
        print(f"Match: {abs(theory5['max_weight'] - jax5['derived_omega']) < 0.1}")


def debug_matrix_construction():
    """Debug matrix construction for different graph types."""
    print("\n" + "=" * 60)
    print("DEBUGGING MATRIX CONSTRUCTION")
    print("=" * 60)
    
    # Test on triangle
    print("Triangle K3 matrix construction:")
    G = nx.complete_graph(3)
    weights = {0: 1.0, 1: 2.0, 2: 3.0}
    
    B, node_list = construct_gibbons_matrix(G, weights)
    print(f"Nodes: {node_list}")
    print(f"Weights: {weights}")
    print("Matrix B:")
    print(B)
    print(f"Expected: diagonal matrix with [1.0, 0.5, 0.333...]")
    print()
    
    # Verify theoretical calculation
    total_weight = sum(weights.values())
    x_theory = np.array([weights[i] / total_weight for i in range(3)])
    obj_theory = x_theory.T @ B @ x_theory
    
    print(f"Theoretical optimal solution: x = {x_theory}")
    print(f"Theoretical objective: x^T B x = {obj_theory:.6f}")
    print(f"Expected 1/ω = 1/{total_weight} = {1/total_weight:.6f}")
    print(f"Match: {abs(obj_theory - 1/total_weight) < 1e-6}")


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


def generate_random_graphs(num_graphs: int = 5) -> List[Tuple[str, nx.Graph, Dict[int, float]]]:
    """Generate random graphs for testing."""
    test_cases = []
    
    # Test case 1: Small Erdős–Rényi graphs
    for i, (n, p) in enumerate([(10, 0.3), (15, 0.4), (20, 0.5)]):
        G = nx.erdos_renyi_graph(n, p, seed=42 + i)
        # Random weights between 1 and 10
        np.random.seed(100 + i)
        weights = {node: np.random.uniform(1, 10) for node in G.nodes()}
        test_cases.append((f"Erdős-Rényi G({n},{p})", G, weights))
    
    # Test case 2: Scale-free graphs 
    for i, n in enumerate([12, 18]):
        G = nx.barabasi_albert_graph(n, 3, seed=50 + i)
        np.random.seed(200 + i)
        weights = {node: np.random.uniform(1, 5) for node in G.nodes()}
        test_cases.append((f"Barabási-Albert G({n},3)", G, weights))
    
    return test_cases


def test_random_graphs():
    """Test the implementation on random graphs and compare with MILP solver."""
    print("=" * 70)
    print("TESTING ON RANDOM GRAPHS")
    print("=" * 70)
    print()
    
    test_cases = generate_random_graphs()
    
    results = []
    
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
        
        # JAX optimization
        jax_result = None
        if JAX_AVAILABLE:
            try:
                print("  JAX-PGD Solver:")
                jax_result = solve_maximum_weight_clique_jax(graph, weights, verbose=True)
                print()
            except Exception as e:
                print(f"  JAX failed: {e}")
                print()
        
        # Compare results
        if milp_result and milp_result['success'] and jax_result:
            milp_weight = milp_result['clique_weight']
            jax_weight = jax_result['derived_omega']
            
            print("  Comparison:")
            print(f"    MILP weight:    {milp_weight:.6f}")
            print(f"    JAX-PGD ω:      {jax_weight:.6f}")
            print(f"    Difference:     {abs(milp_weight - jax_weight):.6f}")
            print(f"    Match (±0.1):   {abs(milp_weight - jax_weight) < 0.1}")
            print(f"    MILP runtime:   {milp_result['runtime']:.4f}s")
            print(f"    JAX runtime:    {jax_result['runtime']:.4f}s")
            
            # Check if cliques are valid
            print(f"    MILP valid:     {milp_result['is_valid_clique']}")
            print(f"    JAX valid:      {jax_result['is_valid_clique']}")
            
            results.append({
                'name': name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'milp_weight': milp_weight,
                'jax_weight': jax_weight,
                'match': abs(milp_weight - jax_weight) < 0.1,
                'milp_time': milp_result['runtime'],
                'jax_time': jax_result['runtime']
            })
        
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
        print(f"{'Graph':<20} {'Nodes':<5} {'MILP':<8} {'JAX':<8} {'Match':<5} {'MILP(s)':<8} {'JAX(s)':<8}")
        print("-" * 70)
        for r in results:
            match_str = "✓" if r['match'] else "✗"
            print(f"{r['name']:<20} {r['nodes']:<5} {r['milp_weight']:<8.3f} {r['jax_weight']:<8.3f} "
                  f"{match_str:<5} {r['milp_time']:<8.4f} {r['jax_time']:<8.4f}")


if __name__ == "__main__":
    print("Maximum Weight Clique Solver")
    print("Using Gibbons' weighted Motzkin-Straus theorem")
    print()
    
    # Run basic tests
    test_known_solutions()
    debug_matrix_construction()
    
    print("\n")
    
    # Test on random graphs
    test_random_graphs()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)