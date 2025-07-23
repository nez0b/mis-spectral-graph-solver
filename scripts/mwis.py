#!/usr/bin/env python3
"""
Maximum Weight Independent Set (MWIS) Solver using Gibbons' Weighted Motzkin-Straus Theorem

Theory: 
- Uses Gibbons' weighted generalization of the Motzkin-Straus theorem
- For MWIS in graph G, we solve maximum weight clique on complement graph Ḡ
- The theorem states: 1/ω(w,Ḡ) = min{x^T B x | e^T x = 1, x ≥ 0}
- Matrix B construction follows Gibbons' Theorem 5:
  * B[i,i] = 1/w[i] for all vertices i (diagonal entries)
  * B[i,j] = 0 for adjacent vertices in Ḡ (adjacent means non-adjacent in G)  
  * B[i,j] = (1/w[i] + 1/w[j])/2 for non-adjacent vertices in Ḡ
- The support of the optimal solution gives the maximum weight independent set

Implementation:
- Correct Gibbons matrix construction on complement graph
- JAX-PGD optimization with multiple restarts for finding global minimum
- MILP solver comparison for ground truth validation
- Dual plotting to visualize both solver results
- Extract independent set from solution support
- Verify solutions are valid independent sets

Solvers:
- JAX-PGD: Projected Gradient Descent on Gibbons matrix (complement graph)
- MILP: Mixed-Integer Linear Programming for exact solution

Usage Examples:

    # Test with dual comparison (MILP vs JAX-PGD) and visualization
    python scripts/mwis.py --test-dual --plot
    
    # Test with JAX-PGD oracle on predefined graphs
    python scripts/mwis.py --test --oracle jax-pgd --verbose
    
    # Test on random graph with MILP comparison
    python scripts/mwis.py --random-test --compare-milp --nodes 15
    
    # JAX-PGD with custom parameters
    python scripts/mwis.py --test --oracle jax-pgd --num-restarts 100
    
    # Compare with NetworkX algorithms
    python scripts/mwis.py --test --compare-networkx --plot

Parameters:
    --test-dual         Test with dual comparison (MILP vs JAX-PGD) and visualization
    --test              Run tests on predefined graphs
    --random-test       Test on random graph with generated weights
    --oracle TYPE       Oracle solver: 'jax-pgd' or 'dirac' (default: jax-pgd)
    --compare-oracles   Compare multiple oracles on same graphs
    --compare-milp      Compare with MILP solver for ground truth
    --threshold T       Support extraction threshold (default: 1e-5)
    --verbose          Enable detailed progress output
    --plot             Generate visualization plots
    --save-plots DIR   Directory to save plots (default: ./plots/)
    --compare-networkx Compare results with NetworkX algorithms
    --nodes N          Number of nodes for random graphs (default: 10)
    --edge-prob P      Edge probability for random graphs (default: 0.5)

JAX-PGD Oracle Parameters:
    --num-restarts N    Number of optimization restarts (default: 50)
    --learning-rate R   Learning rate for gradient descent (default: 0.01)
    --max-iterations N  Maximum iterations per restart (default: 2000)

Dirac-3 Oracle Parameters:
    --num-samples N     Number of Dirac samples (default: 100)
    --relax-schedule N  Relaxation schedule 1-4 (default: 2)
    --solution-precision P  Solution precision (optional)

Superposition Refinement Options:
    --enable-refinement   Enable local search refinement of superposition solutions (default)
    --disable-refinement  Disable refinement to improve performance

Performance Considerations:
- JAX-PGD: Larger graphs require more restarts, local optimization approach
- Dirac-3: Quantum annealing, good for finding global optima, requires QCI account
- Threshold parameter affects precision vs recall trade-off for both oracles
- Sparse graphs (fewer edges) are generally easier for stable set problems than dense graphs
- Superposition refinement: Improves stable set discovery but increases computational cost
- Oracle comparison mode: Provides enhanced visualization with color-coded results
"""

import sys
import os
import networkx as nx
import numpy as np
import time
from typing import Set, List, Tuple, Dict, Any, Optional
import argparse
import random
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from motzkinstraus.algorithms import verify_clique  # Not used directly but needed for oracle system
    print("Successfully imported motzkinstraus modules")
except ImportError as e:
    print(f"Error importing motzkinstraus modules: {e}")
    print("Make sure you're in the correct directory and virtual environment is activated")
    sys.exit(1)

# Import oracle system (optional - for JAX-PGD oracle)
try:
    from oracles import OracleFactory
    ORACLE_FACTORY_AVAILABLE = True
    print("Successfully imported oracle system")
except ImportError as e:
    print(f"Oracle system not available: {e}")
    print("Only Dirac oracle will be available")
    OracleFactory = None
    ORACLE_FACTORY_AVAILABLE = False

# Import corrected Dirac oracle adapter directly
try:
    from oracles.dirac_adapter import DiracAdapter, DiracConfig
    DIRAC_AVAILABLE = True
    print("Successfully imported corrected Dirac oracle adapter")
except ImportError as e:
    print(f"Corrected Dirac oracle adapter not available: {e}")
    DIRAC_AVAILABLE = False
    DiracAdapter = None
    DiracConfig = None

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available for plotting")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - plotting disabled")


def extract_support(solution_vector: np.ndarray, threshold: float = 1e-5) -> Set[int]:
    """
    Extract support (non-zero indices) from solution vector.
    
    The support corresponds to vertices that are candidates for being in a stable set
    (via cliques in the complement graph) based on the Motzkin-Straus theorem.
    
    Args:
        solution_vector: Solution vector from the optimization (x values)
        threshold: Minimum value to consider as "non-zero" (default: 1e-5)
        
    Returns:
        Set of vertex indices where x_i > threshold
    """
    if len(solution_vector) == 0:
        return set()
    
    # Find indices where solution values exceed threshold
    support_indices = np.where(solution_vector > threshold)[0]
    return set(support_indices.tolist())


def verify_maximal_stable_set(graph: nx.Graph, candidate_set: Set[int]) -> bool:
    """
    Verify that a set of vertices is a maximal stable set.

    A stable set is a set of vertices where no two vertices are adjacent.
    A maximal stable set cannot be extended by adding any other vertex from the graph.

    Args:
        graph: The input graph
        candidate_set: Set of vertices to check for maximality

    Returns:
        True if the set is a maximal stable set, False otherwise
    """
    if not candidate_set:
        return False

    # Verify it's a stable set (no edges between vertices in the set)
    for u in candidate_set:
        for v in candidate_set:
            if u != v and graph.has_edge(u, v):
                return False

    # Check for maximality: can any vertex be added?
    for vertex in graph.nodes():
        if vertex not in candidate_set:
            # Check if the new vertex is not connected to any vertex in the stable set
            if all(not graph.has_edge(vertex, member) for member in candidate_set):
                # This vertex can be added, so the set is not maximal
                return False

    return True


def plot_graph_with_dual_mwis_solutions(
    graph: nx.Graph, 
    weights: Dict[int, float], 
    milp_set: Set[int],
    jax_set: Set[int],
    title: str,
    save_path: str = None
) -> bool:
    """
    Plot both MILP and JAX-PGD MWIS solutions with different colors.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        milp_set: Vertices in MILP solution
        jax_set: Vertices in JAX-PGD solution
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
    
    # Categorize vertices
    milp_only = milp_set - jax_set
    jax_only = jax_set - milp_set
    common_vertices = milp_set & jax_set
    other_vertices = set(graph.nodes()) - milp_set - jax_set
    
    # Draw vertices
    if other_vertices:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(other_vertices),
                               node_color='lightgray', node_size=300, alpha=0.6)
    if milp_only:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(milp_only),
                               node_color='red', node_size=500, alpha=0.9, 
                               edgecolors='darkred', linewidths=2)
    if jax_only:
        nx.draw_networkx_nodes(graph, pos, nodelist=list(jax_only),
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
    milp_weight = sum(weights.get(v, 1.0) for v in milp_set)
    jax_weight = sum(weights.get(v, 1.0) for v in jax_set)
    
    # Title
    solutions_match = milp_set == jax_set
    match_text = "MATCH" if solutions_match else "DIFFER"
    
    plt.title(f"{title}\\n" + 
              f"MILP: {milp_set} (weight: {milp_weight:.3f})\\n" + 
              f"JAX-PGD: {jax_set} (weight: {jax_weight:.3f}) - {match_text}", 
              fontsize=11, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                   markersize=12, markeredgecolor='darkmagenta', markeredgewidth=2,
                   label='Both MILP & JAX-PGD'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='darkred', markeredgewidth=2,
                   label='MILP only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, markeredgecolor='darkblue', markeredgewidth=2,
                   label='JAX-PGD only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=8, label='Other vertices')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual MWIS solution plot saved to {save_path}")
    
    plt.close()
    return True


from motzkinstraus.jax_optimizers import adjacency_to_polynomial, matrix_to_polynomial

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

def find_maximum_weight_independent_set(
    graph: nx.Graph,
    weights: Dict[int, float],
    oracle_type: str = 'jax-pgd',
    support_threshold: float = 1e-5,
    verbose: bool = False,
    enable_refinement: bool = True,
    **oracle_config
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Find the maximum weight independent set using the generalized Motzkin-Straus theorem.

    This function applies the weighted Motzkin-Straus formulation to the complement
    of the graph to find the maximum weight independent set.

    Args:
        graph: Input NetworkX graph
        weights: A dictionary mapping node indices to their weights
        oracle_type: Type of oracle solver ('jax-pgd', 'dirac')
        support_threshold: Threshold for extracting support from solution vectors
        verbose: Whether to print detailed progress
        enable_refinement: Whether to enable superposition refinement (default: True)
        **oracle_config: Oracle-specific configuration parameters

    Returns:
        Tuple of (maximal_stable_sets, optimization_details) where:
        - maximal_stable_sets: List of sets, each containing vertices of a maximal stable set
        - optimization_details: Dictionary with oracle-specific optimization information
    """
    if graph.number_of_nodes() == 0:
        return [], {"message": "Empty graph"}

    # Find stable sets by finding cliques in the complement graph
    complement_graph = nx.complement(graph)

    # Apply Gibbons' weighted Motzkin-Straus theorem to the complement graph
    # For MWIS in graph G, we solve maximum weight clique on complement of G
    # The theorem states: 1/ω(w,G) = min{x^T B x | e^T x = 1, x ≥ 0}

    # Construct the correct Gibbons matrix B on the complement graph
    B, node_list = construct_gibbons_matrix(complement_graph, weights)

    # Handle coefficient negation based on oracle type:
    # - Dirac oracle: Minimizes directly (Gibbons is already minimization) - NO NEGATION
    # - JAX-PGD oracle: Maximizes, so needs negation to handle minimization problem
    if oracle_type == 'dirac':
        # Dirac handles Gibbons minimization directly - no negation needed
        if verbose:
            print("Using Dirac oracle: Gibbons minimization (no coefficient negation)")
    else:
        # JAX-PGD and other oracles expect maximization - negate for minimization
        B = -B
        if verbose:
            print("Using maximization oracle: negating coefficients for Gibbons minimization")

    # Convert to polynomial format - use matrix_to_polynomial for weighted matrices
    poly_indices, poly_coefficients = matrix_to_polynomial(B, scale_factor=1.0)

    # Use corrected Dirac oracle directly for 'dirac' type, otherwise use oracle system
    if oracle_type == 'dirac' and DIRAC_AVAILABLE:
        if verbose:
            print("Using corrected Dirac oracle adapter directly")
        
        try:
            # Create DiracConfig from oracle_config
            dirac_config = DiracConfig(
                num_samples=oracle_config.get('num_samples', 100),
                relaxation_schedule=oracle_config.get('relaxation_schedule', 2),
                solution_precision=oracle_config.get('solution_precision', None),
                sum_constraint=oracle_config.get('sum_constraint', 1),
                mean_photon_number=oracle_config.get('mean_photon_number', None),
                quantum_fluctuation_coefficient=oracle_config.get('quantum_fluctuation_coefficient', None),
                save_raw_data=oracle_config.get('save_raw_data', False),
                job_timeout=oracle_config.get('job_timeout', 300)
            )
            
            # Initialize corrected DiracAdapter
            dirac_adapter = DiracAdapter(dirac_config, verbose=verbose, enable_refinement=enable_refinement)
            
            if verbose:
                print(f"Using {dirac_adapter.name} oracle")
                print(f"Finding maximum weight independent set in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                print(f"Complement graph has {complement_graph.number_of_nodes()} nodes, {complement_graph.number_of_edges()} edges")

            # Find maximal cliques in the complement graph (which are stable sets in the original)
            # Pass weights to the oracle for proper Gibbons matrix construction
            maximal_stable_sets = dirac_adapter.find_maximal_cliques(complement_graph, support_threshold, weights)

            # Get optimization details
            optimization_details = dirac_adapter.get_optimization_details()

            if verbose:
                print(f"Found {len(maximal_stable_sets)} unique stable sets from complement graph cliques")

            return maximal_stable_sets, optimization_details
            
        except Exception as e:
            if verbose:
                print(f"Corrected Dirac oracle failed: {e}")
                print("Falling back to legacy JAX-PGD implementation")
            return _legacy_jax_pgd_implementation_stable_set(complement_graph, support_threshold, verbose, poly_coefficients=poly_coefficients, **oracle_config)
    
    # Handle request for Dirac oracle when not available
    elif oracle_type == 'dirac' and not DIRAC_AVAILABLE:
        if verbose:
            print("Dirac oracle requested but not available - falling back to legacy JAX-PGD implementation")
        return _legacy_jax_pgd_implementation_stable_set(complement_graph, support_threshold, verbose, poly_coefficients=poly_coefficients, **oracle_config)
    
    # Use oracle system for other oracle types, or fallback if DiracAdapter not available
    elif ORACLE_FACTORY_AVAILABLE and OracleFactory is not None:
        try:
            oracle = OracleFactory.create_oracle(oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config)

            if verbose:
                print(f"Using {oracle.name} oracle")
                print(f"Finding maximum weight independent set in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Find maximal cliques in the complement graph (which are stable sets in the original)
            # Pass weights to the oracle for proper Gibbons matrix construction
            maximal_stable_sets = oracle.find_maximal_cliques(complement_graph, support_threshold, weights)

            # Get optimization details
            optimization_details = oracle.get_optimization_details()

            if verbose:
                print(f"Found {len(maximal_stable_sets)} unique maximal stable sets")

            return maximal_stable_sets, optimization_details

        except Exception as e:
            if verbose:
                print(f"Oracle {oracle_type} failed: {e}")
                print("Falling back to legacy JAX-PGD implementation")
            return _legacy_jax_pgd_implementation_stable_set(complement_graph, support_threshold, verbose, poly_coefficients=poly_coefficients, **oracle_config)
    
    else:
        # No oracle system available and not using corrected Dirac - use legacy implementation
        if verbose:
            print("Oracle system not available - using legacy JAX-PGD implementation")
        return _legacy_jax_pgd_implementation_stable_set(complement_graph, support_threshold, verbose, poly_coefficients=poly_coefficients, **oracle_config)


# Alias for compatibility with test scripts
def find_maximal_stable_sets_motzkin_straus(
    graph: nx.Graph,
    weights: Dict[int, float],
    oracle_type: str = 'jax-pgd',
    support_threshold: float = 1e-5,
    verbose: bool = False,
    enable_refinement: bool = True,
    **oracle_config
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """Alias for find_maximum_weight_independent_set for test script compatibility."""
    return find_maximum_weight_independent_set(
        graph, weights, oracle_type, support_threshold, verbose, enable_refinement, **oracle_config
    )


def _legacy_jax_pgd_implementation_stable_set(
    graph: nx.Graph, 
    support_threshold: float = 1e-5, 
    verbose: bool = False,
    num_restarts: int = 50,
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    poly_coefficients: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Legacy JAX-PGD implementation for stable set finding using complement graph.
    """
    if verbose:
        print(f"Legacy JAX-PGD: Finding maximal stable sets with {num_restarts} restarts")
        print(f"Legacy JAX-PGD: Operating on complement graph for stable set problem")
    
    # Import required modules
    try:
        from motzkinstraus.jax_optimizers import (
            JAXOptimizerConfig, 
            adjacency_to_polynomial, 
            matrix_to_polynomial,
            run_projected_gradient_descent,
            sample_dirichlet
        )
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print(f"Error importing JAX modules: {e}")
        return [], {"error": str(e)}
    
    # Set up adjacency matrix and polynomial representation for complement
    node_list = list(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)
    
    # Convert to polynomial format
    if poly_coefficients is None:
        poly_indices, poly_coefficients = adjacency_to_polynomial(adj_matrix)
    else:
        poly_indices, _ = adjacency_to_polynomial(adj_matrix)

    
    if len(poly_indices) == 0:
        if verbose:
            print("No polynomial terms in complement - original graph is complete")
        # For a complete graph, only single nodes are stable sets
        return [{node} for node in graph.nodes()], {"message": "Complete graph, only individual nodes are stable sets"}
    
    # Create configuration
    config = JAXOptimizerConfig(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=False
    )
    
    # Generate initializations and run optimization
    key = jax.random.PRNGKey(42)
    alpha = jnp.ones(len(node_list)) * 1.0
    key, subkey = jax.random.split(key)
    initial_states = sample_dirichlet(subkey, alpha, sample_shape=(num_restarts,))
    
    all_final_solutions = []
    for i in range(num_restarts):
        try:
            final_x, _ = run_projected_gradient_descent(
                poly_indices=poly_indices,
                poly_coefficients=poly_coefficients,
                num_vars=len(node_list),
                config=config,
                x_init=initial_states[i],
                seed=42 + i
            )
            all_final_solutions.append(np.array(final_x))
        except Exception as e:
            if verbose:
                print(f"Warning: Restart {i+1} failed: {e}")
            continue
    
    # Extract stable sets
    maximal_stable_sets = set()
    for solution_vector in all_final_solutions:
        support_indices = extract_support(solution_vector, support_threshold)
        if not support_indices:
            continue
        
        candidate_set = {node_list[idx] for idx in support_indices if idx < len(node_list)}
        
        if verify_maximal_stable_set(graph, candidate_set):
            maximal_stable_sets.add(frozenset(candidate_set))
    
    result_sets = [set(s) for s in maximal_stable_sets]
    details = {
        "oracle_type": "legacy-jax-pgd",
        "num_restarts": num_restarts,
        "solutions_processed": len(all_final_solutions),
        "stable_sets_found": len(result_sets)
    }
    
    return result_sets, details


def plot_stable_set_instances(graph: nx.Graph, stable_sets: List[Set[int]], title: str, save_path: str = None) -> bool:
    """
    Plot graph with maximal stable sets highlighted in different colors.
    
    Args:
        graph: NetworkX graph to plot
        stable_sets: List of maximal stable sets (sets of vertices)
        title: Title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        True if plot was created successfully, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - cannot create plots")
        return False
    
    try:
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Generate layout for the graph
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # Define colors for different stable sets
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Draw the graph structure first
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax)
        
        # Draw all nodes in default color first
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7, ax=ax)
        
        # Highlight nodes in each stable set with different colors
        for i, stable_set in enumerate(stable_sets):
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(graph, pos, nodelist=list(stable_set),
                                  node_color=color, node_size=600, 
                                  alpha=0.8, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Create legend for stable sets
        legend_elements = []
        for i, stable_set in enumerate(stable_sets):
            color = colors[i % len(colors)]
            stable_set_str = str(sorted(stable_set))
            legend_elements.append(patches.Patch(color=color, 
                                                label=f'Stable Set {i+1}: {stable_set_str}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set title and formatting
        ax.set_title(f'{title}\nNodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, Stable Sets: {len(stable_sets)}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return False


def plot_oracle_comparison_stable_set_results(
    graph: nx.Graph, 
    oracle_results: Dict[str, Tuple[List[Set[int]], Dict[str, Any], float]], 
    title: str, 
    save_path: str = None
) -> bool:
    """
    Plot graph with maximal stable sets color-coded by oracle solver.
    
    This enhanced plotting function visualizes the results from multiple oracle 
    solvers, using distinct colors and line styles to differentiate between:
    - Oracle types: JAX-PGD (blue family) vs Dirac (red family)  
    - Individual stable sets: Different line styles (solid, dashed, dotted, dashdot)
    - Stable set prominence: Different line widths (2, 3, 4, 5 pixels)
    
    Args:
        graph: NetworkX graph to plot
        oracle_results: Dict mapping oracle names to (stable_sets, details, runtime) tuples
        title: Title for the plot
        save_path: Optional path to save the plot
        
    Returns:
        True if plot was created successfully, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - cannot create oracle comparison plots")
        return False
    
    try:
        # Create figure and axis with larger size for comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Generate layout for the graph
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # Oracle color and style configuration
        oracle_styles = {
            'jax-pgd': {
                'colors': ['blue', 'navy', 'cornflowerblue', 'royalblue'],
                'base_color': 'blue'
            },
            'dirac': {
                'colors': ['red', 'darkred', 'crimson', 'indianred'],
                'base_color': 'red'
            }
        }
        
        # Line styles for stable set differentiation
        linestyles = ['-', '--', ':', '-.']  # solid, dashed, dotted, dashdot
        linewidths = [2, 3, 4, 5]  # different thickness levels
        
        # Draw the base graph structure first
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax)
        
        # Draw all nodes in default color first  
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=600, alpha=0.7, ax=ax)
        
        # Track legend elements for organized display
        legend_elements = []
        oracle_stable_set_counts = {}
        
        # Process each oracle's results
        for oracle_idx, (oracle_name, (stable_sets, details, runtime)) in enumerate(oracle_results.items()):
            oracle_stable_set_counts[oracle_name] = len(stable_sets)
            
            # Get style configuration for this oracle
            if oracle_name.startswith('jax') or 'pgd' in oracle_name.lower():
                style_config = oracle_styles['jax-pgd']
            else:  # Assume dirac or other quantum oracle
                style_config = oracle_styles['dirac']
            
            # Draw stable_sets for this oracle
            for stable_set_idx, stable_set in enumerate(stable_sets):
                # Select color, line style, and width for unique visual signature
                color = style_config['colors'][stable_set_idx % len(style_config['colors'])]
                
                # Draw stable set nodes with oracle base color but varying alpha
                alpha = 0.9 - (stable_set_idx * 0.1) % 0.4  # Vary alpha between 0.5-0.9
                nx.draw_networkx_nodes(graph, pos, nodelist=list(stable_set),
                                      node_color=color, node_size=700, 
                                      alpha=alpha, ax=ax)
                
                # Add legend entry with oracle and stable set information
                stable_set_str = str(sorted(stable_set))
                if len(stable_set_str) > 30:  # Truncate long stable set strings
                    stable_set_str = stable_set_str[:27] + "..."
                    
                legend_elements.append(patches.Patch(
                    color=color, 
                    label=f'{oracle_name.upper()} S{stable_set_idx+1}: {stable_set_str} (size: {len(stable_set)})'
                ))
        
        # Draw node labels on top
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', 
                               font_color='white', ax=ax)
        
        # Create organized legend
        if legend_elements:
            # Sort legend by oracle name for better organization
            legend_elements.sort(key=lambda x: (x.get_label().split()[0], x.get_label()))
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                     fontsize=9, title="Oracle Stable Sets")
        
        # Create informative title and subtitle
        runtime_info = " | ".join([
            f"{oracle}: {runtime:.3f}s, {count} stable sets" 
            for oracle, (_, _, runtime), count in 
            [(k, v, oracle_stable_set_counts[k]) for k, v in oracle_results.items()]
        ])
        
        ax.set_title(f'{title}\nNodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}', 
                    fontsize=14, fontweight='bold')
        ax.text(0.5, -0.05, f'Performance: {runtime_info}', 
               transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Oracle comparison plot saved to: {save_path}")
        
        plt.show()
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error creating oracle comparison plot: {e}")
        return False


def solve_maximum_weight_independent_set_jax(
    graph: nx.Graph,
    weights: Dict[int, float],
    verbose: bool = False,
    num_restarts: int = 50,
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    debug_objective: bool = False
) -> Dict[str, Any]:
    """
    Solve Maximum Weight Independent Set using JAX-PGD on complement graph.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        verbose: Print debug information
        num_restarts: Number of JAX-PGD restarts
        learning_rate: Learning rate for optimization
        max_iterations: Maximum iterations per restart
        tolerance: Convergence tolerance
        
    Returns:
        Dictionary with solution information
    """
    try:
        from motzkinstraus.jax_optimizers import (
            matrix_to_polynomial,
            run_multi_restart_optimization,
            JAXOptimizerConfig
        )
        import jax.numpy as jnp
    except ImportError as e:
        return {
            'success': False,
            'message': f'JAX not available: {e}',
            'runtime': 0.0
        }
    
    # Apply Gibbons' theorem to complement graph for MWIS
    complement_graph = nx.complement(graph)
    
    # Construct Gibbons matrix
    B, node_list = construct_gibbons_matrix(complement_graph, weights)
    n = len(node_list)
    
    if verbose:
        print(f"JAX-PGD: Constructed Gibbons matrix B ({n}x{n}) on complement graph")
        print(f"JAX-PGD: Original graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"JAX-PGD: Complement graph has {complement_graph.number_of_nodes()} nodes, {complement_graph.number_of_edges()} edges")
    
    # Negate for maximization (we want to minimize x^T B x)
    B_opt = -B
    
    # Convert to polynomial format
    poly_indices, poly_coeffs = matrix_to_polynomial(B_opt, scale_factor=1.0)
    
    # Create configuration
    config = JAXOptimizerConfig(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        num_restarts=num_restarts,
        tolerance=tolerance,
        verbose=verbose
    )
    
    if verbose:
        print(f"JAX-PGD: Running optimization with {num_restarts} restarts...")
    
    # Run optimization
    start_time = time.time()
    best_x, best_energy, all_histories, all_energies = run_multi_restart_optimization(
        poly_indices, poly_coeffs, n, config, algorithm="pgd"
    )
    runtime = time.time() - start_time
    
    # Convert results
    objective_value = -best_energy  # Convert back from maximization
    derived_omega = 1.0 / objective_value if objective_value > 0 else float('inf')
    
    # DEBUG: Track the raw objective value for debugging purposes
    raw_motzkin_straus_value = objective_value
    
    # Extract independent set from solution (on complement graph this gives clique, on original graph this gives independent set)
    support_threshold = 1e-5
    support_indices = np.where(best_x > support_threshold)[0]
    independent_set_nodes = {node_list[i] for i in support_indices}
    
    # Verify it's actually an independent set in original graph
    is_valid = True
    for i in independent_set_nodes:
        for j in independent_set_nodes:
            if i != j and graph.has_edge(i, j):
                is_valid = False
                break
        if not is_valid:
            break
    
    set_weight = sum(weights.get(v, 1.0) for v in independent_set_nodes)
    
    if verbose:
        print(f"JAX-PGD: Solution vector: {best_x}")
        print(f"JAX-PGD: Objective value (raw): {objective_value:.8f}")
        print(f"JAX-PGD: Derived ω(w,complement) = {derived_omega:.6f}")
        print(f"JAX-PGD: Independent set: {independent_set_nodes}")
        print(f"JAX-PGD: Is valid independent set: {is_valid}")
        print(f"JAX-PGD: Set weight: {set_weight:.6f}")
        print(f"JAX-PGD: Runtime: {runtime:.4f} seconds")
    
    if debug_objective and verbose:
        print(f"DEBUG: Raw Motzkin-Straus objective value (without support): {raw_motzkin_straus_value:.8f}")
        print(f"DEBUG: Theoretical maximum weight = 1/objective = {1.0/raw_motzkin_straus_value:.8f}" if raw_motzkin_straus_value > 0 else "DEBUG: Invalid objective value")
    
    return {
        'success': True,
        'solution_vector': best_x,
        'objective_value': objective_value,
        'raw_motzkin_straus_value': raw_motzkin_straus_value,
        'derived_omega': derived_omega,
        'selected_vertices': independent_set_nodes,
        'set_weight': set_weight,
        'is_valid_independent_set': is_valid,
        'runtime': runtime,
        'node_list': node_list,
        'matrix_B': B
    }


def debug_motzkin_straus_objective_values(
    graph: nx.Graph,
    weights: Dict[int, float],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Debug function to compare optimal Motzkin-Straus objective values with MILP solutions.
    
    This function checks if the weighted Motzkin-Straus optimization objective value
    (without support extraction) correctly matches the theoretical value derived from
    the MILP solution.
    
    According to Gibbons' theorem:
    1/ω(w,G) = min{x^T B x | e^T x = 1, x ≥ 0}
    
    Where ω(w,G) is the maximum weight of an independent set.
    
    Args:
        graph: Input graph
        weights: Vertex weights
        verbose: Print detailed debugging information
        
    Returns:
        Dictionary with debugging information
    """
    if verbose:
        print("=" * 60)
        print("DEBUGGING MOTZKIN-STRAUS OBJECTIVE VALUES")
        print("=" * 60)
    
    # Get MILP solution
    milp_result = solve_maximum_weight_independent_set_milp(graph, weights, verbose=False)
    if not milp_result['success']:
        return {'error': 'MILP solver failed', 'milp_result': milp_result}
    
    milp_optimal_weight = milp_result['set_weight']
    
    # Get JAX-PGD solution with debug mode
    jax_result = solve_maximum_weight_independent_set_jax(
        graph, weights, verbose=False, debug_objective=True,
        num_restarts=50, max_iterations=2000
    )
    
    if not jax_result['success']:
        return {'error': 'JAX-PGD solver failed', 'jax_result': jax_result}
    
    jax_raw_objective = jax_result['raw_motzkin_straus_value']
    jax_theoretical_weight = 1.0 / jax_raw_objective if jax_raw_objective > 0 else float('inf')
    jax_extracted_weight = jax_result['set_weight']
    
    # According to Gibbons' theorem, 1/ω(w,G) = min{x^T B x}
    # So the theoretical optimal weight should be ω(w,G) = 1 / min_objective
    theoretical_from_milp = 1.0 / milp_optimal_weight if milp_optimal_weight > 0 else float('inf')
    
    # Calculate discrepancies
    objective_discrepancy = abs(jax_raw_objective - theoretical_from_milp)
    weight_discrepancy = abs(jax_theoretical_weight - milp_optimal_weight)
    extraction_discrepancy = abs(jax_extracted_weight - milp_optimal_weight)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Weights: {weights}")
        print()
        
        print("MILP SOLUTION (Ground Truth):")
        print(f"  Optimal independent set: {milp_result['selected_vertices']}")
        print(f"  Optimal weight: {milp_optimal_weight:.8f}")
        print(f"  Expected Motzkin-Straus objective: 1/{milp_optimal_weight:.8f} = {theoretical_from_milp:.8f}")
        print()
        
        print("JAX-PGD MOTZKIN-STRAUS OPTIMIZATION:")
        print(f"  Raw objective value (x^T B x): {jax_raw_objective:.8f}")
        print(f"  Theoretical weight from objective: 1/{jax_raw_objective:.8f} = {jax_theoretical_weight:.8f}")
        print(f"  Extracted independent set: {jax_result['selected_vertices']}")
        print(f"  Extracted set weight: {jax_extracted_weight:.8f}")
        print(f"  Valid independent set: {jax_result['is_valid_independent_set']}")
        print()
        
        print("COMPARISON & VALIDATION:")
        print(f"  Objective discrepancy: |{jax_raw_objective:.8f} - {theoretical_from_milp:.8f}| = {objective_discrepancy:.8f}")
        print(f"  Weight discrepancy (theory): |{jax_theoretical_weight:.8f} - {milp_optimal_weight:.8f}| = {weight_discrepancy:.8f}")
        print(f"  Weight discrepancy (extracted): |{jax_extracted_weight:.8f} - {milp_optimal_weight:.8f}| = {extraction_discrepancy:.8f}")
        print()
        
        # Tolerance for numerical errors
        tolerance = 1e-4
        
        objective_match = objective_discrepancy < tolerance
        weight_theory_match = weight_discrepancy < tolerance  
        weight_extract_match = extraction_discrepancy < tolerance
        
        print("VALIDATION RESULTS:")
        print(f"  Objective values agree (tol={tolerance}): {'PASS' if objective_match else 'FAIL'}")
        print(f"  Theoretical weights agree (tol={tolerance}): {'PASS' if weight_theory_match else 'FAIL'}")
        print(f"  Extracted weights agree (tol={tolerance}): {'PASS' if weight_extract_match else 'FAIL'}")
        
        if objective_match and weight_theory_match:
            print("  STATUS: PASS - MOTZKIN-STRAUS OPTIMIZATION IS MATHEMATICALLY CORRECT")
        elif not objective_match:
            print("  STATUS: FAIL - MOTZKIN-STRAUS OBJECTIVE VALUE IS INCORRECT")
        else:
            print("  STATUS: ⚠ OBJECTIVE CORRECT BUT WEIGHT EXTRACTION ISSUES")
        
        if not weight_extract_match:
            if not jax_result['is_valid_independent_set']:
                print("  ISSUE: JAX-PGD found an invalid independent set")
            else:
                print("  ISSUE: JAX-PGD found valid but suboptimal independent set")
        
        print()
    
    return {
        'milp_optimal_weight': milp_optimal_weight,
        'jax_raw_objective': jax_raw_objective,
        'jax_theoretical_weight': jax_theoretical_weight,
        'jax_extracted_weight': jax_extracted_weight,
        'theoretical_from_milp': theoretical_from_milp,
        'objective_discrepancy': objective_discrepancy,
        'weight_discrepancy': weight_discrepancy,
        'extraction_discrepancy': extraction_discrepancy,
        'objective_match': objective_discrepancy < 1e-4,
        'weight_theory_match': weight_discrepancy < 1e-4,
        'weight_extract_match': extraction_discrepancy < 1e-4,
        'milp_result': milp_result,
        'jax_result': jax_result
    }


def test_mwis_debug_objectives():
    """Test MWIS solver by debugging objective value consistency."""
    print("=" * 70)
    print("DEBUGGING MWIS OBJECTIVE VALUES CONSISTENCY")
    print("=" * 70)
    print()
    
    # Test graphs - focus on the problematic ones
    test_cases = [
        ("Triangle K3", nx.complete_graph(3), {0: 1.0, 1: 2.0, 2: 3.0}),
        ("Two triangles", nx.Graph([(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]), {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}),
        ("Path P4", nx.path_graph(4), {0: 2.0, 1: 1.0, 2: 3.0, 3: 4.0}),
        ("Diamond", nx.Graph([(0,1), (0,2), (1,2), (1,3), (2,3)]), {0: 2.0, 1: 3.0, 2: 1.0, 3: 4.0}),
    ]
    
    debug_results = []
    
    for name, graph, weights in test_cases:
        print(f"Testing: {name}")
        print("-" * 50)
        
        debug_result = debug_motzkin_straus_objective_values(graph, weights, verbose=True)
        
        if 'error' not in debug_result:
            debug_results.append({
                'name': name,
                'objective_match': debug_result['objective_match'],
                'weight_theory_match': debug_result['weight_theory_match'],
                'weight_extract_match': debug_result['weight_extract_match'],
                'objective_discrepancy': debug_result['objective_discrepancy'],
                'weight_discrepancy': debug_result['weight_discrepancy'],
                'extraction_discrepancy': debug_result['extraction_discrepancy']
            })
        else:
            debug_results.append({
                'name': name,
                'error': debug_result['error']
            })
        
        print()
    
    # Summary
    print("SUMMARY:")
    print("=" * 70)
    
    if debug_results:
        valid_results = [r for r in debug_results if 'error' not in r]
        objective_matches = sum(1 for r in valid_results if r['objective_match'])
        weight_theory_matches = sum(1 for r in valid_results if r['weight_theory_match'])
        weight_extract_matches = sum(1 for r in valid_results if r['weight_extract_match'])
        
        print(f"Test cases: {len(debug_results)}")
        print(f"Valid tests: {len(valid_results)}")
        print(f"Objective value matches: {objective_matches}/{len(valid_results)} ({100*objective_matches/len(valid_results):.1f}%)")
        print(f"Theoretical weight matches: {weight_theory_matches}/{len(valid_results)} ({100*weight_theory_matches/len(valid_results):.1f}%)")
        print(f"Extracted weight matches: {weight_extract_matches}/{len(valid_results)} ({100*weight_extract_matches/len(valid_results):.1f}%)")
        print()
        
        print("Detailed Results:")
        print(f"{'Graph':<20} {'Obj.Match':<10} {'W.Theory':<10} {'W.Extract':<10} {'Obj.Diff':<12} {'W.T.Diff':<12} {'W.E.Diff':<12}")
        print("-" * 90)
        for r in debug_results:
            if 'error' not in r:
                obj_match = "PASS" if r['objective_match'] else "FAIL"
                wt_match = "PASS" if r['weight_theory_match'] else "FAIL"
                we_match = "PASS" if r['weight_extract_match'] else "FAIL"
                print(f"{r['name']:<20} {obj_match:<10} {wt_match:<10} {we_match:<10} "
                      f"{r['objective_discrepancy']:<12.8f} {r['weight_discrepancy']:<12.8f} {r['extraction_discrepancy']:<12.8f}")
            else:
                print(f"{r['name']:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {r['error']}")
    
    print(f"\nDebugging complete.")


def test_mwis_with_dual_comparison():
    """Test MWIS solver with MILP vs JAX-PGD comparison and dual plotting."""
    print("=" * 70)
    print("TESTING MAXIMUM WEIGHT INDEPENDENT SET WITH DUAL COMPARISON")
    print("=" * 70)
    print()
    
    # Test graphs with known MWIS solutions
    test_cases = [
        # Simple cases with known solutions
        ("Triangle K3", nx.complete_graph(3), {0: 1.0, 1: 2.0, 2: 3.0}),  # MWIS: {2} (weight: 3.0)
        ("Path P4", nx.path_graph(4), {0: 2.0, 1: 1.0, 2: 3.0, 3: 4.0}),  # MWIS: {0, 2} or {1, 3} (weights: 5.0 or 5.0)
        ("Diamond", nx.Graph([(0,1), (0,2), (1,2), (1,3), (2,3)]), {0: 2.0, 1: 3.0, 2: 1.0, 3: 4.0}),  # MWIS: {0, 3} (weight: 6.0)
        ("Two triangles", nx.Graph([(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]), {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}),  # MWIS: {2, 5} (weight: 9.0)
        ("5-node path", nx.path_graph(5), {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}),  # MWIS: {1, 3} or {0, 2, 4} (weight: 6.0 or 9.0)
    ]
    
    plot_counter = 1
    results = []
    
    for name, graph, weights in test_cases:
        print(f"Testing: {name}")
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        print(f"  Weights: {weights}")
        print()
        
        # MILP solver
        print("  MILP Solver:")
        milp_result = solve_maximum_weight_independent_set_milp(graph, weights, verbose=True)
        print()
        
        # JAX-PGD solver
        print("  JAX-PGD Solver:")
        jax_result = solve_maximum_weight_independent_set_jax(
            graph, weights, verbose=True, num_restarts=50, max_iterations=2000, debug_objective=True
        )
        print()
        
        # DEBUG: Check objective values
        print("  DEBUG: Objective Value Analysis:")
        debug_result = debug_motzkin_straus_objective_values(graph, weights, verbose=False)
        if 'error' not in debug_result:
            obj_match = "PASS" if debug_result['objective_match'] else "FAIL"
            weight_theory_match = "PASS" if debug_result['weight_theory_match'] else "FAIL"
            weight_extract_match = "PASS" if debug_result['weight_extract_match'] else "FAIL"
            
            print(f"    MILP optimal weight: {debug_result['milp_optimal_weight']:.6f}")
            print(f"    JAX raw objective: {debug_result['jax_raw_objective']:.8f}")
            print(f"    Expected objective: {debug_result['theoretical_from_milp']:.8f}")
            print(f"    Objective agreement: {obj_match} (diff: {debug_result['objective_discrepancy']:.8f})")
            print(f"    Theoretical weight agreement: {weight_theory_match} (diff: {debug_result['weight_discrepancy']:.8f})")
            print(f"    Extracted weight agreement: {weight_extract_match} (diff: {debug_result['extraction_discrepancy']:.8f})")
        else:
            print(f"    Debug failed: {debug_result['error']}")
        print()
        
        # Compare results
        if milp_result['success'] and jax_result['success']:
            milp_set = milp_result['selected_vertices']
            jax_set = jax_result['selected_vertices']
            milp_weight = milp_result['set_weight']
            jax_weight = jax_result['set_weight']
            
            print("  Comparison:")
            print(f"    MILP set:       {milp_set} (weight: {milp_weight:.6f})")
            print(f"    JAX-PGD set:    {jax_set} (weight: {jax_weight:.6f})")
            print(f"    Weight diff:    {abs(milp_weight - jax_weight):.6f}")
            print(f"    Sets match:     {milp_set == jax_set}")
            print(f"    Weight match:   {abs(milp_weight - jax_weight) < 0.01}")
            print(f"    MILP runtime:   {milp_result['runtime']:.4f}s")
            print(f"    JAX runtime:    {jax_result['runtime']:.4f}s")
            
            # Validate both are independent sets
            print(f"    MILP valid:     {milp_result['is_valid_independent_set']}")
            print(f"    JAX valid:      {jax_result['is_valid_independent_set']}")
            
            # Create dual visualization
            if MATPLOTLIB_AVAILABLE:
                safe_name = name.lower().replace(" ", "_").replace("-", "_")
                plot_filename = f"mwis_dual_{plot_counter:02d}_{safe_name}.png"
                plot_path = os.path.join("plots", plot_filename)
                plot_title = f"{name} - MWIS Dual Comparison"
                
                success = plot_graph_with_dual_mwis_solutions(
                    graph, weights, milp_set, jax_set, plot_title, plot_path
                )
                if success:
                    print(f"    Dual plot saved: {plot_filename}")
                else:
                    print(f"    Dual plot failed")
            
            results.append({
                'name': name,
                'milp_weight': milp_weight,
                'jax_weight': jax_weight,
                'weight_match': abs(milp_weight - jax_weight) < 0.01,
                'sets_match': milp_set == jax_set,
                'milp_time': milp_result['runtime'],
                'jax_time': jax_result['runtime']
            })
            
        else:
            print("  Error: One or both solvers failed")
            if not milp_result['success']:
                print(f"    MILP error: {milp_result.get('message', 'Unknown error')}")
            if not jax_result['success']:
                print(f"    JAX error: {jax_result.get('message', 'Unknown error')}")
        
        plot_counter += 1
        print("-" * 50)
        print()
    
    # Summary
    if results:
        print("SUMMARY:")
        print("=" * 70)
        weight_matches = sum(1 for r in results if r['weight_match'])
        set_matches = sum(1 for r in results if r['sets_match'])
        
        print(f"Test cases: {len(results)}")
        print(f"Weight matches: {weight_matches}/{len(results)} ({100*weight_matches/len(results):.1f}%)")
        print(f"Set matches: {set_matches}/{len(results)} ({100*set_matches/len(results):.1f}%)")
        print()
        
        print("Detailed Results:")
        print(f"{'Graph':<20} {'MILP Weight':<12} {'JAX Weight':<12} {'W.Match':<8} {'S.Match':<8} {'MILP(s)':<8} {'JAX(s)':<8}")
        print("-" * 80)
        for r in results:
            w_match = "PASS" if r['weight_match'] else "FAIL"
            s_match = "PASS" if r['sets_match'] else "FAIL"
            print(f"{r['name']:<20} {r['milp_weight']:<12.3f} {r['jax_weight']:<12.3f} "
                  f"{w_match:<8} {s_match:<8} {r['milp_time']:<8.4f} {r['jax_time']:<8.4f}")
    
    print(f"\nDual comparison plots saved to plots/ directory")


def create_test_graphs() -> List[Tuple[str, nx.Graph]]:
    """Create a variety of test graphs with known stable set structures."""
    test_graphs = []
    
    # Triangle (stable sets: {0}, {1}, {2} - only single vertices)
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    test_graphs.append(("Triangle", triangle))
    
    # Complete graph K4 (stable sets: only single vertices {0}, {1}, {2}, {3})
    k4 = nx.complete_graph(4)
    test_graphs.append(("Complete K4", k4))
    
    # Two disjoint triangles (stable sets: one vertex from each triangle, e.g., {0, 3}, {0, 4}, etc.)
    two_triangles = nx.Graph()
    two_triangles.add_edges_from([(0, 1), (1, 2), (2, 0),  # First triangle
                                  (3, 4), (4, 5), (5, 3)])  # Second triangle
    test_graphs.append(("Two Triangles", two_triangles))
    
    # Diamond (K4 minus one edge - stable sets include {0, 3} which is maximal)
    diamond = nx.Graph()
    diamond.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])  # Missing edge (0,3)
    test_graphs.append(("Diamond", diamond))
    
    # Path of length 3 (stable sets: {0, 2} and {1, 3} are maximal)
    path = nx.path_graph(4)  # Nodes 0-1-2-3
    test_graphs.append(("Path P4", path))
    
    return test_graphs


def generate_erdos_renyi_graphs(nodes_list: List[int] = [10, 20], 
                                prob_list: List[float] = [0.3, 0.5, 0.7]) -> List[Tuple[str, nx.Graph]]:
    """
    Generate Erdős-Rényi random graphs with different parameters.
    
    Args:
        nodes_list: List of node counts to test
        prob_list: List of edge probabilities to test
        
    Returns:
        List of (name, graph) tuples
    """
    erdos_graphs = []
    
    for n in nodes_list:
        for p in prob_list:
            # Create reproducible random graph
            graph = nx.erdos_renyi_graph(n, p, seed=42 + int(n*10 + p*100))
            name = f"Erdos-Renyi G({n}, {p:.1f})"
            erdos_graphs.append((name, graph))
    
    return erdos_graphs


def compare_oracles_on_graph_stable_sets(
    graph: nx.Graph, 
    oracle_types: List[str], 
    support_threshold: float = 1e-5,
    verbose: bool = False,
    **base_config
) -> Dict[str, Tuple[List[Set[int]], Dict[str, Any], float]]:
    """
    Compare multiple oracles on the same graph for finding stable sets.
    
    Args:
        graph: Input graph to analyze
        oracle_types: List of oracle types to compare
        support_threshold: Support extraction threshold
        verbose: Whether to print detailed progress
        **base_config: Base configuration shared by oracles
        
    Returns:
        Dictionary mapping oracle type to (stable_sets, details, runtime) tuples
    """
    results = {}
    
    for oracle_type in oracle_types:
        if verbose:
            print(f"\n--- Testing {oracle_type} oracle ---")
        
        # Prepare oracle-specific configuration
        oracle_config = {}
        if oracle_type == 'jax-pgd':
            oracle_config = {
                'num_restarts': base_config.get('num_restarts', 50),
                'learning_rate': base_config.get('learning_rate', 0.01),
                'max_iterations': base_config.get('max_iterations', 2000)
            }
        elif oracle_type == 'dirac':
            oracle_config = {
                'num_samples': base_config.get('num_samples', 100),
                'relaxation_schedule': base_config.get('relax_schedule', 2),
                'solution_precision': base_config.get('solution_precision', None)
            }
        
        try:
            start_time = time.time()
            stable_sets, details = find_maximum_weight_independent_set(
                graph,
                oracle_type=oracle_type,
                support_threshold=support_threshold,
                verbose=verbose,
                enable_refinement=base_config.get('enable_refinement', True),
                **oracle_config
            )
            runtime = time.time() - start_time
            
            results[oracle_type] = (stable_sets, details, runtime)
            
            if verbose:
                print(f"{oracle_type}: Found {len(stable_sets)} stable sets in {runtime:.3f}s")
                
        except Exception as e:
            if verbose:
                print(f"{oracle_type}: Failed with error: {e}")
            results[oracle_type] = ([], {"error": str(e)}, 0.0)
    
    return results


def analyze_stable_set_coverage(found_sets: List[Set[int]], 
                          true_sets: List[Set[int]]) -> Tuple[float, List[Set[int]], List[Set[int]]]:
    """
    Analyze coverage of found stable sets compared to ground truth.
    
    Args:
        found_sets: Stable sets found by Motzkin-Straus method
        true_sets: Ground truth stable sets from a reliable source
        
    Returns:
        Tuple of (success_rate, missing_sets, extra_sets)
    """
    found_set = {frozenset(c) for c in found_sets}
    true_set = {frozenset(c) for c in true_sets}
    
    correct = found_set & true_set
    missing = true_set - found_set
    extra = found_set - true_set
    
    success_rate = len(correct) / len(true_set) if true_set else 1.0
    
    return success_rate, [set(c) for c in missing], [set(c) for c in extra]


def solve_maximum_weight_independent_set_milp(
    graph: nx.Graph,
    weights: Dict[int, float],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solve Maximum Weight Independent Set using SciPy MILP solver as ground truth.
    
    Formulation:
    maximize: sum(w[i] * x[i]) for all vertices i
    subject to: x[i] + x[j] <= 1 for all edges (i,j)
                x[i] ∈ {0,1} for all vertices i
    
    Args:
        graph: Input graph
        weights: Vertex weights
        verbose: Print debug information
        
    Returns:
        Dictionary with solution information
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return {
            'success': False,
            'message': 'SciPy not available',
            'runtime': 0.0
        }
    
    node_list = list(graph.nodes())
    n = len(node_list)
    
    if verbose:
        print(f"Setting up MILP for MWIS with {n} vertices, {graph.number_of_edges()} edges")
    
    start_time = time.time()
    
    # Objective: maximize sum(w[i] * x[i])
    c = np.array([-weights.get(node, 1.0) for node in node_list])  # Negative for maximization
    
    # Constraints: x[i] + x[j] <= 1 for all edges (i,j)
    constraints = []
    
    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if i < j and graph.has_edge(node_i, node_j):
                # Adjacent vertices cannot both be selected
                A_row = np.zeros(n)
                A_row[i] = 1
                A_row[j] = 1
                constraints.append((A_row, 1.0))  # x[i] + x[j] <= 1
    
    if constraints:
        A_ub = np.vstack([row for row, _ in constraints])
        b_ub = np.array([bound for _, bound in constraints])
        constraint = LinearConstraint(A_ub, -np.inf, b_ub)
    else:
        # No constraints (no edges - all vertices are independent)
        constraint = None
    
    # Variable bounds: x[i] ∈ {0,1}
    bounds = Bounds(0, 1)
    
    # Integer constraints
    integrality = np.ones(n)  # All variables are integers
    
    if verbose:
        print(f"MILP has {len(constraints)} constraints")
    
    # Solve
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
    
    # Verify it's a valid independent set
    is_valid = True
    for i in selected_vertices:
        for j in selected_vertices:
            if i != j and graph.has_edge(i, j):
                is_valid = False
                break
        if not is_valid:
            break
    
    set_weight = sum(weights.get(v, 1.0) for v in selected_vertices)
    
    if verbose:
        print(f"MILP solution: {solution}")
        print(f"Objective value: {objective_value:.6f}")
        print(f"Selected vertices: {selected_vertices}")
        print(f"Is valid independent set: {is_valid}")
        print(f"Set weight: {set_weight:.6f}")
        print(f"Runtime: {runtime:.4f} seconds")
    
    return {
        'success': True,
        'solution': solution,
        'objective_value': objective_value,
        'selected_vertices': selected_vertices,
        'set_weight': set_weight,
        'is_valid_independent_set': is_valid,
        'runtime': runtime,
        'node_list': node_list
    }


def solve_mwis_milp(graph: nx.Graph, weights: Dict[int, float], verbose: bool = False) -> Tuple[Optional[Set[int]], float, float]:
    """
    Legacy wrapper for backward compatibility.
    Solves the Maximum Weight Independent Set (MWIS) problem using a classical MILP solver.
    """
    result = solve_maximum_weight_independent_set_milp(graph, weights, verbose)
    
    if result['success']:
        return result['selected_vertices'], result['set_weight'], result['runtime']
    else:
        return None, 0.0, result['runtime']


import json

def main():
    """Main function with CLI interface and testing."""
    parser = argparse.ArgumentParser(
        description="Find maximum weight independent sets using the generalized Motzkin-Straus theorem",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main options
    parser.add_argument("--test", action="store_true", 
                       help="Run tests on predefined graphs")
    parser.add_argument("--random-test", action="store_true",
                       help="Test on a larger random graph with internally generated weights")
    parser.add_argument("--oracle", type=str, default="jax-pgd", choices=["jax-pgd", "dirac"],
                       help="Oracle solver type (default: jax-pgd)")
    parser.add_argument("--compare-oracles", action="store_true",
                       help="Compare multiple oracles on same graphs")
    parser.add_argument("--threshold", type=float, default=1e-5,
                       help="Support extraction threshold (default: 1e-5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--save-plots", type=str, default="./plots/",
                       help="Directory to save plots (default: ./plots/)")
    parser.add_argument("--compare-networkx", action="store_true",
                       help="Compare results with NetworkX find_cliques")
    parser.add_argument("--compare-milp", action="store_true",
                       help="Compare Motzkin-Straus results with a classical MILP solver (requires scipy)")
    parser.add_argument("--test-dual", action="store_true",
                       help="Test MWIS with dual comparison (MILP vs JAX-PGD) and visualization")
    parser.add_argument("--debug-objectives", action="store_true",
                       help="Debug Motzkin-Straus objective values vs MILP ground truth")
    parser.add_argument("--nodes", type=int, default=10,
                       help="Number of nodes for Erdős-Rényi graphs (default: 10)")
    parser.add_argument("--edge-prob", type=float, default=0.5,
                       help="Edge probability for random graphs (default: 0.5)")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to a JSON file containing node weights")
    
    # JAX-PGD specific options
    jax_group = parser.add_argument_group('JAX-PGD Oracle Options')
    jax_group.add_argument("--num-restarts", type=int, default=50,
                          help="Number of optimization restarts (default: 50)")
    jax_group.add_argument("--learning-rate", type=float, default=0.01,
                          help="Learning rate for gradient descent (default: 0.01)")
    jax_group.add_argument("--max-iterations", type=int, default=2000,
                          help="Maximum iterations per restart (default: 2000)")
    
    # Dirac-3 specific options
    dirac_group = parser.add_argument_group('Dirac-3 Oracle Options')
    dirac_group.add_argument("--num-samples", type=int, default=100,
                            help="Number of Dirac samples (default: 100)")
    dirac_group.add_argument("--relax-schedule", type=int, default=2, choices=[1, 2, 3, 4],
                            help="Relaxation schedule 1-4 (default: 2)")
    dirac_group.add_argument("--solution-precision", type=float, default=None,
                            help="Solution precision (optional)")
    
    # Refinement control options
    refinement_group = parser.add_argument_group('Superposition Refinement Options')
    refinement_group.add_argument("--enable-refinement", action="store_true", default=True,
                                 help="Enable superposition solution refinement (default: True)")
    refinement_group.add_argument("--disable-refinement", action="store_true", 
                                 help="Disable superposition solution refinement")
    
    args = parser.parse_args()
    
    # Handle refinement flag logic (--disable-refinement overrides --enable-refinement)
    if args.disable_refinement:
        args.enable_refinement = False
    
    # Create plots directory if plotting is enabled
    if args.plot:
        os.makedirs(args.save_plots, exist_ok=True)
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available - plotting disabled")
            args.plot = False
    
    # Check oracle availability if needed
    if OracleFactory and (args.compare_oracles or args.oracle != 'jax-pgd'):
        available_oracles = OracleFactory.list_available_oracles()
        if args.oracle not in available_oracles:
            print(f"Warning: Oracle {args.oracle} is not available. Available oracles: {available_oracles}")
            if not available_oracles:
                print("No oracles available - ensure dependencies are installed")
                return 1
            args.oracle = available_oracles[0]  # Use first available oracle
            print(f"Switching to {args.oracle} oracle")

    # Load weights if provided
    weights = {}
    if args.weights:
        with open(args.weights, 'r') as f:
            weights = json.load(f)
        # Convert keys to integers
        weights = {int(k): v for k, v in weights.items()}

    if args.debug_objectives:
        print("Debugging MWIS Motzkin-Straus objective values")
        print("=" * 60)
        
        # Run objective debugging test
        test_mwis_debug_objectives()
        
    elif args.test_dual:
        print("Testing MWIS solver with dual comparison (MILP vs JAX-PGD)")
        print("=" * 60)
        
        # Create plots directory
        os.makedirs("plots", exist_ok=True)
        
        # Run dual comparison test
        test_mwis_with_dual_comparison()
        
    elif args.test:
        print("Testing maximum weight independent set finder on various graphs")
        print("=" * 60)
        
        test_graphs = create_test_graphs()
        
        for name, graph in test_graphs:
            print(f"\nTesting on {name}")
            print(f"   Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            
            # Single oracle mode
            start_time = time.time()
            
            # Find maximal stable sets using selected oracle
            try:
                # Prepare oracle configuration based on selected oracle
                oracle_config = {}
                if args.oracle == 'jax-pgd':
                    oracle_config = {
                        'num_restarts': args.num_restarts,
                        'learning_rate': args.learning_rate,
                        'max_iterations': args.max_iterations
                    }
                elif args.oracle == 'dirac':
                    oracle_config = {
                        'num_samples': args.num_samples,
                        'relaxation_schedule': args.relax_schedule,
                        'solution_precision': args.solution_precision
                    }
                
                maximal_stable_sets, optimization_details = find_maximum_weight_independent_set(
                    graph,
                    weights,
                    oracle_type=args.oracle,
                    support_threshold=args.threshold,
                    verbose=args.verbose,
                    enable_refinement=args.enable_refinement,
                    **oracle_config
                )
                
                runtime = time.time() - start_time
                
                print(f"   Motzkin-Straus result: {len(maximal_stable_sets)} maximal stable sets found")
                for i, stable_set in enumerate(sorted(maximal_stable_sets, key=lambda x: (len(x), sorted(x)))):
                    set_weight = sum(weights.get(node, 1.0) for node in stable_set)
                    print(f"     Stable Set {i+1}: {sorted(stable_set)} (size: {len(stable_set)}, weight: {set_weight})")
                
                print(f"   Runtime: {runtime:.3f}s")
                
            except Exception as e:
                print(f"   Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    elif args.random_test:
        print("Testing MWIS finder on a random graph")
        print("=" * 60)
        
        # Generate a random graph
        graph = nx.erdos_renyi_graph(n=args.nodes, p=args.edge_prob, seed=random.randint(0, 10000))
        print(f"Generated Erdős-Rényi graph with {args.nodes} nodes and edge probability {args.edge_prob}")
        print(f"Actual graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # Generate random weights if not provided
        if not weights:
            print("Generating random weights for nodes...")
            weights = {node: round(random.uniform(1.0, 10.0), 2) for node in graph.nodes()}
            if args.verbose:
                print(f"Generated weights: {weights}")

        # Find the maximum weight independent set
        start_time = time.time()
        try:
            oracle_config = {}
            if args.oracle == 'jax-pgd':
                oracle_config = {
                    'num_restarts': args.num_restarts,
                    'learning_rate': args.learning_rate,
                    'max_iterations': args.max_iterations
                }
            elif args.oracle == 'dirac':
                oracle_config = {
                    'num_samples': args.num_samples,
                    'relaxation_schedule': args.relax_schedule,
                    'solution_precision': args.solution_precision
                }

            maximal_stable_sets, optimization_details = find_maximum_weight_independent_set(
                graph,
                weights,
                oracle_type=args.oracle,
                support_threshold=args.threshold,
                verbose=args.verbose,
                enable_refinement=args.enable_refinement,
                **oracle_config
            )
            runtime = time.time() - start_time

            if not maximal_stable_sets:
                print("No independent sets found.")
            else:
                # Find the one with the maximum weight
                best_set = max(maximal_stable_sets, key=lambda s: sum(weights.get(node, 1.0) for node in s))
                best_weight = sum(weights.get(node, 1.0) for node in best_set)
                
                print("\n--- Motzkin-Straus Results ---")
                print(f"Found {len(maximal_stable_sets)} maximal independent sets.")
                print(f"Largest weight independent set: {sorted(list(best_set))}")
                print(f"Total weight: {best_weight:.2f}")
                print(f"Size of set: {len(best_set)}")
                print(f"Runtime: {runtime:.3f}s")

            # Compare with MILP if requested
            if args.compare_milp:
                print("\n--- MILP Baseline Comparison ---")
                milp_set, milp_weight, milp_runtime = solve_mwis_milp(graph, weights, verbose=args.verbose)
                if milp_set is not None:
                    print(f"MILP solver found an independent set of size {len(milp_set)} with total weight {milp_weight:.2f}")
                    print(f"Set: {sorted(list(milp_set))}")
                    print(f"Runtime: {milp_runtime:.3f}s")

                    # Compare results
                    if maximal_stable_sets:
                        ms_weight = best_weight
                        print("\n--- Comparison Summary ---")
                        print(f"Motzkin-Straus Weight: {ms_weight:.2f}")
                        print(f"MILP (Optimal) Weight: {milp_weight:.2f}")
                        if abs(ms_weight - milp_weight) < 1e-5:
                            print("Result: Motzkin-Straus found the optimal solution.")
                        else:
                            print("Result: Motzkin-Straus found a sub-optimal solution.")
                else:
                    print("MILP solver failed or is not available.")


        except Exception as e:
            print(f"   Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    else:
        print("Maximum Weight Independent Set Finder using Motzkin-Straus Theorem")
        print("Run with --test to test on predefined graphs")
        print("Run with --help to see all options")


if __name__ == "__main__":
    main()