#!/usr/bin/env python3
"""
Maximal Clique Finder using Motzkin-Straus Theorem with Multiple Oracle Solvers

Theory: 
- The Motzkin-Straus theorem connects the maximum of a quadratic function over the simplex 
  to the clique number of a graph.
- The support of the optimal solution (global maximum) corresponds to vertices of a maximum clique.
- Local maxima correspond to maximal cliques - cliques that cannot be extended by adding vertices.
- Different oracle solvers can explore the optimization landscape in different ways.

Implementation:
- Support multiple oracle solvers: JAX-PGD (local optimization) and Dirac-3 (quantum annealing)
- Extract support from solution vectors (indices where x_i > threshold)
- Verify each support forms a valid clique
- Check maximality (cannot be extended)
- Return collection of unique maximal cliques

Oracle Solvers:
- JAX-PGD: Projected Gradient Descent with multiple restarts for local optimization
- Dirac-3: Quantum annealing solver via QCI cloud service

Usage Examples:

    # Test with JAX-PGD oracle (default)
    python scripts/clique_instance.py --test --oracle jax-pgd --compare-networkx
    
    # Test with Dirac-3 oracle
    python scripts/clique_instance.py --test --oracle dirac --num-samples 100
    
    # Compare different oracles with color-coded visualization
    python scripts/clique_instance.py --test --compare-oracles --plot
    
    # Test Erdős-Rényi graphs with Dirac
    python scripts/clique_instance.py --erdos-test --nodes 10 --oracle dirac
    
    # JAX-PGD with custom parameters
    python scripts/clique_instance.py --test --oracle jax-pgd --num-restarts 100
    
    # Dirac with custom configuration and refinement disabled
    python scripts/clique_instance.py --test --oracle dirac --num-samples 200 --disable-refinement
    
    # Enable refinement for better clique discovery (default behavior)
    python scripts/clique_instance.py --test --oracle dirac --enable-refinement --verbose
    
    # Compare oracles on random graphs with refinement control
    python scripts/clique_instance.py --erdos-test --compare-oracles --disable-refinement --plot

Parameters:
    --test              Run tests on predefined graphs
    --erdos-test        Test on Erdős-Rényi random graphs
    --oracle TYPE       Oracle solver: 'jax-pgd' or 'dirac' (default: jax-pgd)
    --compare-oracles   Compare multiple oracles on same graphs
    --threshold T       Support extraction threshold (default: 1e-5)
    --verbose          Enable detailed progress output
    --plot             Generate visualization plots
    --save-plots DIR   Directory to save plots (default: ./plots/)
    --compare-networkx Compare results with NetworkX ground truth
    --nodes N          Number of nodes for Erdős-Rényi graphs (default: 10)
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
- Dense graphs are generally easier to solve than sparse graphs
- Superposition refinement: Improves clique discovery but increases computational cost
- Oracle comparison mode: Provides enhanced visualization with color-coded results
"""

import sys
import os
import networkx as nx
import numpy as np
import time
from typing import Set, List, Tuple, Dict, Any, Optional
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from motzkinstraus.algorithms import verify_clique
    print("Successfully imported motzkinstraus modules")
except ImportError as e:
    print(f"Error importing motzkinstraus modules: {e}")
    print("Make sure you're in the correct directory and virtual environment is activated")
    sys.exit(1)

# Import oracle system
try:
    from oracles import OracleFactory
    print("Successfully imported oracle system")
except ImportError as e:
    print(f"Error importing oracle system: {e}")
    print("Oracle system not available - falling back to legacy JAX-PGD implementation")
    OracleFactory = None

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
    
    The support corresponds to vertices that are candidates for being in a clique
    based on the Motzkin-Straus theorem.
    
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


def verify_maximal_clique(graph: nx.Graph, candidate_clique: Set[int]) -> bool:
    """
    Verify that a clique is maximal (cannot be extended by adding another vertex).
    
    Args:
        graph: The input graph
        candidate_clique: Set of vertices to check for maximality
        
    Returns:
        True if the clique is maximal, False otherwise
    """
    if not candidate_clique:
        return False
    
    # First verify it's actually a clique
    if not verify_clique(graph, candidate_clique):
        return False
    
    # Check if we can add any vertex to extend the clique
    for vertex in graph.nodes():
        if vertex not in candidate_clique:
            # Check if this vertex is connected to ALL vertices in the current clique
            is_connected_to_all = all(graph.has_edge(vertex, clique_member) 
                                    for clique_member in candidate_clique)
            
            if is_connected_to_all:
                # We can extend the clique, so it's not maximal
                return False
    
    # No vertex can be added, so it's maximal
    return True


def find_maximal_cliques_motzkin_straus(
    graph: nx.Graph,
    oracle_type: str = 'jax-pgd',
    support_threshold: float = 1e-5,
    verbose: bool = False,
    enable_refinement: bool = True,
    **oracle_config
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Find maximal cliques using the Motzkin-Straus theorem with configurable oracle solvers.
    
    This function supports multiple oracle solvers to find local maxima of the 
    Motzkin-Straus quadratic program. Each local maximum potentially corresponds 
    to a maximal clique.
    
    Args:
        graph: Input NetworkX graph
        oracle_type: Type of oracle solver ('jax-pgd', 'dirac')
        support_threshold: Threshold for extracting support from solution vectors
        verbose: Whether to print detailed progress
        enable_refinement: Whether to enable superposition refinement (default: True)
        **oracle_config: Oracle-specific configuration parameters
        
    Returns:
        Tuple of (maximal_cliques, optimization_details) where:
        - maximal_cliques: List of sets, each containing vertices of a maximal clique
        - optimization_details: Dictionary with oracle-specific optimization information
    """
    if graph.number_of_nodes() == 0:
        return [], {"message": "Empty graph"}
    
    # Use oracle system if available, otherwise fallback to legacy implementation
    if OracleFactory is None:
        if verbose:
            print("Oracle system not available - using legacy JAX-PGD implementation")
        return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, **oracle_config)
    
    # Create oracle adapter
    try:
        oracle = OracleFactory.create_oracle(oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config)
        
        if verbose:
            print(f"Using {oracle.name} oracle")
            print(f"Finding maximal cliques in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Find maximal cliques using the oracle
        maximal_cliques = oracle.find_maximal_cliques(graph, support_threshold)
        
        # Get optimization details
        optimization_details = oracle.get_optimization_details()
        
        if verbose:
            print(f"Found {len(maximal_cliques)} unique maximal cliques")
        
        return maximal_cliques, optimization_details
        
    except Exception as e:
        if verbose:
            print(f"Oracle {oracle_type} failed: {e}")
            print("Falling back to legacy JAX-PGD implementation")
        return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, **oracle_config)


def _legacy_jax_pgd_implementation(
    graph: nx.Graph, 
    support_threshold: float = 1e-5, 
    verbose: bool = False,
    num_restarts: int = 50,
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    **kwargs
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Legacy JAX-PGD implementation for backward compatibility.
    """
    if verbose:
        print(f"Legacy JAX-PGD: Finding maximal cliques with {num_restarts} restarts")
    
    # Import required modules
    try:
        from motzkinstraus.jax_optimizers import (
            JAXOptimizerConfig, 
            adjacency_to_polynomial, 
            run_projected_gradient_descent,
            sample_dirichlet
        )
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        print(f"Error importing JAX modules: {e}")
        return [], {"error": str(e)}
    
    # Set up adjacency matrix and polynomial representation
    node_list = list(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)
    
    # Convert to polynomial format
    poly_indices, poly_coefficients = adjacency_to_polynomial(adj_matrix)
    
    if len(poly_indices) == 0:
        if verbose:
            print("No polynomial terms - empty graph or no edges")
        return [], {"message": "No edges"}
    
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
    
    # Extract cliques
    maximal_cliques = set()
    for solution_vector in all_final_solutions:
        support_indices = extract_support(solution_vector, support_threshold)
        if not support_indices:
            continue
        
        candidate_clique = {node_list[idx] for idx in support_indices if idx < len(node_list)}
        
        if verify_clique(graph, candidate_clique) and verify_maximal_clique(graph, candidate_clique):
            maximal_cliques.add(frozenset(candidate_clique))
    
    result_cliques = [set(clique) for clique in maximal_cliques]
    details = {
        "oracle_type": "legacy-jax-pgd",
        "num_restarts": num_restarts,
        "solutions_processed": len(all_final_solutions),
        "cliques_found": len(result_cliques)
    }
    
    return result_cliques, details


def plot_clique_instances(graph: nx.Graph, cliques: List[Set[int]], title: str, save_path: str = None) -> bool:
    """
    Plot graph with maximal cliques highlighted in different colors.
    
    Args:
        graph: NetworkX graph to plot
        cliques: List of maximal cliques (sets of vertices)
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
        
        # Define colors for different cliques
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Draw the graph structure first
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax)
        
        # Draw all nodes in default color first
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7, ax=ax)
        
        # Highlight nodes in each clique with different colors
        for i, clique in enumerate(cliques):
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(graph, pos, nodelist=list(clique),
                                  node_color=color, node_size=600, 
                                  alpha=0.8, ax=ax)
            
            # Add clique edges with same color
            clique_edges = [(u, v) for u in clique for v in clique 
                           if u < v and graph.has_edge(u, v)]
            nx.draw_networkx_edges(graph, pos, edgelist=clique_edges,
                                  edge_color=color, width=3, alpha=0.7, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Create legend for cliques
        legend_elements = []
        for i, clique in enumerate(cliques):
            color = colors[i % len(colors)]
            clique_str = str(sorted(clique))
            legend_elements.append(patches.Patch(color=color, 
                                                label=f'Clique {i+1}: {clique_str}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set title and formatting
        ax.set_title(f'{title}\nNodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, Cliques: {len(cliques)}', 
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


def plot_oracle_comparison_results(
    graph: nx.Graph, 
    oracle_results: Dict[str, Tuple[List[Set[int]], Dict[str, Any], float]], 
    title: str, 
    save_path: str = None
) -> bool:
    """
    Plot graph with maximal cliques color-coded by oracle solver.
    
    This enhanced plotting function visualizes the results from multiple oracle 
    solvers, using distinct colors and line styles to differentiate between:
    - Oracle types: JAX-PGD (blue family) vs Dirac (red family)  
    - Individual cliques: Different line styles (solid, dashed, dotted, dashdot)
    - Clique prominence: Different line widths (2, 3, 4, 5 pixels)
    
    Args:
        graph: NetworkX graph to plot
        oracle_results: Dict mapping oracle names to (cliques, details, runtime) tuples
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
        
        # Line styles for clique differentiation
        linestyles = ['-', '--', ':', '-.']  # solid, dashed, dotted, dashdot
        linewidths = [2, 3, 4, 5]  # different thickness levels
        
        # Draw the base graph structure first
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax)
        
        # Draw all nodes in default color first  
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=600, alpha=0.7, ax=ax)
        
        # Track legend elements for organized display
        legend_elements = []
        oracle_clique_counts = {}
        
        # Process each oracle's results
        for oracle_idx, (oracle_name, (cliques, details, runtime)) in enumerate(oracle_results.items()):
            oracle_clique_counts[oracle_name] = len(cliques)
            
            # Get style configuration for this oracle
            if oracle_name.startswith('jax') or 'pgd' in oracle_name.lower():
                style_config = oracle_styles['jax-pgd']
            else:  # Assume dirac or other quantum oracle
                style_config = oracle_styles['dirac']
            
            # Draw cliques for this oracle
            for clique_idx, clique in enumerate(cliques):
                # Select color, line style, and width for unique visual signature
                color = style_config['colors'][clique_idx % len(style_config['colors'])]
                linestyle = linestyles[clique_idx % len(linestyles)]
                linewidth = linewidths[clique_idx % len(linewidths)]
                
                # Draw clique nodes with oracle base color but varying alpha
                alpha = 0.9 - (clique_idx * 0.1) % 0.4  # Vary alpha between 0.5-0.9
                nx.draw_networkx_nodes(graph, pos, nodelist=list(clique),
                                      node_color=color, node_size=700, 
                                      alpha=alpha, ax=ax)
                
                # Draw clique edges with distinctive style
                clique_edges = [(u, v) for u in clique for v in clique 
                               if u < v and graph.has_edge(u, v)]
                nx.draw_networkx_edges(graph, pos, edgelist=clique_edges,
                                      edge_color=color, width=linewidth, 
                                      style=linestyle, alpha=0.8, ax=ax)
                
                # Add legend entry with oracle and clique information
                clique_str = str(sorted(clique))
                if len(clique_str) > 30:  # Truncate long clique strings
                    clique_str = clique_str[:27] + "..."
                    
                legend_elements.append(patches.Patch(
                    color=color, 
                    label=f'{oracle_name.upper()} C{clique_idx+1}: {clique_str} (size: {len(clique)})'
                ))
        
        # Draw node labels on top
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', 
                               font_color='white', ax=ax)
        
        # Create organized legend
        if legend_elements:
            # Sort legend by oracle name for better organization
            legend_elements.sort(key=lambda x: (x.get_label().split()[0], x.get_label()))
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                     fontsize=9, title="Oracle Cliques")
        
        # Create informative title and subtitle
        runtime_info = " | ".join([
            f"{oracle}: {runtime:.3f}s, {count} cliques" 
            for oracle, (_, _, runtime), count in 
            [(k, v, oracle_clique_counts[k]) for k, v in oracle_results.items()]
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


def create_test_graphs() -> List[Tuple[str, nx.Graph]]:
    """Create a variety of test graphs with known clique structures."""
    test_graphs = []
    
    # Triangle (single maximal clique of size 3)
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    test_graphs.append(("Triangle", triangle))
    
    # Complete graph K4 (single maximal clique of size 4)
    k4 = nx.complete_graph(4)
    test_graphs.append(("Complete K4", k4))
    
    # Two disjoint triangles (two maximal cliques of size 3 each)
    two_triangles = nx.Graph()
    two_triangles.add_edges_from([(0, 1), (1, 2), (2, 0),  # First triangle
                                  (3, 4), (4, 5), (5, 3)])  # Second triangle
    test_graphs.append(("Two Triangles", two_triangles))
    
    # Diamond (K4 minus one edge - should have multiple maximal cliques of size 3)
    diamond = nx.Graph()
    diamond.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])  # Missing edge (0,3)
    test_graphs.append(("Diamond", diamond))
    
    # Path of length 3 (should have maximal cliques of size 2)
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


def compare_oracles_on_graph(
    graph: nx.Graph, 
    oracle_types: List[str], 
    support_threshold: float = 1e-5,
    verbose: bool = False,
    **base_config
) -> Dict[str, Tuple[List[Set[int]], Dict[str, Any], float]]:
    """
    Compare multiple oracles on the same graph.
    
    Args:
        graph: Input graph to analyze
        oracle_types: List of oracle types to compare
        support_threshold: Support extraction threshold
        verbose: Whether to print detailed progress
        **base_config: Base configuration shared by oracles
        
    Returns:
        Dictionary mapping oracle type to (cliques, details, runtime) tuples
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
            cliques, details = find_maximal_cliques_motzkin_straus(
                graph,
                oracle_type=oracle_type,
                support_threshold=support_threshold,
                verbose=verbose,
                enable_refinement=base_config.get('enable_refinement', True),
                **oracle_config
            )
            runtime = time.time() - start_time
            
            results[oracle_type] = (cliques, details, runtime)
            
            if verbose:
                print(f"{oracle_type}: Found {len(cliques)} cliques in {runtime:.3f}s")
                
        except Exception as e:
            if verbose:
                print(f"{oracle_type}: Failed with error: {e}")
            results[oracle_type] = ([], {"error": str(e)}, 0.0)
    
    return results


def analyze_clique_coverage(found_cliques: List[Set[int]], 
                          true_cliques: List[Set[int]]) -> Tuple[float, List[Set[int]], List[Set[int]]]:
    """
    Analyze coverage of found cliques compared to ground truth.
    
    Args:
        found_cliques: Cliques found by Motzkin-Straus method
        true_cliques: Ground truth cliques from NetworkX
        
    Returns:
        Tuple of (success_rate, missing_cliques, extra_cliques)
    """
    found_set = {frozenset(c) for c in found_cliques}
    true_set = {frozenset(c) for c in true_cliques}
    
    correct = found_set & true_set
    missing = true_set - found_set
    extra = found_set - true_set
    
    success_rate = len(correct) / len(true_set) if true_set else 1.0
    
    return success_rate, [set(c) for c in missing], [set(c) for c in extra]


def main():
    """Main function with CLI interface and testing."""
    parser = argparse.ArgumentParser(
        description="Find maximal cliques using Motzkin-Straus theorem with multiple oracle solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main options
    parser.add_argument("--test", action="store_true", 
                       help="Run tests on predefined graphs")
    parser.add_argument("--erdos-test", action="store_true",
                       help="Test on Erdős-Rényi random graphs")
    parser.add_argument("--oracle", type=str, default="jax-pgd", choices=["jax-pgd", "dirac"],
                       help="Oracle solver type (default: jax-pgd)")
    parser.add_argument("--compare-oracles", action="store_true",
                       help="Compare multiple oracles on same graphs")
    parser.add_argument("--threshold", type=float, default=1e-3,
                       help="Support extraction threshold (default: 1e-5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--save-plots", type=str, default="./plots/",
                       help="Directory to save plots (default: ./plots/)")
    parser.add_argument("--compare-networkx", action="store_true",
                       help="Compare results with NetworkX find_cliques")
    parser.add_argument("--nodes", type=int, default=10,
                       help="Number of nodes for Erdős-Rényi graphs (default: 10)")
    parser.add_argument("--edge-prob", type=float, default=0.5,
                       help="Edge probability for random graphs (default: 0.5)")
    
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

    if args.test:
        print("Testing maximal clique finder on various graphs")
        print("=" * 60)
        
        test_graphs = create_test_graphs()
        total_success = 0
        total_graphs = len(test_graphs)
        
        for name, graph in test_graphs:
            print(f"\nTesting on {name}")
            print(f"   Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            
            # Handle oracle comparison mode
            if args.compare_oracles:
                if not OracleFactory:
                    print("   Oracle comparison not available - oracle system not imported")
                    continue
                
                oracle_types = OracleFactory.list_available_oracles()
                if len(oracle_types) < 2:
                    print(f"   Need at least 2 oracles for comparison, found: {oracle_types}")
                    oracle_types = ['jax-pgd']  # Fall back to single oracle
                
                print(f"   Comparing oracles: {oracle_types}")
                
                base_config = {
                    'num_restarts': args.num_restarts,
                    'learning_rate': args.learning_rate,
                    'max_iterations': args.max_iterations,
                    'num_samples': args.num_samples,
                    'relax_schedule': args.relax_schedule,
                    'solution_precision': args.solution_precision,
                    'enable_refinement': args.enable_refinement
                }
                
                oracle_results = compare_oracles_on_graph(
                    graph, oracle_types, args.threshold, args.verbose, **base_config
                )
                
                # Report comparison results
                for oracle_type, (cliques, details, runtime) in oracle_results.items():
                    print(f"   {oracle_type}: {len(cliques)} cliques, {runtime:.3f}s")
                    if args.verbose and cliques:
                        for i, clique in enumerate(sorted(cliques, key=lambda x: (len(x), sorted(x)))):
                            print(f"     Clique {i+1}: {sorted(clique)} (size: {len(clique)})")
                
                # Generate oracle comparison plot if requested
                if args.plot and len(oracle_results) > 1:
                    plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}_oracle_comparison.png")
                    plot_oracle_comparison_results(graph, oracle_results, f"{name} - Oracle Comparison", plot_path)
                
                # Use first successful oracle result for NetworkX comparison
                first_result = None
                for oracle_type, (cliques, details, runtime) in oracle_results.items():
                    if cliques:
                        first_result = (cliques, runtime)
                        break
                
                if first_result:
                    maximal_cliques, runtime = first_result
                else:
                    maximal_cliques, runtime = [], 0.0
                    print("   No oracles succeeded")
                
            else:
                # Single oracle mode
                start_time = time.time()
                
                # Find maximal cliques using selected oracle
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
                    
                    maximal_cliques, optimization_details = find_maximal_cliques_motzkin_straus(
                        graph,
                        oracle_type=args.oracle,
                        support_threshold=args.threshold,
                        verbose=args.verbose,
                        enable_refinement=args.enable_refinement,
                        **oracle_config
                    )
                    
                    runtime = time.time() - start_time
                    
                    print(f"   Motzkin-Straus result: {len(maximal_cliques)} maximal cliques found")
                    for i, clique in enumerate(sorted(maximal_cliques, key=lambda x: (len(x), sorted(x)))):
                        print(f"     Clique {i+1}: {sorted(clique)} (size: {len(clique)})")
                    
                    print(f"   Runtime: {runtime:.3f}s")
                    
                    # Compare with NetworkX if requested
                    success_rate = 1.0  # Default if not comparing
                    if args.compare_networkx:
                        try:
                            nx_cliques = list(nx.find_cliques(graph))
                            print(f"   NetworkX result: {len(nx_cliques)} maximal cliques")
                            for i, clique in enumerate(sorted(nx_cliques, key=lambda x: (len(x), sorted(x)))):
                                print(f"     NX Clique {i+1}: {sorted(clique)} (size: {len(clique)})")
                                
                            # Analyze coverage
                            success_rate, missing, extra = analyze_clique_coverage(maximal_cliques, nx_cliques)
                            print(f"   Success rate: {success_rate:.1%} ({len(maximal_cliques)}/{len(nx_cliques)} cliques found)")
                            
                            if missing:
                                print(f"   Missing: {[sorted(c) for c in missing]}")
                            if extra:
                                print(f"   Extra: {[sorted(c) for c in extra]}")
                            if success_rate == 1.0:
                                print("   Results match NetworkX exactly!")
                                    
                        except Exception as e:
                            print(f"   Error comparing with NetworkX: {e}")
                
                    total_success += success_rate
                    
                    # Generate plot if requested
                    if args.plot:
                        plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_')}.png")
                        plot_clique_instances(graph, maximal_cliques, name, plot_path)
                        
                except Exception as e:
                    print(f"   Error: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
        
        print(f"\nOverall Success Rate: {total_success/total_graphs:.1%}")
    
    elif args.erdos_test:
        print("Testing on Erdős-Rényi random graphs")
        print("=" * 60)
        
        # Generate test graphs
        if args.nodes and args.edge_prob:
            erdos_graphs = generate_erdos_renyi_graphs([args.nodes], [args.edge_prob])
        else:
            erdos_graphs = generate_erdos_renyi_graphs()
        
        total_success = 0
        total_graphs = len(erdos_graphs)
        
        for name, graph in erdos_graphs:
            print(f"\nTesting {name}")
            print(f"   Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            
            try:
                # Handle oracle comparison mode
                if args.compare_oracles:
                    if not OracleFactory:
                        print("   Oracle comparison not available - oracle system not imported")
                        continue
                    
                    oracle_types = OracleFactory.list_available_oracles()
                    if len(oracle_types) < 2:
                        print(f"   Need at least 2 oracles for comparison, found: {oracle_types}")
                        oracle_types = ['jax-pgd']  # Fall back to single oracle
                    
                    print(f"   Comparing oracles: {oracle_types}")
                    
                    base_config = {
                        'num_restarts': args.num_restarts,
                        'learning_rate': args.learning_rate,
                        'max_iterations': args.max_iterations,
                        'num_samples': args.num_samples,
                        'relax_schedule': args.relax_schedule,
                        'solution_precision': args.solution_precision,
                        'enable_refinement': args.enable_refinement
                    }
                    
                    oracle_results = compare_oracles_on_graph(
                        graph, oracle_types, args.threshold, args.verbose, **base_config
                    )
                    
                    # Report comparison results
                    for oracle_type, (cliques, _, runtime) in oracle_results.items():
                        print(f"   {oracle_type}: {len(cliques)} cliques, {runtime:.3f}s")
                        if args.verbose and cliques:
                            sorted_cliques = sorted(cliques, key=lambda x: (len(x), sorted(x)))[:3]  # Show first 3
                            for i, clique in enumerate(sorted_cliques):
                                print(f"     Clique {i+1}: {sorted(clique)} (size: {len(clique)})")
                    
                    # Generate oracle comparison plot if requested
                    if args.plot and len(oracle_results) > 1:
                        plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_')}_oracle_comparison.png")
                        plot_oracle_comparison_results(graph, oracle_results, f"{name} - Oracle Comparison", plot_path)
                    
                    # Use first successful oracle result for NetworkX comparison
                    first_result = None
                    for oracle_type, (cliques, _, runtime) in oracle_results.items():
                        if cliques:
                            first_result = (cliques, runtime)
                            break
                    
                    if first_result:
                        maximal_cliques, runtime = first_result
                    else:
                        maximal_cliques, runtime = [], 0.0
                        print("   No oracles succeeded")
                    
                else:
                    # Single oracle mode
                    start_time = time.time()
                    
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
                    
                    maximal_cliques, _ = find_maximal_cliques_motzkin_straus(
                        graph,
                        oracle_type=args.oracle,
                        support_threshold=args.threshold,
                        verbose=args.verbose,
                        enable_refinement=args.enable_refinement,
                        **oracle_config
                    )
                    
                    runtime = time.time() - start_time
                    
                    print(f"   Motzkin-Straus: {len(maximal_cliques)} cliques, Runtime: {runtime:.3f}s")
                
                success_rate = 1.0
                if args.compare_networkx:
                    try:
                        nx_cliques = list(nx.find_cliques(graph))
                        success_rate, missing, extra = analyze_clique_coverage(maximal_cliques, nx_cliques)
                        print(f"   NetworkX: {len(nx_cliques)} cliques")
                        print(f"   Success rate: {success_rate:.1%}")
                        
                        if missing and args.verbose:
                            print(f"   Missing cliques: {len(missing)}")
                        if extra and args.verbose:
                            print(f"   Extra cliques: {len(extra)}")
                            
                    except Exception as e:
                        print(f"   NetworkX comparison failed: {e}")
                
                total_success += success_rate
                
                # Generate plot if requested
                if args.plot:
                    plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')}.png")
                    plot_clique_instances(graph, maximal_cliques, name, plot_path)
                
            except Exception as e:
                print(f"   Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\nOverall Success Rate: {total_success/total_graphs:.1%}")
    
    else:
        print("Maximal Clique Finder using Motzkin-Straus Theorem")
        print("Run with --test to test on predefined graphs")
        print("Run with --erdos-test to test on random graphs")
        print("Run with --help to see all options")


if __name__ == "__main__":
    main()