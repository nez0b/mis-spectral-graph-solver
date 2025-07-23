#!/usr/bin/env python3
"""
Regularized Clique Finder using Regularized Motzkin-Straus Theorem with Multiple Oracle Solvers

Theory: 
- Uses the regularized Motzkin-Straus theorem: max_{x ∈ Δ} x^T (A + cI) x
- Standard formulation: x^T A x → Regularized formulation: x^T (A + cI) x  
- Regularization eliminates spurious solutions (non-clique local optima)
- Ensures one-to-one correspondence between optima and cliques
- Makes optimization landscape strictly concave for more robust convergence
- Focuses on efficiently generating large cliques (not necessarily maximal or optimal)

Implementation:
- Apply identity regularization A → A + cI before optimization
- Support multiple oracle solvers: JAX-PGD (local optimization) and Dirac-3 (quantum annealing)
- Extract support from solution vectors (indices where x_i > threshold)
- Verify each support forms a valid clique
- Return collection of unique cliques with enhanced convergence properties

Oracle Solvers:
- JAX-PGD: Projected Gradient Descent with multiple restarts for local optimization
- Dirac-3: Quantum annealing solver via QCI cloud service

Regularization Benefits:
- Eliminates spurious solutions and improves convergence to true cliques
- Parameter c controls regularization strength (default: 0.1)
- c=0 reduces to standard Motzkin-Straus formulation
- Larger c values provide stronger regularization but may affect clique size

Usage Examples:

    # Test with regularized JAX-PGD oracle (default c=0.1)
    python scripts/clique_instance_reg.py --test --oracle jax-pgd --compare-networkx
    
    # Test with custom regularization parameter
    python scripts/clique_instance_reg.py --test --regularization-c 0.5
    
    # Test with Dirac-3 oracle and strong regularization
    python scripts/clique_instance_reg.py --test --oracle dirac --regularization-c 1.0 --num-samples 100
    
    # Compare different oracles with regularization
    python scripts/clique_instance_reg.py --test --compare-oracles --regularization-c 0.3 --plot
    
    # Test Erdős-Rényi graphs with regularized Dirac
    python scripts/clique_instance_reg.py --erdos-test --nodes 10 --oracle dirac --regularization-c 0.2
    
    # JAX-PGD with custom regularization and parameters
    python scripts/clique_instance_reg.py --test --oracle jax-pgd --regularization-c 0.8 --num-restarts 100
    
    # No regularization (standard Motzkin-Straus)
    python scripts/clique_instance_reg.py --test --regularization-c 0.0
    
    # Strong regularization for difficult graphs
    python scripts/clique_instance_reg.py --erdos-test --regularization-c 2.0 --oracle dirac --plot

Parameters:
    --test              Run tests on predefined graphs
    --erdos-test        Test on Erdős-Rényi random graphs
    --oracle TYPE       Oracle solver: 'jax-pgd' or 'dirac' (default: jax-pgd)
    --compare-oracles   Compare multiple oracles on same graphs
    --regularization-c C Regularization parameter for A → A + cI (default: 0.1)
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

Regularization Options:
    --regularization-c C  Regularization parameter for identity regularization (default: 0.1)

Performance Considerations:
- Regularization improves convergence reliability and eliminates spurious solutions
- JAX-PGD: Larger graphs require more restarts, local optimization approach
- Dirac-3: Quantum annealing, good for finding global optima, requires QCI account  
- Regularization parameter c: Higher values provide stronger regularization
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
    print("Oracle system not available - falling back to legacy implementations")
    OracleFactory = None
except Exception as e:
    print(f"Oracle system failed to load: {e}")
    print("Oracle system not available - falling back to legacy implementations")
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

# Import regularization functions from regularized_graph_to_omega.py
try:
    # Add scripts directory to path for importing regularization functions
    scripts_dir = os.path.dirname(__file__)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    from regularized_graph_to_omega import (
        apply_identity_regularization, 
        validate_regularization_parameter,
        qplib_to_polynomial_file
    )
    print("Successfully imported regularization functions")
    REGULARIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Error importing regularization functions: {e}")
    print("Regularization functionality not available - falling back to standard Motzkin-Straus")
    REGULARIZATION_AVAILABLE = False

# Import DIMACS to QPLIB conversion and Dirac submission functions
try:
    from dimacs_to_qplib import dimacs_to_qplib
    from graph_to_omega import submit_to_dirac, extract_best_energy, energy_to_omega
    from regularized_graph_to_omega import regularized_energy_to_omega
    print("Successfully imported DIMACS to QPLIB and Dirac submission functions")
    DIRAC_SUBMISSION_AVAILABLE = True
except ImportError as e:
    print(f"Error importing Dirac submission functions: {e}")
    print("Dirac submission functionality not available")
    DIRAC_SUBMISSION_AVAILABLE = False


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
    regularization_c: float = 0.1,
    **oracle_config
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Find cliques using the regularized Motzkin-Straus theorem with configurable oracle solvers.
    
    This function applies identity regularization (A → A + cI) to the adjacency matrix
    before optimization, which eliminates spurious solutions and ensures better 
    convergence to true cliques. Each local maximum corresponds to a clique.
    
    Args:
        graph: Input NetworkX graph
        oracle_type: Type of oracle solver ('jax-pgd', 'dirac')
        support_threshold: Threshold for extracting support from solution vectors
        verbose: Whether to print detailed progress
        enable_refinement: Whether to enable superposition refinement (default: True)
        regularization_c: Regularization parameter for A → A + cI (default: 0.1)
        **oracle_config: Oracle-specific configuration parameters
        
    Returns:
        Tuple of (cliques, optimization_details) where:
        - cliques: List of sets, each containing vertices of a clique
        - optimization_details: Dictionary with oracle-specific optimization information
    """
    if graph.number_of_nodes() == 0:
        return [], {"message": "Empty graph"}
    
    # Validate and apply regularization parameter
    try:
        if REGULARIZATION_AVAILABLE:
            regularization_c = validate_regularization_parameter(regularization_c)
        if verbose and regularization_c != 0.0:
            print(f"🔧 Applying regularization: A → A + {regularization_c}I")
        elif verbose and regularization_c == 0.0:
            print("🔧 No regularization applied (c = 0.0, standard Motzkin-Straus)")
    except Exception as e:
        if verbose:
            print(f"Warning: Regularization parameter validation failed: {e}")
        print("Using default regularization_c = 0.1")
        regularization_c = 0.1
    
    # Use oracle system if available, otherwise fallback to legacy implementation
    if OracleFactory is None:
        if verbose:
            print("Oracle system not available - using legacy JAX-PGD implementation")
        return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, regularization_c, **oracle_config)
    
    # Create oracle adapter or use regularized implementations
    try:
        # Handle regularized cases with custom implementations
        if regularization_c != 0.0:
            if oracle_type == 'dirac':
                if DIRAC_SUBMISSION_AVAILABLE:
                    if verbose:
                        print("🔄 Using regularized Dirac implementation for regularization support")
                    return _regularized_dirac_implementation(graph, support_threshold, verbose, regularization_c, **oracle_config)
                else:
                    if verbose:
                        print("⚠️  Dirac submission not available, falling back to regularized JAX-PGD")
                    return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, regularization_c, **oracle_config)
            else:
                # oracle_type is 'jax-pgd' or other - use JAX-PGD implementation
                if verbose:
                    print("🔄 Using regularized JAX-PGD implementation for regularization support")
                return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, regularization_c, **oracle_config)
        
        # Use standard oracle system for c=0 (no regularization)
        oracle = OracleFactory.create_oracle(oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config)
        
        if verbose:
            print(f"Using {oracle.name} oracle (no regularization)")
            print(f"Finding cliques in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Find maximal cliques using the oracle
        maximal_cliques = oracle.find_maximal_cliques(graph, support_threshold)
        
        # Get optimization details
        optimization_details = oracle.get_optimization_details()
        
        if verbose:
            print(f"Found {len(maximal_cliques)} unique cliques")
        
        return maximal_cliques, optimization_details
        
    except Exception as e:
        if verbose:
            print(f"Oracle {oracle_type} failed: {e}")
            print("Falling back to regularized JAX-PGD implementation")
        return _legacy_jax_pgd_implementation(graph, support_threshold, verbose, regularization_c, **oracle_config)


def _graph_to_qplib_standard(graph: nx.Graph) -> Dict[str, Any]:
    """
    Convert NetworkX graph to QPLIB format using standard adjacency matrix approach.
    
    This function converts the graph to the standard Motzkin-Straus QPLIB format
    where each edge (i,j) becomes a polynomial term with coefficient 1.0.
    This is different from the Gibbons matrix approach used in the oracle system.
    
    IMPORTANT: Ensures node indices are contiguous starting from 0 for QCI compatibility.
    
    Args:
        graph: NetworkX graph to convert
        
    Returns:
        QPLIB data dictionary with poly_indices, poly_coefficients, sum_constraint
    """
    if graph.number_of_nodes() == 0:
        return {
            "poly_indices": [],
            "poly_coefficients": [],
            "sum_constraint": 1
        }
    
    # Create mapping from original node IDs to contiguous 0-based indices
    # This is critical for QCI API which expects 0-based contiguous indexing
    node_list = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
    # Convert edges to polynomial format with remapped indices
    poly_indices = []
    poly_coefficients = []
    
    # For each edge, add it as a polynomial term with coefficient 1.0
    for u, v in graph.edges():
        # Map to 0-based contiguous indices, then convert to 1-based for QCI API
        u_idx = node_to_index[u] + 1  # Convert to 1-indexed for QCI compatibility
        v_idx = node_to_index[v] + 1  # Convert to 1-indexed for QCI compatibility
        
        # Ensure consistent ordering for undirected edges
        if u_idx <= v_idx:
            poly_indices.append([u_idx, v_idx])
            poly_coefficients.append(1.0)
        else:
            poly_indices.append([v_idx, u_idx])
            poly_coefficients.append(1.0)
    
    # Include self-loops if present (diagonal terms)
    for node in graph.nodes():
        if graph.has_edge(node, node):
            node_idx = node_to_index[node] + 1  # Convert to 1-indexed for QCI compatibility
            poly_indices.append([node_idx, node_idx])
            poly_coefficients.append(1.0)
    
    return {
        "poly_indices": poly_indices,
        "poly_coefficients": poly_coefficients,
        "sum_constraint": 1
    }


def _qplib_to_polynomial_file_fixed_degrees(qplib_data: Dict[str, Any], file_name: str = "regularized_qplib_optimization", verbose: bool = False) -> Dict[str, Any]:
    """
    Transform regularized QPLIB data to QCI polynomial file format with proper degree handling.
    
    This is a fixed version of qplib_to_polynomial_file that correctly handles mixed degrees
    (both degree 1 and degree 2 terms) by calculating min/max degrees properly.
    
    CRITICAL: This function converts a MAXIMIZATION problem (regularized Motzkin-Straus) to 
    MINIMIZATION format (Dirac solver) by negating all coefficients.
    
    Args:
        qplib_data: Regularized QPLIB data dictionary with positive coefficients
        file_name: Name for the polynomial file
        
    Returns:
        QCI polynomial file configuration dictionary with negated coefficients
        
    Raises:
        ValueError: If QPLIB data is invalid
    """
    from collections import Counter
    
    try:
        poly_indices = qplib_data['poly_indices']
        poly_coefficients = qplib_data['poly_coefficients']
        
        if len(poly_indices) != len(poly_coefficients):
            raise ValueError("poly_indices and poly_coefficients must have same length")
        
        # Calculate number of variables and degrees properly
        all_indices = np.array(poly_indices).flatten()
        if len(all_indices) == 0:
            raise ValueError("Empty polynomial data")
        
        ind_dict = Counter(all_indices.tolist())
        num_vars = int(max(all_indices)) if len(all_indices) > 0 else 0
        
        # Calculate min and max degrees correctly by examining each term
        degrees = []
        for idx in poly_indices:
            if isinstance(idx, (list, tuple)):
                # Degree is the LENGTH of the index list, not unique indices
                # [0, 0] has degree 2 (x_0^2), [0, 1] has degree 2 (x_0 * x_1)
                degrees.append(len(idx))
            else:
                # Single index means degree 1
                degrees.append(1)
        
        min_degree = min(degrees) if degrees else 2
        max_degree = max(degrees) if degrees else 2
        
        if verbose:
            print(f"Polynomial structure: {num_vars} variables, degree {min_degree}-{max_degree}")
            print(f"Term degrees: {Counter(degrees)}")
        
        # Transform to QCI format: [{"idx": [i,j], "val": coefficient}]
        # Ensure all values are native Python types (not numpy types)
        data = []
        for idx, val in zip(poly_indices, poly_coefficients):
            # Convert indices to native Python ints and coefficients to native Python floats
            if isinstance(idx, (list, tuple)):
                idx_converted = [int(i) for i in idx]
            else:
                idx_converted = int(idx)
            
            val_converted = -float(val)  # Negate for maximization->minimization conversion
            data.append({"idx": idx_converted, "val": val_converted})
        
        # Create QCI polynomial file configuration
        file_config = {
            "file_name": file_name,
            "file_config": {
                "polynomial": {
                    "num_variables": int(num_vars),
                    "min_degree": int(min_degree),
                    "max_degree": int(max_degree),
                    "data": data
                }
            }
        }
        
        if verbose:
            print(f"Created QCI polynomial file config with {len(data)} terms (degrees {min_degree}-{max_degree})")
        
        return file_config
        
    except Exception as e:
        raise ValueError(f"Failed to convert QPLIB data to polynomial file: {e}")


def debug_reconstruct_matrix(qplib_data: Dict[str, Any], graph: nx.Graph = None, title: str = "Matrix") -> np.ndarray:
    """
    Reconstruct full matrix from QPLIB polynomial data for debugging.
    
    This function converts the polynomial representation back to matrix form
    to verify that the objective matrix is constructed correctly.
    
    Args:
        qplib_data: QPLIB data with poly_indices and poly_coefficients
        graph: Optional original graph for validation
        title: Title for debugging output
        
    Returns:
        Reconstructed matrix as numpy array
    """
    poly_indices = qplib_data['poly_indices']
    poly_coefficients = qplib_data['poly_coefficients']
    
    if not poly_indices:
        print(f"❌ {title}: No polynomial terms found")
        return np.array([])
    
    # Find matrix dimensions
    all_indices = np.array(poly_indices).flatten()
    max_index = int(max(all_indices))
    min_index = int(min(all_indices))
    
    # Handle 1-indexed variables: if min_index is 1, we have 1-indexed variables
    if min_index == 1:
        # 1-indexed variables: matrix size is max_index, indices range [1, max_index]
        n = max_index
        index_offset = 1
    else:
        # 0-indexed variables: matrix size is max_index + 1, indices range [0, max_index]  
        n = max_index + 1
        index_offset = 0
    
    # Initialize matrix
    matrix = np.zeros((n, n))
    
    # Fill matrix from polynomial terms
    for indices, coeff in zip(poly_indices, poly_coefficients):
        if len(indices) == 2:
            # Convert from QPLIB indices to matrix indices
            i, j = int(indices[0]) - index_offset, int(indices[1]) - index_offset
            
            # Ensure indices are within bounds
            if 0 <= i < n and 0 <= j < n:
                if i == j:
                    # Diagonal term
                    matrix[i, j] = coeff
                else:
                    # Off-diagonal term (symmetric for undirected graph)
                    matrix[i, j] = coeff
                    matrix[j, i] = coeff
    
    print(f"\n🔍 {title} Reconstruction:")
    print(f"   Dimensions: {n}x{n}")
    print(f"   Polynomial terms: {len(poly_indices)}")
    print(f"   Matrix:\n{matrix}")
    
    # Check symmetry for undirected graphs
    is_symmetric = np.allclose(matrix, matrix.T)
    print(f"   Symmetric: {is_symmetric}")
    
    # Check diagonal values
    diagonal = np.diag(matrix)
    print(f"   Diagonal: {diagonal}")
    
    # If graph provided, validate structure
    if graph is not None:
        print(f"   Graph validation:")
        print(f"     Original nodes: {sorted(graph.nodes())}")
        print(f"     Original edges: {len(graph.edges())}")
        
        # Check that edges match
        edge_count_matrix = 0
        for i in range(n):
            for j in range(i+1, n):  # Upper triangular
                if matrix[i, j] != 0 and i != j:
                    edge_count_matrix += 1
        
        print(f"     Matrix edges: {edge_count_matrix}")
        print(f"     Edge count match: {edge_count_matrix == graph.number_of_edges()}")
    
    return matrix


def debug_inspect_solutions(
    solution_vectors: List[np.ndarray], 
    graph: nx.Graph, 
    threshold: float = 1e-5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Detailed analysis of Dirac solution vectors for debugging.
    
    Args:
        solution_vectors: List of solution vectors from Dirac
        graph: Original graph for clique verification
        threshold: Support extraction threshold
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print(f"\n🔍 Solution Vector Analysis:")
        print(f"   Number of solutions: {len(solution_vectors)}")
        print(f"   Support threshold: {threshold}")
    
    analysis = {
        "num_solutions": len(solution_vectors),
        "threshold": threshold,
        "solutions_analysis": [],
        "valid_cliques": [],
        "threshold_sensitivity": {}
    }
    
    # Create node mapping (matches _graph_to_qplib_standard)
    original_node_list = sorted(graph.nodes())
    index_to_node = {idx: node for idx, node in enumerate(original_node_list)}
    
    for i, solution_vector in enumerate(solution_vectors):
        sol_analysis = {
            "index": i,
            "vector": solution_vector.tolist(),
            "sum": float(np.sum(solution_vector)),
            "max": float(np.max(solution_vector)),
            "min": float(np.min(solution_vector)),
            "std": float(np.std(solution_vector))
        }
        
        if verbose and i < 5:  # Show details for first 5 solutions
            print(f"\n   Solution {i+1}:")
            print(f"     Vector: {solution_vector}")
            print(f"     Sum: {sol_analysis['sum']:.6f}")
            print(f"     Max: {sol_analysis['max']:.6f}")
            print(f"     Non-zero entries: {np.sum(solution_vector > 1e-10)}")
        
        # Extract support with current threshold
        support_indices = extract_support(solution_vector, threshold)
        candidate_clique = {index_to_node[idx + 1] for idx in support_indices if (idx + 1) in index_to_node}
        
        sol_analysis["support_indices"] = list(support_indices)
        sol_analysis["candidate_clique"] = list(candidate_clique)
        sol_analysis["candidate_size"] = len(candidate_clique)
        
        if verbose and i < 5:
            print(f"     Support indices: {list(support_indices)}")
            print(f"     Candidate clique: {sorted(candidate_clique)} (size: {len(candidate_clique)})")
        
        # Verify if it's a valid clique
        is_valid_clique = verify_clique(graph, candidate_clique) if candidate_clique else False
        sol_analysis["is_valid_clique"] = is_valid_clique
        
        if is_valid_clique:
            analysis["valid_cliques"].append(candidate_clique)
            if verbose and i < 5:
                print(f"     ✅ Valid clique!")
        elif verbose and i < 5:
            print(f"     ❌ Not a valid clique")
            
            # Debug why it's not a clique
            if candidate_clique:
                missing_edges = []
                nodes = list(candidate_clique)
                for u_idx, u in enumerate(nodes):
                    for v_idx, v in enumerate(nodes[u_idx+1:], u_idx+1):
                        if not graph.has_edge(u, v):
                            missing_edges.append((u, v))
                
                if missing_edges:
                    print(f"     Missing edges: {missing_edges[:3]}{'...' if len(missing_edges) > 3 else ''}")
        
        analysis["solutions_analysis"].append(sol_analysis)
    
    # Test different thresholds for sensitivity analysis
    test_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    if verbose:
        print(f"\n   Threshold Sensitivity Analysis:")
    
    for test_thresh in test_thresholds:
        valid_cliques_at_thresh = set()
        support_sizes = []
        
        for solution_vector in solution_vectors:
            support_indices = extract_support(solution_vector, test_thresh)
            support_sizes.append(len(support_indices))
            
            if support_indices:
                candidate_clique = {index_to_node[idx + 1] for idx in support_indices if (idx + 1) in index_to_node}
                if verify_clique(graph, candidate_clique):
                    valid_cliques_at_thresh.add(frozenset(candidate_clique))
        
        avg_support_size = np.mean(support_sizes) if support_sizes else 0
        
        analysis["threshold_sensitivity"][test_thresh] = {
            "avg_support_size": float(avg_support_size),
            "valid_cliques": len(valid_cliques_at_thresh),
            "unique_cliques": [set(clique) for clique in valid_cliques_at_thresh]
        }
        
        if verbose:
            marker = " <-- current" if abs(test_thresh - threshold) < 1e-9 else ""
            print(f"     {test_thresh}: avg_support={avg_support_size:.1f}, valid_cliques={len(valid_cliques_at_thresh)}{marker}")
    
    return analysis


def debug_test_small_cases():
    """
    Test known small cases with expected results.
    
    Tests triangle, K4, and path graphs to verify correctness.
    """
    print("\n🧪 Testing Small Cases with Expected Results")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Triangle (K3)",
            "graph": nx.complete_graph(3),
            "expected_clique": {0, 1, 2},
            "expected_clique_size": 3
        },
        {
            "name": "Complete K4", 
            "graph": nx.complete_graph(4),
            "expected_clique": {0, 1, 2, 3},
            "expected_clique_size": 4
        },
        {
            "name": "Path P3",
            "graph": nx.path_graph(3),
            "expected_clique_size": 2  # Any pair of adjacent nodes
        }
    ]
    
    for case in test_cases:
        print(f"\n🔍 Testing {case['name']}:")
        graph = case["graph"]
        
        print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"   Edges: {list(graph.edges())}")
        
        # Test matrix construction
        qplib_data = _graph_to_qplib_standard(graph)
        print(f"   QPLIB terms: {len(qplib_data['poly_indices'])}")
        
        # Reconstruct and display adjacency matrix
        adj_matrix = debug_reconstruct_matrix(qplib_data, graph, "Adjacency Matrix")
        
        # Test regularization
        for c in [0.0, 0.1, 0.5]:
            print(f"\n   Regularization c = {c}:")
            
            if c == 0.0:
                reg_data = qplib_data
            else:
                reg_data = apply_identity_regularization_fixed(qplib_data, c)
            
            # Reconstruct regularized matrix
            reg_matrix = debug_reconstruct_matrix(reg_data, graph, f"Regularized Matrix (c={c})")
            
            # Verify A + cI structure
            if c > 0.0:
                expected_diagonal = c
                actual_diagonal = np.diag(reg_matrix)
                print(f"   Expected diagonal: {expected_diagonal}")
                print(f"   Actual diagonal: {actual_diagonal}")
                print(f"   Diagonal correct: {np.allclose(actual_diagonal, expected_diagonal)}")


def apply_identity_regularization_fixed(qplib_data: Dict[str, Any], c: float) -> Dict[str, Any]:
    """
    Apply identity regularization to QPLIB data: A → A + cI.
    
    FIXED VERSION: Handles 0-indexed variables correctly (unlike the original 1-indexed version).
    
    This transforms the Motzkin-Straus objective from x^T A x to x^T (A + cI) x,
    which eliminates spurious solutions and makes the optimization landscape
    strictly concave.
    
    Args:
        qplib_data: Original QPLIB data dictionary with 0-indexed variables
        c: Regularization parameter (typically 0.1, 0.5, or 1.0)
        
    Returns:
        Regularized QPLIB data with identity matrix added to adjacency matrix
    """
    # Special case: c = 0 means no regularization, return original data
    if c == 0.0:
        print(f"No regularization applied: c = 0.0 (standard Motzkin-Straus)")
        return qplib_data
    
    # Extract original data
    poly_indices = qplib_data['poly_indices']
    poly_coefficients = qplib_data['poly_coefficients']
    
    # Find the number of variables (0-indexed)
    all_indices = np.array(poly_indices).flatten()
    if len(all_indices) == 0:
        # No edges in graph, just return original data
        print(f"WARNING: No edges found, regularization has no effect")
        return qplib_data
    
    max_index = int(max(all_indices))
    min_index = int(min(all_indices))
    
    # For 1-indexed variables, num_vars is just max_index (not max_index + 1)
    if min_index == 1:
        # Variables are 1-indexed: [1, 2, 3], so num_vars = max_index = 3
        num_vars = max_index  
    else:
        # Variables are 0-indexed: [0, 1, 2], so num_vars = max_index + 1 = 3
        num_vars = max_index + 1
    
    # Create regularized polynomial terms
    regularized_indices = list(poly_indices)  # Copy original terms
    regularized_coefficients = list(poly_coefficients)  # Copy original coefficients
    
    # Add diagonal terms for regularization: c * x_i^2 for each variable
    diagonal_terms_added = 0
    if min_index == 1:
        # 1-indexed variables: add diagonal for variables 1, 2, 3, ..., num_vars
        var_range = range(1, num_vars + 1)
    else:
        # 0-indexed variables: add diagonal for variables 0, 1, 2, ..., num_vars-1
        var_range = range(0, num_vars)
    
    for i in var_range:
        # Check if diagonal term already exists
        diagonal_exists = any(
            len(idx) == 2 and idx[0] == i and idx[1] == i 
            for idx in poly_indices
        )
        
        if diagonal_exists:
            # Update existing diagonal term to c (replace, don't add)
            # For regularized formulation A + cI, diagonal should be c, not 1.0 + c
            for j, idx in enumerate(regularized_indices):
                if len(idx) == 2 and idx[0] == i and idx[1] == i:
                    regularized_coefficients[j] = c  # Set to c, not add c
                    diagonal_terms_added += 1
                    break
        else:
            # Add new diagonal term with coefficient c
            regularized_indices.append([i, i])
            regularized_coefficients.append(c)
            diagonal_terms_added += 1
    
    # Create regularized QPLIB data
    regularized_data = qplib_data.copy()
    regularized_data['poly_indices'] = regularized_indices
    regularized_data['poly_coefficients'] = regularized_coefficients
    
    print(f"Applied identity regularization: c = {c}")
    print(f"   Added/updated {diagonal_terms_added} diagonal terms")
    print(f"   Total polynomial terms: {len(regularized_indices)} (was {len(poly_indices)})")
    
    return regularized_data


def _regularized_dirac_implementation(
    graph: nx.Graph,
    support_threshold: float = 1e-5,
    verbose: bool = False,
    regularization_c: float = 0.1,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    solution_precision: Optional[float] = None,
    job_timeout: int = 300,
    **kwargs
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Regularized Dirac implementation using the approach from regularized_graph_to_omega.py.
    
    This function applies regularization A → A + cI at the QPLIB level and submits
    to Dirac-3 using the proven workflow from regularized_graph_to_omega.py.
    
    Args:
        graph: NetworkX graph to analyze
        support_threshold: Threshold for extracting support from solutions
        verbose: Whether to print detailed progress
        regularization_c: Regularization parameter for A → A + cI
        num_samples: Number of Dirac samples
        relaxation_schedule: Relaxation schedule 1-4
        solution_precision: Solution precision (optional)
        job_timeout: Job timeout in seconds
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Tuple of (cliques, optimization_details)
    """
    if not DIRAC_SUBMISSION_AVAILABLE:
        raise RuntimeError("Dirac submission functions not available. Cannot use regularized Dirac implementation.")
    
    if verbose:
        print(f"Regularized Dirac: Finding cliques with regularization c={regularization_c}")
        print(f"Dirac parameters: samples={num_samples}, schedule={relaxation_schedule}")
    
    try:
        # Step 1: Convert graph to standard QPLIB format
        qplib_data = _graph_to_qplib_standard(graph)
        
        if not qplib_data['poly_indices']:
            if verbose:
                print("No polynomial terms - empty graph or no edges")
            return [], {"message": "No edges"}
        
        # DEBUG: Show original adjacency matrix
        if verbose:
            print(f"\n🔍 Matrix Debugging:")
            adj_matrix = debug_reconstruct_matrix(qplib_data, graph, "Original Adjacency Matrix")
        
        # Step 2: Apply regularization A → A + cI
        if regularization_c != 0.0:
            regularized_data = apply_identity_regularization_fixed(qplib_data, regularization_c)
            if verbose:
                print(f"✅ Applied identity regularization: A + {regularization_c}I")
                # DEBUG: Show regularized matrix
                reg_matrix = debug_reconstruct_matrix(regularized_data, graph, f"Regularized Matrix (c={regularization_c})")
        else:
            regularized_data = qplib_data
            if verbose:
                print("✅ No regularization applied (c = 0.0, standard Motzkin-Straus)")
        
        # Step 3: Transform to QCI polynomial file format (with coefficient negation)
        file_name = f"clique_reg_c{regularization_c}_n{graph.number_of_nodes()}_{int(time.time())}"
        polynomial_file = _qplib_to_polynomial_file_fixed_degrees(regularized_data, file_name, verbose)
        
        if verbose:
            print(f"✅ Created QCI polynomial file with {len(regularized_data['poly_indices'])} terms")
        
        # Step 4: Submit to Dirac-3
        job_name = f"clique_reg_c{regularization_c}_{int(time.time())}"
        
        job_response = submit_to_dirac(
            polynomial_file=polynomial_file,
            job_name=job_name,
            num_samples=num_samples,
            relaxation_schedule=relaxation_schedule,
            solution_precision=solution_precision,
            sum_constraint=1,  # Always 1 for Motzkin-Straus
            wait=True,
            job_tags=['regularized_clique', 'motzkin_straus']
        )
        
        if verbose:
            print(f"✅ Dirac job completed: {job_response.get('status', 'unknown')}")
        
        # Step 5: Extract energies and solutions
        best_energy, all_energies, best_solution = extract_best_energy(job_response)
        
        # For regularized Motzkin-Straus, use the correct formula: ω = (1-2c)/(1 + 2*energy)
        theoretical_omega = regularized_energy_to_omega(best_energy, regularization_c)
        
        # Extract all solution vectors
        results = job_response.get('results', {})
        solutions = results.get('solutions', [])
        solution_vectors = [np.array(sol) for sol in solutions]
        
        if verbose:
            print(f"✅ Best energy: {best_energy:.6f}, theoretical omega: {theoretical_omega:.3f}")
            print(f"✅ Processing {len(solution_vectors)} solution vectors")
        
        # DEBUG: Detailed solution analysis
        if verbose:
            analysis = debug_inspect_solutions(solution_vectors, graph, support_threshold, verbose=True)
        
        # Step 6: Extract cliques from solution vectors
        found_cliques = set()  # Use set of frozensets for deduplication
        
        # Create node mapping back to original IDs (matches _graph_to_qplib_standard)
        # NOTE: QPLIB variables are 1-indexed, so we need to map 1-based indices to original nodes
        original_node_list = sorted(graph.nodes())
        index_to_node = {idx + 1: node for idx, node in enumerate(original_node_list)}
        
        for i, solution_vector in enumerate(solution_vectors):
            if verbose and i < 3:  # Show details for first few solutions
                print(f"  Solution {i+1}: sum={np.sum(solution_vector):.6f}, max={np.max(solution_vector):.6f}")
            
            # Extract support (candidate clique vertices)
            support_indices = extract_support(solution_vector, support_threshold)
            
            if not support_indices:
                continue
            
            # Map indices back to actual node IDs using the same mapping as graph conversion
            # NOTE: support_indices are 0-indexed array positions, but QPLIB variables are 1-indexed
            candidate_clique = {index_to_node[idx + 1] for idx in support_indices if (idx + 1) in index_to_node}
            
            if verbose and i < 3:
                print(f"  Candidate clique: {sorted(candidate_clique)} (size: {len(candidate_clique)})")
            
            # Verify it's actually a clique
            if verify_clique(graph, candidate_clique):
                clique_frozen = frozenset(candidate_clique)
                if clique_frozen not in found_cliques:
                    found_cliques.add(clique_frozen)
                    if verbose:
                        print(f"  ✅ Found valid clique: {sorted(candidate_clique)}")
        
        result_cliques = [set(clique) for clique in found_cliques]
        
        # Prepare optimization details
        details = {
            "oracle_type": "regularized-dirac",
            "regularization_c": regularization_c,
            "num_samples": num_samples,
            "relaxation_schedule": relaxation_schedule,
            "best_energy": best_energy,
            "theoretical_omega": theoretical_omega,
            "solutions_processed": len(solution_vectors),
            "unique_cliques_found": len(result_cliques),
            "job_status": job_response.get('status', 'unknown')
        }
        
        if verbose:
            print(f"✅ Regularized Dirac completed: found {len(result_cliques)} unique cliques")
        
        return result_cliques, details
        
    except Exception as e:
        if verbose:
            print(f"❌ Regularized Dirac implementation failed: {e}")
        raise RuntimeError(f"Regularized Dirac optimization failed: {e}")


def _legacy_jax_pgd_implementation(
    graph: nx.Graph, 
    support_threshold: float = 1e-5, 
    verbose: bool = False,
    regularization_c: float = 0.1,
    num_restarts: int = 50,
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    **kwargs
) -> Tuple[List[Set[int]], Dict[str, Any]]:
    """
    Regularized JAX-PGD implementation with identity regularization support.
    
    Applies regularization A → A + cI to the adjacency matrix before optimization.
    """
    if verbose:
        if regularization_c != 0.0:
            print(f"Regularized JAX-PGD: Finding cliques with {num_restarts} restarts (c={regularization_c})")
        else:
            print(f"Standard JAX-PGD: Finding cliques with {num_restarts} restarts (no regularization)")
    
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
    
    # Apply regularization: A → A + cI
    if regularization_c != 0.0:
        regularized_matrix = adj_matrix + regularization_c * np.eye(len(node_list))
        if verbose:
            print(f"✅ Applied identity regularization: A + {regularization_c}I")
    else:
        regularized_matrix = adj_matrix
        if verbose:
            print("✅ Using standard adjacency matrix (no regularization)")
    
    # Convert regularized matrix to polynomial format
    poly_indices, poly_coefficients = adjacency_to_polynomial(regularized_matrix)
    
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
    regularization_c: float = 0.1,
    **base_config
) -> Dict[str, Tuple[List[Set[int]], Dict[str, Any], float]]:
    """
    Compare multiple oracles on the same graph with regularization.
    
    Args:
        graph: Input graph to analyze
        oracle_types: List of oracle types to compare
        support_threshold: Support extraction threshold
        verbose: Whether to print detailed progress
        regularization_c: Regularization parameter for A → A + cI
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
                regularization_c=regularization_c,
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
        description="Find cliques using regularized Motzkin-Straus theorem with multiple oracle solvers",
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
    parser.add_argument("--regularization-c", type=float, default=0.1,
                       help="Regularization parameter for A → A + cI (default: 0.1)")
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
                    graph, oracle_types, args.threshold, args.verbose, args.regularization_c, **base_config
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
                        regularization_c=args.regularization_c,
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
                        graph, oracle_types, args.threshold, args.verbose, args.regularization_c, **base_config
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
                        regularization_c=args.regularization_c,
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