#!/usr/bin/env python3
"""
Quantum Column Generation Algorithm for Minimum Vertex Graph Coloring

This module implements the hybrid quantum-classical column generation algorithm 
described in "Quantum-enhanced column generation for the antenna frequency assignment problem" 
by Wesley da Silva Coelho et al. (arxiv:2301.02637).

The algorithm combines:
- Classical Master Problem: Linear programming for color assignment optimization  
- Quantum Pricing Subproblem: Maximum Weight Independent Set using Motzkin-Straus oracle

Key Components:
1. Classical/Quantum Column Generation Solvers: Main solver classes implementing the iterative algorithm
2. Three Pricing Subproblem Solvers: 
   - ClassicalPricingSubproblemSolver: Exact MILP-based PSP solver
   - QuantumPricingSubproblemSolver: Quantum sampling with classical post-processing  
   - MWISBasedPricingSubproblemSolver: Direct MWIS oracle integration
3. Integration with JAX-PGD and Dirac-3 quantum oracles
4. Paper-compliant negative weight filtering (V' = {u ∈ V | w_u > 0})
5. Adaptive 1/N support threshold for N-node graphs

Algorithm Flow:
1. Initialize with singleton independent sets (each vertex is a color)
2. Iterate until convergence:
   a. Solve Reduced Master Problem (RMP) using linear programming
   b. Extract dual variables from RMP solution  
   c. Solve Pricing Subproblem (PSP) using quantum/classical MWIS oracle
   d. Add profitable columns (reduced cost < 0) to RMP
3. Solve final integer linear program for optimal coloring

Test Graphs Available (--test option):
- Triangle: 3-node complete graph (chromatic number = 3)
- Complete K4: 4-node complete graph (chromatic number = 4)  
- Path P4: 4-node path graph (chromatic number = 2)
- Cycle C5: 5-node cycle graph (chromatic number = 3)
- Wheel W5: 6-node wheel graph with center (chromatic number = 4)

Superposition Refinement:
When quantum oracles produce superposition states (mixed solutions), refinement
extracts multiple constituent independent sets using "Greedy Clique Peeling":
- Enabled (default): Better column discovery, finds more profitable columns
- Disabled: Faster execution, may miss columns from quantum superposition states

Usage:
    # Basic Tests:
    # Run tests on predefined graphs (Triangle, K4, Path P4, Cycle C5, Wheel W5)
    python scripts/column_gen.py --test --oracle jax-pgd --verbose
    
    # Test with Dirac oracle with refinement disabled for faster execution
    python scripts/column_gen.py --test --oracle dirac --disable-refinement
    
    # Test only classical column generation (no quantum oracle needed)
    python scripts/column_gen.py --test --classical-only --verbose
    
    # Random Graph Tests:
    # Test on Erdős-Rényi random graphs with custom parameters
    python scripts/column_gen.py --erdos-test --nodes 10 --edge-prob 0.5 --plot
    
    # Comparison Options:
    # Compare quantum vs classical column generation
    python scripts/column_gen.py --test --oracle jax-pgd --compare-classical --verbose
    
    # Compare different oracles (JAX-PGD vs Dirac)
    python scripts/column_gen.py --test --compare-oracles --verbose
    
    # Compare with NetworkX greedy coloring baseline
    python scripts/column_gen.py --test --oracle dirac --compare-networkx --plot
    
    # Oracle Configuration:
    # JAX-PGD with custom parameters
    python scripts/column_gen.py --test --oracle jax-pgd --num-restarts 100 --learning-rate 0.005 --max-iter 3000
    
    # Dirac-3 with custom quantum parameters
    python scripts/column_gen.py --test --oracle dirac --num-samples 200 --relax-schedule 3
    
    # Column Generation Parameters:
    # Custom convergence settings
    python scripts/column_gen.py --test --max-iterations 100 --tolerance 1e-8 --verbose
    
    # Superposition Refinement Control:
    # Enhanced column discovery with superposition refinement (default behavior)
    python scripts/column_gen.py --test --oracle dirac --enable-refinement --verbose
    
    # Faster execution with refinement disabled
    python scripts/column_gen.py --test --oracle dirac --disable-refinement --verbose
    
    # Complete Workflow Examples:
    # Full comparison with visualization
    python scripts/column_gen.py --erdos-test --nodes 12 --edge-prob 0.3 --compare-classical --compare-networkx --plot --save-plots ./results/
    
    # Performance analysis with multiple configurations
    python scripts/column_gen.py --test --compare-oracles --compare-classical --disable-refinement --verbose

Available Options:
    Main Options:
    --test                    Run tests on predefined graphs (Triangle, K4, Path P4, Cycle C5, Wheel W5)
    --erdos-test             Test on Erdős-Rényi random graphs  
    --oracle {jax-pgd,dirac} Oracle type for pricing subproblem (default: jax-pgd)
    --classical-only         Run only classical column generation
    --verbose, -v            Enable verbose output
    --plot                   Generate visualization plots
    --save-plots DIR         Directory to save plots (default: ./plots/)
    
    Comparison Options:
    --compare-oracles        Compare JAX-PGD vs Dirac oracles
    --compare-classical      Compare quantum vs classical column generation
    --compare-networkx       Compare with NetworkX greedy coloring baseline
    
    Column Generation Parameters:
    --max-iterations N       Maximum column generation iterations (default: 50)
    --tolerance FLOAT        Convergence tolerance (default: 1e-6)
    
    Erdős-Rényi Graph Parameters:
    --nodes N                Number of nodes (default: 8)
    --edge-prob FLOAT        Edge probability (default: 0.4)
    
    JAX-PGD Oracle Parameters:
    --num-restarts N         Number of restarts (default: 50)
    --learning-rate FLOAT    Learning rate (default: 0.01)
    --max-iter N             Maximum iterations per restart (default: 2000)
    
    Dirac-3 Oracle Parameters:
    --num-samples N          Number of samples (default: 100)
    --relax-schedule {1,2,3,4} Relaxation schedule (default: 2)
    
    Superposition Refinement Options:
    --enable-refinement      Enable superposition solution refinement (default: True)
    --disable-refinement     Disable superposition solution refinement for faster execution
"""

import sys
import os
import time
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Any, Optional, Tuple, FrozenSet
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import optimization libraries
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings("ignore")

# Import existing oracle system
try:
    from oracles import OracleFactory
    ORACLE_SYSTEM_AVAILABLE = True
    print("Oracle system successfully imported")
except (ImportError, AttributeError) as e:
    print(f"Warning: Oracle system not available: {e}")
    ORACLE_SYSTEM_AVAILABLE = False
    OracleFactory = None

# Import MWIS solver functions from scripts/mwis.py
try:
    # Try different import paths for mwis.py
    try:
        # First try relative import from same directory
        from mwis import (
            find_maximum_weight_independent_set,
            construct_gibbons_matrix,
            extract_support,
            verify_maximal_stable_set,
            solve_maximum_weight_independent_set_milp
        )
    except ImportError:
        # Try absolute import
        from scripts.mwis import (
            find_maximum_weight_independent_set,
            construct_gibbons_matrix,
            extract_support,
            verify_maximal_stable_set,
            solve_maximum_weight_independent_set_milp
        )
    MWIS_SOLVER_AVAILABLE = True
    print("MWIS solver successfully imported from scripts/mwis.py")
except (ImportError, AttributeError) as e:
    print(f"Warning: MWIS solver not available: {e}")
    MWIS_SOLVER_AVAILABLE = False

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available for plotting")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - plotting disabled")


def verify_independent_set(graph: nx.Graph, candidate_set: Set[int]) -> bool:
    """
    Verify that a set of vertices forms a valid independent set (stable set).
    
    An independent set is a set of vertices where no two vertices are adjacent.
    Unlike verify_maximal_stable_set, this function only checks validity, not maximality.
    
    Args:
        graph: The input graph
        candidate_set: Set of vertices to check
        
    Returns:
        True if the set is a valid independent set, False otherwise
    """
    if not candidate_set:
        return True  # Empty set is trivially independent
    
    # Verify it's an independent set (no edges between vertices in the set)
    for u in candidate_set:
        for v in candidate_set:
            if u != v and graph.has_edge(u, v):
                return False
    
    return True


class ClassicalPricingSubproblemSolver:
    """
    Classical Pricing Subproblem Solver using scipy MILP.
    
    This class implements the classical PSP for column generation by solving
    the Maximum Weight Independent Set problem using Mixed Integer Linear Programming.
    This provides an exact solution to the PSP, serving as a baseline for comparison
    with the quantum approach.
    
    Algorithm:
    1. Formulate MWIS as binary integer program with edge constraints
    2. Use scipy.optimize.milp for exact optimization  
    3. Return the optimal independent set if profitable (reduced cost < 0)
    """
    
    def __init__(self, verbose: bool = False, timeout: int = 60):
        """
        Initialize classical pricing subproblem solver.
        
        Args:
            verbose: Enable detailed logging
            timeout: Maximum solving time per PSP call in seconds
        """
        self.verbose = verbose
        self.timeout = timeout
        self.name = "Classical MILP PSP"
    
    def solve_pricing_subproblem(self, graph: nx.Graph, dual_weights: np.ndarray, 
                                support_threshold: float = 1e-5) -> List[Set[int]]:
        """
        Solve Pricing Subproblem using exact MILP formulation.
        
        Algorithm:
        1. Formulate MWIS as binary integer program:
           - Variables: x[v] ∈ {0,1} for each vertex v
           - Objective: maximize Σ dual_weights[v] * x[v] 
           - Constraints: x[u] + x[v] ≤ 1 for all edges (u,v)
        2. Check profitability: sum of dual weights > 1
        3. Return profitable independent set
        
        Args:
            graph: NetworkX graph 
            dual_weights: Array of dual variables from master problem
            support_threshold: Ignored in classical solver
            
        Returns:
            List containing single optimal independent set if profitable, empty list otherwise
        """
        if self.verbose:
            print(f"  Classical PSP: Solving MWIS using scipy MILP")
            print(f"  Dual weights: {dual_weights}")
        
        node_list = sorted(list(graph.nodes()))
        num_vertices = len(node_list)
        
        if num_vertices == 0:
            return []
        
        # Filter vertices according to the paper's specification:
        # V' = {u ∈ V | w_u > 0}
        # Only consider vertices with positive dual weights
        positive_weight_indices = [i for i, w in enumerate(dual_weights) if w > 0]
        
        if len(positive_weight_indices) == 0:
            if self.verbose:
                print(f"  No vertices with positive dual weights - returning empty set")
            return []
        
        # Create filtered node list and mapping
        filtered_nodes = [node_list[i] for i in positive_weight_indices]
        filtered_dual_weights = dual_weights[positive_weight_indices]
        filtered_node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}
        
        # Create subgraph with only positive-weight vertices
        filtered_graph = graph.subgraph(filtered_nodes)
        
        if self.verbose:
            print(f"  Original nodes: {len(node_list)}, Filtered nodes: {len(filtered_nodes)}")
            weighted_nodes = [(filtered_nodes[i], filtered_dual_weights[i]) for i in range(len(filtered_nodes))]
            print(f"  Node weights: {weighted_nodes}")
        
        try:
            # Objective: maximize sum of dual weights (minimize negative)
            c_psp = -filtered_dual_weights  # Maximize dual weights (minimize negative)
            
            # Variables: x[v] ∈ {0,1} for each filtered vertex v
            num_filtered = len(filtered_nodes)
            integrality_psp = np.ones(num_filtered, dtype=int)
            bounds_psp = ([0] * num_filtered, [1] * num_filtered)
            
            # Constraints: x[u] + x[v] ≤ 1 for each edge (u,v) in filtered graph
            edges = list(filtered_graph.edges())
            num_edges = len(edges)
            
            if num_edges > 0:
                A_ub_psp = lil_matrix((num_edges, num_filtered), dtype=float)
                for i, (u, v) in enumerate(edges):
                    u_idx = filtered_node_to_idx[u]
                    v_idx = filtered_node_to_idx[v]
                    A_ub_psp[i, u_idx] = 1
                    A_ub_psp[i, v_idx] = 1
                b_ub_psp = np.ones(num_edges)
                
                psp_constraints = [LinearConstraint(A_ub_psp.toarray(), -np.inf, b_ub_psp)]
            else:
                # No edges: all vertices form an independent set
                psp_constraints = []
            
            # Solve MILP with timeout
            from scipy.optimize import Bounds
            psp_result = milp(
                c=c_psp, 
                constraints=psp_constraints, 
                integrality=integrality_psp, 
                bounds=Bounds(lb=0, ub=1),
                options={'time_limit': self.timeout}
            )
            
            if not psp_result.success:
                if self.verbose:
                    print(f"  Classical PSP failed: {psp_result.message}")
                return []
            
            # Extract solution from filtered indices
            selected_filtered_indices = [v for v, val in enumerate(psp_result.x) if val > 0.5]
            if not selected_filtered_indices:
                if self.verbose:
                    print(f"  Classical PSP: Empty independent set found")
                return []
            
            # Map back to original node labels
            independent_set = {filtered_nodes[i] for i in selected_filtered_indices}
            
            # Calculate objective value (sum of filtered dual weights)
            total_weight = sum(filtered_dual_weights[i] for i in selected_filtered_indices)
            reduced_cost = 1 - total_weight
            
            if self.verbose:
                print(f"  Classical PSP: Found IS {sorted(independent_set)} with weight {total_weight:.4f}")
                print(f"  Reduced cost: {reduced_cost:.4f}")
            
            # Check profitability: reduced cost < 0, i.e., total_weight > 1
            if total_weight > 1.0 + 1e-6:  # Profitable column
                if self.verbose:
                    print(f"  Classical PSP: Profitable column found")
                return [independent_set]
            else:
                if self.verbose:
                    print(f"  Classical PSP: No profitable column (weight {total_weight:.4f} ≤ 1)")
                return []
            
        except Exception as e:
            if self.verbose:
                print(f"  Classical PSP error: {e}")
            return []


class QuantumPricingSubproblemSolver:
    """
    Quantum Pricing Subproblem Solver using sample-then-filter approach.
    
    This class implements the quantum PSP for column generation by:
    1. Using unweighted Motzkin-Straus oracle to sample multiple independent sets
    2. Post-processing samples to find profitable columns (sum of dual weights > 1)
    
    This approach leverages quantum algorithms' natural sampling behavior rather than
    trying to solve a weighted optimization problem directly.
    
    Refinement Control:
    - When enable_refinement=True: Oracle uses superposition refinement to extract
      multiple independent sets from quantum superposition states, potentially
      finding more profitable columns but with increased computational cost
    - When enable_refinement=False: Faster execution, may miss some profitable 
      columns from superposition solutions
    """
    
    def __init__(self, oracle_type: str = 'jax-pgd', num_samples: int = 100,
                 verbose: bool = False, enable_refinement: bool = True, **oracle_config):
        """
        Initialize quantum pricing subproblem solver.
        
        Args:
            oracle_type: Type of oracle ('jax-pgd', 'dirac')
            num_samples: Number of samples to request from quantum oracle
            verbose: Enable detailed logging
            enable_refinement: Enable superposition refinement for better solution extraction
            **oracle_config: Oracle-specific configuration parameters
        """
        self.oracle_type = oracle_type
        self.num_samples = num_samples
        self.verbose = verbose
        self.enable_refinement = enable_refinement
        self.oracle_config = oracle_config
        
        # Create underlying oracle adapter
        if ORACLE_SYSTEM_AVAILABLE:
            try:
                self.oracle = OracleFactory.create_oracle(
                    oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config
                )
            except Exception as e:
                if verbose:
                    print(f"Failed to create {oracle_type} oracle: {e}")
                    print("Falling back to JAX-PGD oracle")
                self.oracle = OracleFactory.create_oracle('jax-pgd', verbose=verbose, **oracle_config)
        else:
            raise ImportError("Oracle system not available - cannot create quantum PSP solver")
    
    def solve_pricing_subproblem(self, graph: nx.Graph, dual_weights: np.ndarray, 
                                support_threshold: float = 1e-5) -> List[Set[int]]:
        """
        Solve Pricing Subproblem using quantum sampling + classical filtering.
        
        Algorithm:
        1. Use unweighted Motzkin-Straus oracle to sample independent sets
        2. For each sampled independent set, calculate sum of dual weights
        3. Keep only sets where sum > 1 (profitable columns)
        
        Args:
            graph: NetworkX graph 
            dual_weights: Array of dual variables from master problem
            support_threshold: Threshold for solution extraction
            
        Returns:
            List of profitable independent sets (reduced cost < 0)
        """
        if self.verbose:
            print(f"  Quantum PSP: Sampling with {self.oracle.name}")
            print(f"  Dual weights: {dual_weights}")
            print(f"  Target samples: {self.num_samples}")
            print(f"  Refinement: {'enabled' if self.enable_refinement else 'disabled'}")
        
        node_list = sorted(list(graph.nodes()))
        
        # Filter vertices according to the paper's specification:
        # V' = {u ∈ V | w_u > 0}
        # Only consider vertices with positive dual weights
        positive_weight_indices = [i for i, w in enumerate(dual_weights) if w > 0]
        
        if len(positive_weight_indices) == 0:
            if self.verbose:
                print(f"  No vertices with positive dual weights - returning empty set")
            return []
        
        # Create filtered node list and dual weights
        filtered_nodes = [node_list[i] for i in positive_weight_indices]
        filtered_dual_weights = dual_weights[positive_weight_indices]
        
        # Create subgraph with only positive-weight vertices
        filtered_graph = graph.subgraph(filtered_nodes)
        
        if self.verbose:
            print(f"  Original nodes: {len(node_list)}, Filtered nodes: {len(filtered_nodes)}")
            weighted_nodes = [(filtered_nodes[i], filtered_dual_weights[i]) for i in range(len(filtered_nodes))]
            print(f"  Node weights: {weighted_nodes}")
        
        # Step 1: Use unweighted oracle to sample independent sets
        # The oracle finds cliques in complement graph = stable sets in original
        complement_graph = nx.complement(filtered_graph)
        
        if self.verbose:
            print(f"  Filtered complement graph: {complement_graph.number_of_nodes()} nodes, "
                  f"{complement_graph.number_of_edges()} edges")
        
        try:
            # Sample independent sets using existing oracle
            # For JAX-PGD: multiple restarts give multiple samples
            # For Dirac: multiple solutions from quantum annealer
            sampled_sets = self.oracle.find_maximal_cliques(complement_graph, support_threshold)
            
            if self.verbose:
                print(f"  Oracle returned {len(sampled_sets)} candidate independent sets")
            
            # Step 2: Classical post-processing - filter for profitable columns
            profitable_sets = []
            
            for i, indep_set in enumerate(sampled_sets):
                # Calculate total weight for this independent set using filtered dual weights
                total_weight = 0.0
                for node in indep_set:
                    if node in filtered_nodes:
                        node_idx = filtered_nodes.index(node)
                        if node_idx < len(filtered_dual_weights):
                            total_weight += filtered_dual_weights[node_idx]
                
                # Check profitability: reduced cost = 1 - total_weight
                # Profitable if reduced cost < 0, i.e., total_weight > 1
                if total_weight > 1.0 + 1e-6:  # Profitable column
                    if self.verbose:
                        print(f"    Sample {i}: IS {sorted(indep_set)} profitable with weight {total_weight:.4f}")
                    profitable_sets.append(indep_set)
                elif self.verbose:
                    print(f"    Sample {i}: IS {sorted(indep_set)} not profitable (weight {total_weight:.4f} ≤ 1)")
            
            if self.verbose:
                print(f"  Found {len(profitable_sets)} profitable columns from {len(sampled_sets)} samples")
            
            return profitable_sets
            
        except Exception as e:
            if self.verbose:
                print(f"  Quantum PSP failed: {e}")
            return []


class MWISBasedPricingSubproblemSolver:
    """
    Enhanced MWIS-Based Pricing Subproblem Solver using scripts/mwis.py.
    
    This class integrates the proven Maximum Weight Independent Set solver from
    scripts/mwis.py, which has been extensively debugged and enhanced with:
    - Proper Dirac oracle integration with coefficient handling
    - Enhanced debugging output with support value analysis
    - Threshold sensitivity analysis
    - Clean output without emojis
    
    This solver directly addresses the weighted pricing subproblem by:
    1. Creating a weighted graph using dual variables as node weights
    2. Calling the MWIS solver with the specified oracle (Dirac, JAX-PGD)
    3. Extracting profitable independent sets (reduced cost < 0)
    4. Providing detailed debugging output for threshold analysis
    """
    
    def __init__(self, oracle_type: str = 'dirac', verbose: bool = False, 
                 enable_refinement: bool = True, **oracle_config):
        """
        Initialize MWIS-based pricing subproblem solver.
        
        Args:
            oracle_type: Type of oracle ('dirac', 'jax-pgd')
            verbose: Enable detailed logging
            enable_refinement: Enable superposition refinement for better solution extraction
            **oracle_config: Oracle-specific configuration parameters
        """
        self.oracle_type = oracle_type
        self.verbose = verbose
        self.enable_refinement = enable_refinement
        self.oracle_config = oracle_config
        self.name = f"MWIS-based PSP ({oracle_type})"
        
        # Validate that MWIS solver is available
        if not MWIS_SOLVER_AVAILABLE:
            raise ImportError("MWIS solver from scripts/mwis.py not available")
    
    def solve_pricing_subproblem(self, graph: nx.Graph, dual_weights: np.ndarray, 
                                support_threshold: float = 1e-5) -> List[Set[int]]:
        """
        Solve Pricing Subproblem using enhanced MWIS solver from scripts/mwis.py.
        
        This method directly solves the weighted pricing subproblem:
        - Maximize: Σ dual_weights[v] * x[v] 
        - Subject to: x[u] + x[v] ≤ 1 for all edges (u,v)
        - Profitable if: sum of dual weights > 1 (reduced cost < 0)
        
        Args:
            graph: NetworkX graph 
            dual_weights: Array of dual variables from master problem
            support_threshold: Threshold for solution extraction
            
        Returns:
            List of profitable independent sets (reduced cost < 0)
        """
        if self.verbose:
            print(f"  {self.name}: Solving weighted MWIS using scripts/mwis.py")
            print(f"  Dual weights: {dual_weights}")
            print(f"  Oracle: {self.oracle_type}")
            print(f"  Refinement: {'enabled' if self.enable_refinement else 'disabled'}")
        
        node_list = sorted(list(graph.nodes()))
        num_vertices = len(node_list)
        
        if num_vertices == 0:
            return []
        
        # Step 1: Create weighted graph for MWIS following the paper's approach
        # Map dual variables to node weights dictionary
        weights = {}
        for i, node in enumerate(node_list):
            if i < len(dual_weights):
                weights[node] = float(dual_weights[i])
            else:
                weights[node] = 0.0
        
        # Filter vertices and edges according to the paper's specification:
        # V' = {u ∈ V | w_u > 0}
        # E' = {(u,v) ∈ E | w_u > 0 and w_v > 0}
        positive_weight_nodes = [node for node in node_list if weights[node] > 0]
        
        if len(positive_weight_nodes) == 0:
            if self.verbose:
                print(f"  No vertices with positive weights - returning empty set")
            return []
        
        # Create subgraph with only positive-weight vertices  
        filtered_graph = graph.subgraph(positive_weight_nodes)
        
        # Create filtered weights dictionary
        filtered_weights = {node: weights[node] for node in positive_weight_nodes}
        
        if self.verbose:
            print(f"  Original nodes: {len(node_list)}, Filtered nodes: {len(positive_weight_nodes)}")
            weighted_nodes = [(node, filtered_weights[node]) for node in positive_weight_nodes]
            print(f"  Node weights: {weighted_nodes}")
        
        try:
            # Step 2: Call enhanced MWIS solver with filtered graph and weights
            start_time = time.time()
            oracle_result, oracle_details = find_maximum_weight_independent_set(
                graph=filtered_graph,
                weights=filtered_weights,
                oracle_type=self.oracle_type,
                verbose=self.verbose,
                enable_refinement=self.enable_refinement,
                support_threshold=support_threshold,
                **self.oracle_config
            )
            runtime = time.time() - start_time
            
            if self.verbose:
                print(f"  MWIS solver runtime: {runtime:.4f}s")
                if oracle_details:
                    print(f"  Oracle details: {oracle_details}")
            
            # Step 3: Process results and extract profitable columns
            profitable_sets = []
            
            if self.verbose:
                print(f"  (2) MWIS SOLUTION FROM PSP:")
            
            # Handle the case where oracle_result is a list of sets (from MWIS solver)
            if oracle_result and isinstance(oracle_result, list):
                candidate_sets = oracle_result
                
                if self.verbose:
                    print(f"      Found {len(candidate_sets)} candidate independent sets from MWIS solver:")
                
                for i, candidate_set in enumerate(candidate_sets):
                    if not isinstance(candidate_set, set):
                        candidate_set = set(candidate_set)
                    
                    # Verify this is actually an independent set
                    if not verify_independent_set(graph, candidate_set):
                        if self.verbose:
                            print(f"        MWIS Solution {i+1}: {sorted(candidate_set)} - INVALID (not independent set)")
                        continue
                    
                    # Calculate total weight for this independent set
                    total_weight = sum(weights.get(node, 0.0) for node in candidate_set)
                    reduced_cost = 1.0 - total_weight
                    
                    if self.verbose:
                        print(f"        MWIS Solution {i+1}: {sorted(candidate_set)}")
                        print(f"          Weight (Σ dual_weights): {total_weight:.4f}")
                        print(f"          Reduced Cost (1 - weight): {reduced_cost:.4f}")
                    
                    # Check profitability: reduced cost < 0, i.e., total_weight > 1
                    if total_weight > 1.0 + 1e-6:  # Profitable column
                        profitable_sets.append(candidate_set)
                        if self.verbose:
                            print(f"          Status: ✓ PROFITABLE (reduced cost < 0)")
                    elif self.verbose:
                        print(f"          Status: ✗ Not profitable (reduced cost ≥ 0)")
            
            # Handle the case where oracle_result is a dictionary with 'candidate_sets' key
            elif oracle_result and isinstance(oracle_result, dict) and 'candidate_sets' in oracle_result:
                candidate_sets = oracle_result['candidate_sets']
                
                if self.verbose:
                    print(f"      Found {len(candidate_sets)} candidate independent sets from MWIS solver:")
                
                for i, candidate_set in enumerate(candidate_sets):
                    if not isinstance(candidate_set, set):
                        candidate_set = set(candidate_set)
                    
                    # Verify this is actually an independent set
                    if not verify_independent_set(graph, candidate_set):
                        if self.verbose:
                            print(f"    PSP Solution {i+1}: {sorted(candidate_set)} - INVALID (not independent set)")
                        continue
                    
                    # Calculate total weight for this independent set
                    total_weight = sum(weights.get(node, 0.0) for node in candidate_set)
                    reduced_cost = 1.0 - total_weight
                    
                    if self.verbose:
                        print(f"        MWIS Solution {i+1}: {sorted(candidate_set)}")
                        print(f"          Weight (Σ dual_weights): {total_weight:.4f}")
                        print(f"          Reduced Cost (1 - weight): {reduced_cost:.4f}")
                    
                    # Check profitability: reduced cost < 0, i.e., total_weight > 1
                    if total_weight > 1.0 + 1e-6:  # Profitable column
                        profitable_sets.append(candidate_set)
                        if self.verbose:
                            print(f"          Status: ✓ PROFITABLE (reduced cost < 0)")
                    elif self.verbose:
                        print(f"          Status: ✗ Not profitable (reduced cost ≥ 0)")
            
            # Handle the case where oracle_result is a dictionary with 'independent_set' key  
            elif oracle_result and isinstance(oracle_result, dict) and 'independent_set' in oracle_result:
                # Handle single solution format
                independent_set = oracle_result['independent_set']
                if not isinstance(independent_set, set):
                    independent_set = set(independent_set)
                
                if verify_independent_set(graph, independent_set):
                    total_weight = sum(weights.get(node, 0.0) for node in independent_set)
                    reduced_cost = 1.0 - total_weight
                    
                    if self.verbose:
                        print(f"        MWIS Solution: {sorted(independent_set)}")
                        print(f"          Weight (Σ dual_weights): {total_weight:.4f}")
                        print(f"          Reduced Cost (1 - weight): {reduced_cost:.4f}")
                    
                    if total_weight > 1.0 + 1e-6:  # Profitable column
                        profitable_sets.append(independent_set)
                        if self.verbose:
                            print(f"          Status: ✓ PROFITABLE (reduced cost < 0)")
                    elif self.verbose:
                        print(f"          Status: ✗ Not profitable (reduced cost ≥ 0)")
                else:
                    if self.verbose:
                        print(f"  Single solution: {sorted(independent_set)} - INVALID (not independent)")
            
            else:
                if self.verbose:
                    print(f"      No valid solutions found from MWIS solver")
                    print(f"      oracle_result type: {type(oracle_result)}")
                    print(f"      oracle_result value: {oracle_result}")
            
            if self.verbose:
                print(f"  Found {len(profitable_sets)} profitable columns from MWIS solver")
            
            return profitable_sets
            
        except Exception as e:
            if self.verbose:
                print(f"  MWIS-based PSP failed: {e}")
                import traceback
                traceback.print_exc()
            return []


class ClassicalColumnGenerationSolver:
    """
    Classical Column Generation Solver for Minimum Vertex Coloring Problem.
    
    This class implements the traditional column generation algorithm using
    classical linear programming for the master problem and classical MILP
    for the pricing subproblem. This serves as the baseline for comparison
    with the quantum-enhanced version.
    """
    
    def __init__(self, verbose: bool = False, max_cg_iterations: int = 50, 
                 tolerance: float = 1e-6, psp_timeout: int = 60):
        """
        Initialize classical column generation solver.
        
        Args:
            verbose: Enable detailed logging
            max_cg_iterations: Maximum column generation iterations
            tolerance: Convergence tolerance for reduced cost
            psp_timeout: Maximum time per PSP solve in seconds
        """
        self.verbose = verbose
        self.max_cg_iterations = max_cg_iterations
        self.tolerance = tolerance
        
        # Initialize classical PSP solver
        self.psp_solver = ClassicalPricingSubproblemSolver(
            verbose=verbose, timeout=psp_timeout
        )
        
        # Performance tracking
        self.solving_stats = {
            'total_time': 0.0,
            'rmp_time': 0.0,
            'psp_time': 0.0,
            'iterations': 0,
            'columns_generated': 0,
            'psp_calls': 0
        }
    
    def solve(self, graph: nx.Graph) -> Tuple[Optional[int], List[FrozenSet[int]], Dict[str, Any]]:
        """
        Solve minimum vertex coloring using classical column generation.
        
        Args:
            graph: Input NetworkX graph
            
        Returns:
            Tuple of (num_colors, coloring_solution, solving_details)
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Solving MVCP using Classical Column Generation")
            print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Step 1: Initialize with singleton independent sets
        node_list = sorted(list(graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        num_vertices = len(node_list)
        
        # Current columns (independent sets) - start with singletons
        current_columns = [frozenset([i]) for i in range(num_vertices)]
        known_signatures = {tuple(sorted(col)) for col in current_columns}
        
        if self.verbose:
            print(f"Initial columns: {len(current_columns)} singleton sets")
        
        # Step 2: Column Generation Loop
        iteration = 0
        rmp_time = 0.0
        psp_time = 0.0
        
        while iteration < self.max_cg_iterations:
            iteration += 1
            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")
            
            # Step 2a: Solve Reduced Master Problem
            rmp_start = time.time()
            dual_variables = self._solve_rmp(current_columns, num_vertices)
            rmp_time += time.time() - rmp_start
            
            if dual_variables is None:
                if self.verbose:
                    print("RMP failed - stopping column generation")
                break
            
            if self.verbose:
                print(f"  Dual variables: {dual_variables}")
            
            # Step 2b: Solve Pricing Subproblem  
            psp_start = time.time()
            new_columns = self._solve_psp(graph, dual_variables)
            psp_time += time.time() - psp_start
            self.solving_stats['psp_calls'] += 1
            
            if not new_columns:
                if self.verbose:
                    print("  No profitable columns found - converged")
                break
            
            # Step 2c: Add new columns
            added_count = 0
            for new_col in new_columns:
                # Convert back to node indices for column storage
                col_indices = frozenset(node_to_idx[node] for node in new_col if node in node_to_idx)
                col_signature = tuple(sorted(col_indices))
                
                if col_signature not in known_signatures:
                    current_columns.append(col_indices)
                    known_signatures.add(col_signature)
                    added_count += 1
            
            if self.verbose:
                print(f"  Added {added_count} new columns, total: {len(current_columns)}")
            
            if added_count == 0:
                if self.verbose:
                    print("  All profitable columns already known - converged")
                break
        
        # Step 3: Solve Final Integer Linear Program
        if self.verbose:
            print(f"\n--- Solving Final ILP ---")
        
        final_result = self._solve_final_ilp(current_columns, num_vertices)
        
        total_time = time.time() - start_time
        
        # Update solving statistics
        self.solving_stats.update({
            'total_time': total_time,
            'rmp_time': rmp_time,
            'psp_time': psp_time,
            'iterations': iteration,
            'columns_generated': len(current_columns)
        })
        
        if final_result is None:
            return None, [], self.solving_stats
        
        # Extract coloring solution
        coloring_solution = []
        if final_result['success']:
            for col_idx in final_result['selected_columns']:
                # Convert indices back to original node labels
                color_set = frozenset(node_list[idx] for idx in current_columns[col_idx])
                coloring_solution.append(color_set)
        
        num_colors = len(coloring_solution) if final_result['success'] else None
        
        if self.verbose:
            print(f"\n--- Solution Summary ---")
            print(f"Optimal colors: {num_colors}")
            print(f"Total time: {total_time:.3f}s")
            print(f"  RMP time: {rmp_time:.3f}s ({rmp_time/total_time*100:.1f}%)")
            print(f"  PSP time: {psp_time:.3f}s ({psp_time/total_time*100:.1f}%)")
            print(f"Iterations: {iteration}")
            print(f"Columns generated: {len(current_columns)}")
        
        return num_colors, coloring_solution, self.solving_stats
    
    def _solve_rmp(self, columns: List[FrozenSet[int]], num_vertices: int) -> Optional[np.ndarray]:
        """Solve Reduced Master Problem using linear programming."""
        num_columns = len(columns)
        
        # Build constraint matrix: A[v,s] = 1 if vertex v in column s
        A = lil_matrix((num_vertices, num_columns))
        for col_idx, col in enumerate(columns):
            for vertex in col:
                A[vertex, col_idx] = 1
        A = A.tocsc()
        
        # RHS: each vertex covered exactly once
        b = np.ones(num_vertices)
        
        # Objective: minimize number of colors
        c = np.ones(num_columns)
        
        # Variable bounds
        bounds = [(0, 1) for _ in range(num_columns)]
        
        try:
            result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
            
            if result.success:
                # Extract dual variables
                if hasattr(result, 'eqlin') and result.eqlin is not None:
                    duals = result.eqlin['marginals'] if isinstance(result.eqlin, dict) else result.eqlin.marginals
                    return np.array(duals) if duals is not None else np.zeros(num_vertices)
                else:
                    return np.zeros(num_vertices)
            else:
                if self.verbose:
                    print(f"  RMP failed: {result.message}")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"  RMP error: {e}")
            return None
    
    def _solve_psp(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        """Solve Pricing Subproblem using classical MILP."""
        try:
            # Use 1/N threshold for N-node graphs
            n_nodes = len(graph.nodes())
            threshold = 1.0 / n_nodes
            return self.psp_solver.solve_pricing_subproblem(graph, dual_vars, support_threshold=threshold)
        except Exception as e:
            if self.verbose:
                print(f"  PSP failed: {e}")
            return []
    
    def _solve_final_ilp(self, columns: List[FrozenSet[int]], num_vertices: int) -> Optional[Dict[str, Any]]:
        """Solve final integer linear program."""
        num_columns = len(columns)
        
        # Build constraint matrix
        A = np.zeros((num_vertices, num_columns))
        for col_idx, col in enumerate(columns):
            for vertex in col:
                A[vertex, col_idx] = 1
        
        # Objective and constraints
        c = np.ones(num_columns)
        b = np.ones(num_vertices)
        constraints = [LinearConstraint(A, b, b)]
        
        # Integer variables
        integrality = np.ones(num_columns, dtype=int)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if result.success:
                # Find selected columns
                selected_columns = [i for i, val in enumerate(result.x) if val > 0.5]
                
                return {
                    'success': True,
                    'objective': int(round(result.fun)),
                    'selected_columns': selected_columns,
                    'solution': result.x
                }
            else:
                if self.verbose:
                    print(f"  Final ILP failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            if self.verbose:
                print(f"  Final ILP error: {e}")
            return {'success': False, 'error': str(e)}


class QuantumColumnGenerationSolver:
    """
    Quantum Column Generation Solver for Minimum Vertex Coloring Problem.
    
    This class implements the hybrid quantum-classical column generation algorithm,
    combining classical linear programming for the master problem with the enhanced
    MWIS-based quantum optimization for the pricing subproblem.
    
    Enhanced Features:
    - Uses proven MWIS solver from scripts/mwis.py with extensive debugging
    - Proper Dirac oracle integration with coefficient handling
    - Enhanced debugging output with support value analysis
    - Threshold sensitivity analysis for better column discovery
    - Clean output without emojis
    
    Refinement Control:
    - When enable_refinement=True: Quantum oracle uses superposition refinement to
      extract multiple independent sets from mixed quantum states, potentially
      discovering more profitable columns but with increased computational overhead
    - When enable_refinement=False: Faster pricing subproblem solving, may miss
      some profitable columns from superposition quantum solutions
    """
    
    def __init__(self, oracle_type: str = 'dirac', verbose: bool = False, 
                 max_cg_iterations: int = 50, tolerance: float = 1e-6, 
                 enable_refinement: bool = True, **oracle_config):
        """
        Initialize quantum column generation solver.
        
        Args:
            oracle_type: Type of oracle for PSP ('dirac', 'jax-pgd') - defaults to 'dirac'
            verbose: Enable detailed logging
            max_cg_iterations: Maximum column generation iterations
            tolerance: Convergence tolerance for reduced cost
            enable_refinement: Enable superposition refinement for better column discovery
            **oracle_config: Oracle-specific parameters
        """
        self.oracle_type = oracle_type
        self.verbose = verbose
        self.max_cg_iterations = max_cg_iterations
        self.tolerance = tolerance
        self.enable_refinement = enable_refinement
        
        # Initialize enhanced MWIS-based PSP solver
        if MWIS_SOLVER_AVAILABLE:
            self.psp_solver = MWISBasedPricingSubproblemSolver(
                oracle_type=oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config
            )
        else:
            # Fallback to original quantum PSP solver if MWIS solver not available
            if self.verbose:
                print("Warning: MWIS solver not available, falling back to original quantum PSP solver")
            self.psp_solver = QuantumPricingSubproblemSolver(
                oracle_type=oracle_type, verbose=verbose, enable_refinement=enable_refinement, **oracle_config
            )
        
        # Performance tracking
        self.solving_stats = {
            'total_time': 0.0,
            'rmp_time': 0.0,
            'psp_time': 0.0,
            'iterations': 0,
            'columns_generated': 0,
            'oracle_calls': 0
        }
    
    def solve(self, graph: nx.Graph) -> Tuple[Optional[int], List[FrozenSet[int]], Dict[str, Any]]:
        """
        Solve minimum vertex coloring using quantum column generation.
        
        Args:
            graph: Input NetworkX graph
            
        Returns:
            Tuple of (num_colors, coloring_solution, solving_details)
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Solving MVCP using Quantum Column Generation")
            print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"Oracle: {self.oracle_type}")
            print(f"Superposition refinement: {'enabled' if self.enable_refinement else 'disabled'}")
        
        # Step 1: Initialize with singleton independent sets
        node_list = sorted(list(graph.nodes()))
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        num_vertices = len(node_list)
        
        # Current columns (independent sets) - start with singletons
        current_columns = [frozenset([i]) for i in range(num_vertices)]
        known_signatures = {tuple(sorted(col)) for col in current_columns}
        
        if self.verbose:
            print(f"Initial columns: {len(current_columns)} singleton sets")
        
        # Step 2: Column Generation Loop
        iteration = 0
        rmp_time = 0.0
        psp_time = 0.0
        
        while iteration < self.max_cg_iterations:
            iteration += 1
            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")
            
            # Step 2a: Solve Reduced Master Problem
            rmp_start = time.time()
            dual_variables = self._solve_rmp(current_columns, num_vertices)
            rmp_time += time.time() - rmp_start
            
            if dual_variables is None:
                if self.verbose:
                    print("RMP failed - stopping column generation")
                break
            
            if self.verbose:
                print(f"  (1) DUAL WEIGHTS FROM RMP: {dual_variables}")
                print(f"      These are the dual variables π_v for each vertex v")
            
            # Step 2b: Solve Pricing Subproblem  
            psp_start = time.time()
            new_columns = self._solve_psp(graph, dual_variables)
            psp_time += time.time() - psp_start
            self.solving_stats['oracle_calls'] += 1
            
            if self.verbose:
                print(f"  (3) NEW COLUMNS FOUND: {'YES' if new_columns else 'NO'}")
                if new_columns:
                    print(f"      Found {len(new_columns)} profitable columns:")
                    for i, col in enumerate(new_columns):
                        sorted_col = sorted(col)
                        weight = sum(dual_variables[node_to_idx[node]] for node in col if node in node_to_idx)
                        print(f"        Column {i+1}: {sorted_col} (weight: {weight:.4f})")
                else:
                    print(f"      No profitable columns - algorithm converged")
            
            if not new_columns:
                break
            
            # Step 2c: Add new columns
            added_count = 0
            for new_col in new_columns:
                # Convert back to node indices for column storage
                col_indices = frozenset(node_to_idx[node] for node in new_col if node in node_to_idx)
                col_signature = tuple(sorted(col_indices))
                
                if col_signature not in known_signatures:
                    current_columns.append(col_indices)
                    known_signatures.add(col_signature)
                    added_count += 1
            
            if self.verbose:
                print(f"  Added {added_count} new columns, total: {len(current_columns)}")
                
                # Print current column set (similar to ref(2) format)
                print(f"  === CURRENT COLUMN SET (like ref(2)) ===")
                for col_idx, column in enumerate(current_columns):
                    # Convert indices back to original node labels for display
                    node_labels = sorted([node_list[idx] for idx in column])
                    print(f"    Column {col_idx+1}: {node_labels}")
                print(f"  ========================================")
            
            if added_count == 0:
                if self.verbose:
                    print("  All profitable columns already known - converged")
                break
        
        # Step 3: Solve Final Integer Linear Program
        if self.verbose:
            print(f"\n--- Solving Final ILP ---")
        
        final_result = self._solve_final_ilp(current_columns, num_vertices)
        
        total_time = time.time() - start_time
        
        # Update solving statistics
        self.solving_stats.update({
            'total_time': total_time,
            'rmp_time': rmp_time,
            'psp_time': psp_time,
            'iterations': iteration,
            'columns_generated': len(current_columns)
        })
        
        if final_result is None:
            return None, [], self.solving_stats
        
        # Extract coloring solution
        coloring_solution = []
        if final_result['success']:
            for col_idx in final_result['selected_columns']:
                # Convert indices back to original node labels
                color_set = frozenset(node_list[idx] for idx in current_columns[col_idx])
                coloring_solution.append(color_set)
        
        num_colors = len(coloring_solution) if final_result['success'] else None
        
        if self.verbose:
            print(f"\n--- Solution Summary ---")
            print(f"Optimal colors: {num_colors}")
            print(f"Total time: {total_time:.3f}s")
            print(f"  RMP time: {rmp_time:.3f}s ({rmp_time/total_time*100:.1f}%)")
            print(f"  PSP time: {psp_time:.3f}s ({psp_time/total_time*100:.1f}%)")
            print(f"Iterations: {iteration}")
            print(f"Columns generated: {len(current_columns)}")
        
        return num_colors, coloring_solution, self.solving_stats
    
    def _solve_rmp(self, columns: List[FrozenSet[int]], num_vertices: int) -> Optional[np.ndarray]:
        """Solve Reduced Master Problem using linear programming."""
        num_columns = len(columns)
        
        # Build constraint matrix: A[v,s] = 1 if vertex v in column s
        A = lil_matrix((num_vertices, num_columns))
        for col_idx, col in enumerate(columns):
            for vertex in col:
                A[vertex, col_idx] = 1
        A = A.tocsc()
        
        # RHS: each vertex covered exactly once
        b = np.ones(num_vertices)
        
        # Objective: minimize number of colors
        c = np.ones(num_columns)
        
        # Variable bounds
        bounds = [(0, 1) for _ in range(num_columns)]
        
        try:
            result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
            
            if result.success:
                # Extract dual variables
                if hasattr(result, 'eqlin') and result.eqlin is not None:
                    duals = result.eqlin['marginals'] if isinstance(result.eqlin, dict) else result.eqlin.marginals
                    return np.array(duals) if duals is not None else np.zeros(num_vertices)
                else:
                    return np.zeros(num_vertices)
            else:
                if self.verbose:
                    print(f"  RMP failed: {result.message}")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"  RMP error: {e}")
            return None
    
    def _solve_psp(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        """Solve Pricing Subproblem using quantum sampling + classical filtering."""
        try:
            # Use 1/N threshold for N-node graphs
            n_nodes = len(graph.nodes())
            threshold = 1.0 / n_nodes
            if self.verbose:
                print(f"  Using 1/N threshold: {threshold:.4f} for {n_nodes}-node graph")
            return self.psp_solver.solve_pricing_subproblem(graph, dual_vars, support_threshold=threshold)
        except Exception as e:
            if self.verbose:
                print(f"  PSP failed: {e}")
            return []
    
    def _solve_final_ilp(self, columns: List[FrozenSet[int]], num_vertices: int) -> Optional[Dict[str, Any]]:
        """Solve final integer linear program."""
        num_columns = len(columns)
        
        # Build constraint matrix
        A = np.zeros((num_vertices, num_columns))
        for col_idx, col in enumerate(columns):
            for vertex in col:
                A[vertex, col_idx] = 1
        
        # Objective and constraints
        c = np.ones(num_columns)
        b = np.ones(num_vertices)
        constraints = [LinearConstraint(A, b, b)]
        
        # Integer variables
        integrality = np.ones(num_columns, dtype=int)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if result.success:
                # Find selected columns
                selected_columns = [i for i, val in enumerate(result.x) if val > 0.5]
                
                return {
                    'success': True,
                    'objective': int(round(result.fun)),
                    'selected_columns': selected_columns,
                    'solution': result.x
                }
            else:
                if self.verbose:
                    print(f"  Final ILP failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            if self.verbose:
                print(f"  Final ILP error: {e}")
            return {'success': False, 'error': str(e)}


def plot_coloring_solution(graph: nx.Graph, coloring: List[FrozenSet[int]], 
                          title: str = "Graph Coloring", save_path: str = None):
    """Plot graph with coloring solution."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - cannot create plots")
        return False
    
    try:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Generate layout
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # Color palette
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'cyan', 'magenta', 'lime', 'yellow']
        
        # Draw graph structure
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax)
        
        # Draw all nodes in default color first
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7, ax=ax)
        
        # Color nodes by their assigned color
        for i, color_set in enumerate(coloring):
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(graph, pos, nodelist=list(color_set),
                                  node_color=color, node_size=600, alpha=0.8, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold', ax=ax)
        
        # Create legend
        legend_elements = []
        for i, color_set in enumerate(coloring):
            color = colors[i % len(colors)]
            label = f'Color {i+1}: {sorted(color_set)}'
            legend_elements.append(patches.Patch(color=color, label=label))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.set_title(f'{title}\nNodes: {graph.number_of_nodes()}, '
                    f'Edges: {graph.number_of_edges()}, Colors: {len(coloring)}', 
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


def create_test_graphs() -> List[Tuple[str, nx.Graph]]:
    """Create test graphs for validation."""
    test_graphs = []
    
    # Triangle (chromatic number = 3)
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    test_graphs.append(("Triangle", triangle))
    
    # Complete graph K4 (chromatic number = 4)
    k4 = nx.complete_graph(4)
    test_graphs.append(("Complete K4", k4))
    
    # Path graph P4 (chromatic number = 2)
    path = nx.path_graph(4)
    test_graphs.append(("Path P4", path))
    
    # Cycle C5 (chromatic number = 3) 
    cycle = nx.cycle_graph(5)
    test_graphs.append(("Cycle C5", cycle))
    
    # Wheel graph (chromatic number = 4 for n >= 3)
    wheel = nx.wheel_graph(6)  # 6 nodes including center
    test_graphs.append(("Wheel W5", wheel))
    
    return test_graphs


def compare_classical_vs_quantum(graph: nx.Graph, classical_solution: List[FrozenSet[int]], 
                                quantum_solution: List[FrozenSet[int]]) -> Dict[str, Any]:
    """Compare classical column generation with quantum column generation."""
    try:
        classical_colors = len(classical_solution) if classical_solution else float('inf')
        quantum_colors = len(quantum_solution) if quantum_solution else float('inf')
        
        comparison = {
            'classical_colors': classical_colors,
            'quantum_colors': quantum_colors,
            'difference': quantum_colors - classical_colors,
            'classical_valid': classical_colors != float('inf'),
            'quantum_valid': quantum_colors != float('inf')
        }
        
        print(f"  Classical CG: {classical_colors if classical_colors != float('inf') else 'FAILED'} colors")
        print(f"  Quantum CG: {quantum_colors if quantum_colors != float('inf') else 'FAILED'} colors")
        
        if comparison['classical_valid'] and comparison['quantum_valid']:
            if comparison['difference'] > 0:
                print(f"  Classical CG outperforms by {abs(comparison['difference'])} colors")
            elif comparison['difference'] == 0:
                print(f"  Both methods achieve the same solution quality")
            else:
                print(f"  Quantum CG outperforms by {abs(comparison['difference'])} colors")
        elif comparison['classical_valid']:
            print(f"  Only classical CG succeeded")
        elif comparison['quantum_valid']:
            print(f"  Only quantum CG succeeded")
        else:
            print(f"  Both methods failed")
        
        return comparison
        
    except Exception as e:
        print(f"  Error comparing classical vs quantum: {e}")
        return {'error': str(e)}


def compare_with_networkx(graph: nx.Graph, cg_solution: List[FrozenSet[int]], 
                         method_name: str = "Column Generation") -> Dict[str, Any]:
    """Compare column generation solution with NetworkX greedy coloring."""
    try:
        # NetworkX greedy coloring (not optimal but fast)
        nx_coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
        nx_colors = max(nx_coloring.values()) + 1 if nx_coloring else 0
        
        cg_colors = len(cg_solution)
        
        comparison = {
            'cg_colors': cg_colors,
            'nx_colors': nx_colors,
            'improvement': nx_colors - cg_colors,
            'improvement_pct': ((nx_colors - cg_colors) / nx_colors * 100) if nx_colors > 0 else 0
        }
        
        print(f"  {method_name}: {cg_colors} colors")
        print(f"  NetworkX greedy: {nx_colors} colors")
        if comparison['improvement'] > 0:
            print(f"  Improvement: {comparison['improvement']} colors ({comparison['improvement_pct']:.1f}%)")
        elif comparison['improvement'] == 0:
            print(f"  Same as NetworkX greedy coloring")
        else:
            print(f"  NetworkX outperforms by {abs(comparison['improvement'])} colors")
        
        return comparison
        
    except Exception as e:
        print(f"  Error comparing with NetworkX: {e}")
        return {'error': str(e)}


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Quantum Column Generation for Minimum Vertex Graph Coloring",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main options
    parser.add_argument("--test", action="store_true",
                       help="Run tests on predefined graphs")
    parser.add_argument("--erdos-test", action="store_true", 
                       help="Test on Erdős-Rényi random graphs")
    parser.add_argument("--oracle", type=str, default="jax-pgd", choices=["jax-pgd", "dirac"],
                       help="Oracle type for pricing subproblem (default: jax-pgd)")
    parser.add_argument("--compare-oracles", action="store_true",
                       help="Compare different oracles")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--save-plots", type=str, default="./plots/",
                       help="Directory to save plots")
    parser.add_argument("--compare-networkx", action="store_true",
                       help="Compare with NetworkX greedy coloring")
    parser.add_argument("--compare-classical", action="store_true",
                       help="Compare quantum CG with classical CG")
    parser.add_argument("--classical-only", action="store_true",
                       help="Run only classical CG (useful when oracle system unavailable)")
    
    # Column generation parameters
    cg_group = parser.add_argument_group('Column Generation Parameters')
    cg_group.add_argument("--max-iterations", type=int, default=50,
                         help="Maximum column generation iterations (default: 50)")
    cg_group.add_argument("--tolerance", type=float, default=1e-6,
                         help="Convergence tolerance (default: 1e-6)")
    
    # Erdős-Rényi parameters
    er_group = parser.add_argument_group('Erdős-Rényi Graph Parameters')
    er_group.add_argument("--nodes", type=int, default=8,
                         help="Number of nodes (default: 8)")
    er_group.add_argument("--edge-prob", type=float, default=0.4,
                         help="Edge probability (default: 0.4)")
    
    # Oracle parameters
    jax_group = parser.add_argument_group('JAX-PGD Oracle Parameters')
    jax_group.add_argument("--num-restarts", type=int, default=50,
                          help="Number of restarts (default: 50)")
    jax_group.add_argument("--learning-rate", type=float, default=0.01,
                          help="Learning rate (default: 0.01)")
    jax_group.add_argument("--max-iter", type=int, default=2000,
                          help="Maximum iterations per restart (default: 2000)")
    
    dirac_group = parser.add_argument_group('Dirac-3 Oracle Parameters')
    dirac_group.add_argument("--num-samples", type=int, default=20,
                            help="Number of samples (default: 20)")
    dirac_group.add_argument("--relax-schedule", type=int, default=2, choices=[1,2,3,4],
                            help="Relaxation schedule (default: 2)")
    
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
    
    # Create plots directory
    if args.plot:
        os.makedirs(args.save_plots, exist_ok=True)
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available - plotting disabled")
            args.plot = False
    
    # Check oracle availability for quantum solvers
    if not ORACLE_SYSTEM_AVAILABLE and not (args.compare_classical or args.classical_only):
        print("Error: Oracle system not available for quantum column generation")
        print("Use --compare-classical or --classical-only to run classical column generation")
        return 1
    
    # If only classical comparison is requested and oracles are not available, that's ok
    if not ORACLE_SYSTEM_AVAILABLE and (args.compare_classical or args.classical_only):
        print("Warning: Oracle system not available - running only classical column generation")
        args.classical_only = True  # Force classical-only mode
    
    if args.test:
        print("Testing Quantum Column Generation on predefined graphs")
        print("=" * 60)
        
        test_graphs = create_test_graphs()
        
        for name, graph in test_graphs:
            print(f"\n--- Testing {name} ---")
            print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            
            # Prepare oracle configuration
            oracle_config = {}
            if args.oracle == 'jax-pgd':
                oracle_config = {
                    'num_restarts': args.num_restarts,
                    'learning_rate': args.learning_rate,
                    'max_iterations': args.max_iter
                }
            elif args.oracle == 'dirac':
                oracle_config = {
                    'num_samples': args.num_samples,
                    'relaxation_schedule': args.relax_schedule
                }
            
            # Create quantum solver only if not in classical-only mode and oracles are available
            quantum_solver = None
            if not args.classical_only and ORACLE_SYSTEM_AVAILABLE:
                quantum_solver = QuantumColumnGenerationSolver(
                    oracle_type=args.oracle,
                    verbose=args.verbose,
                    max_cg_iterations=args.max_iterations,
                    tolerance=args.tolerance,
                    enable_refinement=args.enable_refinement,
                    **oracle_config
                )
            
            try:
                # Initialize variables
                num_colors = None
                coloring = []
                stats = {}
                classical_num_colors = None
                classical_coloring = None
                classical_stats = None
                
                # Run classical solver if requested or in classical-only mode
                if args.compare_classical or args.classical_only:
                    if args.verbose:
                        print(f"\n--- Running Classical CG ---")
                    
                    classical_solver = ClassicalColumnGenerationSolver(
                        verbose=args.verbose,
                        max_cg_iterations=args.max_iterations,
                        tolerance=args.tolerance
                    )
                    
                    try:
                        classical_num_colors, classical_coloring, classical_stats = classical_solver.solve(graph)
                    except Exception as e:
                        if args.verbose:
                            print(f"Classical CG failed: {e}")
                        classical_num_colors, classical_coloring, classical_stats = None, [], {}
                
                # Run quantum solver if not in classical-only mode and oracles are available
                if not args.classical_only and ORACLE_SYSTEM_AVAILABLE:
                    if args.verbose:
                        print(f"\n--- Running Quantum CG ---")
                    num_colors, coloring, stats = quantum_solver.solve(graph)
                
                # Display results
                success = False
                
                if args.classical_only:
                    # Classical-only mode
                    if classical_num_colors is not None:
                        print(f"\n--- Classical CG Results ---")
                        print(f"Solution: {classical_num_colors} colors")
                        print(f"Runtime: {classical_stats['total_time']:.3f}s")
                        print(f"Iterations: {classical_stats['iterations']}")
                        success = True
                        
                        # Use classical results for plotting
                        num_colors, coloring = classical_num_colors, classical_coloring
                        
                    else:
                        print("Classical CG solution failed")
                        
                elif num_colors is not None:
                    # Quantum solver succeeded
                    print(f"\n--- Quantum CG Results ---")
                    print(f"Solution: {num_colors} colors")
                    print(f"Runtime: {stats['total_time']:.3f}s")
                    print(f"Iterations: {stats['iterations']}")
                    success = True
                    
                    # Classical vs Quantum comparison if requested  
                    if args.compare_classical and classical_coloring is not None:
                        if classical_num_colors is not None:
                            print(f"\n--- Classical CG Results ---")
                            print(f"Solution: {classical_num_colors} colors")
                            print(f"Runtime: {classical_stats['total_time']:.3f}s")
                            print(f"Iterations: {classical_stats['iterations']}")
                            
                            print(f"\n--- Performance Comparison ---")
                            compare_classical_vs_quantum(graph, classical_coloring, coloring)
                            
                            # Performance metrics comparison
                            print(f"\n--- Timing Comparison ---")
                            print(f"Classical total time: {classical_stats['total_time']:.3f}s")
                            print(f"Quantum total time: {stats['total_time']:.3f}s")
                            if classical_stats['total_time'] > 0:
                                speedup = classical_stats['total_time'] / stats['total_time']
                                print(f"Speedup: {speedup:.2f}x")
                        else:
                            print(f"\n--- Classical CG failed, only Quantum CG succeeded ---")
                            
                else:
                    print("Quantum CG solution failed")
                    
                    # Fallback to classical if available
                    if classical_num_colors is not None:
                        print(f"\n--- Fallback to Classical CG Results ---")
                        print(f"Solution: {classical_num_colors} colors")
                        print(f"Runtime: {classical_stats['total_time']:.3f}s")
                        print(f"Iterations: {classical_stats['iterations']}")
                        num_colors, coloring = classical_num_colors, classical_coloring
                        success = True
                    else:
                        print("Both solvers failed")
                
                # Compare with NetworkX if requested
                if success and args.compare_networkx:
                    method_name = "Classical CG" if args.classical_only else "Quantum CG" 
                    compare_with_networkx(graph, coloring, method_name)
                
                # Generate plot if requested
                if success and args.plot:
                    plot_suffix = "classical_cg" if args.classical_only else "quantum_cg"
                    plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_')}_{plot_suffix}.png")
                    plot_title = f"{name} - {'Classical' if args.classical_only else 'Quantum'} Column Generation"
                    plot_coloring_solution(graph, coloring, plot_title, plot_path)
                    
                    # Also plot classical solution if comparing
                    if args.compare_classical and not args.classical_only and classical_coloring and classical_num_colors is not None:
                        classical_plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_')}_classical_cg.png")
                        plot_coloring_solution(graph, classical_coloring, f"{name} - Classical Column Generation", classical_plot_path)
                    
            except Exception as e:
                print(f"Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    elif args.erdos_test:
        print("Testing Quantum Column Generation on Erdős-Rényi graphs")
        print("=" * 60)
        
        # Generate random graph
        graph = nx.erdos_renyi_graph(args.nodes, args.edge_prob, seed=42)
        name = f"Erdős-Rényi G({args.nodes}, {args.edge_prob})"
        
        print(f"\n--- Testing {name} ---")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        # Configure oracle settings and create quantum solver if needed
        oracle_config = {}
        quantum_solver = None
        
        if not args.classical_only and ORACLE_SYSTEM_AVAILABLE:
            if args.oracle == 'jax-pgd':
                oracle_config = {
                    'num_restarts': args.num_restarts,
                    'learning_rate': args.learning_rate,
                    'max_iterations': args.max_iter
                }
            elif args.oracle == 'dirac':
                oracle_config = {
                    'num_samples': args.num_samples,
                    'relaxation_schedule': args.relax_schedule
                }
            
            quantum_solver = QuantumColumnGenerationSolver(
                oracle_type=args.oracle,
                verbose=args.verbose,
                max_cg_iterations=args.max_iterations,
                tolerance=args.tolerance,
                enable_refinement=args.enable_refinement,
                **oracle_config
            )
        
        try:
            # Initialize variables
            num_colors = None
            coloring = []
            stats = {}
            classical_num_colors = None
            classical_coloring = None
            classical_stats = None
            
            # Run classical solver if requested or in classical-only mode
            if args.compare_classical or args.classical_only:
                if args.verbose:
                    print(f"\n--- Running Classical CG ---")
                
                classical_solver = ClassicalColumnGenerationSolver(
                    verbose=args.verbose,
                    max_cg_iterations=args.max_iterations,
                    tolerance=args.tolerance
                )
                
                try:
                    classical_num_colors, classical_coloring, classical_stats = classical_solver.solve(graph)
                except Exception as e:
                    if args.verbose:
                        print(f"Classical CG failed: {e}")
                    classical_num_colors, classical_coloring, classical_stats = None, [], {}
            
            # Run quantum solver if available
            if quantum_solver is not None:
                if args.verbose:
                    print(f"\n--- Running Quantum CG ---")
                num_colors, coloring, stats = quantum_solver.solve(graph)
            
            # Display results
            success = False
            
            if args.classical_only:
                # Classical-only mode
                if classical_num_colors is not None:
                    print(f"\n--- Classical CG Results ---")
                    print(f"Solution: {classical_num_colors} colors")
                    print(f"Runtime: {classical_stats['total_time']:.3f}s")
                    print(f"Iterations: {classical_stats['iterations']}")
                    success = True
                    
                    # Use classical results for plotting
                    num_colors, coloring = classical_num_colors, classical_coloring
                    
                else:
                    print("Classical CG solution failed")
                    
            elif num_colors is not None:
                # Quantum solver succeeded
                print(f"\n--- Quantum CG Results ---")
                print(f"Solution: {num_colors} colors")
                print(f"Runtime: {stats['total_time']:.3f}s")
                print(f"Iterations: {stats['iterations']}")
                success = True
                
                # Classical vs Quantum comparison if requested  
                if args.compare_classical and classical_coloring is not None:
                    if classical_num_colors is not None:
                        print(f"\n--- Classical CG Results ---")
                        print(f"Solution: {classical_num_colors} colors")
                        print(f"Runtime: {classical_stats['total_time']:.3f}s")
                        print(f"Iterations: {classical_stats['iterations']}")
                        
                        print(f"\n--- Performance Comparison ---")
                        compare_classical_vs_quantum(graph, classical_coloring, coloring)
                        
                        # Performance metrics comparison
                        print(f"\n--- Timing Comparison ---")
                        print(f"Classical total time: {classical_stats['total_time']:.3f}s")
                        print(f"Quantum total time: {stats['total_time']:.3f}s")
                        if classical_stats['total_time'] > 0:
                            speedup = classical_stats['total_time'] / stats['total_time']
                            print(f"Speedup: {speedup:.2f}x")
                    else:
                        print(f"\n--- Classical CG failed, only Quantum CG succeeded ---")
                        
            else:
                print("Quantum CG solution failed")
                
                # Fallback to classical if available
                if classical_num_colors is not None:
                    print(f"\n--- Fallback to Classical CG Results ---")
                    print(f"Solution: {classical_num_colors} colors")
                    print(f"Runtime: {classical_stats['total_time']:.3f}s")
                    print(f"Iterations: {classical_stats['iterations']}")
                    num_colors, coloring = classical_num_colors, classical_coloring
                    success = True
                else:
                    print("Both solvers failed")
            
            # Compare with NetworkX if requested
            if success and args.compare_networkx:
                method_name = "Classical CG" if args.classical_only else "Quantum CG" 
                compare_with_networkx(graph, coloring, method_name)
            
            # Generate plot if requested
            if success and args.plot:
                plot_suffix = "classical_cg" if args.classical_only else "quantum_cg"
                plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}_{plot_suffix}.png")
                plot_title = f"{name} - {'Classical' if args.classical_only else 'Quantum'} Column Generation"
                plot_coloring_solution(graph, coloring, plot_title, plot_path)
                
                # Also plot classical solution if comparing
                if args.compare_classical and not args.classical_only and classical_coloring and classical_num_colors is not None:
                    classical_plot_path = os.path.join(args.save_plots, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')}_classical_cg.png")
                    plot_coloring_solution(graph, classical_coloring, f"{name} - Classical Column Generation", classical_plot_path)
                
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    else:
        print("Quantum Column Generation for Minimum Vertex Graph Coloring")
        print("="*60)
        print("Use --test to run on predefined graphs")
        print("Use --erdos-test to run on random graphs") 
        print("Use --compare-classical to compare quantum vs classical column generation")
        print("Use --help to see all options")
        print("\nExample: python scripts/column_gen.py --test --compare-classical --verbose")
    
    return 0


if __name__ == "__main__":
    exit(main())