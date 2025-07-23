#!/usr/bin/env python3
"""
Non-Weighted Column Generation Algorithm for Independent Set Problem using Regularized Motzkin-Straus

Theory:
- Column generation approach to the set covering formulation of the independent set problem
- RMP: minimize Σ λⱼ subject to Σⱼ aᵢⱼλⱼ ≥ 1 for all vertices i, λⱼ ≥ 0
- PSP: Find independent sets with negative reduced cost using regularized Motzkin-Straus sampling
- Uses pure regularized Motzkin-Straus (no linear terms): max x^T (A + cI) x
- Post-sampling evaluation: reduced_cost = 1 - Σᵢ∈S πᵢ where πᵢ are dual values

Algorithm:
1. Initialize RMP with singleton independent sets
2. Solve RMP to get dual values πᵢ
3. Filter nodes: remove vertices with πᵢ < 0 (cannot contribute to profitable columns)
4. Sample independent sets on filtered complement graph using regularized Motzkin-Straus
5. Evaluate profitability and add negative reduced cost columns
6. Repeat until convergence (no profitable columns found)

Implementation Features:
- Support for both JAX-PGD and Dirac-3 oracles for PSP sampling
- Node filtering optimization for improved efficiency
- Multiple sampling per iteration for diversity
- Automatic column deduplication
- Comprehensive testing and visualization

Usage Examples:

    # Test with JAX-PGD oracle
    python scripts/colgen_reg.py --test --oracle jax-pgd --regularization-c 0.1
    
    # Test with Dirac-3 oracle and more samples
    python scripts/colgen_reg.py --test --oracle dirac --num-samples 50 --regularization-c 0.2
    
    # Compare oracles on Erdős-Rényi graphs
    python scripts/colgen_reg.py --erdos-test --nodes 15 --compare-oracles --verbose
    
    # Custom convergence and iteration limits
    python scripts/colgen_reg.py --test --max-iterations 20 --convergence-tol 1e-4
    
Parameters:
    --test                  Run tests on predefined graphs
    --erdos-test           Test on Erdős-Rényi random graphs  
    --oracle TYPE          Oracle solver: 'jax-pgd' or 'dirac' (default: jax-pgd)
    --compare-oracles      Compare both oracles on same instances
    --regularization-c C   Regularization parameter for A → A + cI (default: 0.1)
    --max-iterations N     Maximum column generation iterations (default: 100)
    --num-samples N        Samples per PSP call (default: 20)
    --convergence-tol T    Convergence tolerance for reduced cost (default: 1e-6)
    --verbose             Detailed output with iteration progress
    --plot                Generate convergence and solution quality plots
    --nodes N             Number of nodes for Erdős-Rényi graphs (default: 10)
    --edge-prob P         Edge probability for random graphs (default: 0.5)

Oracle-Specific Parameters:
    --num-restarts N      Number of JAX-PGD restarts (default: 10)
    --learning-rate R     JAX-PGD learning rate (default: 0.01)
    --max-oracle-iter N   Maximum iterations per oracle call (default: 1000)
    --relax-schedule N    Dirac relaxation schedule 1-4 (default: 2)
"""

import sys
import os
import networkx as nx
import numpy as np
import time
from typing import Set, List, Tuple, Dict, Any, Optional
import argparse
from dataclasses import dataclass
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from motzkinstraus.algorithms import verify_clique
    from motzkinstraus.io import read_dimacs_graph
    print("Successfully imported motzkinstraus modules")
except ImportError as e:
    print(f"Error importing motzkinstraus modules: {e}")
    print("Make sure you're in the correct directory and virtual environment is activated")
    sys.exit(1)

# Try to import scipy for linear programming
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
    print("SciPy available for RMP solving")
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available - RMP solving disabled")
    sys.exit(1)

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib available for plotting")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available - plotting disabled")

# Import existing oracle functionality
try:
    # Add scripts directory to path for importing existing functions
    scripts_dir = os.path.dirname(__file__)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    from clique_instance_reg import (
        _graph_to_qplib_standard,
        _qplib_to_polynomial_file_fixed_degrees,
        _regularized_dirac_implementation,
        _legacy_jax_pgd_implementation,
        extract_support,
    )
    print("Successfully imported existing oracle functions")
except ImportError as e:
    print(f"Error importing oracle functions: {e}")
    sys.exit(1)

# Import regularization functions
try:
    from regularized_graph_to_omega import apply_identity_regularization
    print("Successfully imported regularization functions")
except ImportError as e:
    print(f"Error importing regularization functions: {e}")
    sys.exit(1)


@dataclass
class ColumnGenerationResult:
    """Result of column generation algorithm."""
    independent_sets: List[Set[int]]
    objective_value: float
    iterations: int
    total_columns: int
    convergence_history: List[Dict[str, Any]]
    solve_time: float


class IndependentSetColumn:
    """Represents a column (independent set) in the RMP."""
    
    def __init__(self, vertices: Set[int]):
        """
        Initialize an independent set column.
        
        Args:
            vertices: Set of vertices forming the independent set
        """
        self.vertices = frozenset(vertices)
        self._hash = hash(self.vertices)
    
    def covers_vertex(self, vertex: int) -> bool:
        """Check if this column covers a given vertex."""
        return vertex in self.vertices
    
    def reduced_cost(self, dual_values: Dict[int, float]) -> float:
        """Calculate reduced cost: 1 - Σ(πᵢ for i in vertices)."""
        return 1.0 - sum(dual_values.get(v, 0.0) for v in self.vertices)
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, IndependentSetColumn):
            return False
        return self.vertices == other.vertices
    
    def __repr__(self) -> str:
        return f"IndependentSetColumn({sorted(self.vertices)})"


class RMPSolver:
    """Restricted Master Problem solver using linear programming."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize RMP solver.
        
        Args:
            graph: Original graph for independent set problem
        """
        self.graph = graph
        self.vertices = list(graph.nodes())
        self.columns: List[IndependentSetColumn] = []
        self.vertex_to_index = {v: i for i, v in enumerate(self.vertices)}
        
    def add_column(self, column: IndependentSetColumn):
        """Add a new column to the RMP."""
        if column not in self.columns:
            self.columns.append(column)
    
    def solve(self) -> Tuple[Dict[int, float], float]:
        """
        Solve the RMP and return dual values and objective.
        
        Returns:
            Tuple of (dual_values, objective_value)
        """
        if not self.columns:
            # No columns - return zero duals
            return {v: 0.0 for v in self.vertices}, float('inf')
        
        # Build constraint matrix A
        # A[i,j] = 1 if column j covers vertex i, 0 otherwise
        num_vertices = len(self.vertices)
        num_columns = len(self.columns)
        
        A_ub = np.zeros((num_vertices, num_columns))
        for j, column in enumerate(self.columns):
            for vertex in column.vertices:
                if vertex in self.vertex_to_index:
                    i = self.vertex_to_index[vertex]
                    A_ub[i, j] = -1.0  # Convert >= to <=
        
        # RHS for constraints: Σⱼ aᵢⱼλⱼ ≥ 1 becomes -Σⱼ aᵢⱼλⱼ ≤ -1
        b_ub = -np.ones(num_vertices)
        
        # Objective: minimize Σ λⱼ
        c = np.ones(num_columns)
        
        # Bounds: λⱼ ≥ 0
        bounds = [(0, None) for _ in range(num_columns)]
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                objective_value = result.fun
                # Extract dual values (negative of shadow prices for >= constraints)
                dual_values = {self.vertices[i]: -result.ineqlin.marginals[i] 
                             for i in range(num_vertices)}
                return dual_values, objective_value
            else:
                print(f"LP solver failed: {result.message}")
                return {v: 0.0 for v in self.vertices}, float('inf')
                
        except Exception as e:
            print(f"Error solving RMP: {e}")
            return {v: 0.0 for v in self.vertices}, float('inf')
    
    def is_converged(self, min_reduced_cost: float, tolerance: float = 1e-6) -> bool:
        """Check if column generation has converged."""
        return min_reduced_cost >= -tolerance


class PSPSamplingOracle:
    """Pricing Subproblem oracle using sampling-based approach with node filtering."""
    
    def __init__(self, oracle_type: str, regularization_c: float, num_samples: int = 20, debug: bool = False):
        """
        Initialize PSP oracle.
        
        Args:
            oracle_type: 'jax-pgd' or 'dirac'
            regularization_c: Regularization parameter c
            num_samples: Number of samples per PSP call
            debug: Enable detailed oracle debugging output
        """
        self.oracle_type = oracle_type.lower()
        self.regularization_c = regularization_c
        self.num_samples = num_samples
        self.debug = debug
        
        if self.oracle_type not in ['jax-pgd', 'dirac']:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
    
    def _filter_negative_dual_nodes(self, graph: nx.Graph, dual_values: Dict[int, float]) -> nx.Graph:
        """
        Remove vertices with negative dual values to improve PSP efficiency.
        
        Args:
            graph: Original graph
            dual_values: Dual values from RMP
            
        Returns:
            Filtered subgraph containing only vertices with non-negative dual values
        """
        positive_nodes = {v for v in graph.nodes() if dual_values.get(v, 0.0) >= 0}
        
        if not positive_nodes:
            # If all nodes have negative duals, return empty graph
            return nx.Graph()
        
        filtered_graph = graph.subgraph(positive_nodes).copy()
        return filtered_graph
    
    def _sample_independent_sets_jax_pgd(self, graph: nx.Graph) -> List[Set[int]]:
        """Sample independent sets using JAX-PGD oracle."""
        try:
            # Use existing JAX-PGD implementation - returns (cliques, details)
            cliques, details = _legacy_jax_pgd_implementation(
                graph=graph,
                regularization_c=self.regularization_c,
                num_restarts=self.num_samples,
                support_threshold=1e-5,
                learning_rate=0.01,
                max_iterations=1000,
                verbose=self.debug
            )
            # Convert cliques to sets and return
            return [set(clique) for clique in cliques if clique]
        except Exception as e:
            print(f"JAX-PGD sampling failed: {e}")
            return []
    
    def _sample_independent_sets_dirac(self, graph: nx.Graph) -> List[Set[int]]:
        """Sample independent sets using Dirac oracle."""
        try:
            # Use existing Dirac implementation - returns (cliques, details)
            cliques, details = _regularized_dirac_implementation(
                graph=graph,
                regularization_c=self.regularization_c,
                num_samples=self.num_samples,
                relax_schedule=2,
                verbose=self.debug
            )
            # Convert cliques to sets and return
            return [set(clique) for clique in cliques if clique]
        except Exception as e:
            print(f"Dirac sampling failed: {e}")
            return []
    
    def sample_independent_sets(self, graph: nx.Graph, dual_values: Dict[int, float], debug: bool = False) -> List[Set[int]]:
        """
        Sample independent sets using node filtering and regularized Motzkin-Straus.
        
        Args:
            graph: Original graph
            dual_values: Dual values from RMP
            debug: Enable detailed debugging output
            
        Returns:
            List of sampled independent sets
        """
        if debug:
            print(f"  🔍 PSP Debug - Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"  🔍 PSP Debug - Graph edges: {list(graph.edges())}")
            print(f"  🔍 PSP Debug - Dual values: {dual_values}")
        
        # Step 1: Filter nodes with negative dual values
        filtered_graph = self._filter_negative_dual_nodes(graph, dual_values)
        
        if debug:
            positive_nodes = {v for v in graph.nodes() if dual_values.get(v, 0.0) >= 0}
            negative_nodes = {v for v in graph.nodes() if dual_values.get(v, 0.0) < 0}
            print(f"  🔍 PSP Debug - Positive dual nodes: {positive_nodes}")
            print(f"  🔍 PSP Debug - Negative dual nodes (filtered out): {negative_nodes}")
            print(f"  🔍 PSP Debug - Filtered graph: {filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges")
        
        if filtered_graph.number_of_nodes() == 0:
            if debug:
                print("  🔍 PSP Debug - All nodes filtered out, returning empty")
            return []
        
        # Step 2: Sample on complement graph (independent sets become cliques)
        complement_graph = nx.complement(filtered_graph)
        
        if debug:
            print(f"  🔍 PSP Debug - Complement graph: {complement_graph.number_of_nodes()} nodes, {complement_graph.number_of_edges()} edges")
            print(f"  🔍 PSP Debug - Complement edges: {list(complement_graph.edges())}")
        
        if complement_graph.number_of_nodes() == 0:
            return []
        
        # Step 3: Use appropriate oracle to sample
        if self.oracle_type == 'jax-pgd':
            independent_sets = self._sample_independent_sets_jax_pgd(complement_graph)
        else:  # dirac
            independent_sets = self._sample_independent_sets_dirac(complement_graph)
        
        if debug:
            print(f"  🔍 PSP Debug - Oracle sampled {len(independent_sets)} sets: {independent_sets}")
        
        # Step 4: Validate that all returned sets are actually independent sets
        valid_independent_sets = []
        for is_set in independent_sets:
            is_valid = self._is_independent_set(graph, is_set)
            if debug:
                print(f"  🔍 PSP Debug - Checking {is_set}: valid={is_valid}")
            if is_valid:
                valid_independent_sets.append(is_set)
        
        if debug:
            print(f"  🔍 PSP Debug - Valid independent sets: {valid_independent_sets}")
        
        return valid_independent_sets
    
    def _is_independent_set(self, graph: nx.Graph, vertices: Set[int]) -> bool:
        """Check if a set of vertices forms an independent set."""
        if not vertices:
            return True
        
        # Check that no two vertices in the set are adjacent
        for u in vertices:
            for v in vertices:
                if u != v and graph.has_edge(u, v):
                    return False
        return True
    
    def evaluate_profitability(self, samples: List[Set[int]], dual_values: Dict[int, float], debug: bool = False) -> List[Tuple[Set[int], float]]:
        """
        Evaluate profitability of sampled independent sets.
        
        Args:
            samples: List of independent sets
            dual_values: Dual values from RMP
            debug: Enable detailed debugging output
            
        Returns:
            List of (independent_set, reduced_cost) tuples for profitable columns only
        """
        if debug:
            print(f"  🔍 Profitability Debug - Evaluating {len(samples)} samples")
            
        profitable_columns = []
        seen_sets = set()
        
        for is_set in samples:
            if not is_set:
                continue
            
            # Avoid duplicates
            frozen_set = frozenset(is_set)
            if frozen_set in seen_sets:
                if debug:
                    print(f"  🔍 Profitability Debug - Skipping duplicate: {is_set}")
                continue
            seen_sets.add(frozen_set)
            
            # Calculate reduced cost: 1 - Σ(πᵢ for i in IS)
            sum_of_duals = sum(dual_values.get(v, 0.0) for v in is_set)
            reduced_cost = 1.0 - sum_of_duals
            
            if debug:
                dual_breakdown = [f"{v}:{dual_values.get(v, 0.0)}" for v in sorted(is_set)]
                print(f"  🔍 Profitability Debug - IS {sorted(is_set)}: duals=[{', '.join(dual_breakdown)}], sum={sum_of_duals:.3f}, reduced_cost={reduced_cost:.6f}")
            
            # Only keep profitable columns (negative reduced cost)
            if reduced_cost < 0:
                profitable_columns.append((is_set, reduced_cost))
                if debug:
                    print(f"  🔍 Profitability Debug - ✅ PROFITABLE: {sorted(is_set)} with reduced cost {reduced_cost:.6f}")
            elif debug:
                print(f"  🔍 Profitability Debug - ❌ Not profitable: {sorted(is_set)} with reduced cost {reduced_cost:.6f}")
        
        # Sort by most profitable (lowest reduced cost)
        profitable_columns.sort(key=lambda x: x[1])
        
        if debug:
            print(f"  🔍 Profitability Debug - Found {len(profitable_columns)} profitable columns")
        
        return profitable_columns


class ColumnGenerationSolver:
    """Main column generation solver for the independent set problem."""
    
    def __init__(self, graph: nx.Graph, oracle_type: str = 'jax-pgd', 
                 regularization_c: float = 0.1, max_iterations: int = 100,
                 num_samples: int = 20, convergence_tol: float = 1e-6, debug: bool = False):
        """
        Initialize column generation solver.
        
        Args:
            graph: Graph for independent set problem
            oracle_type: PSP oracle type ('jax-pgd' or 'dirac')
            regularization_c: Regularization parameter
            max_iterations: Maximum CG iterations
            num_samples: Samples per PSP call
            convergence_tol: Convergence tolerance
            debug: Enable detailed oracle debugging output
        """
        self.graph = graph
        self.oracle_type = oracle_type
        self.regularization_c = regularization_c
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        
        # Initialize components
        self.rmp_solver = RMPSolver(graph)
        self.psp_oracle = PSPSamplingOracle(oracle_type, regularization_c, num_samples, debug)
        
        # Track convergence history
        self.convergence_history: List[Dict[str, Any]] = []
    
    def _initialize_columns(self):
        """Initialize RMP with singleton independent sets."""
        for vertex in self.graph.nodes():
            singleton_column = IndependentSetColumn({vertex})
            self.rmp_solver.add_column(singleton_column)
    
    def solve(self) -> ColumnGenerationResult:
        """
        Solve the independent set problem using column generation.
        
        Returns:
            ColumnGenerationResult with solution and statistics
        """
        start_time = time.time()
        
        # Step 1: Initialize with singleton columns
        self._initialize_columns()
        print(f"Initialized with {len(self.rmp_solver.columns)} singleton columns")
        
        # Step 2: Column generation loop
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            debug_enabled = (iteration <= 1)  # Enable debug for first 2 iterations
            
            # Solve RMP
            dual_values, objective_value = self.rmp_solver.solve()
            
            if objective_value == float('inf'):
                print(f"Iteration {iteration}: RMP infeasible - need more columns")
                # Continue to PSP to generate more columns
            else:
                print(f"Iteration {iteration}: RMP objective = {objective_value:.6f}")
                if debug_enabled:
                    print(f"  🔍 RMP Debug - Dual values: {dual_values}")
                    print(f"  🔍 RMP Debug - Current columns:")
                    for i, col in enumerate(self.rmp_solver.columns):
                        print(f"    Column {i}: {sorted(col.vertices)}")
            
            # Solve PSP: sample independent sets
            sampled_sets = self.psp_oracle.sample_independent_sets(self.graph, dual_values, debug=debug_enabled)
            print(f"  Sampled {len(sampled_sets)} independent sets")
            
            # Evaluate profitability
            profitable_columns = self.psp_oracle.evaluate_profitability(sampled_sets, dual_values, debug=debug_enabled)
            print(f"  Found {len(profitable_columns)} profitable columns")
            
            if not profitable_columns:
                print(f"  No profitable columns found - converged!")
                break
            
            # Add profitable columns to RMP
            columns_added = 0
            min_reduced_cost = float('inf')
            
            for is_set, reduced_cost in profitable_columns:
                column = IndependentSetColumn(is_set)
                self.rmp_solver.add_column(column)
                columns_added += 1
                min_reduced_cost = min(min_reduced_cost, reduced_cost)
            
            iter_time = time.time() - iter_start
            
            # Record iteration statistics
            iter_stats = {
                'iteration': iteration,
                'objective_value': objective_value,
                'num_columns': len(self.rmp_solver.columns),
                'columns_added': columns_added,
                'min_reduced_cost': min_reduced_cost,
                'sampled_sets': len(sampled_sets),
                'profitable_columns': len(profitable_columns),
                'iteration_time': iter_time
            }
            self.convergence_history.append(iter_stats)
            
            print(f"  Added {columns_added} columns, min reduced cost: {min_reduced_cost:.6f}")
            print(f"  Total columns: {len(self.rmp_solver.columns)}, iteration time: {iter_time:.2f}s")
            
            # Check convergence
            if self.rmp_solver.is_converged(min_reduced_cost, self.convergence_tol):
                print(f"  Converged! (reduced cost {min_reduced_cost:.6f} >= -{self.convergence_tol})")
                break
        
        # Final solve to get solution
        final_dual_values, final_objective = self.rmp_solver.solve()
        total_time = time.time() - start_time
        
        # Extract independent sets from solution
        # For now, just return all columns (could extract actual solution values)
        independent_sets = [list(col.vertices) for col in self.rmp_solver.columns]
        
        result = ColumnGenerationResult(
            independent_sets=independent_sets,
            objective_value=final_objective,
            iterations=len(self.convergence_history),
            total_columns=len(self.rmp_solver.columns),
            convergence_history=self.convergence_history,
            solve_time=total_time
        )
        
        print(f"\nColumn Generation Complete:")
        print(f"  Final objective: {final_objective:.6f}")
        print(f"  Total iterations: {result.iterations}")
        print(f"  Total columns: {result.total_columns}")
        print(f"  Total time: {total_time:.2f}s")
        
        return result


def test_basic_functionality():
    """Test basic classes with simple examples."""
    print("Testing Basic Functionality")
    print("-" * 40)
    
    # Create a simple triangle graph
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    print(f"Test graph: {triangle.number_of_nodes()} nodes, {triangle.number_of_edges()} edges")
    
    # Test IndependentSetColumn
    col1 = IndependentSetColumn({0})
    col2 = IndependentSetColumn({1})
    col3 = IndependentSetColumn({0})  # Duplicate
    
    print(f"Column 1: {col1}")
    print(f"Column 2: {col2}")
    print(f"Column 3: {col3}")
    print(f"col1 == col3: {col1 == col3}")
    
    # Test RMPSolver with all singleton columns
    rmp = RMPSolver(triangle)
    rmp.add_column(IndependentSetColumn({0}))
    rmp.add_column(IndependentSetColumn({1}))
    rmp.add_column(IndependentSetColumn({2}))  # Add missing column
    
    print(f"RMP has {len(rmp.columns)} columns")
    
    dual_values, objective = rmp.solve()
    print(f"Dual values: {dual_values}")
    print(f"Objective: {objective}")
    
    # Test reduced cost calculation
    print(f"Column {0} reduced cost: {col1.reduced_cost(dual_values)}")
    print(f"Column {1} reduced cost: {col2.reduced_cost(dual_values)}")
    print()


def test_column_generation_algorithm():
    """Test the complete column generation algorithm."""
    print("Testing Column Generation Algorithm")
    print("-" * 40)
    
    # Test on a path graph (easier than triangle for independent set)
    path = nx.path_graph(4)  # Path: 0-1-2-3, optimal IS = {0,2} or {1,3}
    print(f"Path graph: {path.number_of_nodes()} nodes, {path.number_of_edges()} edges")
    
    # Initialize solver
    solver = ColumnGenerationSolver(
        graph=path,
        oracle_type='jax-pgd',  # Start with JAX-PGD (more reliable)
        regularization_c=0.1,
        max_iterations=10,
        num_samples=5,  # Small for testing
        convergence_tol=1e-6
    )
    
    try:
        # Solve
        result = solver.solve()
        
        print(f"\nSolution found:")
        print(f"  Independent sets: {result.independent_sets[:10]}")  # Show first 10
        print(f"  Best objective: {result.objective_value}")
        print(f"  Convergence iterations: {result.iterations}")
        print(f"  Total solve time: {result.solve_time:.2f}s")
        
    except Exception as e:
        print(f"Column generation failed: {e}")
        import traceback
        traceback.print_exc()


def debug_path4_case():
    """Debug the specific Path-4 case with detailed output."""
    print("Debugging Path-4 Case")
    print("=" * 50)
    
    # Create path graph
    path = nx.path_graph(4)  # 0-1-2-3
    print(f"Path graph edges: {list(path.edges())}")
    print(f"Expected optimal independent sets: {{0,2}} or {{1,3}} with objective = 2.0")
    print(f"Current suboptimal: 4 singletons with objective = 4.0")
    print()
    
    # Test the PSP oracle directly
    oracle = PSPSamplingOracle('jax-pgd', regularization_c=0.1, num_samples=10)
    
    # Simulate the dual values that would come from the singleton RMP solution
    dual_values = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}  # All constraints have dual 1.0
    
    print("Testing PSP Oracle with all-ones dual values:")
    print(f"Dual values: {dual_values}")
    print()
    
    # Sample independent sets with debugging
    sampled_sets = oracle.sample_independent_sets(path, dual_values, debug=True)
    print()
    
    # Evaluate profitability with debugging  
    profitable_columns = oracle.evaluate_profitability(sampled_sets, dual_values, debug=True)
    print()
    
    print("Summary:")
    print(f"  Sampled sets: {len(sampled_sets)}")
    print(f"  Profitable columns: {len(profitable_columns)}")
    if profitable_columns:
        print("  Most profitable:")
        for is_set, reduced_cost in profitable_columns[:3]:
            print(f"    {sorted(is_set)}: reduced_cost = {reduced_cost:.6f}")
    else:
        print("  ❌ No profitable columns found - this explains the convergence issue!")
    
    print()
    print("Expected profitable columns:")
    for expected_set in [{0, 2}, {1, 3}]:
        sum_duals = sum(dual_values[v] for v in expected_set)
        expected_rc = 1.0 - sum_duals
        print(f"  {sorted(expected_set)}: sum_duals={sum_duals}, reduced_cost={expected_rc:.6f} ({'✅ Profitable' if expected_rc < 0 else '❌ Not profitable'})")


def create_test_graphs() -> List[Tuple[str, nx.Graph]]:
    """Create test graphs for validation."""
    graphs = []
    
    # Triangle graph (clique)
    triangle = nx.Graph()
    triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graphs.append(("Triangle", triangle))
    
    # Path graph  
    path4 = nx.path_graph(4)
    graphs.append(("Path-4", path4))
    
    # Complete graph K4
    k4 = nx.complete_graph(4)
    graphs.append(("K4", k4))
    
    # Star graph
    star5 = nx.star_graph(4)
    graphs.append(("Star-5", star5))
    
    return graphs


def generate_erdos_renyi_graphs(nodes_list: List[int] = [8, 12], edge_prob: float = 0.5) -> List[Tuple[str, nx.Graph]]:
    """Generate Erdős-Rényi random graphs."""
    graphs = []
    
    for n in nodes_list:
        graph = nx.erdos_renyi_graph(n, edge_prob, seed=42)
        graphs.append((f"ER-{n}-{edge_prob}", graph))
    
    return graphs


def run_test_suite(oracle_type: str, regularization_c: float, max_iterations: int, 
                  num_samples: int, convergence_tol: float, verbose: bool, debug: bool = False):
    """Run column generation on test graphs."""
    print(f"\nRunning Test Suite with {oracle_type.upper()} Oracle")
    print(f"Parameters: c={regularization_c}, max_iter={max_iterations}, samples={num_samples}")
    print("=" * 80)
    
    test_graphs = create_test_graphs()
    
    for graph_name, graph in test_graphs:
        print(f"\nTesting {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print("-" * 60)
        
        try:
            solver = ColumnGenerationSolver(
                graph=graph,
                oracle_type=oracle_type,
                regularization_c=regularization_c,
                max_iterations=max_iterations,
                num_samples=num_samples,
                convergence_tol=convergence_tol,
                debug=debug
            )
            
            result = solver.solve()
            
            print(f"  ✅ Success: objective={result.objective_value:.3f}, iterations={result.iterations}, time={result.solve_time:.2f}s")
            
            if verbose:
                print(f"  Independent sets found: {len(result.independent_sets)}")
                for i, is_set in enumerate(result.independent_sets[:5]):  # Show first 5
                    print(f"    IS {i+1}: {sorted(is_set)}")
                if len(result.independent_sets) > 5:
                    print(f"    ... and {len(result.independent_sets)-5} more")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def run_erdos_test(oracle_type: str, regularization_c: float, max_iterations: int,
                  num_samples: int, convergence_tol: float, nodes: int, edge_prob: float, verbose: bool, debug: bool = False):
    """Run column generation on Erdős-Rényi graphs."""
    print(f"\nRunning Erdős-Rényi Test with {oracle_type.upper()} Oracle")
    print(f"Parameters: n={nodes}, p={edge_prob}, c={regularization_c}")
    print("=" * 80)
    
    erdos_graphs = generate_erdos_renyi_graphs([nodes], edge_prob)
    
    for graph_name, graph in erdos_graphs:
        print(f"\nTesting {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print("-" * 60)
        
        try:
            solver = ColumnGenerationSolver(
                graph=graph,
                oracle_type=oracle_type,
                regularization_c=regularization_c,
                max_iterations=max_iterations,
                num_samples=num_samples,
                convergence_tol=convergence_tol,
                debug=debug
            )
            
            result = solver.solve()
            
            print(f"  ✅ Success: objective={result.objective_value:.3f}, iterations={result.iterations}, time={result.solve_time:.2f}s")
            
            if verbose:
                print(f"  Convergence history:")
                for i, stats in enumerate(result.convergence_history[-3:]):  # Show last 3 iterations  
                    print(f"    Iter {stats['iteration']}: obj={stats['objective_value']:.3f}, "
                          f"cols_added={stats['columns_added']}, reduced_cost={stats['min_reduced_cost']:.6f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


def compare_oracles(graphs: List[Tuple[str, nx.Graph]], regularization_c: float, 
                   max_iterations: int, num_samples: int, convergence_tol: float, debug: bool = False):
    """Compare JAX-PGD and Dirac oracles on the same graphs."""
    print(f"\nComparing Oracles")
    print("=" * 80)
    
    oracles = ['jax-pgd', 'dirac']
    
    for graph_name, graph in graphs:
        print(f"\n{graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print("-" * 60)
        
        results = {}
        
        for oracle_type in oracles:
            try:
                solver = ColumnGenerationSolver(
                    graph=graph,
                    oracle_type=oracle_type,
                    regularization_c=regularization_c,
                    max_iterations=max_iterations,
                    num_samples=num_samples,
                    convergence_tol=convergence_tol
                )
                
                result = solver.solve()
                results[oracle_type] = result
                
                print(f"  {oracle_type.upper():8}: obj={result.objective_value:.3f}, "
                      f"iter={result.iterations:2d}, time={result.solve_time:.2f}s")
                
            except Exception as e:
                print(f"  {oracle_type.upper():8}: FAILED - {e}")
                results[oracle_type] = None
        
        # Compare results
        if all(r is not None for r in results.values()):
            jax_obj = results['jax-pgd'].objective_value
            dirac_obj = results['dirac'].objective_value
            diff = abs(jax_obj - dirac_obj)
            print(f"  Difference: {diff:.6f} ({'✅ Match' if diff < 1e-3 else '⚠️ Differ'})")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Non-Weighted Column Generation for Independent Set Problem",
        epilog="Examples:\n"
               "  python scripts/colgen_reg.py --test --oracle jax-pgd\n"
               "  python scripts/colgen_reg.py --erdos-test --nodes 12 --oracle dirac\n"
               "  python scripts/colgen_reg.py --test --compare-oracles --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test modes
    parser.add_argument('--test', action='store_true',
                       help='Run tests on predefined graphs')
    parser.add_argument('--erdos-test', action='store_true', 
                       help='Test on Erdős-Rényi random graphs')
    parser.add_argument('--compare-oracles', action='store_true',
                       help='Compare JAX-PGD and Dirac oracles')
    
    # Oracle and algorithm parameters
    parser.add_argument('--oracle', choices=['jax-pgd', 'dirac'], default='jax-pgd',
                       help='PSP oracle type (default: jax-pgd)')
    parser.add_argument('--regularization-c', type=float, default=0.1,
                       help='Regularization parameter (default: 0.1)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum column generation iterations (default: 50)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Samples per PSP call (default: 10)')
    parser.add_argument('--convergence-tol', type=float, default=1e-6,
                       help='Convergence tolerance (default: 1e-6)')
    
    # Graph parameters for Erdős-Rényi test
    parser.add_argument('--nodes', type=int, default=10,
                       help='Number of nodes for random graphs (default: 10)')
    parser.add_argument('--edge-prob', type=float, default=0.5,
                       help='Edge probability for random graphs (default: 0.5)')
    
    # Output control
    parser.add_argument('--verbose', action='store_true',
                       help='Detailed output with iteration progress')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed oracle debugging output')
    
    args = parser.parse_args()
    
    print("Column Generation for Independent Set Problem")
    print("=" * 60)
    
    if not any([args.test, args.erdos_test]):
        # Default: run basic tests
        print("No test specified, running basic functionality test...")
        test_basic_functionality()
        test_column_generation_algorithm()
        print()
        debug_path4_case()
        return
    
    if args.test:
        if args.compare_oracles:
            test_graphs = create_test_graphs()
            compare_oracles(test_graphs, args.regularization_c, args.max_iterations,
                          args.num_samples, args.convergence_tol, args.debug)
        else:
            run_test_suite(args.oracle, args.regularization_c, args.max_iterations,
                          args.num_samples, args.convergence_tol, args.verbose, args.debug)
    
    if args.erdos_test:
        run_erdos_test(args.oracle, args.regularization_c, args.max_iterations,
                      args.num_samples, args.convergence_tol, args.nodes, 
                      args.edge_prob, args.verbose, args.debug)


if __name__ == "__main__":
    main()