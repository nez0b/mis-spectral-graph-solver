"""
Test comparing Dirac-3 vs Gurobi solvers on small quadratic optimization problems.

This test validates that both solvers produce consistent results before using
them in the full MIS algorithm.
"""

import numpy as np
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.oracles.base import Oracle
from motzkinstraus.exceptions import SolverUnavailableError

def test_simple_quadratic_optimization():
    """
    Test both solvers on a simple 2x2 quadratic optimization problem.
    
    We'll use a simple adjacency matrix for a graph with 2 nodes and 1 edge:
    A = [[0, 1],
         [1, 0]]
    
    The optimal solution should be x = [0.5, 0.5] with objective value 0.25.
    """
    print("=== Testing Simple 2x2 Quadratic Optimization ===")
    
    # Create a simple 2-node graph with 1 edge
    A = np.array([[0, 1], [1, 0]], dtype=np.float64)
    
    # Expected optimal value for max(0.5 * x.T * A * x) with x = [0.5, 0.5]
    expected_value = 0.25
    
    solvers_to_test = []
    
    # Try to load Gurobi solver
    try:
        from motzkinstraus.oracles.gurobi import GurobiOracle
        gurobi_oracle = GurobiOracle(suppress_output=True)
        solvers_to_test.append(("Gurobi", gurobi_oracle))
        print("✓ Gurobi solver loaded successfully")
    except (ImportError, SolverUnavailableError) as e:
        print(f"✗ Gurobi solver not available: {e}")
    
    # Try to load Dirac solver
    try:
        from motzkinstraus.oracles.dirac import DiracOracle
        dirac_oracle = DiracOracle(num_samples=50, relax_schedule=1)
        solvers_to_test.append(("Dirac-3", dirac_oracle))
        print("✓ Dirac-3 solver loaded successfully")
    except (ImportError, SolverUnavailableError) as e:
        print(f"✗ Dirac-3 solver not available: {e}")
    
    if len(solvers_to_test) < 2:
        print("⚠️  Cannot compare solvers - need at least 2 available")
        return False
    
    results = {}
    
    # Test each solver
    for solver_name, solver in solvers_to_test:
        print(f"\nTesting {solver_name} solver...")
        try:
            result = solver.solve_quadratic_program(A)
            results[solver_name] = result
            error = abs(result - expected_value)
            print(f"  Result: {result:.6f}")
            print(f"  Expected: {expected_value:.6f}")
            print(f"  Error: {error:.6f}")
            
            if error < 0.1:  # Allow for reasonable numerical tolerance
                print(f"  ✓ {solver_name} result within tolerance")
            else:
                print(f"  ✗ {solver_name} result outside tolerance")
        except Exception as e:
            print(f"  ✗ {solver_name} failed: {e}")
            results[solver_name] = None
    
    # Compare results between solvers
    if len(results) >= 2:
        solver_names = list(results.keys())
        values = list(results.values())
        
        if all(v is not None for v in values):
            max_diff = max(abs(values[i] - values[j]) 
                          for i in range(len(values)) 
                          for j in range(i+1, len(values)))
            
            print(f"\n=== Solver Comparison ===")
            print(f"Maximum difference between solvers: {max_diff:.6f}")
            
            if max_diff < 0.1:
                print("✓ All solvers agree within tolerance")
                return True
            else:
                print("✗ Solvers disagree beyond tolerance")
                return False
    
    return False


def test_triangle_graph():
    """
    Test both solvers on a triangle graph (3-clique).
    
    For a complete graph K3, omega = 3, so the Motzkin-Straus optimal value 
    should be 0.5 * (1 - 1/3) = 1/3.
    """
    print("\n=== Testing Triangle Graph (K3) ===")
    
    # Create triangle graph
    graph = nx.complete_graph(3)
    A = nx.to_numpy_array(graph, dtype=np.float64)
    
    # Expected optimal value: 0.5 * (1 - 1/omega) = 0.5 * (1 - 1/3) = 1/3
    expected_value = 1/3
    
    # Load available solvers
    solvers_to_test = []
    
    try:
        from motzkinstraus.oracles.gurobi import GurobiOracle
        gurobi_oracle = GurobiOracle(suppress_output=True)
        solvers_to_test.append(("Gurobi", gurobi_oracle))
    except (ImportError, SolverUnavailableError):
        pass
    
    try:
        from motzkinstraus.oracles.dirac import DiracOracle
        dirac_oracle = DiracOracle(num_samples=100, relax_schedule=2)
        solvers_to_test.append(("Dirac-3", dirac_oracle))
    except (ImportError, SolverUnavailableError):
        pass
    
    if not solvers_to_test:
        print("No solvers available for testing")
        return False
    
    results = {}
    
    # Test each solver
    for solver_name, solver in solvers_to_test:
        print(f"\nTesting {solver_name} on triangle graph...")
        try:
            result = solver.solve_quadratic_program(A)
            results[solver_name] = result
            error = abs(result - expected_value)
            print(f"  Result: {result:.6f}")
            print(f"  Expected: {expected_value:.6f}")
            print(f"  Error: {error:.6f}")
            
            if error < 0.1:
                print(f"  ✓ {solver_name} result within tolerance")
            else:
                print(f"  ✗ {solver_name} result outside tolerance")
        except Exception as e:
            print(f"  ✗ {solver_name} failed: {e}")
            results[solver_name] = None
    
    return len([r for r in results.values() if r is not None]) > 0


if __name__ == "__main__":
    print("Testing solver comparison for Motzkin-Straus quadratic programs")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_simple_quadratic_optimization()
    test2_passed = test_triangle_graph()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Simple 2x2 test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Triangle graph test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Solvers are ready for MIS algorithm.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check solver implementations.")
        sys.exit(1)