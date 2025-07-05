"""
Performance comparison and demonstration tests for SciPy MILP vs Gurobi solvers.
"""

import pytest
import time
import networkx as nx

from motzkinstraus import (
    solve_max_clique_scipy,
    solve_mis_scipy,
    verify_clique,
    verify_independent_set,
    GUROBI_AVAILABLE,
    SCIPY_MILP_AVAILABLE
)

if GUROBI_AVAILABLE:
    from motzkinstraus import (
        solve_max_clique_gurobi,
        solve_mis_gurobi
    )


@pytest.mark.skipif(not SCIPY_MILP_AVAILABLE, reason="SciPy MILP not available")
class TestScipyMILPDemo:
    """Test and demonstrate SciPy MILP solver functionality."""
    
    @pytest.fixture
    def demo_graphs(self):
        """Create various test graphs for comparison."""
        graphs = []
        
        # Small graphs for validation
        graphs.append(("Triangle K3", nx.complete_graph(3)))
        graphs.append(("4-cycle", nx.cycle_graph(4)))
        graphs.append(("Complete K5", nx.complete_graph(5)))
        graphs.append(("Petersen", nx.petersen_graph()))
        graphs.append(("Wheel 8", nx.wheel_graph(8)))
        graphs.append(("Grid 3x3", nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))))
        graphs.append(("Random G(12,0.4)", nx.erdos_renyi_graph(12, 0.4, seed=42)))
        
        return graphs
    
    def test_scipy_solver_correctness(self, demo_graphs):
        """Test that SciPy solver produces valid solutions."""
        for name, graph in demo_graphs:
            # SciPy solver
            clique_scipy = solve_max_clique_scipy(graph, suppress_output=True)
            mis_scipy = solve_mis_scipy(graph, suppress_output=True)
            
            # Verify solutions are valid
            assert verify_clique(graph, clique_scipy), f"Invalid clique for {name}"
            assert verify_independent_set(graph, mis_scipy), f"Invalid MIS for {name}"
    
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available for comparison")
    def test_scipy_vs_gurobi_comparison(self, demo_graphs):
        """Compare SciPy and Gurobi solver results."""
        total_scipy_time = 0
        total_gurobi_time = 0
        
        for name, graph in demo_graphs:
            # SciPy solver
            start_time = time.time()
            clique_scipy = solve_max_clique_scipy(graph, suppress_output=True)
            mis_scipy = solve_mis_scipy(graph, suppress_output=True)
            scipy_time = time.time() - start_time
            total_scipy_time += scipy_time
            
            # Gurobi solver
            start_time = time.time()
            clique_gurobi = solve_max_clique_gurobi(graph, suppress_output=True)
            mis_gurobi = solve_mis_gurobi(graph, suppress_output=True)
            gurobi_time = time.time() - start_time
            total_gurobi_time += gurobi_time
            
            # Verify solutions are valid
            assert verify_clique(graph, clique_scipy), f"Invalid SciPy clique for {name}"
            assert verify_independent_set(graph, mis_scipy), f"Invalid SciPy MIS for {name}"
            assert verify_clique(graph, clique_gurobi), f"Invalid Gurobi clique for {name}"
            assert verify_independent_set(graph, mis_gurobi), f"Invalid Gurobi MIS for {name}"
            
            # Check if results match (should be optimal)
            assert len(clique_scipy) == len(clique_gurobi), \
                f"Clique size mismatch for {name}: SciPy={len(clique_scipy)}, Gurobi={len(clique_gurobi)}"
            assert len(mis_scipy) == len(mis_gurobi), \
                f"MIS size mismatch for {name}: SciPy={len(mis_scipy)}, Gurobi={len(mis_gurobi)}"
        
        # Performance comparison
        speed_ratio = total_gurobi_time / total_scipy_time if total_scipy_time > 0 else float('inf')
        print(f"\nPerformance Summary:")
        print(f"Total SciPy time: {total_scipy_time:.4f}s")
        print(f"Total Gurobi time: {total_gurobi_time:.4f}s")
        print(f"Speed ratio (Gurobi/SciPy): {speed_ratio:.2f}x")
    
    def test_edge_cases(self):
        """Test SciPy solver on edge cases."""
        # Empty graph
        empty = nx.empty_graph(3)
        clique = solve_max_clique_scipy(empty)
        mis = solve_mis_scipy(empty)
        assert len(clique) == 1, "Empty graph should have clique size 1"
        assert len(mis) == 3, "Empty graph should have MIS size 3"
        
        # Complete graph
        complete = nx.complete_graph(4)
        clique = solve_max_clique_scipy(complete)
        mis = solve_mis_scipy(complete)
        assert len(clique) == 4, "Complete K4 should have clique size 4"
        assert len(mis) == 1, "Complete K4 should have MIS size 1"
        
        # Single node
        single = nx.Graph()
        single.add_node(0)
        clique = solve_max_clique_scipy(single)
        mis = solve_mis_scipy(single)
        assert len(clique) == 1, "Single node should have clique size 1"
        assert len(mis) == 1, "Single node should have MIS size 1"
    
    @pytest.mark.slow
    def test_performance_on_larger_graph(self):
        """Performance test on a larger graph."""
        large_graph = nx.erdos_renyi_graph(20, 0.3, seed=123)
        
        start_time = time.time()
        clique = solve_max_clique_scipy(large_graph)
        mis = solve_mis_scipy(large_graph)
        scipy_time = time.time() - start_time
        
        # Verify solutions
        assert verify_clique(large_graph, clique), "Invalid clique on large graph"
        assert verify_independent_set(large_graph, mis), "Invalid MIS on large graph"
        
        # Should complete in reasonable time (< 1 second for 20 nodes)
        assert scipy_time < 1.0, f"SciPy solver too slow: {scipy_time:.4f}s"
        
        if GUROBI_AVAILABLE:
            start_time = time.time()
            clique_g = solve_max_clique_gurobi(large_graph)
            mis_g = solve_mis_gurobi(large_graph)
            gurobi_time = time.time() - start_time
            
            # Results should match
            assert len(clique) == len(clique_g), "Clique size mismatch on large graph"
            assert len(mis) == len(mis_g), "MIS size mismatch on large graph"