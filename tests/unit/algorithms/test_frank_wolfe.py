"""
Test Frank-Wolfe oracle implementation against JAX PGD on small graphs.

Tests correctness and performance comparison between optimization methods.
"""

import pytest
import numpy as np
import networkx as nx

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

try:
    from motzkinstraus.oracles.jax_frank_wolfe import FrankWolfeOracle
    FRANK_WOLFE_AVAILABLE = True
except ImportError:
    FRANK_WOLFE_AVAILABLE = False

from motzkinstraus.benchmarks.networkx_comparison import run_algorithm_comparison


@pytest.mark.skipif(not FRANK_WOLFE_AVAILABLE, reason="Frank-Wolfe oracle not available")
class TestFrankWolfeOracle:
    """Test Frank-Wolfe oracle implementation."""
    
    @pytest.fixture
    def jax_pgd_oracle(self):
        """Create JAX PGD oracle for comparison."""
        return ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1000,
            num_restarts=10,
            tolerance=1e-6,
            verbose=False
        )
    
    @pytest.fixture
    def frank_wolfe_oracle(self):
        """Create Frank-Wolfe oracle."""
        return FrankWolfeOracle(
            num_restarts=10,
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )
    
    @pytest.fixture
    def frank_wolfe_test_graphs(self):
        """Graphs for Frank-Wolfe testing."""
        return [
            ("2-node complete", nx.complete_graph(2), 2),
            ("Triangle K3", nx.complete_graph(3), 3),
            ("4-clique K4", nx.complete_graph(4), 4),
            ("6-cycle", nx.cycle_graph(6), 2),
            ("12-node BA", nx.barabasi_albert_graph(12, 2, seed=42), None),
        ]
    
    def _compute_omega_from_result(self, qp_result, graph_nodes):
        """Compute omega from quadratic program result."""
        if qp_result < 0.5:
            omega = 1.0 / (1.0 - 2.0 * qp_result)
        else:
            omega = graph_nodes
        return omega
    
    def test_frank_wolfe_basic_functionality(self, frank_wolfe_test_graphs, frank_wolfe_oracle):
        """Test basic Frank-Wolfe oracle functionality."""
        for name, graph, expected_omega in frank_wolfe_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result = frank_wolfe_oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            # Basic sanity checks
            assert 0.0 <= result <= 1.0, f"QP result out of bounds for {name}: {result}"
            assert omega >= 1.0, f"Omega less than 1 for {name}: {omega}"
            
            # Check against expected omega if known
            if expected_omega is not None:
                assert omega_rounded == expected_omega, \
                    f"Frank-Wolfe oracle wrong omega for {name}: got {omega_rounded}, expected {expected_omega}"
    
    def test_frank_wolfe_vs_pgd_comparison(self, frank_wolfe_test_graphs, jax_pgd_oracle, frank_wolfe_oracle):
        """Compare Frank-Wolfe vs JAX PGD oracle results."""
        for name, graph, expected_omega in frank_wolfe_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            # Get JAX PGD result
            jax_result = jax_pgd_oracle.solve_quadratic_program(adj_matrix)
            jax_omega = self._compute_omega_from_result(jax_result, graph.number_of_nodes())
            jax_omega_rounded = round(jax_omega)
            
            # Get Frank-Wolfe result
            fw_result = frank_wolfe_oracle.solve_quadratic_program(adj_matrix)
            fw_omega = self._compute_omega_from_result(fw_result, graph.number_of_nodes())
            fw_omega_rounded = round(fw_omega)
            
            # Check omega agreement (the key metric)
            if expected_omega is not None:
                assert jax_omega_rounded == fw_omega_rounded == expected_omega, \
                    f"Oracle disagreement on {name}: PGD={jax_omega_rounded}, FW={fw_omega_rounded}, expected={expected_omega}"
            else:
                # For unknown cases, just check they agree
                assert jax_omega_rounded == fw_omega_rounded, \
                    f"Oracle disagreement on {name}: PGD={jax_omega_rounded}, FW={fw_omega_rounded}"
            
            # Check that QP results are reasonably close (within 5%)
            if jax_result > 0:
                percentage_diff = abs(fw_result - jax_result) / jax_result * 100
                assert percentage_diff < 5.0, \
                    f"Large QP value difference on {name}: {percentage_diff:.2f}% (PGD={jax_result:.6f}, FW={fw_result:.6f})"
    
    def test_frank_wolfe_convergence(self, frank_wolfe_oracle):
        """Test Frank-Wolfe convergence on a simple case."""
        # Use triangle where we know the exact answer
        graph = nx.complete_graph(3)
        adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
        
        # Test with different iteration limits
        for max_iterations in [100, 500, 1000]:
            oracle = FrankWolfeOracle(
                num_restarts=5,
                max_iterations=max_iterations,
                tolerance=1e-6,
                verbose=False
            )
            
            result = oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            # Should converge to correct answer regardless of iteration count
            assert omega_rounded == 3, \
                f"Frank-Wolfe failed to converge with {max_iterations} iterations: got omega={omega_rounded}"
    
    @pytest.mark.integration
    def test_frank_wolfe_benchmark_integration(self):
        """Test Frank-Wolfe integration in benchmark framework."""
        G = nx.cycle_graph(6)
        
        try:
            results = run_algorithm_comparison(
                G, 
                "test_cycle_6",
                algorithms=["nx_exact", "jax_pgd", "jax_fw"],
                benchmark_config={'num_random_runs': 3}
            )
            
            # Check that all algorithms succeeded
            for alg, result in results.items():
                assert result.success, f"Algorithm {alg} failed: {result.error_message}"
            
            # Check if all algorithms agree on MIS size
            sizes = [result.set_size for result in results.values()]
            assert len(set(sizes)) == 1, f"Algorithms disagree on MIS size: {sizes}"
            
        except ImportError:
            pytest.skip("Benchmark framework components not available")
    
    def test_frank_wolfe_edge_cases(self, frank_wolfe_oracle):
        """Test Frank-Wolfe on edge cases."""
        # Single node
        single_node = nx.Graph()
        single_node.add_node(0)
        adj_matrix = nx.to_numpy_array(single_node, dtype=np.float64)
        
        result = frank_wolfe_oracle.solve_quadratic_program(adj_matrix)
        omega = self._compute_omega_from_result(result, 1)
        assert round(omega) == 1, "Single node should have omega=1"
        
        # Empty graph (multiple nodes, no edges)
        empty_graph = nx.empty_graph(3)
        adj_matrix = nx.to_numpy_array(empty_graph, dtype=np.float64)
        
        result = frank_wolfe_oracle.solve_quadratic_program(adj_matrix)
        omega = self._compute_omega_from_result(result, 3)
        assert round(omega) == 1, "Empty graph should have omega=1"