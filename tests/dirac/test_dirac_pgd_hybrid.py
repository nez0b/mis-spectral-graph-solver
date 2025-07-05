"""
Test the Dirac/PGD hybrid oracle implementation for high precision.

Tests hybrid solver that combines Dirac quantum computing with JAX PGD for precision.
"""

import pytest
import numpy as np
import networkx as nx

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

try:
    from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle
    DIRAC_PGD_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_PGD_HYBRID_AVAILABLE = False

try:
    from motzkinstraus.benchmarks.networkx_comparison import run_algorithm_comparison
    BENCHMARK_FRAMEWORK_AVAILABLE = True
except ImportError:
    BENCHMARK_FRAMEWORK_AVAILABLE = False


@pytest.mark.skipif(not DIRAC_PGD_HYBRID_AVAILABLE, reason="Dirac PGD hybrid oracle not available")
class TestDiracPGDHybridOracle:
    """Test Dirac/PGD hybrid oracle for high precision optimization."""
    
    @pytest.fixture
    def pgd_oracle(self):
        """High precision PGD oracle for comparison."""
        return ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=5000,
            num_restarts=10,
            tolerance=1e-8,
            verbose=False
        )
    
    @pytest.fixture
    def hybrid_oracle(self):
        """Dirac/PGD hybrid oracle."""
        return DiracPGDHybridOracle(
            nx_threshold=35,  # Use NetworkX for small graphs
            pgd_tolerance=1e-8,  # High precision
            pgd_max_iterations=5000,
            verbose=False
        )
    
    @pytest.fixture
    def precision_test_graphs(self):
        """Test graphs for precision validation."""
        return [
            (nx.complete_graph(3), "Triangle (K3)", 3, 0.5 * (1.0 - 1.0/3)),
            (nx.complete_graph(4), "4-clique (K4)", 4, 0.5 * (1.0 - 1.0/4)),
            (nx.cycle_graph(6), "6-cycle", 3, 0.5 * (1.0 - 1.0/3)),
            (nx.path_graph(7), "7-path", 4, 0.5 * (1.0 - 1.0/4)),
            (nx.barabasi_albert_graph(10, 2, seed=42), "10-node BA graph", None, None),
        ]
    
    def _compute_omega_from_result(self, qp_result, graph_nodes):
        """Compute omega from quadratic program result."""
        if qp_result < 0.5:
            omega = 1.0 / (1.0 - 2.0 * qp_result)
        else:
            omega = graph_nodes
        return omega
    
    def test_hybrid_oracle_correctness(self, hybrid_oracle, precision_test_graphs):
        """Test hybrid oracle correctness on known graphs."""
        for graph, description, expected_omega, expected_ms_value in precision_test_graphs:
            if expected_omega is None:
                continue  # Skip graphs with unknown omega
                
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result = hybrid_oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds for {description}: {result}"
            assert omega >= 1.0, f"Omega less than 1 for {description}: {omega}"
            
            # Check correctness
            assert omega_rounded == expected_omega, \
                f"Hybrid oracle wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    def test_hybrid_vs_pgd_precision(self, hybrid_oracle, pgd_oracle, precision_test_graphs):
        """Test hybrid oracle precision compared to PGD oracle."""
        tolerance = 1e-7
        
        for graph, description, expected_omega, expected_ms_value in precision_test_graphs:
            if expected_ms_value is None:
                continue  # Skip graphs with unknown theoretical value
                
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            # Get PGD result
            pgd_result = pgd_oracle.solve_quadratic_program(adj_matrix)
            
            # Get hybrid result
            hybrid_result = hybrid_oracle.solve_quadratic_program(adj_matrix)
            
            # Check precision difference
            precision_diff = abs(hybrid_result - pgd_result)
            
            # Check if results are within tolerance OR hybrid is more precise
            pgd_error = abs(pgd_result - expected_ms_value)
            hybrid_error = abs(hybrid_result - expected_ms_value)
            
            precision_acceptable = (precision_diff <= tolerance) or (hybrid_error <= pgd_error)
            
            assert precision_acceptable, \
                f"Precision test failed for {description}: " \
                f"diff={precision_diff:.2e}, tolerance={tolerance:.0e}, " \
                f"PGD_error={pgd_error:.2e}, hybrid_error={hybrid_error:.2e}"
    
    def test_hybrid_oracle_solve_info(self, hybrid_oracle):
        """Test that hybrid oracle provides solve information."""
        G = nx.cycle_graph(6)
        adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
        
        # Solve and get info
        result = hybrid_oracle.solve_quadratic_program(adj_matrix)
        solve_info = hybrid_oracle.get_last_solve_info()
        
        # Basic validation
        assert 0.0 <= result <= 1.0, "QP result out of bounds"
        assert isinstance(solve_info, dict), "Solve info should be a dictionary"
        
        # Should contain method information
        assert 'method' in solve_info, "Solve info should contain method"
        method = solve_info['method']
        assert isinstance(method, str), "Method should be a string"
        assert len(method) > 0, "Method should not be empty"
    
    def test_threshold_behavior(self):
        """Test hybrid oracle threshold behavior."""
        G_small = nx.cycle_graph(8)   # Below threshold
        G_large = nx.cycle_graph(40)  # Above threshold
        
        # Small threshold oracle (prefers non-NetworkX methods)
        oracle_small_thresh = DiracPGDHybridOracle(nx_threshold=10)
        
        # Large threshold oracle (prefers NetworkX)
        oracle_large_thresh = DiracPGDHybridOracle(nx_threshold=50)
        
        # Test small graph
        adj_matrix_small = nx.to_numpy_array(G_small, dtype=np.float64)
        result_small = oracle_large_thresh.solve_quadratic_program(adj_matrix_small)
        
        # Should use NetworkX for small graph with large threshold
        solve_info_small = oracle_large_thresh.get_last_solve_info()
        
        assert 0.0 <= result_small <= 1.0, "Small graph result out of bounds"
        assert isinstance(solve_info_small, dict), "Solve info should be dict"
        
        # Test large graph behavior (may vary based on availability)
        adj_matrix_large = nx.to_numpy_array(G_large, dtype=np.float64)
        
        try:
            result_large = oracle_small_thresh.solve_quadratic_program(adj_matrix_large)
            assert 0.0 <= result_large <= 1.0, "Large graph result out of bounds"
        except Exception as e:
            # Dirac might not be available, check error message
            assert "dirac" in str(e).lower() or "unavailable" in str(e).lower(), \
                f"Unexpected error: {e}"
    
    def test_hybrid_oracle_edge_cases(self, hybrid_oracle):
        """Test hybrid oracle on edge cases."""
        edge_cases = [
            (nx.Graph([[0]]), "Single node", 1),
            (nx.empty_graph(3), "Empty 3-graph", 1),
            (nx.complete_graph(2), "K2", 2),
        ]
        
        for graph, description, expected_omega in edge_cases:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result = hybrid_oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            assert 0.0 <= result <= 1.0, f"Result out of bounds for {description}: {result}"
            assert omega_rounded == expected_omega, \
                f"Wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    def test_hybrid_oracle_consistency(self, hybrid_oracle):
        """Test that hybrid oracle gives consistent results."""
        G = nx.cycle_graph(8)
        adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
        
        # Run multiple times
        results = []
        for i in range(3):
            result = hybrid_oracle.solve_quadratic_program(adj_matrix)
            results.append(result)
        
        # Results should be very close (allowing for small numerical differences)
        for i in range(1, len(results)):
            diff = abs(results[i] - results[0])
            assert diff < 1e-6, f"Inconsistent results: {results}"
    
    @pytest.mark.skipif(not BENCHMARK_FRAMEWORK_AVAILABLE, reason="Benchmark framework not available")
    @pytest.mark.integration
    def test_benchmark_integration(self):
        """Test hybrid oracle integration in benchmark framework."""
        G = nx.cycle_graph(8)
        
        results = run_algorithm_comparison(
            G, 
            "test_cycle_8_hybrid",
            algorithms=["nx_exact", "jax_pgd", "dirac_pgd_hybrid"],
            benchmark_config={'num_random_runs': 2}
        )
        
        # Check that hybrid oracle ran successfully
        hybrid_result = results.get("dirac_pgd_hybrid")
        assert hybrid_result is not None, "No hybrid result returned"
        
        if hybrid_result.success:
            assert hybrid_result.set_size > 0, "MIS size should be positive"
            assert hybrid_result.runtime_seconds >= 0, "Runtime should be non-negative"
            
            # Check against NetworkX exact if available
            nx_result = results.get("nx_exact")
            if nx_result and nx_result.success:
                assert hybrid_result.set_size == nx_result.set_size, \
                    f"MIS size mismatch: hybrid={hybrid_result.set_size}, nx={nx_result.set_size}"
        else:
            # If it fails, should be due to missing dependencies
            error_msg = hybrid_result.error_message.lower()
            assert any(keyword in error_msg for keyword in ["dirac", "unavailable", "import"]), \
                f"Unexpected error: {hybrid_result.error_message}"
    
    def test_verbose_output_control(self):
        """Test verbose output control."""
        G = nx.cycle_graph(6)
        adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
        
        # Test with verbose off
        oracle_quiet = DiracPGDHybridOracle(verbose=False)
        result1 = oracle_quiet.solve_quadratic_program(adj_matrix)
        
        # Test with verbose on
        oracle_verbose = DiracPGDHybridOracle(verbose=True)
        result2 = oracle_verbose.solve_quadratic_program(adj_matrix)
        
        # Results should be the same
        assert abs(result1 - result2) < 1e-10, "Verbose setting should not affect results"
    
    @pytest.mark.slow
    def test_high_precision_validation(self, hybrid_oracle, pgd_oracle):
        """Test high precision capabilities of hybrid oracle."""
        # Use triangle where exact value is known
        G = nx.complete_graph(3)
        adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
        
        expected_value = 0.5 * (1.0 - 1.0/3)  # Exact Motzkin-Straus value
        
        # Get results
        hybrid_result = hybrid_oracle.solve_quadratic_program(adj_matrix)
        pgd_result = pgd_oracle.solve_quadratic_program(adj_matrix)
        
        # Check errors
        hybrid_error = abs(hybrid_result - expected_value)
        pgd_error = abs(pgd_result - expected_value)
        
        # Hybrid should be at least as good as PGD
        assert hybrid_error <= pgd_error + 1e-8, \
            f"Hybrid not as precise as PGD: hybrid_error={hybrid_error:.2e}, pgd_error={pgd_error:.2e}"
        
        # Both should be quite accurate
        assert hybrid_error < 1e-6, f"Hybrid oracle not sufficiently precise: error={hybrid_error:.2e}"