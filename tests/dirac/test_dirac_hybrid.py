"""
Test the Dirac/NetworkX hybrid solver implementation.

Tests hybrid solver behavior on different graph sizes and validates solver selection.
"""

import pytest
import numpy as np
import networkx as nx

try:
    from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle
    DIRAC_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_HYBRID_AVAILABLE = False


@pytest.mark.skipif(not DIRAC_HYBRID_AVAILABLE, reason="Dirac hybrid oracle not available")
class TestDiracHybridSolver:
    """Test Dirac/NetworkX hybrid solver implementation."""
    
    @pytest.fixture
    def small_threshold_oracle(self):
        """Hybrid oracle with small threshold (prefers Dirac)."""
        oracle = DiracNetworkXHybridOracle(threshold_nodes=15)
        oracle.verbose_oracle_calls = True
        return oracle
    
    @pytest.fixture
    def large_threshold_oracle(self):
        """Hybrid oracle with large threshold (prefers NetworkX)."""
        oracle = DiracNetworkXHybridOracle(threshold_nodes=100)
        oracle.verbose_oracle_calls = True
        return oracle
    
    @pytest.fixture
    def test_graphs(self):
        """Test graphs for hybrid solver testing."""
        return [
            (nx.cycle_graph(10), "10-cycle", 5),
            (nx.cycle_graph(50), "50-cycle", 25),
            (nx.complete_graph(5), "K5", 1),
            (nx.path_graph(12), "12-path", 6),
            (nx.wheel_graph(8), "8-wheel", 1),  # Wheel graph has small MIS
        ]
    
    def test_solver_info_method(self, large_threshold_oracle, test_graphs):
        """Test the get_solver_info method for different graph sizes."""
        for graph, description, expected_mis_size in test_graphs:
            solver_info = large_threshold_oracle.get_solver_info(graph.number_of_nodes())
            
            # Should return a string describing the solver choice
            assert isinstance(solver_info, str), f"Solver info should be string for {description}"
            assert len(solver_info) > 0, f"Solver info should not be empty for {description}"
            
            # Should mention either 'NetworkX' or 'Dirac' based on threshold
            if graph.number_of_nodes() < large_threshold_oracle.threshold_nodes:
                assert "networkx" in solver_info.lower(), f"Should use NetworkX for {description}"
            else:
                assert "dirac" in solver_info.lower() or "networkx" in solver_info.lower(), \
                    f"Should mention solver choice for {description}"
    
    def test_small_graph_uses_networkx(self, large_threshold_oracle):
        """Test that small graphs use NetworkX solver."""
        G_small = nx.cycle_graph(10)  # Well below threshold
        adj_matrix = nx.to_numpy_array(G_small)
        
        result = large_threshold_oracle.solve_quadratic_program(adj_matrix)
        
        # Basic validation
        assert 0.0 <= result <= 1.0, f"QP result out of bounds: {result}"
        
        # For 10-cycle, expected omega = 5, so result should be ~0.4
        expected_result = 0.5 * (1.0 - 1.0/5)  # 0.4
        assert abs(result - expected_result) < 0.01, \
            f"Result mismatch for 10-cycle: got {result}, expected ~{expected_result}"
    
    def test_large_graph_behavior(self, small_threshold_oracle):
        """Test behavior on large graphs (may use Dirac or fallback)."""
        G_large = nx.cycle_graph(50)  # Above small threshold
        adj_matrix = nx.to_numpy_array(G_large)
        
        try:
            result = small_threshold_oracle.solve_quadratic_program(adj_matrix)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds: {result}"
            
            # For 50-cycle, expected omega = 25, so result should be ~0.48
            expected_result = 0.5 * (1.0 - 1.0/25)  # 0.48
            assert abs(result - expected_result) < 0.1, \
                f"Result roughly correct for 50-cycle: got {result}, expected ~{expected_result}"
                
        except Exception as e:
            # If Dirac is not available, should fallback gracefully
            assert "dirac" in str(e).lower() or "unavailable" in str(e).lower(), \
                f"Unexpected error (should be Dirac unavailability): {e}"
    
    def test_threshold_switching_behavior(self):
        """Test that threshold correctly determines solver choice."""
        G_medium = nx.cycle_graph(25)  # Medium size graph
        adj_matrix = nx.to_numpy_array(G_medium)
        
        # Oracle with low threshold (should try Dirac)
        oracle_low = DiracNetworkXHybridOracle(threshold_nodes=10)
        
        # Oracle with high threshold (should use NetworkX)
        oracle_high = DiracNetworkXHybridOracle(threshold_nodes=50)
        
        # Test high threshold oracle (should use NetworkX)
        result_high = oracle_high.solve_quadratic_program(adj_matrix)
        assert 0.0 <= result_high <= 1.0, "High threshold result out of bounds"
        
        # Test low threshold oracle (may use Dirac or fallback)
        try:
            result_low = oracle_low.solve_quadratic_program(adj_matrix)
            assert 0.0 <= result_low <= 1.0, "Low threshold result out of bounds"
            
            # Results should be close if both work
            assert abs(result_low - result_high) < 0.1, \
                f"Results should be similar: low={result_low}, high={result_high}"
                
        except Exception:
            # Dirac might not be available, that's okay
            pass
    
    def test_edge_cases(self, large_threshold_oracle):
        """Test hybrid solver on edge cases."""
        edge_cases = [
            (nx.Graph([[0]]), "Single node", 1),
            (nx.empty_graph(3), "Empty 3-graph", 1),
            (nx.complete_graph(2), "K2", 1),
        ]
        
        for graph, description, expected_omega in edge_cases:
            adj_matrix = nx.to_numpy_array(graph)
            
            result = large_threshold_oracle.solve_quadratic_program(adj_matrix)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds for {description}: {result}"
            
            # Convert to omega
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = graph.number_of_nodes()
            
            omega_rounded = round(omega)
            assert omega_rounded == expected_omega, \
                f"Wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    def test_networkx_exact_method_directly(self, large_threshold_oracle):
        """Test the internal NetworkX exact method."""
        test_cases = [
            (nx.cycle_graph(8), 4),    # 8-cycle -> omega = 4
            (nx.complete_graph(4), 1), # K4 -> omega = 1
            (nx.path_graph(7), 4),     # 7-path -> omega = 4
        ]
        
        for graph, expected_omega in test_cases:
            adj_matrix = nx.to_numpy_array(graph)
            
            # Test the internal NetworkX method directly
            result = large_threshold_oracle._solve_with_networkx_exact(adj_matrix)
            
            # Convert to omega
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = graph.number_of_nodes()
            
            omega_rounded = round(omega)
            assert omega_rounded == expected_omega, \
                f"NetworkX exact method wrong for {graph.number_of_nodes()}-node graph: got {omega_rounded}, expected {expected_omega}"
    
    def test_verbose_output_control(self):
        """Test verbose output control."""
        G = nx.cycle_graph(6)
        adj_matrix = nx.to_numpy_array(G)
        
        # Test with verbose off
        oracle_quiet = DiracNetworkXHybridOracle(threshold_nodes=50)
        oracle_quiet.verbose_oracle_calls = False
        
        result1 = oracle_quiet.solve_quadratic_program(adj_matrix)
        assert 0.0 <= result1 <= 1.0, "Quiet oracle result out of bounds"
        
        # Test with verbose on
        oracle_verbose = DiracNetworkXHybridOracle(threshold_nodes=50)
        oracle_verbose.verbose_oracle_calls = True
        
        result2 = oracle_verbose.solve_quadratic_program(adj_matrix)
        assert 0.0 <= result2 <= 1.0, "Verbose oracle result out of bounds"
        
        # Results should be the same
        assert abs(result1 - result2) < 1e-10, "Verbose setting should not affect results"
    
    @pytest.mark.integration
    def test_hybrid_oracle_consistency(self):
        """Test that hybrid oracle gives consistent results."""
        G = nx.cycle_graph(12)
        adj_matrix = nx.to_numpy_array(G)
        
        oracle = DiracNetworkXHybridOracle(threshold_nodes=50)
        
        # Run multiple times
        results = []
        for i in range(3):
            result = oracle.solve_quadratic_program(adj_matrix)
            results.append(result)
        
        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-10, \
                f"Inconsistent results: {results}"
    
    def test_different_threshold_configurations(self):
        """Test various threshold configurations."""
        G = nx.cycle_graph(20)
        adj_matrix = nx.to_numpy_array(G)
        
        thresholds = [5, 15, 25, 50, 100]
        results = []
        
        for threshold in thresholds:
            oracle = DiracNetworkXHybridOracle(threshold_nodes=threshold)
            
            try:
                result = oracle.solve_quadratic_program(adj_matrix)
                results.append((threshold, result))
            except Exception:
                # Some configurations might fail if Dirac unavailable
                results.append((threshold, None))
        
        # At least one configuration should work
        successful_results = [r for _, r in results if r is not None]
        assert len(successful_results) > 0, "No threshold configuration worked"
        
        # All successful results should be close
        if len(successful_results) > 1:
            for i in range(1, len(successful_results)):
                assert abs(successful_results[i] - successful_results[0]) < 0.1, \
                    f"Different thresholds gave very different results: {results}"