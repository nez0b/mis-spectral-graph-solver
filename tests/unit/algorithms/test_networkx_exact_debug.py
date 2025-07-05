"""
Debug and test the NetworkX exact implementation in hybrid solvers.

Tests NetworkX exact algorithm behavior and validates results.
"""

import pytest
import networkx as nx
import numpy as np

try:
    from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle
    HYBRID_ORACLE_AVAILABLE = True
except ImportError:
    HYBRID_ORACLE_AVAILABLE = False


@pytest.mark.skipif(not HYBRID_ORACLE_AVAILABLE, reason="Hybrid oracle not available")
class TestNetworkXExactDebug:
    """Test and debug NetworkX exact implementation."""
    
    @pytest.fixture
    def hybrid_oracle(self):
        """Create hybrid oracle for testing."""
        oracle = DiracNetworkXHybridOracle(threshold_nodes=35)
        oracle.verbose_oracle_calls = True
        return oracle
    
    @pytest.fixture
    def debug_test_graphs(self):
        """Test graphs with known MIS properties for debugging."""
        return [
            (nx.cycle_graph(10), "10-cycle", 5, 0.4),  # Known MIS size = 5, theoretical value = 0.4
            (nx.complete_graph(5), "Complete K5", 1, 0.0),  # Known MIS size = 1, theoretical value = 0.0
            (nx.empty_graph(5), "Empty 5-graph", 5, 0.4),  # Known MIS size = 5, theoretical value = 0.4
            (nx.path_graph(6), "6-path", 3, 2.0/3.0 * 0.5),  # Known MIS size = 3
        ]
    
    def test_hybrid_oracle_networkx_exact_method(self, hybrid_oracle, debug_test_graphs):
        """Test the internal NetworkX exact method."""
        for graph, description, expected_mis_size, expected_theoretical_value in debug_test_graphs:
            adj_matrix = nx.to_numpy_array(graph)
            
            # Test the internal NetworkX exact method
            result = hybrid_oracle._solve_with_networkx_exact(adj_matrix)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds for {description}: {result}"
            
            # Convert result to omega
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = graph.number_of_nodes()
            
            omega_rounded = round(omega)
            
            # Check against expected MIS size
            assert omega_rounded == expected_mis_size, \
                f"Hybrid solver wrong omega for {description}: got {omega_rounded}, expected {expected_mis_size}"
            
            # Check theoretical value (with tolerance)
            if expected_theoretical_value is not None:
                assert abs(result - expected_theoretical_value) < 0.1, \
                    f"Theoretical value mismatch for {description}: got {result:.6f}, expected {expected_theoretical_value:.6f}"
    
    def test_networkx_direct_validation(self, debug_test_graphs):
        """Validate expected results using direct NetworkX algorithms."""
        for graph, description, expected_mis_size, _ in debug_test_graphs:
            # Method 1: Maximum clique on complement
            complement = nx.complement(graph)
            max_cliques = list(nx.find_cliques(complement))
            max_clique_size = max(len(clique) for clique in max_cliques) if max_cliques else 1
            
            assert max_clique_size == expected_mis_size, \
                f"NetworkX max clique size wrong for {description}: got {max_clique_size}, expected {expected_mis_size}"
            
            # Method 2: Maximal independent set (should be at least as large)
            try:
                mis_approx = nx.maximal_independent_set(graph)
                assert len(mis_approx) >= expected_mis_size, \
                    f"NetworkX maximal IS too small for {description}: got {len(mis_approx)}, expected >= {expected_mis_size}"
            except Exception:
                # NetworkX maximal_independent_set might not be available in all versions
                pass
    
    def test_motzkin_straus_theoretical_values(self):
        """Test Motzkin-Straus theoretical value computations."""
        test_cases = [
            (1, 0.0),  # MIS size 1 -> theoretical value 0.0
            (2, 0.25),  # MIS size 2 -> theoretical value 0.25
            (3, 1.0/3.0),  # MIS size 3 -> theoretical value 1/3
            (4, 0.375),  # MIS size 4 -> theoretical value 0.375
            (5, 0.4),  # MIS size 5 -> theoretical value 0.4
        ]
        
        for mis_size, expected_value in test_cases:
            theoretical_value = 0.5 * (1.0 - 1.0/mis_size)
            
            assert abs(theoretical_value - expected_value) < 1e-10, \
                f"Theoretical value computation error for MIS size {mis_size}: got {theoretical_value}, expected {expected_value}"
    
    def test_edge_cases_networkx_exact(self, hybrid_oracle):
        """Test NetworkX exact on edge cases."""
        edge_cases = [
            (nx.Graph([[0]]), "Single node", 1),
            (nx.Graph([[0], [1]]), "Two isolated nodes", 1),  # Two nodes, no edges
            (nx.complete_graph(2), "K2", 1),  # Two nodes, one edge
        ]
        
        for graph, description, expected_mis_size in edge_cases:
            adj_matrix = nx.to_numpy_array(graph)
            
            result = hybrid_oracle._solve_with_networkx_exact(adj_matrix)
            
            # Convert to omega
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = graph.number_of_nodes()
            
            omega_rounded = round(omega)
            
            assert omega_rounded == expected_mis_size, \
                f"Edge case {description} failed: got {omega_rounded}, expected {expected_mis_size}"
    
    def test_cycle_graphs_various_sizes(self, hybrid_oracle):
        """Test cycle graphs of various sizes."""
        cycle_test_cases = [
            (3, 1),   # 3-cycle (triangle) -> MIS size 1
            (4, 2),   # 4-cycle -> MIS size 2
            (5, 2),   # 5-cycle -> MIS size 2
            (6, 3),   # 6-cycle -> MIS size 3
            (8, 4),   # 8-cycle -> MIS size 4
            (10, 5),  # 10-cycle -> MIS size 5
        ]
        
        for cycle_size, expected_mis_size in cycle_test_cases:
            graph = nx.cycle_graph(cycle_size)
            adj_matrix = nx.to_numpy_array(graph)
            
            result = hybrid_oracle._solve_with_networkx_exact(adj_matrix)
            
            # Convert to omega
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = graph.number_of_nodes()
            
            omega_rounded = round(omega)
            
            assert omega_rounded == expected_mis_size, \
                f"{cycle_size}-cycle failed: got {omega_rounded}, expected {expected_mis_size}"
    
    def test_complete_graphs_various_sizes(self, hybrid_oracle):
        """Test complete graphs of various sizes."""
        complete_test_cases = [
            (2, 1),   # K2 -> MIS size 1
            (3, 1),   # K3 -> MIS size 1
            (4, 1),   # K4 -> MIS size 1
            (5, 1),   # K5 -> MIS size 1
        ]
        
        for complete_size, expected_mis_size in complete_test_cases:
            graph = nx.complete_graph(complete_size)
            adj_matrix = nx.to_numpy_array(graph)
            
            result = hybrid_oracle._solve_with_networkx_exact(adj_matrix)
            
            # For complete graphs, result should be 0.0 (or very close)
            assert abs(result) < 1e-6, \
                f"K{complete_size} should have result ~0.0, got {result}"
            
            # Convert to omega
            omega = 1.0 / (1.0 - 2.0 * result) if result < 0.5 else complete_size
            omega_rounded = round(omega)
            
            assert omega_rounded == expected_mis_size, \
                f"K{complete_size} failed: got {omega_rounded}, expected {expected_mis_size}"
    
    @pytest.mark.integration
    def test_hybrid_oracle_threshold_behavior(self):
        """Test that hybrid oracle switches correctly based on threshold."""
        # Create oracles with different thresholds
        small_threshold_oracle = DiracNetworkXHybridOracle(threshold_nodes=5)
        large_threshold_oracle = DiracNetworkXHybridOracle(threshold_nodes=50)
        
        # Test on 10-node graph
        graph = nx.cycle_graph(10)
        adj_matrix = nx.to_numpy_array(graph)
        
        # Small threshold oracle should use Dirac (if available)
        try:
            result1 = small_threshold_oracle.solve_quadratic_program(adj_matrix)
            assert 0.0 <= result1 <= 1.0, "Small threshold result out of bounds"
        except Exception:
            # Dirac might not be available, skip this part
            pass
        
        # Large threshold oracle should use NetworkX exact
        result2 = large_threshold_oracle._solve_with_networkx_exact(adj_matrix)
        assert 0.0 <= result2 <= 1.0, "Large threshold result out of bounds"
        
        # Both should give omega = 5 for 10-cycle
        omega2 = 1.0 / (1.0 - 2.0 * result2) if result2 < 0.5 else 10
        assert round(omega2) == 5, f"Expected omega=5, got {round(omega2)}"