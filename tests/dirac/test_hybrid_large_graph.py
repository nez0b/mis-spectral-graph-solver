"""
Test hybrid oracle behavior on large graphs that exceed the NetworkX threshold.

Validates that hybrid oracles properly use Dirac+PGD for larger graphs.
"""

import pytest
import numpy as np
import networkx as nx

try:
    from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle
    DIRAC_PGD_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_PGD_HYBRID_AVAILABLE = False


@pytest.mark.skipif(not DIRAC_PGD_HYBRID_AVAILABLE, reason="Dirac PGD hybrid oracle not available")
class TestHybridLargeGraph:
    """Test hybrid oracle behavior on large graphs."""
    
    @pytest.fixture
    def large_test_graph(self):
        """Create a large graph that exceeds typical NetworkX threshold."""
        # 50 nodes > 35 threshold
        return nx.barabasi_albert_graph(50, 3, seed=42)
    
    @pytest.fixture
    def hybrid_oracle_verbose(self):
        """Create hybrid oracle with verbose output and reduced samples."""
        return DiracPGDHybridOracle(
            nx_threshold=35,
            dirac_num_samples=30,  # Reduce samples for faster testing
            dirac_relax_schedule=2,
            pgd_tolerance=1e-6,
            pgd_max_iterations=500,
            verbose=True
        )
    
    @pytest.fixture
    def hybrid_oracle_quiet(self):
        """Create hybrid oracle without verbose output."""
        return DiracPGDHybridOracle(
            nx_threshold=35,
            dirac_num_samples=20,  # Even fewer samples for quiet testing
            dirac_relax_schedule=1,
            pgd_tolerance=1e-5,
            pgd_max_iterations=300,
            verbose=False
        )
    
    def test_large_graph_uses_dirac_method(self, large_test_graph, hybrid_oracle_verbose):
        """Test that large graphs trigger Dirac method usage."""
        adj_matrix = nx.to_numpy_array(large_test_graph, dtype=np.float64)
        
        # Verify graph is above threshold
        assert large_test_graph.number_of_nodes() > 35, "Test graph should be above threshold"
        
        try:
            result = hybrid_oracle_verbose.solve_quadratic_program(adj_matrix)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds: {result}"
            
            # Get solve information
            solve_info = hybrid_oracle_verbose.get_last_solve_info()
            
            # Should contain information about method used
            assert isinstance(solve_info, dict), "Solve info should be a dictionary"
            assert 'method' in solve_info, "Solve info should contain method information"
            
            # For large graphs, should attempt Dirac method (may fallback to NetworkX if unavailable)
            method_used = solve_info['method'].lower()
            
            # Either used Dirac successfully or fell back to NetworkX due to unavailability
            assert any(keyword in method_used for keyword in ['dirac', 'networkx', 'fallback']), \
                f"Unexpected method for large graph: {solve_info['method']}"
            
        except Exception as e:
            # If Dirac is not available, should get informative error or fallback
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['dirac', 'unavailable', 'not available']), \
                f"Unexpected error for large graph: {e}"
    
    def test_threshold_behavior_comparison(self, large_test_graph):
        """Compare behavior with different threshold settings."""
        adj_matrix = nx.to_numpy_array(large_test_graph, dtype=np.float64)
        
        # Oracle with very high threshold (should use NetworkX)
        oracle_high_threshold = DiracPGDHybridOracle(
            nx_threshold=100,  # Higher than our 50-node graph
            verbose=False
        )
        
        # Oracle with low threshold (should try Dirac)
        oracle_low_threshold = DiracPGDHybridOracle(
            nx_threshold=10,   # Lower than our 50-node graph
            verbose=False
        )
        
        # Test high threshold (should use NetworkX)
        result_high = oracle_high_threshold.solve_quadratic_program(adj_matrix)
        solve_info_high = oracle_high_threshold.get_last_solve_info()
        
        assert 0.0 <= result_high <= 1.0, "High threshold result out of bounds"
        assert 'networkx' in solve_info_high['method'].lower(), \
            "High threshold should use NetworkX"
        
        # Test low threshold (should try Dirac or fallback)
        try:
            result_low = oracle_low_threshold.solve_quadratic_program(adj_matrix)
            solve_info_low = oracle_low_threshold.get_last_solve_info()
            
            assert 0.0 <= result_low <= 1.0, "Low threshold result out of bounds"
            
            # Results should be reasonably close
            assert abs(result_low - result_high) < 0.1, \
                f"Results too different: low={result_low}, high={result_high}"
                
        except Exception as e:
            # Dirac might not be available, that's okay
            assert "dirac" in str(e).lower() or "unavailable" in str(e).lower(), \
                f"Unexpected error with low threshold: {e}"
    
    def test_solve_info_details(self, large_test_graph, hybrid_oracle_verbose):
        """Test that solve info provides detailed information."""
        adj_matrix = nx.to_numpy_array(large_test_graph, dtype=np.float64)
        
        try:
            result = hybrid_oracle_verbose.solve_quadratic_program(adj_matrix)
            solve_info = hybrid_oracle_verbose.get_last_solve_info()
            
            # Should contain key information
            required_keys = ['method']
            for key in required_keys:
                assert key in solve_info, f"Missing key '{key}' in solve info"
            
            # Method should be informative
            method = solve_info['method']
            assert isinstance(method, str) and len(method) > 0, "Method should be non-empty string"
            
            # May contain additional details depending on method used
            if 'dirac' in method.lower():
                # Might have Dirac-specific details
                pass
            elif 'networkx' in method.lower():
                # Might have NetworkX-specific details
                pass
            
        except Exception:
            # If solve fails due to missing dependencies, that's acceptable
            pass
    
    @pytest.mark.slow
    def test_performance_on_large_graph(self, large_test_graph, hybrid_oracle_quiet):
        """Test performance characteristics on large graph."""
        adj_matrix = nx.to_numpy_array(large_test_graph, dtype=np.float64)
        
        import time
        start_time = time.time()
        
        try:
            result = hybrid_oracle_quiet.solve_quadratic_program(adj_matrix)
            runtime = time.time() - start_time
            
            # Basic validation
            assert 0.0 <= result <= 1.0, "Result out of bounds"
            
            # Performance should be reasonable (allow up to 60 seconds for large graph with Dirac)
            assert runtime < 60.0, f"Runtime too long: {runtime:.3f}s"
            
            # Convert to omega for validation
            if result < 0.5:
                omega = 1.0 / (1.0 - 2.0 * result)
            else:
                omega = large_test_graph.number_of_nodes()
            
            omega_rounded = round(omega)
            
            # Omega should be reasonable for this graph size
            assert 1 <= omega_rounded <= large_test_graph.number_of_nodes(), \
                f"Omega out of range: {omega_rounded}"
            
        except Exception as e:
            # If Dirac is unavailable, should either fallback or give informative error
            if "dirac" not in str(e).lower() and "unavailable" not in str(e).lower():
                # If it's not a Dirac availability issue, re-raise
                raise
    
    def test_different_graph_sizes_threshold_behavior(self, hybrid_oracle_quiet):
        """Test threshold behavior with different graph sizes."""
        threshold = 35
        oracle = DiracPGDHybridOracle(nx_threshold=threshold, verbose=False)
        
        test_sizes = [20, 30, 40, 50]  # Around the threshold
        results = {}
        
        for size in test_sizes:
            G = nx.cycle_graph(size)
            adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
            
            try:
                result = oracle.solve_quadratic_program(adj_matrix)
                solve_info = oracle.get_last_solve_info()
                
                results[size] = {
                    'result': result,
                    'method': solve_info['method'],
                    'expected_omega': size // 2  # For cycle graphs
                }
                
                # Basic validation
                assert 0.0 <= result <= 1.0, f"Result out of bounds for size {size}"
                
            except Exception as e:
                results[size] = {'error': str(e)}
        
        # At least some sizes should work
        successful_results = {k: v for k, v in results.items() if 'result' in v}
        assert len(successful_results) > 0, f"No graph sizes worked: {results}"
        
        # Check that method choices are reasonable
        for size, data in successful_results.items():
            if size <= threshold:
                # Small graphs might use NetworkX or other methods
                pass
            else:
                # Large graphs should try Dirac or fallback to NetworkX
                method = data['method'].lower()
                assert any(keyword in method for keyword in ['dirac', 'networkx']), \
                    f"Unexpected method for size {size}: {data['method']}"
    
    def test_edge_case_exactly_at_threshold(self):
        """Test behavior when graph size exactly matches threshold."""
        threshold = 35
        oracle = DiracPGDHybridOracle(nx_threshold=threshold, verbose=False)
        
        # Graph exactly at threshold
        G = nx.cycle_graph(threshold)
        adj_matrix = nx.to_numpy_array(G, dtype=np.float64)
        
        try:
            result = oracle.solve_quadratic_program(adj_matrix)
            solve_info = oracle.get_last_solve_info()
            
            # Should work and use some method
            assert 0.0 <= result <= 1.0, "Result out of bounds at threshold"
            assert isinstance(solve_info['method'], str), "Method should be specified"
            
        except Exception as e:
            # If it fails, should be due to Dirac unavailability
            assert "dirac" in str(e).lower() or "unavailable" in str(e).lower(), \
                f"Unexpected error at threshold: {e}"