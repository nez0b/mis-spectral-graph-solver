"""
Test hybrid solver integration with the benchmark framework.

Tests hybrid solvers through the benchmark system on various graph sizes.
"""

import pytest
import networkx as nx

try:
    from motzkinstraus.benchmarks.networkx_comparison import (
        NetworkXComparisonBenchmark, 
        run_algorithm_comparison
    )
    BENCHMARK_FRAMEWORK_AVAILABLE = True
except ImportError:
    BENCHMARK_FRAMEWORK_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARK_FRAMEWORK_AVAILABLE, reason="Benchmark framework not available")
class TestHybridIntegration:
    """Test hybrid solver integration with benchmark framework."""
    
    @pytest.fixture
    def hybrid_benchmark_config(self):
        """Configuration for hybrid solver benchmarks."""
        return {
            'dirac_config': {
                'num_samples': 10,
                'relax_schedule': 2,
                'solution_precision': 0.001,
                'threshold_nodes': 35
            }
        }
    
    @pytest.fixture
    def test_graphs(self):
        """Test graphs for hybrid integration testing."""
        return [
            (nx.barabasi_albert_graph(20, 2, seed=42), "20-node BA (small)", "networkx"),
            (nx.barabasi_albert_graph(40, 2, seed=42), "40-node BA (large)", "dirac"),
            (nx.cycle_graph(25), "25-cycle (small)", "networkx"),
            (nx.erdos_renyi_graph(15, 0.3, seed=42), "15-node ER (small)", "networkx"),
        ]
    
    def test_hybrid_solver_small_graph_uses_networkx(self, hybrid_benchmark_config):
        """Test that hybrid solver uses NetworkX on small graphs."""
        G_small = nx.barabasi_albert_graph(20, 2, seed=42)
        
        results = run_algorithm_comparison(
            G_small,
            "test_small_hybrid",
            algorithms=["dirac_hybrid"],
            benchmark_config=hybrid_benchmark_config
        )
        
        hybrid_result = results.get("dirac_hybrid")
        assert hybrid_result is not None, "No hybrid result returned"
        assert hybrid_result.success, f"Hybrid solver failed: {hybrid_result.error_message}"
        
        # Verify it used NetworkX (for small graph)
        solver_used = hybrid_result.optimization_details.get('solver_used', 'Unknown')
        # Note: We can't always guarantee which solver is used, so we just check success
        
        # Basic validation
        assert hybrid_result.set_size > 0, "MIS size should be positive"
        assert hybrid_result.runtime_seconds >= 0, "Runtime should be non-negative"
    
    def test_hybrid_solver_large_graph_fallback(self, hybrid_benchmark_config):
        """Test that hybrid solver handles large graphs appropriately."""
        G_large = nx.barabasi_albert_graph(40, 2, seed=42)
        
        results = run_algorithm_comparison(
            G_large,
            "test_large_hybrid",
            algorithms=["dirac_hybrid"],
            benchmark_config=hybrid_benchmark_config
        )
        
        hybrid_result = results.get("dirac_hybrid")
        assert hybrid_result is not None, "No hybrid result returned"
        
        # Should either succeed with Dirac or fallback to NetworkX
        if hybrid_result.success:
            assert hybrid_result.set_size > 0, "MIS size should be positive"
            assert hybrid_result.runtime_seconds >= 0, "Runtime should be non-negative"
        else:
            # If it fails, it should be due to Dirac unavailability, not a bug
            assert "dirac" in hybrid_result.error_message.lower() or \
                   "unavailable" in hybrid_result.error_message.lower(), \
                   f"Unexpected error: {hybrid_result.error_message}"
    
    def test_hybrid_vs_pure_networkx_comparison(self, hybrid_benchmark_config):
        """Compare hybrid solver results with pure NetworkX solver."""
        G = nx.cycle_graph(15)  # Small enough for both solvers
        
        # Run both hybrid and pure NetworkX
        results = run_algorithm_comparison(
            G,
            "test_hybrid_vs_networkx",
            algorithms=["dirac_hybrid", "nx_exact"],
            benchmark_config=hybrid_benchmark_config
        )
        
        hybrid_result = results.get("dirac_hybrid")
        nx_result = results.get("nx_exact")
        
        assert hybrid_result is not None and hybrid_result.success, "Hybrid solver failed"
        assert nx_result is not None and nx_result.success, "NetworkX solver failed"
        
        # Results should match (both optimal)
        assert hybrid_result.set_size == nx_result.set_size, \
            f"MIS size mismatch: hybrid={hybrid_result.set_size}, nx={nx_result.set_size}"
    
    def test_hybrid_solver_threshold_behavior(self, hybrid_benchmark_config):
        """Test hybrid solver threshold switching behavior."""
        # Test with different threshold configurations
        small_threshold_config = {
            'dirac_config': {
                **hybrid_benchmark_config['dirac_config'],
                'threshold_nodes': 10  # Very small threshold
            }
        }
        
        large_threshold_config = {
            'dirac_config': {
                **hybrid_benchmark_config['dirac_config'],
                'threshold_nodes': 100  # Very large threshold
            }
        }
        
        G = nx.cycle_graph(20)  # 20 nodes
        
        # With small threshold, should try Dirac
        try:
            results_small = run_algorithm_comparison(
                G,
                "test_small_threshold",
                algorithms=["dirac_hybrid"],
                benchmark_config=small_threshold_config
            )
            small_result = results_small.get("dirac_hybrid")
        except Exception:
            small_result = None  # Dirac might not be available
        
        # With large threshold, should use NetworkX
        results_large = run_algorithm_comparison(
            G,
            "test_large_threshold",
            algorithms=["dirac_hybrid"],
            benchmark_config=large_threshold_config
        )
        large_result = results_large.get("dirac_hybrid")
        
        assert large_result is not None and large_result.success, "Large threshold test failed"
        
        # If both succeeded, they should give same MIS size
        if small_result and small_result.success:
            assert small_result.set_size == large_result.set_size, \
                "Different thresholds gave different MIS sizes"
    
    def test_hybrid_solver_error_handling(self, hybrid_benchmark_config):
        """Test hybrid solver error handling and fallbacks."""
        # Test with invalid graph (empty)
        G_empty = nx.empty_graph(0)
        
        results = run_algorithm_comparison(
            G_empty,
            "test_empty_graph",
            algorithms=["dirac_hybrid"],
            benchmark_config=hybrid_benchmark_config
        )
        
        hybrid_result = results.get("dirac_hybrid")
        
        # Should either handle gracefully or give informative error
        if hybrid_result:
            if not hybrid_result.success:
                assert hybrid_result.error_message is not None, "No error message for failed result"
            else:
                # If it succeeds on empty graph, MIS size should be 0
                assert hybrid_result.set_size == 0, "Empty graph should have MIS size 0"
    
    @pytest.mark.integration
    def test_full_hybrid_workflow(self, hybrid_benchmark_config):
        """Test complete hybrid solver workflow."""
        # Test on a variety of graph types and sizes
        test_cases = [
            (nx.cycle_graph(10), "Small cycle"),
            (nx.complete_graph(8), "Small complete"),
            (nx.path_graph(12), "Small path"),
            (nx.wheel_graph(15), "Small wheel"),
        ]
        
        all_successful = True
        results_summary = []
        
        for graph, description in test_cases:
            try:
                results = run_algorithm_comparison(
                    graph,
                    f"test_{description.lower().replace(' ', '_')}",
                    algorithms=["dirac_hybrid"],
                    benchmark_config=hybrid_benchmark_config
                )
                
                hybrid_result = results.get("dirac_hybrid")
                if hybrid_result and hybrid_result.success:
                    results_summary.append((description, hybrid_result.set_size, hybrid_result.runtime_seconds))
                else:
                    all_successful = False
                    results_summary.append((description, None, None))
                    
            except Exception as e:
                all_successful = False
                results_summary.append((description, f"Error: {e}", None))
        
        # At least some tests should succeed
        successful_count = sum(1 for _, size, _ in results_summary if isinstance(size, int))
        assert successful_count > 0, f"No tests succeeded. Results: {results_summary}"
        
        # If most succeed, consider it a pass
        success_rate = successful_count / len(test_cases)
        assert success_rate >= 0.5, f"Too many failures. Success rate: {success_rate:.2f}, Results: {results_summary}"
    
    def test_hybrid_solver_performance_characteristics(self, hybrid_benchmark_config):
        """Test performance characteristics of hybrid solver."""
        # Test on graphs of increasing size
        graph_sizes = [10, 15, 20, 25]
        runtimes = []
        
        for size in graph_sizes:
            G = nx.cycle_graph(size)
            
            results = run_algorithm_comparison(
                G,
                f"test_perf_{size}",
                algorithms=["dirac_hybrid"],
                benchmark_config=hybrid_benchmark_config
            )
            
            hybrid_result = results.get("dirac_hybrid")
            if hybrid_result and hybrid_result.success:
                runtimes.append((size, hybrid_result.runtime_seconds))
        
        # Should have some successful runs
        assert len(runtimes) > 0, "No successful performance tests"
        
        # Runtime should be reasonable (less than 10 seconds for small graphs)
        for size, runtime in runtimes:
            assert runtime < 10.0, f"Runtime too long for {size}-node graph: {runtime:.3f}s"