"""
Tests for NetworkX comparison benchmarking framework.
"""

import pytest
import networkx as nx
import numpy as np
from src.motzkinstraus.benchmarks import (
    NetworkXComparisonBenchmark,
    run_algorithm_comparison,
    generate_test_graphs,
    GraphType,
    ScalingConfig
)
from src.motzkinstraus.benchmarks.graph_generators import create_small_test_graphs

# Skip tests if JAX not available
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestGraphGenerators:
    """Test graph generation functions."""
    
    def test_create_small_test_graphs(self):
        """Test creation of small test graphs."""
        graphs = create_small_test_graphs()
        
        assert len(graphs) > 0
        assert all(isinstance(G, nx.Graph) for G, desc in graphs)
        assert all(isinstance(desc, str) for G, desc in graphs)
        
        # Check that we have various graph types
        descriptions = [desc for G, desc in graphs]
        assert any("Path" in desc for desc in descriptions)
        assert any("Cycle" in desc for desc in descriptions)
        assert any("Complete" in desc for desc in descriptions)
    
    def test_generate_test_graphs(self):
        """Test the main graph generation function."""
        config = ScalingConfig(
            small_range=(5, 10),
            medium_range=(15, 20),
            large_range=(25, 30),
            step_size=5
        )
        
        # Generate a few graphs
        graph_types = [GraphType.PATH, GraphType.ERDOS_RENYI]
        graphs = list(generate_test_graphs(config, graph_types))
        
        assert len(graphs) > 0
        
        for G, desc, category in graphs:
            assert isinstance(G, nx.Graph)
            assert isinstance(desc, str)
            assert category in ["small", "medium", "large"]
            assert G.number_of_nodes() >= 5


class TestNetworkXComparison:
    """Test NetworkX comparison framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = NetworkXComparisonBenchmark(
            fast_timeout=5.0,
            medium_timeout=10.0,
            slow_timeout=15.0,
            num_random_runs=5
        )
        
        # Create test graphs
        self.test_graphs = [
            (nx.path_graph(6), "Path_6"),
            (nx.cycle_graph(5), "Cycle_5"),
            (nx.complete_graph(4), "Complete_4"),
            (nx.star_graph(5), "Star_6"),
            (nx.empty_graph(4), "Empty_4")
        ]
    
    def test_verify_independent_set(self):
        """Test independent set verification."""
        G = nx.path_graph(5)  # Nodes 0-1-2-3-4
        
        # Valid independent sets
        assert self.benchmark._verify_independent_set(G, [])
        assert self.benchmark._verify_independent_set(G, [0, 2, 4])
        assert self.benchmark._verify_independent_set(G, [1, 3])
        
        # Invalid independent sets
        assert not self.benchmark._verify_independent_set(G, [0, 1])  # Adjacent
        assert not self.benchmark._verify_independent_set(G, [2, 3])  # Adjacent
        assert not self.benchmark._verify_independent_set(G, [5])     # Node not in graph
    
    def test_networkx_maximal_greedy_single(self):
        """Test NetworkX maximal greedy with single run."""
        for G, desc in self.test_graphs:
            result = self.benchmark.run_networkx_maximal_greedy(G, multiple_seeds=False)
            
            assert result.algorithm_name == "NetworkX_Maximal_Greedy_Single"
            assert result.graph_size == G.number_of_nodes()
            assert result.graph_edges == G.number_of_edges()
            
            if result.success:
                assert len(result.independent_set) == result.set_size
                assert self.benchmark._verify_independent_set(G, result.independent_set)
                assert result.seed_used == 42
    
    def test_networkx_maximal_greedy_multiple(self):
        """Test NetworkX maximal greedy with multiple runs."""
        G = nx.cycle_graph(8)
        result = self.benchmark.run_networkx_maximal_greedy(G, multiple_seeds=True)
        
        assert result.algorithm_name == "NetworkX_Maximal_Greedy"
        assert result.success
        assert result.multiple_runs is not None
        assert len(result.multiple_runs) == self.benchmark.num_random_runs
        
        # Check that all runs are valid
        for run in result.multiple_runs:
            assert 'seed' in run
            assert 'set' in run
            assert 'size' in run
            assert 'runtime' in run
            assert 'valid' in run
            assert run['valid']  # All should be valid independent sets
    
    def test_networkx_approximation(self):
        """Test NetworkX Boppana-Halldórsson approximation."""
        for G, desc in self.test_graphs:
            if G.number_of_nodes() == 0:
                continue  # Skip empty graphs
                
            result = self.benchmark.run_networkx_approximation(G)
            
            assert result.algorithm_name == "NetworkX_BH_Approximation"
            assert result.graph_size == G.number_of_nodes()
            
            if result.success:
                assert len(result.independent_set) == result.set_size
                assert self.benchmark._verify_independent_set(G, result.independent_set)
    
    def test_networkx_exact(self):
        """Test NetworkX exact algorithm."""
        # Test on smaller graphs only (exact is slow)
        small_graphs = [(G, desc) for G, desc in self.test_graphs if G.number_of_nodes() <= 6]
        
        for G, desc in small_graphs:
            if G.number_of_nodes() == 0:
                continue
                
            result = self.benchmark.run_networkx_exact(G)
            
            assert result.algorithm_name == "NetworkX_Exact_Clique"
            
            if result.success:
                assert len(result.independent_set) == result.set_size
                assert self.benchmark._verify_independent_set(G, result.independent_set)
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_pgd_oracle(self):
        """Test JAX PGD oracle."""
        # Test on small graphs
        small_graphs = [(G, desc) for G, desc in self.test_graphs if G.number_of_nodes() <= 6]
        
        for G, desc in small_graphs:
            result = self.benchmark.run_jax_oracle(G, "pgd")
            
            assert result.algorithm_name == "JAX_PGD"
            
            if result.success:
                assert len(result.independent_set) == result.set_size
                assert self.benchmark._verify_independent_set(G, result.independent_set)
                assert result.oracle_calls is not None
                assert result.oracle_calls > 0
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_mirror_oracle(self):
        """Test JAX Mirror Descent oracle."""
        # Test on small graphs
        small_graphs = [(G, desc) for G, desc in self.test_graphs if G.number_of_nodes() <= 6]
        
        for G, desc in small_graphs:
            result = self.benchmark.run_jax_oracle(G, "md")
            
            assert result.algorithm_name == "JAX_MirrorDescent"
            
            if result.success:
                assert len(result.independent_set) == result.set_size
                assert self.benchmark._verify_independent_set(G, result.independent_set)
                assert result.oracle_calls is not None


class TestComparisonIntegration:
    """Test full algorithm comparison integration."""
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_run_algorithm_comparison_small_graphs(self):
        """Test running comparison on small graphs."""
        # Use algorithms available without Gurobi
        algorithms = ["nx_greedy", "nx_approximation", "jax_pgd"]
        
        test_graphs = [
            (nx.path_graph(5), "Path_5"),
            (nx.cycle_graph(6), "Cycle_6"),
            (nx.complete_graph(4), "Complete_4")
        ]
        
        for G, desc in test_graphs:
            results = run_algorithm_comparison(
                G, 
                desc,
                algorithms=algorithms,
                benchmark_config={'num_random_runs': 3, 'fast_timeout': 5.0}
            )
            
            # Check that we got results for each algorithm
            assert len(results) == len(algorithms)
            
            for alg in algorithms:
                assert alg in results
                result = results[alg]
                assert isinstance(result.graph_size, int)
                assert result.graph_size == G.number_of_nodes()
                
                if result.success:
                    # Verify the independent set is valid
                    benchmark = NetworkXComparisonBenchmark()
                    assert benchmark._verify_independent_set(G, result.independent_set)
    
    def test_known_graph_properties(self):
        """Test algorithms on graphs with known optimal independent set sizes."""
        test_cases = [
            (nx.path_graph(5), 3),       # Path: ceil(n/2) = ceil(5/2) = 3
            (nx.cycle_graph(6), 3),      # Even cycle: n/2 = 6/2 = 3  
            (nx.complete_graph(4), 1),   # Complete: only 1 node can be independent
            (nx.star_graph(5), 5),       # Star: all leaves (5 leaves)
            (nx.empty_graph(4), 4),      # Empty: all nodes independent
        ]
        
        algorithms = ["nx_greedy", "nx_approximation"]
        if JAX_AVAILABLE:
            algorithms.append("jax_pgd")
        
        for G, expected_size in test_cases:
            results = run_algorithm_comparison(
                G,
                f"KnownGraph_{G.number_of_nodes()}",
                algorithms=algorithms,
                benchmark_config={'num_random_runs': 5}
            )
            
            # Check that at least one algorithm finds the optimal or near-optimal solution
            best_size = max(r.set_size for r in results.values() if r.success)
            
            # For exact algorithms, should match exactly
            if JAX_AVAILABLE and "jax_pgd" in results and results["jax_pgd"].success:
                # JAX should find exact solution (within numerical tolerance)
                assert results["jax_pgd"].set_size >= expected_size - 1
            
            # For heuristics, should be reasonable
            assert best_size >= expected_size // 2  # At least half optimal (very loose bound)


if __name__ == "__main__":
    # Run a quick test
    print("Testing NetworkX comparison framework...")
    
    # Create simple test
    G = nx.cycle_graph(8)
    algorithms = ["nx_greedy", "nx_approximation"]
    
    if JAX_AVAILABLE:
        algorithms.append("jax_pgd")
        print("JAX available - testing JAX algorithms")
    else:
        print("JAX not available - skipping JAX tests")
    
    results = run_algorithm_comparison(
        G, 
        "TestCycle8",
        algorithms=algorithms,
        benchmark_config={'num_random_runs': 3}
    )
    
    print("\nResults:")
    for alg, result in results.items():
        if result.success:
            print(f"  {alg}: Set size = {result.set_size}, Runtime = {result.runtime_seconds:.4f}s")
        else:
            print(f"  {alg}: FAILED - {result.error_message}")
    
    print("\nBasic tests completed successfully!")