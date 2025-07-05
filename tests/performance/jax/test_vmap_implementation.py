"""
Test JAX vmap batching implementation for oracle optimizations.

Validates vmap batching behavior and performance for PGD and Mirror Descent oracles.
"""

import pytest
import time
import networkx as nx

try:
    from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
    JAX_PGD_AVAILABLE = True
except ImportError:
    JAX_PGD_AVAILABLE = False

try:
    from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
    JAX_MIRROR_AVAILABLE = True
except ImportError:
    JAX_MIRROR_AVAILABLE = False


@pytest.mark.skipif(not JAX_PGD_AVAILABLE, reason="JAX PGD oracle not available")
class TestVmapImplementation:
    """Test JAX vmap batching implementation."""
    
    @pytest.fixture
    def test_graph(self):
        """Create a test graph for vmap testing."""
        return nx.erdos_renyi_graph(10, 0.3, seed=42)
    
    @pytest.fixture
    def pgd_oracle(self):
        """Create PGD oracle for testing."""
        return ProjectedGradientDescentOracle(
            num_restarts=5,
            max_iterations=100,
            verbose=False  # Disable verbose for automated testing
        )
    
    @pytest.fixture
    def mirror_oracle(self):
        """Create Mirror Descent oracle for testing."""
        if not JAX_MIRROR_AVAILABLE:
            pytest.skip("JAX Mirror Descent oracle not available")
        return MirrorDescentOracle(
            num_restarts=5,
            max_iterations=100,
            verbose=False  # Disable verbose for automated testing
        )
    
    def test_pgd_vmap_batching(self, test_graph, pgd_oracle):
        """Test PGD oracle with vmap batching."""
        start_time = time.time()
        omega_pgd = pgd_oracle.get_omega(test_graph)
        pgd_time = time.time() - start_time
        
        # Basic validation
        assert isinstance(omega_pgd, (int, float)), "Omega should be numeric"
        assert omega_pgd >= 1, "Omega should be at least 1"
        assert omega_pgd <= test_graph.number_of_nodes(), "Omega should not exceed node count"
        
        # Performance validation
        assert pgd_time < 10.0, f"PGD should complete quickly: {pgd_time:.4f}s"
        
        # Oracle call count validation
        assert hasattr(pgd_oracle, 'call_count'), "Oracle should track call count"
        assert pgd_oracle.call_count > 0, "Should make at least one oracle call"
    
    @pytest.mark.skipif(not JAX_MIRROR_AVAILABLE, reason="JAX Mirror Descent oracle not available")
    def test_mirror_descent_vmap_batching(self, test_graph, mirror_oracle):
        """Test Mirror Descent oracle with vmap batching."""
        start_time = time.time()
        omega_md = mirror_oracle.get_omega(test_graph)
        md_time = time.time() - start_time
        
        # Basic validation
        assert isinstance(omega_md, (int, float)), "Omega should be numeric"
        assert omega_md >= 1, "Omega should be at least 1"
        assert omega_md <= test_graph.number_of_nodes(), "Omega should not exceed node count"
        
        # Performance validation
        assert md_time < 10.0, f"Mirror Descent should complete quickly: {md_time:.4f}s"
        
        # Oracle call count validation
        assert hasattr(mirror_oracle, 'call_count'), "Oracle should track call count"
        assert mirror_oracle.call_count > 0, "Should make at least one oracle call"
    
    @pytest.mark.skipif(not (JAX_PGD_AVAILABLE and JAX_MIRROR_AVAILABLE), 
                       reason="Both JAX oracles required for comparison")
    def test_pgd_vs_mirror_descent_consistency(self, test_graph):
        """Test that PGD and Mirror Descent give consistent results."""
        pgd_oracle = ProjectedGradientDescentOracle(
            num_restarts=8,  # More restarts for better consistency
            max_iterations=200,
            verbose=False
        )
        
        mirror_oracle = MirrorDescentOracle(
            num_restarts=8,  # More restarts for better consistency
            max_iterations=200,
            verbose=False
        )
        
        # Get results from both oracles
        omega_pgd = pgd_oracle.get_omega(test_graph)
        omega_md = mirror_oracle.get_omega(test_graph)
        
        # Results should be consistent (allowing for rounding differences)
        assert abs(omega_pgd - omega_md) <= 1, \
            f"Oracle results should be close: PGD={omega_pgd}, MD={omega_md}"
    
    def test_vmap_batching_different_restart_counts(self, test_graph):
        """Test vmap batching with different numbers of restarts."""
        restart_counts = [1, 3, 5, 10]
        results = []
        
        for num_restarts in restart_counts:
            oracle = ProjectedGradientDescentOracle(
                num_restarts=num_restarts,
                max_iterations=50,  # Reduced for speed
                verbose=False
            )
            
            start_time = time.time()
            omega = oracle.get_omega(test_graph)
            runtime = time.time() - start_time
            
            results.append({
                'restarts': num_restarts,
                'omega': omega,
                'runtime': runtime,
                'call_count': oracle.call_count
            })
            
            # Basic validation
            assert omega >= 1, f"Omega should be at least 1 with {num_restarts} restarts"
            assert runtime < 15.0, f"Runtime too long with {num_restarts} restarts: {runtime:.4f}s"
        
        # More restarts should generally give more consistent results
        omegas = [r['omega'] for r in results]
        omega_range = max(omegas) - min(omegas)
        assert omega_range <= 2, f"Omega range too large across restart counts: {omega_range}"
    
    def test_vmap_batching_performance_scaling(self, pgd_oracle):
        """Test that vmap batching scales appropriately."""
        # Test on graphs of different sizes
        graph_sizes = [8, 12, 16]
        runtimes = []
        
        for size in graph_sizes:
            G = nx.cycle_graph(size)  # Use cycle graphs for predictable structure
            
            start_time = time.time()
            omega = pgd_oracle.get_omega(G)
            runtime = time.time() - start_time
            
            runtimes.append(runtime)
            
            # Validate result
            expected_omega = size // 2 if size % 2 == 0 else (size + 1) // 2
            assert abs(omega - expected_omega) <= 1, \
                f"Unexpected omega for {size}-cycle: got {omega}, expected ~{expected_omega}"
        
        # Runtime should scale reasonably (not exponentially)
        assert all(t < 5.0 for t in runtimes), f"Runtimes too long: {runtimes}"
    
    def test_vmap_batching_correctness_on_known_graphs(self, pgd_oracle):
        """Test vmap batching correctness on graphs with known omega."""
        known_graphs = [
            (nx.complete_graph(4), 4),     # K4 has omega = 4
            (nx.cycle_graph(6), 3),        # 6-cycle has omega = 3
            (nx.path_graph(5), 3),         # 5-path has omega = 3
            (nx.empty_graph(5), 1),        # Empty graph has omega = 1
        ]
        
        for graph, expected_omega in known_graphs:
            omega = pgd_oracle.get_omega(graph)
            
            # Should get correct result
            assert omega == expected_omega, \
                f"Wrong omega for {graph.number_of_nodes()}-node graph: got {omega}, expected {expected_omega}"
    
    def test_vmap_batching_with_different_tolerances(self, test_graph):
        """Test vmap batching with different convergence tolerances."""
        tolerances = [1e-4, 1e-6, 1e-8]
        results = []
        
        for tol in tolerances:
            oracle = ProjectedGradientDescentOracle(
                num_restarts=5,
                max_iterations=200,
                tolerance=tol,
                verbose=False
            )
            
            start_time = time.time()
            omega = oracle.get_omega(test_graph)
            runtime = time.time() - start_time
            
            results.append({
                'tolerance': tol,
                'omega': omega,
                'runtime': runtime
            })
            
            # Basic validation
            assert omega >= 1, f"Omega should be at least 1 with tolerance {tol}"
        
        # Tighter tolerances might take longer but should give consistent results
        omegas = [r['omega'] for r in results]
        omega_range = max(omegas) - min(omegas)
        assert omega_range <= 1, f"Omega range too large across tolerances: {omega_range}"
    
    @pytest.mark.slow
    def test_vmap_batching_stress_test(self, pgd_oracle):
        """Stress test vmap batching with many restarts."""
        # Use many restarts to stress test the vmap implementation
        oracle = ProjectedGradientDescentOracle(
            num_restarts=20,  # High number of restarts
            max_iterations=100,
            verbose=False
        )
        
        G = nx.erdos_renyi_graph(15, 0.4, seed=42)
        
        start_time = time.time()
        omega = oracle.get_omega(G)
        runtime = time.time() - start_time
        
        # Should complete in reasonable time even with many restarts
        assert runtime < 30.0, f"Stress test took too long: {runtime:.4f}s"
        
        # Should get reasonable result
        assert 1 <= omega <= G.number_of_nodes(), f"Omega out of range: {omega}"
    
    def test_oracle_call_count_tracking(self, test_graph, pgd_oracle):
        """Test that oracle call count is properly tracked with vmap."""
        # Reset call count
        pgd_oracle.call_count = 0
        
        # Get omega
        omega = pgd_oracle.get_omega(test_graph)
        
        # Should have made exactly 1 oracle call (internally may use multiple restarts)
        assert pgd_oracle.call_count == 1, \
            f"Should make exactly 1 oracle call, got {pgd_oracle.call_count}"
        
        # Get omega again
        omega2 = pgd_oracle.get_omega(test_graph)
        
        # Should have made 2 oracle calls total
        assert pgd_oracle.call_count == 2, \
            f"Should make 2 oracle calls total, got {pgd_oracle.call_count}"
        
        # Results should be consistent
        assert omega == omega2, f"Repeated calls should give same result: {omega} vs {omega2}"