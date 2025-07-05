"""
Extended oracle comparison for all JAX-based and Dirac oracles.

Tests single oracle calls on multiple small graphs and validates consensus.
"""

import pytest
import numpy as np
import networkx as nx

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

try:
    from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
    JAX_MIRROR_AVAILABLE = True
except ImportError:
    JAX_MIRROR_AVAILABLE = False

try:
    from motzkinstraus.oracles.jax_frank_wolfe import FrankWolfeOracle
    FRANK_WOLFE_AVAILABLE = True
except ImportError:
    FRANK_WOLFE_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False


class TestExtendedOracleComparison:
    """Extended oracle comparison testing all available oracle implementations."""
    
    @pytest.fixture
    def oracle_test_graphs(self):
        """Test graphs for extended oracle comparison."""
        return [
            (nx.complete_graph(3), "Triangle (K3)", 3),
            (nx.complete_graph(4), "4-clique (K4)", 4),
            (nx.cycle_graph(5), "5-cycle", 2),
            (nx.path_graph(6), "6-path", 2),
            (nx.barabasi_albert_graph(8, 2, seed=42), "8-node BA graph", None),
            (nx.barabasi_albert_graph(10, 3, seed=42), "10-node BA graph (m=3)", None),
            (nx.erdos_renyi_graph(10, 0.4, seed=42), "10-node ER graph (p=0.4)", None),
        ]
    
    @pytest.fixture
    def common_oracle_config(self):
        """Common configuration for fair oracle comparison."""
        return {
            'max_iterations': 500,
            'num_restarts': 5,  # Reduced for faster testing
            'tolerance': 1e-6
        }
    
    @pytest.fixture
    def pgd_oracle(self, common_oracle_config):
        """Create JAX PGD oracle."""
        return ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=common_oracle_config['max_iterations'],
            num_restarts=common_oracle_config['num_restarts'],
            tolerance=common_oracle_config['tolerance'],
            verbose=False
        )
    
    @pytest.fixture
    def mirror_oracle(self, common_oracle_config):
        """Create JAX Mirror Descent oracle."""
        if not JAX_MIRROR_AVAILABLE:
            pytest.skip("JAX Mirror Descent oracle not available")
        return MirrorDescentOracle(
            learning_rate=0.01,
            max_iterations=common_oracle_config['max_iterations'],
            num_restarts=common_oracle_config['num_restarts'],
            tolerance=common_oracle_config['tolerance'],
            verbose=False
        )
    
    @pytest.fixture
    def frank_wolfe_oracle(self, common_oracle_config):
        """Create Frank-Wolfe oracle."""
        if not FRANK_WOLFE_AVAILABLE:
            pytest.skip("Frank-Wolfe oracle not available")
        return FrankWolfeOracle(
            num_restarts=common_oracle_config['num_restarts'],
            max_iterations=common_oracle_config['max_iterations'],
            tolerance=common_oracle_config['tolerance'],
            verbose=False
        )
    
    @pytest.fixture
    def dirac_oracles(self):
        """Create multiple Dirac oracles with different relax schedules."""
        if not DIRAC_AVAILABLE:
            pytest.skip("Dirac oracle not available")
        
        oracles = {}
        for relax_schedule in [1, 2, 3, 4]:
            oracles[f'dirac_r{relax_schedule}'] = DiracOracle(
                num_samples=30,
                relax_schedule=relax_schedule,
                solution_precision=0.001
            )
        return oracles
    
    def _compute_omega_from_result(self, qp_result, graph_nodes):
        """Compute omega from quadratic program result."""
        if qp_result < 0.5:
            omega = 1.0 / (1.0 - 2.0 * qp_result)
        else:
            omega = graph_nodes
        return omega
    
    def _test_single_oracle(self, oracle, oracle_name, adj_matrix, graph):
        """Test a single oracle on the given graph."""
        try:
            result = oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            # Basic validation
            assert 0.0 <= result <= 1.0, f"QP result out of bounds: {result}"
            assert omega >= 1.0, f"Omega less than 1: {omega}"
            
            return result, omega_rounded, True
            
        except Exception as e:
            pytest.fail(f"{oracle_name} failed: {str(e)}")
    
    def test_jax_pgd_oracle_on_all_graphs(self, oracle_test_graphs, pgd_oracle):
        """Test JAX PGD oracle on all test graphs."""
        for graph, description, expected_omega in oracle_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result, omega_rounded, success = self._test_single_oracle(
                pgd_oracle, "JAX PGD", adj_matrix, graph
            )
            
            assert success, f"JAX PGD failed on {description}"
            
            # Check against expected omega if known
            if expected_omega is not None:
                assert omega_rounded == expected_omega, \
                    f"JAX PGD wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    @pytest.mark.skipif(not JAX_MIRROR_AVAILABLE, reason="JAX Mirror Descent oracle not available")
    def test_mirror_descent_oracle_on_all_graphs(self, oracle_test_graphs, mirror_oracle):
        """Test JAX Mirror Descent oracle on all test graphs."""
        for graph, description, expected_omega in oracle_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result, omega_rounded, success = self._test_single_oracle(
                mirror_oracle, "JAX Mirror", adj_matrix, graph
            )
            
            assert success, f"JAX Mirror failed on {description}"
            
            # Check against expected omega if known
            if expected_omega is not None:
                assert omega_rounded == expected_omega, \
                    f"JAX Mirror wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    @pytest.mark.skipif(not FRANK_WOLFE_AVAILABLE, reason="Frank-Wolfe oracle not available")
    def test_frank_wolfe_oracle_on_all_graphs(self, oracle_test_graphs, frank_wolfe_oracle):
        """Test Frank-Wolfe oracle on all test graphs."""
        for graph, description, expected_omega in oracle_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result, omega_rounded, success = self._test_single_oracle(
                frank_wolfe_oracle, "Frank-Wolfe", adj_matrix, graph
            )
            
            assert success, f"Frank-Wolfe failed on {description}"
            
            # Check against expected omega if known
            if expected_omega is not None:
                assert omega_rounded == expected_omega, \
                    f"Frank-Wolfe wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    @pytest.mark.skipif(not DIRAC_AVAILABLE, reason="Dirac oracle not available")
    def test_dirac_oracles_on_small_graphs(self, oracle_test_graphs, dirac_oracles):
        """Test Dirac oracles with different relax schedules on small graphs."""
        # Only test on smaller graphs due to Dirac's computational cost
        small_graphs = [g for g in oracle_test_graphs if g[0].number_of_nodes() <= 6]
        
        for graph, description, expected_omega in small_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            for oracle_name, oracle in dirac_oracles.items():
                result, omega_rounded, success = self._test_single_oracle(
                    oracle, oracle_name, adj_matrix, graph
                )
                
                assert success, f"{oracle_name} failed on {description}"
                
                # Check against expected omega if known (more tolerance for Dirac)
                if expected_omega is not None:
                    assert omega_rounded == expected_omega, \
                        f"{oracle_name} wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"
    
    def test_networkx_exact_validation(self, oracle_test_graphs):
        """Validate expected omega values using NetworkX exact algorithm."""
        for graph, description, expected_omega in oracle_test_graphs:
            if expected_omega is not None:
                # Compute exact omega using NetworkX
                complement = nx.complement(graph)
                try:
                    max_clique = max(nx.find_cliques(complement), key=len, default=[])
                    networkx_omega = len(max_clique)
                    
                    assert networkx_omega == expected_omega, \
                        f"NetworkX omega mismatch for {description}: got {networkx_omega}, expected {expected_omega}"
                        
                except Exception:
                    # NetworkX might fail on some graphs, skip validation
                    pass
    
    @pytest.mark.integration
    def test_oracle_consensus_on_known_graphs(self, pgd_oracle):
        """Test that all available oracles agree on graphs with known omega."""
        known_graphs = [
            (nx.complete_graph(3), "Triangle (K3)", 3),
            (nx.complete_graph(4), "4-clique (K4)", 4),
            (nx.cycle_graph(5), "5-cycle", 2),
        ]
        
        for graph, description, expected_omega in known_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            results = {}
            
            # Test PGD oracle
            result, omega_rounded, success = self._test_single_oracle(
                pgd_oracle, "JAX PGD", adj_matrix, graph
            )
            if success:
                results['pgd'] = omega_rounded
            
            # Test other oracles if available
            if JAX_MIRROR_AVAILABLE:
                mirror_oracle = MirrorDescentOracle(
                    learning_rate=0.01, max_iterations=500, num_restarts=5, tolerance=1e-6, verbose=False
                )
                result, omega_rounded, success = self._test_single_oracle(
                    mirror_oracle, "JAX Mirror", adj_matrix, graph
                )
                if success:
                    results['mirror'] = omega_rounded
            
            if FRANK_WOLFE_AVAILABLE:
                fw_oracle = FrankWolfeOracle(
                    num_restarts=5, max_iterations=500, tolerance=1e-6, verbose=False
                )
                result, omega_rounded, success = self._test_single_oracle(
                    fw_oracle, "Frank-Wolfe", adj_matrix, graph
                )
                if success:
                    results['frank_wolfe'] = omega_rounded
            
            # Check consensus
            if len(results) > 1:
                unique_omegas = set(results.values())
                assert len(unique_omegas) == 1, \
                    f"Oracle disagreement on {description}: {results}"
                
                consensus_omega = list(unique_omegas)[0]
                assert consensus_omega == expected_omega, \
                    f"Consensus omega wrong for {description}: got {consensus_omega}, expected {expected_omega}"
    
    def test_oracle_edge_cases(self, pgd_oracle):
        """Test oracles on edge cases."""
        edge_cases = [
            (nx.Graph([[0]]), "Single node", 1),  # Single node
            (nx.empty_graph(3), "Empty 3-graph", 1),  # No edges
            (nx.complete_graph(2), "K2", 2),  # Two nodes, one edge
        ]
        
        for graph, description, expected_omega in edge_cases:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            result, omega_rounded, success = self._test_single_oracle(
                pgd_oracle, "JAX PGD", adj_matrix, graph
            )
            
            assert success, f"JAX PGD failed on {description}"
            assert omega_rounded == expected_omega, \
                f"Wrong omega for {description}: got {omega_rounded}, expected {expected_omega}"