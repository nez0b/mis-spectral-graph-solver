"""
Direct comparison of JAX PGD vs Dirac oracle on single oracle calls.

Tests oracle precision and agreement across different graph types.
"""

import pytest
import numpy as np
import networkx as nx

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

try:
    from motzkinstraus.oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False


@pytest.mark.skipif(not DIRAC_AVAILABLE, reason="Dirac oracle not available")
class TestOracleComparison:
    """Compare JAX PGD and Dirac oracles on single oracle calls."""
    
    @pytest.fixture
    def jax_oracle(self):
        """Create JAX PGD oracle for testing."""
        return ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1000,
            num_restarts=10,
            tolerance=1e-6,
            verbose=False
        )
    
    @pytest.fixture
    def dirac_oracle_configs(self):
        """Different Dirac oracle configurations to test."""
        return [
            {"name": "Conservative", "num_samples": 20, "relax_schedule": 2, "solution_precision": 0.001},
            {"name": "More Samples", "num_samples": 100, "relax_schedule": 3, "solution_precision": 0.001},
            {"name": "High Precision", "num_samples": 50, "relax_schedule": 4, "solution_precision": 0.0001},
        ]
    
    @pytest.fixture
    def oracle_test_graphs(self):
        """Graphs for oracle comparison testing."""
        graphs = []
        
        # Simple 2-node graph
        G1 = nx.Graph()
        G1.add_edge(0, 1)
        graphs.append(("2-node edge", G1, 2))
        
        # Triangle
        G2 = nx.complete_graph(3)
        graphs.append(("Triangle K3", G2, 3))
        
        # 4-clique
        G3 = nx.complete_graph(4)
        graphs.append(("4-clique K4", G3, 4))
        
        # Small BA graph
        G4 = nx.barabasi_albert_graph(10, 3, seed=42)
        graphs.append(("10-node BA", G4, None))  # Unknown omega
        
        return graphs
    
    def _compute_omega_from_result(self, qp_result, graph_nodes):
        """Compute omega from quadratic program result."""
        if qp_result < 0.5:
            omega = 1.0 / (1.0 - 2.0 * qp_result)
        else:
            omega = graph_nodes
        return omega
    
    def test_jax_oracle_single_calls(self, oracle_test_graphs, jax_oracle):
        """Test JAX oracle on various graphs."""
        for name, graph, expected_omega in oracle_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            # Reset call count
            jax_oracle.call_count = 0
            result = jax_oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            omega_rounded = round(omega)
            
            # Basic sanity checks
            assert 0.0 <= result <= 1.0, f"QP result out of bounds for {name}: {result}"
            assert omega >= 1.0, f"Omega less than 1 for {name}: {omega}"
            
            # Check against expected omega if known
            if expected_omega is not None:
                assert omega_rounded == expected_omega, \
                    f"JAX oracle wrong omega for {name}: got {omega_rounded}, expected {expected_omega}"
    
    def test_dirac_oracle_single_calls(self, oracle_test_graphs, dirac_oracle_configs):
        """Test Dirac oracle with different configurations."""
        for config in dirac_oracle_configs:
            dirac_oracle = DiracOracle(
                num_samples=config['num_samples'],
                relax_schedule=config['relax_schedule'],
                solution_precision=config['solution_precision']
            )
            
            for name, graph, expected_omega in oracle_test_graphs:
                adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
                
                # Reset call count
                dirac_oracle.call_count = 0
                result = dirac_oracle.solve_quadratic_program(adj_matrix)
                omega = self._compute_omega_from_result(result, graph.number_of_nodes())
                omega_rounded = round(omega)
                
                # Basic sanity checks
                assert 0.0 <= result <= 1.0, f"QP result out of bounds for {name} with {config['name']}: {result}"
                assert omega >= 1.0, f"Omega less than 1 for {name} with {config['name']}: {omega}"
                
                # Check against expected omega if known (more tolerance for Dirac)
                if expected_omega is not None and expected_omega <= 4:  # Only check small graphs
                    assert omega_rounded == expected_omega, \
                        f"Dirac oracle wrong omega for {name} with {config['name']}: got {omega_rounded}, expected {expected_omega}"
    
    def test_oracle_agreement(self, oracle_test_graphs, jax_oracle, dirac_oracle_configs):
        """Test agreement between JAX and Dirac oracles."""
        # Use conservative Dirac config for comparison
        dirac_oracle = DiracOracle(
            num_samples=50,
            relax_schedule=3,
            solution_precision=0.001
        )
        
        for name, graph, expected_omega in oracle_test_graphs:
            adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
            
            # Get JAX result
            jax_oracle.call_count = 0
            jax_result = jax_oracle.solve_quadratic_program(adj_matrix)
            jax_omega = self._compute_omega_from_result(jax_result, graph.number_of_nodes())
            jax_omega_rounded = round(jax_omega)
            
            # Get Dirac result
            dirac_oracle.call_count = 0
            dirac_result = dirac_oracle.solve_quadratic_program(adj_matrix)
            dirac_omega = self._compute_omega_from_result(dirac_result, graph.number_of_nodes())
            dirac_omega_rounded = round(dirac_omega)
            
            # Check agreement on omega (the key metric)
            if expected_omega is not None and expected_omega <= 4:  # Only check small graphs
                assert jax_omega_rounded == dirac_omega_rounded, \
                    f"Oracle disagreement on {name}: JAX={jax_omega_rounded}, Dirac={dirac_omega_rounded}"
            
            # Check that results are reasonably close (within 10% for QP value)
            if jax_result > 0:
                percentage_diff = abs(dirac_result - jax_result) / jax_result * 100
                assert percentage_diff < 10.0, \
                    f"Large QP value difference on {name}: {percentage_diff:.2f}% (JAX={jax_result:.6f}, Dirac={dirac_result:.6f})"
    
    def test_oracle_precision_convergence(self):
        """Test that higher precision configs give better results."""
        graph = nx.complete_graph(3)  # Known omega = 3
        adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
        
        configs = [
            {"num_samples": 20, "relax_schedule": 2, "solution_precision": 0.01},
            {"num_samples": 50, "relax_schedule": 3, "solution_precision": 0.001},
            {"num_samples": 100, "relax_schedule": 4, "solution_precision": 0.0001},
        ]
        
        results = []
        for config in configs:
            dirac_oracle = DiracOracle(**config)
            result = dirac_oracle.solve_quadratic_program(adj_matrix)
            omega = self._compute_omega_from_result(result, graph.number_of_nodes())
            results.append((config['solution_precision'], omega))
        
        # Higher precision should give more accurate results (closer to 3.0)
        for i in range(len(results) - 1):
            prec1, omega1 = results[i]
            prec2, omega2 = results[i + 1]
            
            # Higher precision (smaller value) should be more accurate
            if prec2 < prec1:
                error1 = abs(omega1 - 3.0)
                error2 = abs(omega2 - 3.0)
                # Allow some tolerance as quantum results can vary
                assert error2 <= error1 + 0.1, \
                    f"Higher precision didn't improve result: {prec1}→{omega1:.3f}, {prec2}→{omega2:.3f}"