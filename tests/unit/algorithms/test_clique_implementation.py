"""
Comprehensive test suite for Maximum Clique and MIS implementation.

Tests:
1. Maximum clique algorithms (oracle-based and brute force)
2. MILP solvers for both MIS and max clique
3. Cross-validation between different approaches
4. Small graph instances with known solutions
"""

import pytest
import networkx as nx
import numpy as np

from motzkinstraus import (
    find_max_clique_with_oracle,
    find_max_clique_brute_force,
    find_mis_with_oracle,
    find_mis_brute_force,
    verify_clique,
    verify_independent_set,
    MILP_AVAILABLE
)

# Import MILP solvers if available
if MILP_AVAILABLE:
    from motzkinstraus import (
        solve_max_clique_gurobi,
        solve_mis_gurobi,
        get_clique_number_gurobi,
        get_independence_number_gurobi,
        solve_max_clique_scipy,
        solve_mis_scipy,
        get_clique_number_scipy,
        get_independence_number_scipy,
        GUROBI_AVAILABLE,
        SCIPY_MILP_AVAILABLE
    )

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle


class TestVerificationFunctions:
    """Test clique and independent set verification functions."""
    
    def test_clique_verification(self):
        """Test clique verification on complete graph."""
        G = nx.complete_graph(3)
        
        assert verify_clique(G, {0, 1, 2}) == True
        assert verify_clique(G, {0, 1}) == True
        assert verify_clique(G, {0}) == True
        assert verify_clique(G, set()) == True
    
    def test_independent_set_verification(self):
        """Test independent set verification."""
        G = nx.complete_graph(3)
        
        assert verify_independent_set(G, {0}) == True
        assert verify_independent_set(G, set()) == True
        assert verify_independent_set(G, {0, 1}) == False
        
        # Test on cycle graph
        G2 = nx.cycle_graph(4)
        assert verify_independent_set(G2, {0, 2}) == True
        assert verify_independent_set(G2, {0, 1}) == False


class TestBruteForceAlgorithms:
    """Test brute force algorithms on small graphs."""
    
    def test_brute_force_on_test_graphs(self, small_test_graphs):
        """Test brute force algorithms on all small test graphs."""
        for name, graph, expected in small_test_graphs:
            # Test max clique brute force
            clique = find_max_clique_brute_force(graph)
            clique_size = len(clique)
            
            assert verify_clique(graph, clique), f"Invalid clique found for {name}"
            assert clique_size == expected["clique_size"], \
                f"Wrong clique size for {name}: got {clique_size}, expected {expected['clique_size']}"
            
            # Test MIS brute force
            mis = find_mis_brute_force(graph)
            mis_size = len(mis)
            
            assert verify_independent_set(graph, mis), f"Invalid MIS found for {name}"
            assert mis_size == expected["mis_size"], \
                f"Wrong MIS size for {name}: got {mis_size}, expected {expected['mis_size']}"


class TestOracleAlgorithms:
    """Test oracle-based algorithms."""
    
    @pytest.fixture
    def jax_oracle(self, jax_oracle_config):
        """Create JAX PGD oracle for testing."""
        return ProjectedGradientDescentOracle(
            learning_rate=jax_oracle_config['learning_rate_pgd'],
            max_iterations=jax_oracle_config['max_iterations'],
            num_restarts=jax_oracle_config['num_restarts'],
            tolerance=jax_oracle_config['tolerance'],
            verbose=jax_oracle_config['verbose']
        )
    
    def test_oracle_on_test_graphs(self, small_test_graphs, jax_oracle):
        """Test oracle-based algorithms on all small test graphs."""
        for name, graph, expected in small_test_graphs:
            # Test max clique with oracle
            clique, oracle_calls = find_max_clique_with_oracle(graph, jax_oracle, verbose=False)
            clique_size = len(clique)
            
            assert verify_clique(graph, clique), f"Invalid clique found for {name}"
            assert clique_size == expected["clique_size"], \
                f"Wrong clique size for {name}: got {clique_size}, expected {expected['clique_size']}"
            
            # Test MIS with oracle
            mis, oracle_calls_mis = find_mis_with_oracle(graph, jax_oracle, verbose=False)
            mis_size = len(mis)
            
            assert verify_independent_set(graph, mis), f"Invalid MIS found for {name}"
            assert mis_size == expected["mis_size"], \
                f"Wrong MIS size for {name}: got {mis_size}, expected {expected['mis_size']}"


@pytest.mark.skipif(not MILP_AVAILABLE, reason="MILP solvers not available")
class TestMILPSolvers:
    """Test MILP solvers if available."""
    
    @pytest.mark.skipif(not MILP_AVAILABLE or not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_milp_solvers(self, small_test_graphs):
        """Test Gurobi MILP solvers."""
        for name, graph, expected in small_test_graphs:
            # Test max clique MILP
            clique = solve_max_clique_gurobi(graph, suppress_output=True)
            clique_size = len(clique)
            
            assert verify_clique(graph, clique), f"Invalid clique found for {name} with Gurobi"
            assert clique_size == expected["clique_size"], \
                f"Wrong clique size for {name} with Gurobi: got {clique_size}, expected {expected['clique_size']}"
            
            # Test MIS MILP
            mis = solve_mis_gurobi(graph, suppress_output=True)
            mis_size = len(mis)
            
            assert verify_independent_set(graph, mis), f"Invalid MIS found for {name} with Gurobi"
            assert mis_size == expected["mis_size"], \
                f"Wrong MIS size for {name} with Gurobi: got {mis_size}, expected {expected['mis_size']}"
            
            # Test number functions
            clique_num = get_clique_number_gurobi(graph, suppress_output=True)
            mis_num = get_independence_number_gurobi(graph, suppress_output=True)
            
            assert clique_num == expected["clique_size"], f"Wrong clique number for {name} with Gurobi"
            assert mis_num == expected["mis_size"], f"Wrong independence number for {name} with Gurobi"
    
    @pytest.mark.skipif(not MILP_AVAILABLE or not SCIPY_MILP_AVAILABLE, reason="SciPy MILP not available")
    def test_scipy_milp_solvers(self, small_test_graphs):
        """Test SciPy MILP solvers."""
        for name, graph, expected in small_test_graphs:
            # Test max clique MILP
            clique = solve_max_clique_scipy(graph, suppress_output=True)
            clique_size = len(clique)
            
            assert verify_clique(graph, clique), f"Invalid clique found for {name} with SciPy"
            assert clique_size == expected["clique_size"], \
                f"Wrong clique size for {name} with SciPy: got {clique_size}, expected {expected['clique_size']}"
            
            # Test MIS MILP
            mis = solve_mis_scipy(graph, suppress_output=True)
            mis_size = len(mis)
            
            assert verify_independent_set(graph, mis), f"Invalid MIS found for {name} with SciPy"
            assert mis_size == expected["mis_size"], \
                f"Wrong MIS size for {name} with SciPy: got {mis_size}, expected {expected['mis_size']}"
            
            # Test number functions
            clique_num = get_clique_number_scipy(graph, suppress_output=True)
            mis_num = get_independence_number_scipy(graph, suppress_output=True)
            
            assert clique_num == expected["clique_size"], f"Wrong clique number for {name} with SciPy"
            assert mis_num == expected["mis_size"], f"Wrong independence number for {name} with SciPy"


class TestCrossValidation:
    """Cross-validate all approaches against each other."""
    
    @pytest.fixture
    def simple_test_graphs(self):
        """Simple graphs for cross-validation."""
        return [
            ("Triangle", nx.complete_graph(3)),
            ("4-cycle", nx.cycle_graph(4)),
            ("4-path", nx.path_graph(4)),
        ]
    
    @pytest.fixture
    def jax_oracle(self, jax_oracle_config):
        """Create JAX PGD oracle for testing."""
        config = jax_oracle_config.copy()
        config['max_iterations'] = 500
        config['num_restarts'] = 3
        return ProjectedGradientDescentOracle(**config)
    
    def test_brute_force_vs_oracle(self, simple_test_graphs, jax_oracle):
        """Cross-validate brute force vs oracle algorithms."""
        for name, graph in simple_test_graphs:
            # Get results from both methods
            clique_bf = find_max_clique_brute_force(graph)
            clique_oracle, _ = find_max_clique_with_oracle(graph, jax_oracle)
            
            mis_bf = find_mis_brute_force(graph)
            mis_oracle, _ = find_mis_with_oracle(graph, jax_oracle)
            
            # Compare sizes
            assert len(clique_bf) == len(clique_oracle), \
                f"Clique size mismatch for {name}: BF={len(clique_bf)}, Oracle={len(clique_oracle)}"
            assert len(mis_bf) == len(mis_oracle), \
                f"MIS size mismatch for {name}: BF={len(mis_bf)}, Oracle={len(mis_oracle)}"
    
    @pytest.mark.skipif(not MILP_AVAILABLE or not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_brute_force_vs_gurobi(self, simple_test_graphs):
        """Cross-validate brute force vs Gurobi MILP."""
        for name, graph in simple_test_graphs:
            clique_bf = find_max_clique_brute_force(graph)
            clique_gurobi = solve_max_clique_gurobi(graph, suppress_output=True)
            
            mis_bf = find_mis_brute_force(graph)
            mis_gurobi = solve_mis_gurobi(graph, suppress_output=True)
            
            assert len(clique_bf) == len(clique_gurobi), \
                f"Clique size mismatch for {name}: BF={len(clique_bf)}, Gurobi={len(clique_gurobi)}"
            assert len(mis_bf) == len(mis_gurobi), \
                f"MIS size mismatch for {name}: BF={len(mis_bf)}, Gurobi={len(mis_gurobi)}"
    
    @pytest.mark.skipif(not MILP_AVAILABLE or not SCIPY_MILP_AVAILABLE, reason="SciPy MILP not available")
    def test_brute_force_vs_scipy(self, simple_test_graphs):
        """Cross-validate brute force vs SciPy MILP."""
        for name, graph in simple_test_graphs:
            clique_bf = find_max_clique_brute_force(graph)
            clique_scipy = solve_max_clique_scipy(graph, suppress_output=True)
            
            mis_bf = find_mis_brute_force(graph)
            mis_scipy = solve_mis_scipy(graph, suppress_output=True)
            
            assert len(clique_bf) == len(clique_scipy), \
                f"Clique size mismatch for {name}: BF={len(clique_bf)}, SciPy={len(clique_scipy)}"
            assert len(mis_bf) == len(mis_scipy), \
                f"MIS size mismatch for {name}: BF={len(mis_bf)}, SciPy={len(mis_scipy)}"


class TestMotzkinStrausRelationship:
    """Test the fundamental MIS-clique duality relationship."""
    
    @pytest.fixture
    def duality_test_graphs(self):
        """Graphs for testing MIS-clique duality."""
        return [
            ("Triangle", nx.complete_graph(3)),
            ("4-cycle", nx.cycle_graph(4)),
            ("Star", nx.star_graph(3)),
        ]
    
    def test_mis_clique_duality(self, duality_test_graphs):
        """Test α(G) = ω(Ḡ) and ω(G) = α(Ḡ)."""
        for name, graph in duality_test_graphs:
            # Find MIS of G and clique of complement(G)
            mis = find_mis_brute_force(graph)
            mis_size = len(mis)
            
            complement = nx.complement(graph)
            clique_complement = find_max_clique_brute_force(complement)
            clique_complement_size = len(clique_complement)
            
            # They should be equal: α(G) = ω(Ḡ)
            assert mis_size == clique_complement_size, \
                f"Duality violated for {name}: MIS(G)={mis_size}, Clique(Ḡ)={clique_complement_size}"
            
            # Find clique of G and MIS of complement(G)
            clique = find_max_clique_brute_force(graph)
            clique_size = len(clique)
            
            mis_complement = find_mis_brute_force(complement)
            mis_complement_size = len(mis_complement)
            
            # They should be equal: ω(G) = α(Ḡ)
            assert clique_size == mis_complement_size, \
                f"Duality violated for {name}: Clique(G)={clique_size}, MIS(Ḡ)={mis_complement_size}"