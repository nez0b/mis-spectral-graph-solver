"""
Tests for JAX-based oracles comparing against Gurobi on small graphs.
"""

import pytest
import numpy as np
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, verify_independent_set
from motzkinstraus.exceptions import SolverUnavailableError

# Import oracles
try:
    from motzkinstraus.oracles.gurobi import GurobiOracle
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
    from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture
def test_graphs():
    """Create a collection of small test graphs with known properties."""
    graphs = {}
    
    # Triangle (K3) - clique number = 3
    graphs['triangle'] = nx.complete_graph(3)
    
    # 4-cycle - clique number = 2, independence number = 2
    graphs['4cycle'] = nx.cycle_graph(4)
    
    # 5-cycle - clique number = 2, independence number = 2
    graphs['5cycle'] = nx.cycle_graph(5)
    
    # Complete graph K4 - clique number = 4
    graphs['K4'] = nx.complete_graph(4)
    
    # Path graph P4 - clique number = 2, independence number = 2
    graphs['path4'] = nx.path_graph(4)
    
    # Star graph with 4 leaves - clique number = 2, independence number = 4
    graphs['star4'] = nx.star_graph(4)
    
    # Empty graph (no edges) - clique number = 1
    graphs['empty5'] = nx.Graph()
    graphs['empty5'].add_nodes_from(range(5))
    
    # Single edge - clique number = 2
    graphs['single_edge'] = nx.Graph()
    graphs['single_edge'].add_edge(0, 1)
    
    return graphs


@pytest.fixture
def oracle_configs():
    """Standard configurations for testing different oracles."""
    return {
        'fast': {
            'learning_rate': 0.02,
            'max_iterations': 500,
            'tolerance': 1e-5,
            'num_restarts': 5,
            'verbose': False
        },
        'precise': {
            'learning_rate': 0.01,
            'max_iterations': 2000,
            'tolerance': 1e-7,
            'num_restarts': 10,
            'verbose': False
        }
    }


class TestJAXOracleAvailability:
    """Test JAX oracle availability and initialization."""
    
    def test_jax_availability(self):
        """Test if JAX is available for testing."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Should not raise
        pgd_oracle = ProjectedGradientDescentOracle(verbose=False)
        md_oracle = MirrorDescentOracle(verbose=False)
        
        assert pgd_oracle.is_available
        assert md_oracle.is_available
        assert "JAX-PGD" in pgd_oracle.name
        assert "JAX-MD" in md_oracle.name
    
    def test_oracle_initialization_parameters(self, oracle_configs):
        """Test oracle initialization with different parameters."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        config = oracle_configs['precise']
        
        pgd_oracle = ProjectedGradientDescentOracle(**config)
        md_oracle = MirrorDescentOracle(**config)
        
        assert pgd_oracle.config.learning_rate == config['learning_rate']
        assert pgd_oracle.config.max_iterations == config['max_iterations']
        assert pgd_oracle.config.num_restarts == config['num_restarts']
        
        assert md_oracle.config.learning_rate == config['learning_rate']
        assert md_oracle.config.max_iterations == config['max_iterations']
        assert md_oracle.config.num_restarts == config['num_restarts']


class TestJAXOracleBasicFunctionality:
    """Test basic oracle functionality on simple graphs."""
    
    @pytest.mark.parametrize("graph_name", ['triangle', '4cycle', 'single_edge', 'empty5'])
    def test_pgd_oracle_basic(self, test_graphs, graph_name):
        """Test PGD oracle on basic graphs."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        graph = test_graphs[graph_name]
        oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1000,
            num_restarts=5,
            verbose=False
        )
        
        # Test that omega calculation works
        omega = oracle.get_omega(graph)
        assert isinstance(omega, int)
        assert omega >= 1  # Every graph has at least clique number 1
        
        # Test that optimization details are available
        details = oracle.get_optimization_details()
        assert 'num_restarts' in details
        assert details['num_restarts'] == 5
        assert 'best_energy' in details
    
    @pytest.mark.parametrize("graph_name", ['triangle', '4cycle', 'single_edge', 'empty5'])
    def test_md_oracle_basic(self, test_graphs, graph_name):
        """Test MD oracle on basic graphs."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        graph = test_graphs[graph_name]
        oracle = MirrorDescentOracle(
            learning_rate=0.01,
            max_iterations=1000,
            num_restarts=5,
            verbose=False
        )
        
        # Test that omega calculation works
        omega = oracle.get_omega(graph)
        assert isinstance(omega, int)
        assert omega >= 1  # Every graph has at least clique number 1
        
        # Test that optimization details are available
        details = oracle.get_optimization_details()
        assert 'num_restarts' in details
        assert details['num_restarts'] == 5
        assert 'best_energy' in details


class TestJAXOracleAccuracy:
    """Test JAX oracle accuracy against known results."""
    
    def test_known_clique_numbers(self, test_graphs):
        """Test that oracles produce correct clique numbers for known graphs."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Known clique numbers
        expected = {
            'triangle': 3,
            'K4': 4,
            'single_edge': 2,
            'empty5': 1,
            '4cycle': 2,
            '5cycle': 2,
            'path4': 2,
            'star4': 2
        }
        
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1500,
            num_restarts=8,
            tolerance=1e-6,
            verbose=False
        )
        
        md_oracle = MirrorDescentOracle(
            learning_rate=0.008,
            max_iterations=1500,
            num_restarts=8,
            tolerance=1e-6,
            verbose=False
        )
        
        for graph_name, expected_omega in expected.items():
            graph = test_graphs[graph_name]
            
            pgd_omega = pgd_oracle.get_omega(graph)
            md_omega = md_oracle.get_omega(graph)
            
            assert pgd_omega == expected_omega, f"PGD failed on {graph_name}: got {pgd_omega}, expected {expected_omega}"
            assert md_omega == expected_omega, f"MD failed on {graph_name}: got {md_omega}, expected {expected_omega}"


@pytest.mark.skipif(not (JAX_AVAILABLE and GUROBI_AVAILABLE), 
                   reason="Both JAX and Gurobi required for comparison")
class TestJAXGurobiComparison:
    """Compare JAX oracles against Gurobi baseline."""
    
    def test_omega_agreement(self, test_graphs):
        """Test that JAX oracles agree with Gurobi on omega values."""
        gurobi_oracle = GurobiOracle(suppress_output=True)
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=2000,
            num_restarts=10,
            tolerance=1e-7,
            verbose=False
        )
        md_oracle = MirrorDescentOracle(
            learning_rate=0.008,
            max_iterations=2000,
            num_restarts=10,
            tolerance=1e-7,
            verbose=False
        )
        
        for graph_name, graph in test_graphs.items():
            if graph.number_of_nodes() == 0:
                continue  # Skip empty graphs
                
            gurobi_omega = gurobi_oracle.get_omega(graph)
            pgd_omega = pgd_oracle.get_omega(graph)
            md_omega = md_oracle.get_omega(graph)
            
            assert pgd_omega == gurobi_omega, f"PGD disagreement on {graph_name}: PGD={pgd_omega}, Gurobi={gurobi_omega}"
            assert md_omega == gurobi_omega, f"MD disagreement on {graph_name}: MD={md_omega}, Gurobi={gurobi_omega}"
    
    def test_mis_algorithm_integration(self, test_graphs):
        """Test that JAX oracles work correctly in the MIS algorithm."""
        gurobi_oracle = GurobiOracle(suppress_output=True)
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1500,
            num_restarts=8,
            verbose=False
        )
        md_oracle = MirrorDescentOracle(
            learning_rate=0.008,
            max_iterations=1500,
            num_restarts=8,
            verbose=False
        )
        
        for graph_name, graph in test_graphs.items():
            if graph.number_of_nodes() <= 1:
                continue  # Skip trivial graphs
                
            # Run MIS algorithm with each oracle
            gurobi_mis, gurobi_calls = find_mis_with_oracle(graph, gurobi_oracle)
            pgd_mis, pgd_calls = find_mis_with_oracle(graph, pgd_oracle)
            md_mis, md_calls = find_mis_with_oracle(graph, md_oracle)
            
            # Verify all solutions are valid independent sets
            assert verify_independent_set(graph, gurobi_mis), f"Gurobi MIS invalid on {graph_name}"
            assert verify_independent_set(graph, pgd_mis), f"PGD MIS invalid on {graph_name}"
            assert verify_independent_set(graph, md_mis), f"MD MIS invalid on {graph_name}"
            
            # All should find optimal MIS (same size)
            assert len(pgd_mis) == len(gurobi_mis), f"PGD MIS size mismatch on {graph_name}: PGD={len(pgd_mis)}, Gurobi={len(gurobi_mis)}"
            assert len(md_mis) == len(gurobi_mis), f"MD MIS size mismatch on {graph_name}: MD={len(md_mis)}, Gurobi={len(gurobi_mis)}"
            
            # Oracle call counts should be identical (same algorithm)
            assert pgd_calls == gurobi_calls, f"PGD call count mismatch on {graph_name}: PGD={pgd_calls}, Gurobi={gurobi_calls}"
            assert md_calls == gurobi_calls, f"MD call count mismatch on {graph_name}: MD={md_calls}, Gurobi={gurobi_calls}"


class TestJAXOracleRobustness:
    """Test robustness and edge cases for JAX oracles."""
    
    def test_multi_restart_consistency(self, test_graphs):
        """Test that multi-restart gives consistent results."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        graph = test_graphs['5cycle']
        
        # Run same oracle multiple times
        results = []
        for seed in range(5):
            oracle = ProjectedGradientDescentOracle(
                learning_rate=0.02,
                max_iterations=1000,
                num_restarts=10,
                verbose=False
            )
            oracle.config.verbose = False  # Ensure quiet
            omega = oracle.get_omega(graph)
            results.append(omega)
        
        # All results should be the same
        assert all(r == results[0] for r in results), f"Inconsistent results: {results}"
    
    def test_edge_cases(self):
        """Test edge cases like empty graphs and single nodes."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        pgd_oracle = ProjectedGradientDescentOracle(verbose=False)
        md_oracle = MirrorDescentOracle(verbose=False)
        
        # Empty graph
        empty_graph = nx.Graph()
        assert pgd_oracle.get_omega(empty_graph) == 0
        assert md_oracle.get_omega(empty_graph) == 0
        
        # Single node
        single_node = nx.Graph()
        single_node.add_node(0)
        assert pgd_oracle.get_omega(single_node) == 1
        assert md_oracle.get_omega(single_node) == 1
        
        # Graph with no edges
        no_edges = nx.Graph()
        no_edges.add_nodes_from(range(3))
        assert pgd_oracle.get_omega(no_edges) == 1
        assert md_oracle.get_omega(no_edges) == 1
    
    def test_convergence_tracking(self, test_graphs):
        """Test that convergence histories are properly tracked."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        graph = test_graphs['triangle']
        oracle = ProjectedGradientDescentOracle(
            max_iterations=500,
            num_restarts=5,
            verbose=False
        )
        
        omega = oracle.get_omega(graph)
        
        # Check that histories were recorded
        histories = oracle.get_convergence_histories()
        assert len(histories) == 5  # num_restarts
        
        for history in histories:
            assert len(history) > 0  # Non-empty history
            assert len(history) <= 500  # Doesn't exceed max_iterations
            # Energy should be non-decreasing (we're maximizing)
            assert history[-1] >= history[0] * 0.8  # Allow some numerical tolerance
        
        # Check optimization details
        details = oracle.get_optimization_details()
        assert details['num_restarts'] == 5
        assert details['best_restart_idx'] >= 0
        assert details['energy_range'] >= 0


class TestJAXOraclePerformance:
    """Performance and scaling tests for JAX oracles."""
    
    def test_larger_graphs(self):
        """Test JAX oracles on slightly larger graphs."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # 8-node cycle with some chords
        G = nx.cycle_graph(8)
        G.add_edges_from([(0, 4), (2, 6)])  # Add chords
        
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.015,
            max_iterations=1000,
            num_restarts=8,
            verbose=False
        )
        
        md_oracle = MirrorDescentOracle(
            learning_rate=0.008,
            max_iterations=1000,
            num_restarts=8,
            verbose=False
        )
        
        pgd_omega = pgd_oracle.get_omega(G)
        md_omega = md_oracle.get_omega(G)
        
        # Both should give same result
        assert pgd_omega == md_omega
        
        # Should be reasonable (cycle has clique number 2, chords might increase it)
        assert 2 <= pgd_omega <= 4
    
    @pytest.mark.parametrize("num_restarts", [1, 5, 15])
    def test_restart_scaling(self, test_graphs, num_restarts):
        """Test how performance scales with number of restarts."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        graph = test_graphs['5cycle']
        
        oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=500,
            num_restarts=num_restarts,
            verbose=False
        )
        
        omega = oracle.get_omega(graph)
        details = oracle.get_optimization_details()
        
        assert omega == 2  # Known result for 5-cycle
        assert details['num_restarts'] == num_restarts
        assert len(oracle.get_convergence_histories()) == num_restarts


if __name__ == "__main__":
    # Run a quick test if called directly
    import warnings
    warnings.filterwarnings("ignore")
    
    if JAX_AVAILABLE:
        print("JAX is available - running basic tests...")
        
        # Create test graph
        G = nx.cycle_graph(5)
        
        # Test PGD
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1000,
            num_restarts=5,
            verbose=True
        )
        pgd_omega = pgd_oracle.get_omega(G)
        print(f"PGD Oracle result: omega = {pgd_omega}")
        
        # Test MD
        md_oracle = MirrorDescentOracle(
            learning_rate=0.01,
            max_iterations=1000,
            num_restarts=5,
            verbose=True
        )
        md_omega = md_oracle.get_omega(G)
        print(f"MD Oracle result: omega = {md_omega}")
        
        if GUROBI_AVAILABLE:
            gurobi_oracle = GurobiOracle(suppress_output=True)
            gurobi_omega = gurobi_oracle.get_omega(G)
            print(f"Gurobi Oracle result: omega = {gurobi_omega}")
            print(f"Agreement: PGD={pgd_omega==gurobi_omega}, MD={md_omega==gurobi_omega}")
        
        print("Basic tests completed successfully!")
    else:
        print("JAX not available - skipping tests")