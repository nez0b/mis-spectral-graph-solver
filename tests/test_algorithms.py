"""
Unit tests for MIS algorithms.
"""

import unittest
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles.base import Oracle


class MockOracle(Oracle):
    """Mock oracle that returns predetermined values."""
    
    def __init__(self, omega_values: dict):
        """
        Args:
            omega_values: Dict mapping graph descriptions to omega values.
                         Keys should be tuples (num_nodes, num_edges).
        """
        self.omega_values = omega_values
    
    @property
    def name(self) -> str:
        return "Mock"
    
    @property
    def is_available(self) -> bool:
        return True
    
    def solve_quadratic_program(self, adjacency_matrix):
        # We don't actually use this in the mock
        return 0.0
    
    def get_omega(self, graph):
        """Return predetermined omega value based on graph structure."""
        key = (graph.number_of_nodes(), graph.number_of_edges())
        if key in self.omega_values:
            return self.omega_values[key]
        else:
            # Default fallback
            return 1


class TestBruteForceAlgorithm(unittest.TestCase):
    """Test the brute force MIS algorithm."""
    
    def test_empty_graph(self):
        """Test empty graph."""
        G = nx.empty_graph(0)
        result = find_mis_brute_force(G)
        self.assertEqual(result, set())
    
    def test_single_node(self):
        """Test single node graph."""
        G = nx.empty_graph(1)
        result = find_mis_brute_force(G)
        self.assertEqual(result, {0})
    
    def test_two_isolated_nodes(self):
        """Test two isolated nodes."""
        G = nx.empty_graph(2)
        result = find_mis_brute_force(G)
        self.assertEqual(result, {0, 1})
    
    def test_triangle_graph(self):
        """Test triangle graph - MIS size should be 1."""
        G = nx.complete_graph(3)
        result = find_mis_brute_force(G)
        self.assertEqual(len(result), 1)
        self.assertTrue(verify_independent_set(G, result))
    
    def test_cycle_graph(self):
        """Test 5-cycle - MIS size should be 2."""
        G = nx.cycle_graph(5)
        result = find_mis_brute_force(G)
        self.assertEqual(len(result), 2)
        self.assertTrue(verify_independent_set(G, result))
    
    def test_path_graph(self):
        """Test path graph - MIS size should be ceil(n/2)."""
        G = nx.path_graph(5)  # 0-1-2-3-4
        result = find_mis_brute_force(G)
        self.assertEqual(len(result), 3)  # {0, 2, 4} or {1, 3}
        self.assertTrue(verify_independent_set(G, result))


class TestOracleBasedAlgorithm(unittest.TestCase):
    """Test the oracle-based MIS algorithm."""
    
    def test_empty_graph(self):
        """Test empty graph with mock oracle."""
        G = nx.empty_graph(0)
        mock_oracle = MockOracle({(0, 0): 0})
        result = find_mis_with_oracle(G, mock_oracle)
        self.assertEqual(result, set())
    
    def test_single_node(self):
        """Test single node graph."""
        G = nx.empty_graph(1)
        # Complement of single node is also single node, omega = 1
        mock_oracle = MockOracle({(1, 0): 1})
        result = find_mis_with_oracle(G, mock_oracle)
        self.assertEqual(result, {0})
    
    def test_triangle_graph_with_mock(self):
        """Test triangle graph with mock oracle."""
        G = nx.complete_graph(3)
        G_complement = nx.complement(G)  # Complement of K3 is empty graph on 3 nodes
        
        # For the search algorithm, we need to provide omega values for various subgraphs
        # The complement of K3 is an empty graph with 3 nodes, so omega = 1
        # During search, we'll also query subgraphs
        omega_values = {
            (3, 0): 1,  # Complement of K3 (empty graph)
            (2, 0): 1,  # 2-node empty graph
            (1, 0): 1,  # 1-node graph
            (0, 0): 0,  # Empty graph
        }
        
        mock_oracle = MockOracle(omega_values)
        result = find_mis_with_oracle(G, mock_oracle)
        
        # Should get exactly one node
        self.assertEqual(len(result), 1)
        self.assertTrue(verify_independent_set(G, result))
    
    def test_cycle_graph_with_mock(self):
        """Test 5-cycle with mock oracle."""
        G = nx.cycle_graph(5)
        
        # For C5, the complement is C5 itself, and omega(C5) = 2
        # We need to provide omega values for the complement graph and subgraphs
        omega_values = {
            (5, 5): 2,  # C5 complement (which is also C5)
            (4, 2): 2,  # Various 4-node subgraphs
            (3, 1): 2,  # 3-node path
            (2, 0): 1,  # 2-node empty graph
            (1, 0): 1,  # 1-node graph
            (0, 0): 0,  # Empty graph
        }
        
        mock_oracle = MockOracle(omega_values)
        result = find_mis_with_oracle(G, mock_oracle)
        
        # Should get MIS of size 2
        self.assertEqual(len(result), 2)
        self.assertTrue(verify_independent_set(G, result))


class TestVerifyIndependentSet(unittest.TestCase):
    """Test the independent set verification function."""
    
    def test_empty_set(self):
        """Test empty set is always independent."""
        G = nx.complete_graph(5)
        self.assertTrue(verify_independent_set(G, set()))
    
    def test_single_node(self):
        """Test single node is always independent."""
        G = nx.complete_graph(5)
        self.assertTrue(verify_independent_set(G, {0}))
        self.assertTrue(verify_independent_set(G, {2}))
    
    def test_independent_set_in_cycle(self):
        """Test valid independent set in cycle."""
        G = nx.cycle_graph(5)
        self.assertTrue(verify_independent_set(G, {0, 2}))  # Non-adjacent in C5
        self.assertTrue(verify_independent_set(G, {1, 3}))  # Non-adjacent in C5
    
    def test_non_independent_set(self):
        """Test invalid independent set."""
        G = nx.cycle_graph(5)
        self.assertFalse(verify_independent_set(G, {0, 1}))  # Adjacent in C5
        self.assertFalse(verify_independent_set(G, {1, 2}))  # Adjacent in C5
    
    def test_complete_graph(self):
        """Test independent sets in complete graph."""
        G = nx.complete_graph(3)
        self.assertTrue(verify_independent_set(G, {0}))      # Single node OK
        self.assertFalse(verify_independent_set(G, {0, 1}))  # Two nodes not OK
        self.assertFalse(verify_independent_set(G, {0, 1, 2}))  # All nodes not OK


if __name__ == "__main__":
    unittest.main()