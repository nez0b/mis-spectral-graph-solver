"""
Unit tests for oracle implementations.
"""

import unittest
import numpy as np
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.oracles.base import Oracle
from motzkinstraus.exceptions import SolverUnavailableError, OracleError


class MockOracle(Oracle):
    """Mock oracle for testing purposes."""
    
    def __init__(self, return_value: float = 0.25):
        self.return_value = return_value
    
    @property
    def name(self) -> str:
        return "Mock"
    
    @property
    def is_available(self) -> bool:
        return True
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        return self.return_value


class TestOracleBase(unittest.TestCase):
    """Test the base Oracle class functionality."""
    
    def setUp(self):
        self.mock_oracle = MockOracle(return_value=0.25)
    
    def test_empty_graph(self):
        """Test oracle behavior on empty graph."""
        G = nx.empty_graph(0)
        result = self.mock_oracle.get_omega(G)
        self.assertEqual(result, 0)
    
    def test_single_node_graph(self):
        """Test oracle behavior on single node graph."""
        G = nx.empty_graph(1)
        result = self.mock_oracle.get_omega(G)
        self.assertEqual(result, 1)
    
    def test_no_edges_graph(self):
        """Test oracle behavior on graph with nodes but no edges."""
        G = nx.empty_graph(5)
        result = self.mock_oracle.get_omega(G)
        self.assertEqual(result, 1)
    
    def test_callable_interface(self):
        """Test that oracle can be called directly as a function."""
        G = nx.complete_graph(2)
        result1 = self.mock_oracle.get_omega(G)
        result2 = self.mock_oracle(G)
        self.assertEqual(result1, result2)
    
    def test_motzkin_straus_formula(self):
        """Test the Motzkin-Straus formula calculation."""
        # For optimal_value = 0.25, omega should be 1/(1-2*0.25) = 1/0.5 = 2
        mock_oracle = MockOracle(return_value=0.25)
        G = nx.complete_graph(2)
        result = mock_oracle.get_omega(G)
        self.assertEqual(result, 2)
    
    def test_near_complete_graph_case(self):
        """Test handling of numerical edge case near 0.5."""
        # When optimal_value is very close to 0.5, should return n
        mock_oracle = MockOracle(return_value=0.5 - 1e-10)
        G = nx.complete_graph(5)
        result = mock_oracle.get_omega(G)
        self.assertEqual(result, 5)


class TestGurobiOracle(unittest.TestCase):
    """Test Gurobi oracle implementation."""
    
    def setUp(self):
        try:
            from motzkinstraus.oracles.gurobi import GurobiOracle
            self.oracle = GurobiOracle(suppress_output=True)
            self.gurobi_available = True
        except (ImportError, SolverUnavailableError):
            self.gurobi_available = False
            self.skipTest("Gurobi not available")
    
    def test_triangle_graph(self):
        """Test on triangle graph (K3) where omega = 3."""
        if not self.gurobi_available:
            self.skipTest("Gurobi not available")
        
        G = nx.complete_graph(3)
        result = self.oracle.get_omega(G)
        self.assertEqual(result, 3)
    
    def test_cycle_graph(self):
        """Test on 5-cycle where omega = 2."""
        if not self.gurobi_available:
            self.skipTest("Gurobi not available")
        
        G = nx.cycle_graph(5)
        result = self.oracle.get_omega(G)
        self.assertEqual(result, 2)
    
    def test_path_graph(self):
        """Test on path graph where omega = 2."""
        if not self.gurobi_available:
            self.skipTest("Gurobi not available")
        
        G = nx.path_graph(5)  # Path: 0-1-2-3-4
        result = self.oracle.get_omega(G)
        self.assertEqual(result, 2)  # Maximum clique size is 2 (any edge)


class TestDiracOracle(unittest.TestCase):
    """Test Dirac oracle implementation."""
    
    def setUp(self):
        try:
            from motzkinstraus.oracles.dirac import DiracOracle
            self.oracle = DiracOracle(num_samples=50, relax_schedule=1)
            self.dirac_available = True
        except (ImportError, SolverUnavailableError):
            self.dirac_available = False
            self.skipTest("Dirac not available")
    
    def test_small_graph(self):
        """Test on small graph - just check it doesn't crash."""
        if not self.dirac_available:
            self.skipTest("Dirac not available")
        
        G = nx.complete_graph(2)
        try:
            result = self.oracle.get_omega(G)
            # Just check that we get a reasonable integer result
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 1)
        except OracleError:
            # Dirac might not be properly configured, which is okay for this test
            self.skipTest("Dirac solver not properly configured")


if __name__ == "__main__":
    unittest.main()