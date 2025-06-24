"""
Abstract base class for Motzkin-Straus oracles.
"""

from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from typing import Union


class Oracle(ABC):
    """
    Abstract base class for oracles that solve the Motzkin-Straus quadratic program.
    
    An oracle takes a graph and returns its clique number omega(G) by solving
    the non-convex quadratic program:
    
    max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
    
    where A is the adjacency matrix of the graph.
    """
    
    def __init__(self):
        self.call_count = 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the oracle implementation."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the oracle's dependencies are available."""
        pass
    
    @abstractmethod
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If the solver fails to find a solution.
        """
        pass
    
    def get_omega(self, graph: nx.Graph) -> int:
        """
        Find the clique number omega(G) of a graph using the Motzkin-Straus theorem.
        
        Args:
            graph: A networkx graph.
            
        Returns:
            The clique number omega(G).
            
        Raises:
            OracleError: If the solver fails.
        """
        # Increment call counter
        self.call_count += 1
        
        n = graph.number_of_nodes()
        
        # Handle trivial cases
        if n == 0:
            return 0
        if graph.number_of_edges() == 0:
            return 1
        
        # Get adjacency matrix
        adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
        
        # Solve the quadratic program
        optimal_value = self.solve_quadratic_program(adj_matrix)
        
        # Apply Motzkin-Straus theorem: optimal_value = 0.5 * (1 - 1/omega)
        # Solving for omega: omega = 1 / (1 - 2 * optimal_value)
        if abs(0.5 - optimal_value) < 1e-9:
            # Handle numerical case where omega is very large
            omega_val = n
        else:
            omega_val = 1.0 / (1.0 - 2.0 * optimal_value)
        
        # Round to nearest integer to handle floating point precision
        return round(omega_val)
    
    def __call__(self, graph: nx.Graph) -> int:
        """
        Allow the oracle to be called directly as a function.
        
        Args:
            graph: A networkx graph.
            
        Returns:
            The clique number omega(G).
        """
        return self.get_omega(graph)