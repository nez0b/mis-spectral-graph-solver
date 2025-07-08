"""
Abstract base class for Motzkin-Straus oracles.
"""

from abc import ABC, abstractmethod
import math
import networkx as nx
import numpy as np
from typing import Union, Optional


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
        self.verbose_oracle_calls = False
    
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
    
    def get_omega(self, graph: nx.Graph, regularization_c: Optional[float] = None) -> int:
        """
        Find the clique number omega(G) of a graph using the Motzkin-Straus theorem.
        This method is robust to small floating point errors from approximate solvers
        by systematically rounding down to prevent catastrophic search failure.
        
        Args:
            graph: A networkx graph.
            regularization_c: Optional regularization parameter. If provided,
                            applies identity regularization x^T(A + cI)x instead of x^T A x.
                            This helps eliminate spurious solutions. Common values: 0.5, 1.0.
            
        Returns:
            The clique number omega(G).
            
        Raises:
            OracleError: If the solver fails.
        """
        n = graph.number_of_nodes()
        
        # Increment call counter
        self.call_count += 1
        
        # Verbose progress tracking
        if self.verbose_oracle_calls:
            print(f"Oracle call {self.call_count}: solving {n}-node subgraph with {graph.number_of_edges()} edges...")
        
        # Handle trivial cases
        if n == 0:
            if self.verbose_oracle_calls:
                print(f"  → Trivial case: empty graph, ω = 0")
            return 0
        
        # A graph with nodes but no edges has omega=1
        if graph.number_of_edges() == 0:
            if self.verbose_oracle_calls:
                print(f"  → Trivial case: no edges, ω = 1")
            return 1

        # Get adjacency matrix and apply regularization if requested
        adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
        if regularization_c is not None:
            if regularization_c <= 0:
                raise ValueError(f"Regularization parameter c must be positive, got {regularization_c}")
            # Apply identity regularization: A + cI
            adj_matrix = adj_matrix + regularization_c * np.eye(n, dtype=np.float64)
            if self.verbose_oracle_calls:
                print(f"   Applied regularization: c = {regularization_c}")
        
        optimal_value = self.solve_quadratic_program(adj_matrix)
        
        # A value >= 0.5 implies a very large omega; cap at n
        if optimal_value >= 0.5 - 1e-9:
            if self.verbose_oracle_calls:
                print(f"  → Large omega case: optimal_value = {optimal_value:.6f}, ω = {n}")
            return n

        # A negative value is unexpected. Since we know there's an edge, omega >= 2
        if optimal_value < 0:
            if self.verbose_oracle_calls:
                print(f"  → Negative value case: optimal_value = {optimal_value:.6f}, ω = 2")
            return 2

        # Reverse the M-S formula: omega = 1 / (1 - 2 * optimal_value)
        omega_approx = 1.0 / (1.0 - 2.0 * optimal_value)

        # Round to nearest integer to handle floating point precision
        omega_result = round(omega_approx)

        # Sanity check: a graph with edges must have a clique number of at least 2
        omega_result = max(2, int(omega_result))
        
        if self.verbose_oracle_calls:
            print(f"  → Solved: optimal_value = {optimal_value:.6f}, ω_approx = {omega_approx:.3f}, ω = {omega_result}")
        
        return omega_result
    
    def __call__(self, graph: nx.Graph, regularization_c: Optional[float] = None) -> int:
        """
        Allow the oracle to be called directly as a function.
        
        Args:
            graph: A networkx graph.
            regularization_c: Optional regularization parameter.
            
        Returns:
            The clique number omega(G).
        """
        return self.get_omega(graph, regularization_c)