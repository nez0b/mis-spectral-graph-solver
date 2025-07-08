"""
Regularized oracle base class for Motzkin-Straus formulations.

This module provides the abstract base class for implementing regularized
versions of the Motzkin-Straus quadratic program. Regularization helps eliminate
spurious solutions by modifying the objective function from x^T A x to 
x^T (A + R) x, where R is a regularization matrix.

The canonical regularization adds a scaled identity matrix: R = cI, transforming
the objective to x^T (A + cI) x. This makes the optimization landscape strictly
concave, eliminating spurious solutions and ensuring a one-to-one correspondence
between optimal solutions and maximum cliques.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union
import numpy as np
from .base import Oracle


class RegularizationFunction(ABC):
    """
    Abstract base class for regularization functions.
    
    This enables extensible regularization beyond the simple cI case,
    allowing for more sophisticated regularization terms in the future.
    """
    
    @abstractmethod
    def apply(self, adjacency_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply regularization to an adjacency matrix.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            **kwargs: Regularization-specific parameters
            
        Returns:
            Regularized matrix A + R
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the regularization function."""
        pass


class IdentityRegularization(RegularizationFunction):
    """
    Canonical identity matrix regularization: R = cI.
    
    This is the most common and well-studied regularization for Motzkin-Straus
    programs. It adds a scaled identity matrix to the adjacency matrix,
    effectively adding self-loops with weight c/2 to each vertex.
    
    Mathematical effect:
    - Original objective: x^T A x  
    - Regularized objective: x^T (A + cI) x = x^T A x + c * ||x||^2
    
    The additional term c * ||x||^2 is strictly convex, making the overall
    objective strictly concave and eliminating spurious solutions.
    """
    
    def __init__(self, c: float = 0.5):
        """
        Initialize identity regularization.
        
        Args:
            c: Regularization parameter. Must be in [0, 1] for theoretical guarantees:
               - c = 0.0: No regularization (equivalent to standard Motzkin-Straus)
               - c = 0.5: Default recommended value
               - c = 1.0: Maximum regularization
               - c ∈ [0, 1]: Valid range for theoretical guarantees
        """
        if c < 0 or c > 1:
            raise ValueError(f"Regularization parameter c must be in [0, 1], got {c}")
        self.c = c
    
    def apply(self, adjacency_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply identity regularization: A + cI.
        
        Args:
            adjacency_matrix: Original adjacency matrix A
            **kwargs: Ignored for identity regularization
            
        Returns:
            Regularized matrix A + cI
        """
        n = adjacency_matrix.shape[0]
        if adjacency_matrix.shape != (n, n):
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency_matrix.shape}")
        
        regularization_matrix = self.c * np.eye(n, dtype=adjacency_matrix.dtype)
        return adjacency_matrix + regularization_matrix
    
    @property
    def name(self) -> str:
        return f"Identity(c={self.c})"


class RegularizedOracle(Oracle):
    """
    Abstract base class for regularized Motzkin-Straus oracles.
    
    This class extends the base Oracle class to support regularization of the
    objective function. It provides infrastructure for applying regularization
    functions and maintains the same interface as standard oracles.
    
    The regularized quadratic program is:
    max x^T (A + R) x subject to sum(x_i) = 1, x_i >= 0
    
    where R is the regularization matrix computed by a RegularizationFunction.
    
    IMPORTANT: This class correctly handles the omega conversion for regularized objectives.
    """
    
    def __init__(self, regularization_function: Optional[RegularizationFunction] = None):
        """
        Initialize regularized oracle.
        
        Args:
            regularization_function: Function to compute regularization matrix.
                                   If None, uses default IdentityRegularization(c=0.5)
        """
        super().__init__()
        if regularization_function is None:
            regularization_function = IdentityRegularization(c=0.5)
        self.regularization_function = regularization_function
    
    def apply_regularization(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Apply regularization to an adjacency matrix.
        
        Args:
            adjacency_matrix: Original adjacency matrix A
            
        Returns:
            Regularized matrix A + R
        """
        return self.regularization_function.apply(adjacency_matrix)
    
    def get_omega(self, graph, regularization_c: Optional[float] = None) -> int:
        """
        Override get_omega to properly handle regularized objective values.
        
        For regularized objectives x^T (A + cI) x, the relationship to omega changes:
        - Original: max(x^T A x) = 1 - 1/ω  
        - Regularized: max(x^T (A + cI) x) = (1 - 1/ω) + c/ω = 1 + (c-1)/ω
        
        We need to subtract the regularization contribution before applying the standard formula.
        """
        import networkx as nx
        
        n = graph.number_of_nodes()
        
        # Increment call counter
        self.call_count += 1
        
        # Verbose progress tracking
        if self.verbose_oracle_calls:
            print(f"Regularized Oracle call {self.call_count}: solving {n}-node subgraph with {graph.number_of_edges()} edges...")
        
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

        # Get adjacency matrix and solve the regularized quadratic program
        adj_matrix = nx.to_numpy_array(graph, dtype=np.float64)
        regularized_optimal_value = self.solve_regularized_quadratic_program(adj_matrix)
        
        # Get regularization parameter
        if isinstance(self.regularization_function, IdentityRegularization):
            c = self.regularization_function.c
        else:
            # For other regularization types, we'd need different handling
            # For now, assume identity regularization
            c = 0.5  # fallback
        
        if self.verbose_oracle_calls:
            print(f"   Regularization parameter c = {c}")
            print(f"   Regularized optimal value = {regularized_optimal_value:.6f}")
        
        # Convert back to omega using the correct regularized Motzkin-Straus formula
        # 
        # JAX gives us: 0.5 * x^T (A + cI) x (the "half" formulation)
        # Full formulation: x^T (A + cI) x = 2 * regularized_optimal_value
        # 
        # Theory: max(x^T (A + cI) x) = (1 - 1/ω) + c/ω = 1 + (c-1)/ω
        # Solving for ω: ω = (c-1)/(regularized_full - 1)
        
        # Convert from half to full formulation
        regularized_full = 2.0 * regularized_optimal_value
        
        if self.verbose_oracle_calls:
            print(f"   Regularized full value = {regularized_full:.6f}")
        
        # Handle edge cases
        if abs(regularized_full - 1.0) < 1e-10:
            # Very close to 1 means very large omega
            if self.verbose_oracle_calls:
                print(f"  → Very large omega case: regularized_full ≈ 1, ω = {n}")
            return n
        
        if regularized_full < 0:
            # Unexpected negative value
            if self.verbose_oracle_calls:
                print(f"  → Negative value case: regularized_full = {regularized_full:.6f}, ω = 2")
            return 2
        
        # Apply the correct formula for c ∈ [0, 1]:
        # Theory: regularized_full = 1 - (1-c)/ω  
        # Solving for ω: ω = (1-c)/(1 - regularized_full)
        
        # Special case: c = 0 (no regularization)
        if c == 0.0:
            if self.verbose_oracle_calls:
                print(f"  → Special case: c = 0 (no regularization), using standard Motzkin-Straus formula")
            # Standard Motzkin-Straus: regularized_full = 1 - 1/ω
            # Solving: ω = 1/(1 - regularized_full)
            denominator = 1.0 - regularized_full
            if abs(denominator) < 1e-10:
                return n
            omega_float = 1.0 / denominator
            omega_candidate = max(2, round(omega_float))
        
        # Special case: c = 1 (maximum regularization)
        elif c == 1.0:
            if self.verbose_oracle_calls:
                print(f"  → Special case: c = 1 (maximum regularization)")
            # When c=1: regularized_full = 1, so ω = ∞ (but limited by graph size)
            # In practice, this should give the number of nodes
            return n
        
        # General case: c ∈ (0, 1)
        else:
            if regularized_full >= 1.0:
                # Edge case: regularized_full ≥ 1 suggests very large omega
                if self.verbose_oracle_calls:
                    print(f"  → Large omega case: regularized_full = {regularized_full:.6f}, ω = {n}")
                return n
            
            # Normal case for c ∈ (0, 1)
            denominator = 1.0 - regularized_full
            if abs(denominator) < 1e-10:
                # Very close to 1 means very large omega
                if self.verbose_oracle_calls:
                    print(f"  → Very large omega case: denominator ≈ 0, ω = {n}")
                return n
            
            omega_float = (1.0 - c) / denominator
            omega_candidate = max(2, round(omega_float))
        
        if self.verbose_oracle_calls:
            print(f"   Calculated ω = (1-c)/(1-reg_full) = ({1-c:.3f})/(1-{regularized_full:.6f}) = {omega_float:.6f}")
            print(f"   Rounded to ω = {omega_candidate}")
        
        # Sanity check: omega shouldn't exceed number of nodes
        omega_result = min(omega_candidate, n)
        
        if self.verbose_oracle_calls:
            print(f"  → Solved: regularized_value = {regularized_optimal_value:.6f}, ω = {omega_result}")
        
        return omega_result
    
    def _fallback_omega_search(self, regularized_optimal_value: float, c: float, n: int) -> int:
        """
        Fallback method for omega calculation when c is outside (0,1) or other edge cases.
        
        Uses iterative search to find the best matching omega value.
        """
        regularized_full = 2.0 * regularized_optimal_value
        
        best_omega = 2
        best_error = float('inf')
        
        for trial_omega in range(2, n + 5):  # Try reasonable omega values
            # Expected regularized value for this omega: 1 - (1-c)/ω
            if c < 1:
                expected_regularized_full = 1 - (1-c)/trial_omega
            else:
                # For c ≥ 1, use: 1 + (c-1)/ω  
                expected_regularized_full = 1 + (c-1)/trial_omega
            
            error = abs(expected_regularized_full - regularized_full)
            if error < best_error:
                best_error = error
                best_omega = trial_omega
                
            if self.verbose_oracle_calls and trial_omega <= 6:
                print(f"   Trial ω={trial_omega}: expected={expected_regularized_full:.6f}, error={error:.6f}")
        
        if self.verbose_oracle_calls:
            print(f"  → Fallback search result: ω = {best_omega}")
        
        return best_omega
    
    @abstractmethod
    def solve_regularized_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the regularized Motzkin-Straus quadratic program.
        
        This method should optimize x^T (A + R) x over the simplex, where
        R is the regularization matrix.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            
        Returns:
            The optimal value of the regularized quadratic program
            
        Raises:
            OracleError: If the solver fails to find a solution
        """
        pass
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the quadratic program by applying regularization first.
        
        This method implements the Oracle interface by applying regularization
        and then solving the resulting program.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            
        Returns:
            The optimal value of the regularized quadratic program
        """
        return self.solve_regularized_quadratic_program(adjacency_matrix)
    
    @property
    def name(self) -> str:
        """Name of the regularized oracle."""
        base_name = getattr(self, '_base_name', 'Regularized')
        return f"{base_name}+{self.regularization_function.name}"


def create_identity_regularization(c: float = 0.5) -> IdentityRegularization:
    """
    Convenience function to create identity regularization.
    
    Args:
        c: Regularization parameter (default: 0.5)
        
    Returns:
        IdentityRegularization instance
    """
    return IdentityRegularization(c)


def validate_regularization_parameter(c: Union[float, int]) -> float:
    """
    Validate and convert regularization parameter.
    
    Args:
        c: Regularization parameter. Must be in [0, 1] for theoretical guarantees.
        
    Returns:
        Validated float parameter
        
    Raises:
        ValueError: If parameter is invalid or outside [0, 1]
    """
    try:
        c_float = float(c)
        if c_float < 0 or c_float > 1:
            raise ValueError(f"Regularization parameter must be in [0, 1], got {c_float}")
        return c_float
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid regularization parameter: {c}") from e