"""
Regularized JAX-based Projected Gradient Descent oracle for the Motzkin-Straus quadratic program.

This oracle extends the standard JAX PGD oracle to support regularization, solving:
max x^T (A + R) x subject to sum(x_i) = 1, x_i >= 0

where R is a regularization matrix (typically R = cI) that eliminates spurious solutions.
"""

import numpy as np
from typing import List, Optional
from .jax_pgd import ProjectedGradientDescentOracle
from .regularized_base import RegularizedOracle, IdentityRegularization, RegularizationFunction
from ..exceptions import OracleError, SolverUnavailableError

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


class RegularizedJAXPGDOracle(RegularizedOracle, ProjectedGradientDescentOracle):
    """
    Regularized JAX-based Projected Gradient Descent oracle.
    
    This oracle combines the regularization framework with JAX-based PGD optimization
    to solve the regularized Motzkin-Straus program:
    max x^T (A + R) x subject to sum(x_i) = 1, x_i >= 0
    
    Key benefits:
    - Eliminates spurious solutions through regularization
    - Ensures one-to-one correspondence between optima and cliques
    - Fast JAX-compiled optimization with multi-restart strategy
    - Extensible regularization framework for future research
    
    Args:
        regularization_function: Function to compute regularization matrix.
                               If None, uses IdentityRegularization(c=0.5)
        learning_rate: Step size for gradient descent (default: 0.01).
        max_iterations: Maximum optimization iterations (default: 2000).
        tolerance: Convergence tolerance for early stopping (default: 1e-6).
        min_iterations: Minimum iterations before stopping (default: 50).
        num_restarts: Number of Dirichlet initializations (default: 10).
        dirichlet_alpha: Concentration parameter for Dirichlet sampling (default: 1.0).
        verbose: Whether to print optimization progress (default: False).
    """
    
    def __init__(
        self,
        regularization_function: Optional[RegularizationFunction] = None,
        learning_rate: float = 0.01,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
        min_iterations: int = 50,
        num_restarts: int = 10,
        dirichlet_alpha: float = 1.0,
        verbose: bool = False
    ):
        # Initialize regularization first
        if regularization_function is None:
            regularization_function = IdentityRegularization(c=0.5)
        
        # Initialize both parent classes
        RegularizedOracle.__init__(self, regularization_function)
        ProjectedGradientDescentOracle.__init__(
            self,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            min_iterations=min_iterations,
            num_restarts=num_restarts,
            dirichlet_alpha=dirichlet_alpha,
            verbose=verbose
        )
        
        # Set base name for oracle naming
        self._base_name = "JAX-PGD"
    
    @property
    def name(self) -> str:
        """Name of the regularized oracle."""
        base_name = f"JAX-PGD(lr={self.config.learning_rate},restarts={self.config.num_restarts})"
        return f"{base_name}+{self.regularization_function.name}"
    
    def solve_regularized_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the regularized Motzkin-Straus quadratic program.
        
        This method applies regularization to the adjacency matrix and then
        uses the standard JAX PGD solver on the regularized problem.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            
        Returns:
            The optimal value of the regularized quadratic program
            
        Raises:
            OracleError: If the optimization fails
        """
        # Apply regularization to get A + R
        regularized_matrix = self.apply_regularization(adjacency_matrix)
        
        if self.config.verbose:
            reg_name = self.regularization_function.name
            print(f"Regularized PGD: Applied {reg_name} regularization")
            print(f"Regularized PGD: Matrix diagonal increased by average {np.mean(np.diag(regularized_matrix - adjacency_matrix)):.4f}")
        
        # Use parent class's solve_quadratic_program method on regularized matrix
        return ProjectedGradientDescentOracle.solve_quadratic_program(self, regularized_matrix)
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray, x0: Optional[np.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> float:
        """
        Solve the quadratic program by first applying regularization.
        
        This method maintains compatibility with the base Oracle interface
        while implementing regularization.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            x0: Optional initial point for single-start refinement
            key: Optional JAX random key for reproducible randomness
            
        Returns:
            The optimal value of the regularized quadratic program
        """
        # Apply regularization and solve
        regularized_matrix = self.apply_regularization(adjacency_matrix)
        return ProjectedGradientDescentOracle.solve_quadratic_program(self, regularized_matrix, x0, key)
    
    def compare_with_unregularized(self, adjacency_matrix: np.ndarray) -> dict:
        """
        Compare regularized and unregularized optimization results.
        
        This is useful for analysis and validation of regularization effects.
        
        Args:
            adjacency_matrix: The original adjacency matrix A
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Solve unregularized problem
            unreg_value = ProjectedGradientDescentOracle.solve_quadratic_program(self, adjacency_matrix)
            unreg_histories = [np.array(hist) for hist in self.last_histories]
            unreg_final_energies = list(self.last_final_energies)
            unreg_solution = np.array(self.x_current) if hasattr(self, 'x_current') else None
            
            # Solve regularized problem
            reg_value = self.solve_regularized_quadratic_program(adjacency_matrix)
            reg_histories = [np.array(hist) for hist in self.last_histories]
            reg_final_energies = list(self.last_final_energies)
            reg_solution = np.array(self.x_current) if hasattr(self, 'x_current') else None
            
            # Compute differences
            value_diff = reg_value - unreg_value
            energy_variance_unreg = np.var(unreg_final_energies) if len(unreg_final_energies) > 1 else 0.0
            energy_variance_reg = np.var(reg_final_energies) if len(reg_final_energies) > 1 else 0.0
            
            # Solution sparsity analysis
            unreg_sparsity = np.sum(unreg_solution > 1e-6) if unreg_solution is not None else 0
            reg_sparsity = np.sum(reg_solution > 1e-6) if reg_solution is not None else 0
            
            return {
                "regularization": self.regularization_function.name,
                "unregularized": {
                    "optimal_value": float(unreg_value),
                    "final_energies": unreg_final_energies,
                    "energy_variance": float(energy_variance_unreg),
                    "solution_sparsity": int(unreg_sparsity),
                    "solution": unreg_solution.tolist() if unreg_solution is not None else None
                },
                "regularized": {
                    "optimal_value": float(reg_value),
                    "final_energies": reg_final_energies,
                    "energy_variance": float(energy_variance_reg),
                    "solution_sparsity": int(reg_sparsity),
                    "solution": reg_solution.tolist() if reg_solution is not None else None
                },
                "comparison": {
                    "value_difference": float(value_diff),
                    "energy_variance_reduction": float(energy_variance_unreg - energy_variance_reg),
                    "sparsity_change": int(reg_sparsity - unreg_sparsity),
                    "regularization_benefit": float(energy_variance_reg) < float(energy_variance_unreg)
                }
            }
            
        except Exception as e:
            raise OracleError(f"Comparison analysis failed: {e}")
    
    def plot_regularization_comparison(self, adjacency_matrix: np.ndarray, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot comparison between regularized and unregularized optimization.
        
        Args:
            adjacency_matrix: The adjacency matrix to optimize
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        # Get comparison data
        comparison = self.compare_with_unregularized(adjacency_matrix)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot energy distributions
        unreg_energies = comparison["unregularized"]["final_energies"]
        reg_energies = comparison["regularized"]["final_energies"]
        
        ax1.hist(unreg_energies, bins=10, alpha=0.7, label='Unregularized', color='blue')
        ax1.hist(reg_energies, bins=10, alpha=0.7, label='Regularized', color='red')
        ax1.set_xlabel('Final Energy')
        ax1.set_ylabel('Count')
        ax1.set_title('Final Energy Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot energy variance comparison
        categories = ['Unregularized', 'Regularized']
        variances = [comparison["unregularized"]["energy_variance"], 
                    comparison["regularized"]["energy_variance"]]
        colors = ['blue', 'red']
        
        ax2.bar(categories, variances, color=colors, alpha=0.7)
        ax2.set_ylabel('Energy Variance')
        ax2.set_title('Energy Variance Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot solution sparsity
        sparsities = [comparison["unregularized"]["solution_sparsity"], 
                     comparison["regularized"]["solution_sparsity"]]
        
        ax3.bar(categories, sparsities, color=colors, alpha=0.7)
        ax3.set_ylabel('Solution Sparsity (# non-zero elements)')
        ax3.set_title('Solution Sparsity Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Plot solution vectors if available
        unreg_sol = comparison["unregularized"]["solution"]
        reg_sol = comparison["regularized"]["solution"]
        
        if unreg_sol and reg_sol:
            x_indices = range(len(unreg_sol))
            width = 0.35
            
            ax4.bar([x - width/2 for x in x_indices], unreg_sol, width, 
                   label='Unregularized', alpha=0.7, color='blue')
            ax4.bar([x + width/2 for x in x_indices], reg_sol, width,
                   label='Regularized', alpha=0.7, color='red')
            
            ax4.set_xlabel('Variable Index')
            ax4.set_ylabel('Solution Value')
            ax4.set_title('Solution Vectors')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Solution vectors not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Solution Vectors')
        
        # Add overall statistics as text
        reg_name = comparison["regularization"]
        value_diff = comparison["comparison"]["value_difference"]
        variance_reduction = comparison["comparison"]["energy_variance_reduction"]
        
        fig.suptitle(f'Regularization Analysis: {reg_name}\n'
                    f'Value difference: {value_diff:.6f}, '
                    f'Variance reduction: {variance_reduction:.6f}', 
                    fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def create_regularized_jax_pgd_oracle(
    c: float = 0.5,
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    num_restarts: int = 10,
    verbose: bool = False
) -> RegularizedJAXPGDOracle:
    """
    Convenience function to create a regularized JAX PGD oracle with identity regularization.
    
    Args:
        c: Regularization parameter for identity regularization (default: 0.5)
        learning_rate: Step size for gradient descent (default: 0.01)
        max_iterations: Maximum optimization iterations (default: 2000)
        tolerance: Convergence tolerance (default: 1e-6)
        num_restarts: Number of random restarts (default: 10)
        verbose: Whether to print optimization progress (default: False)
        
    Returns:
        Configured RegularizedJAXPGDOracle instance
    """
    regularization_function = IdentityRegularization(c=c)
    return RegularizedJAXPGDOracle(
        regularization_function=regularization_function,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        num_restarts=num_restarts,
        verbose=verbose
    )