"""
JAX-based Projected Gradient Descent oracle for the Motzkin-Straus quadratic program.
"""

import numpy as np
from typing import List, Optional
from .base import Oracle
from ..exceptions import OracleError, SolverUnavailableError
from ..jax_optimizers import (
    JAXOptimizerConfig, 
    adjacency_to_polynomial, 
    run_multi_restart_optimization,
    JAX_AVAILABLE
)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


class ProjectedGradientDescentOracle(Oracle):
    """
    Oracle implementation using JAX-based Projected Gradient Descent.
    
    This oracle uses gradient-based optimization with simplex projection
    to solve the Motzkin-Straus quadratic program:
    max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
    
    Features:
    - Multi-restart strategy with Dirichlet initializations
    - Early stopping based on energy tolerance
    - Complete convergence history tracking
    - JIT-compiled optimization for performance
    
    Args:
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
        learning_rate: float = 0.01,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
        min_iterations: int = 50,
        num_restarts: int = 10,
        dirichlet_alpha: float = 1.0,
        verbose: bool = False
    ):
        super().__init__()
        if not self.is_available:
            raise SolverUnavailableError(
                "JAX is not available. Please install with: pip install jax jaxlib"
            )
        
        self.config = JAXOptimizerConfig(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            min_iterations=min_iterations,
            num_restarts=num_restarts,
            dirichlet_alpha=dirichlet_alpha,
            verbose=verbose
        )
        
        # Enable verbose oracle call tracking if verbose is True
        self.verbose_oracle_calls = verbose
        
        # Store optimization history for debugging
        self.last_histories: List[jnp.ndarray] = []
        self.last_final_energies: List[float] = []
        self.last_best_restart_idx: int = -1
    
    @property
    def name(self) -> str:
        return f"JAX-PGD(lr={self.config.learning_rate},restarts={self.config.num_restarts})"
    
    @property
    def is_available(self) -> bool:
        return JAX_AVAILABLE
    
    def get_omega(self, graph) -> int:
        """
        Override to ensure optimization details are populated even for trivial cases.
        """
        import networkx as nx
        
        n = graph.number_of_nodes()
        
        # Handle trivial cases but still populate optimization details
        if n == 0:
            # Increment call counter for this trivial case
            self.call_count += 1
            self.last_histories = []
            self.last_final_energies = []
            self.last_best_restart_idx = 0
            return 0
        if graph.number_of_edges() == 0:
            # Increment call counter for this trivial case
            self.call_count += 1
            # Graph with no edges - populate dummy optimization details
            dummy_history = jnp.array([0.0])
            self.last_histories = [dummy_history] * self.config.num_restarts
            self.last_final_energies = [0.0] * self.config.num_restarts
            self.last_best_restart_idx = 0
            return 1
        
        # For non-trivial cases, use the parent implementation (which handles call counting)
        return super().get_omega(graph)
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray, x0: Optional[np.ndarray] = None, key: Optional[jax.random.PRNGKey] = None) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Projected Gradient Descent.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            x0: Optional initial point. If provided, runs single-start refinement mode.
                If None, runs multi-restart discovery mode.
            key: Optional JAX random key for reproducible randomness. If None, uses default seed.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If the optimization fails.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            # Handle empty graph case
            self.last_histories = []
            self.last_final_energies = []
            self.last_best_restart_idx = 0
            self.x_current = jnp.array([])  # Empty solution for empty graph
            return 0.0
        if np.sum(adjacency_matrix) == 0:  # No edges
            # Handle graph with no edges - populate dummy optimization details
            dummy_history = jnp.array([0.0])
            self.last_histories = [dummy_history] * self.config.num_restarts
            self.last_final_energies = [0.0] * self.config.num_restarts
            self.last_best_restart_idx = 0
            self.x_current = jnp.ones(n) / n  # Uniform solution for graph with no edges
            return 0.0
        
        try:
            # Convert adjacency matrix to polynomial format
            poly_indices, poly_coefficients = adjacency_to_polynomial(adjacency_matrix)
            
            if len(poly_indices) == 0:
                # Handle case with no polynomial terms - populate dummy optimization details
                dummy_history = jnp.array([0.0])
                self.last_histories = [dummy_history] * self.config.num_restarts
                self.last_final_energies = [0.0] * self.config.num_restarts
                self.last_best_restart_idx = 0
                self.x_current = jnp.ones(n) / n  # Uniform solution for case with no terms
                return 0.0
            
            if x0 is not None:
                # Single-start refinement mode
                if self.config.verbose:
                    print(f"PGD Oracle: Running single-start refinement from provided x0")
                
                # Convert x0 to JAX array
                x0_jax = jnp.array(x0, dtype=jnp.float32)
                
                # Import the function locally to avoid import issues
                from ..jax_optimizers import run_projected_gradient_descent
                
                # Run single PGD optimization from x0
                best_x, energy_history = run_projected_gradient_descent(
                    poly_indices=poly_indices,
                    poly_coefficients=poly_coefficients,
                    num_vars=n,
                    config=self.config,
                    x_init=x0_jax,
                    seed=42
                )
                
                best_energy = float(energy_history[-1])
                
                # Store results for debugging (single run)
                self.last_histories = [energy_history]
                self.last_final_energies = [best_energy]
                self.last_best_restart_idx = 0
                self.x_current = best_x  # Store final solution for experiment access
                
            else:
                # Multi-restart discovery mode
                if self.config.verbose:
                    print(f"PGD Oracle: Running multi-restart discovery mode")
                
                best_x, best_energy, all_histories, all_final_energies = run_multi_restart_optimization(
                    poly_indices=poly_indices,
                    poly_coefficients=poly_coefficients,
                    num_vars=n,
                    config=self.config,
                    algorithm="pgd",
                    seed=42,  # Fixed seed for reproducibility (used if key is None)
                    key=key   # Pass the provided key for true randomization
                )
                
                # Store results for debugging
                self.last_histories = all_histories
                self.last_final_energies = all_final_energies
                self.last_best_restart_idx = all_final_energies.index(max(all_final_energies))
                self.x_current = best_x  # Store final solution for experiment access
            
            if self.config.verbose:
                print(f"PGD Oracle: Best energy = {best_energy:.6f} from restart {self.last_best_restart_idx}")
                print(f"PGD Oracle: Final solution sum = {float(jnp.sum(best_x)):.6f}")
                print(f"PGD Oracle: Solution sparsity = {float(jnp.sum(best_x > 1e-6))}/{n}")
            
            return float(best_energy)
            
        except Exception as e:
            raise OracleError(f"JAX PGD optimization failed: {e}")
    
    def get_optimization_details(self) -> dict:
        """
        Get detailed information about the last optimization run.
        
        Returns:
            Dictionary with optimization statistics and histories.
        """
        if not self.last_histories:
            return {"message": "No optimization run yet"}
        
        return {
            "num_restarts": len(self.last_histories),
            "best_restart_idx": self.last_best_restart_idx,
            "best_energy": max(self.last_final_energies),
            "worst_energy": min(self.last_final_energies),
            "energy_std": float(np.std(self.last_final_energies)),
            "energy_range": max(self.last_final_energies) - min(self.last_final_energies),
            "convergence_iterations": [len(hist) for hist in self.last_histories],
            "config": {
                "learning_rate": self.config.learning_rate,
                "max_iterations": self.config.max_iterations,
                "tolerance": self.config.tolerance,
                "num_restarts": self.config.num_restarts,
                "dirichlet_alpha": self.config.dirichlet_alpha
            }
        }
    
    def get_convergence_histories(self) -> List[np.ndarray]:
        """
        Get convergence histories from the last optimization run.
        
        Returns:
            List of energy histories, one per restart.
        """
        return [np.array(hist) for hist in self.last_histories]
    
    def plot_convergence_analysis(self, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Plot convergence analysis for the last optimization run.
        
        Args:
            save_path: Path to save the plot (optional).
            show_plot: Whether to display the plot.
        """
        if not self.last_histories:
            print("No optimization history available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot all convergence curves
        for i, history in enumerate(self.last_histories):
            alpha = 0.8 if i == self.last_best_restart_idx else 0.3
            linewidth = 2 if i == self.last_best_restart_idx else 1
            color = 'red' if i == self.last_best_restart_idx else 'blue'
            
            ax1.plot(history, alpha=alpha, linewidth=linewidth, color=color,
                    label='Best' if i == self.last_best_restart_idx else None)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.set_title(f'PGD Convergence Curves ({len(self.last_histories)} restarts)')
        ax1.grid(True, alpha=0.3)
        if self.last_best_restart_idx >= 0:
            ax1.legend()
        
        # Plot final energy distribution
        ax2.hist(self.last_final_energies, bins=min(10, len(self.last_final_energies)), 
                alpha=0.7, edgecolor='black')
        ax2.axvline(max(self.last_final_energies), color='red', linestyle='--', 
                   label=f'Best: {max(self.last_final_energies):.6f}')
        ax2.set_xlabel('Final Energy')
        ax2.set_ylabel('Count')
        ax2.set_title('Final Energy Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()