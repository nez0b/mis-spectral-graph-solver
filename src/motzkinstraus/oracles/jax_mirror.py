"""
JAX-based Mirror Descent oracle for the Motzkin-Straus quadratic program.
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
    import jax.numpy as jnp
except ImportError:
    jnp = None


class MirrorDescentOracle(Oracle):
    """
    Oracle implementation using JAX-based Mirror Descent.
    
    This oracle uses exponentiated gradient updates (Mirror Descent)
    to solve the Motzkin-Straus quadratic program:
    max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
    
    Mirror Descent is particularly well-suited for optimization over the simplex
    as it naturally maintains the simplex constraints through the exponential map.
    
    Features:
    - Multi-restart strategy with Dirichlet initializations
    - Exponentiated gradient updates with numerical stabilization
    - Early stopping based on energy tolerance
    - Complete convergence history tracking
    - JIT-compiled optimization for performance
    
    Args:
        learning_rate: Step size for gradient descent (default: 0.005).
        max_iterations: Maximum optimization iterations (default: 2000).
        tolerance: Convergence tolerance for early stopping (default: 1e-6).
        min_iterations: Minimum iterations before stopping (default: 50).
        num_restarts: Number of Dirichlet initializations (default: 10).
        dirichlet_alpha: Concentration parameter for Dirichlet sampling (default: 1.0).
        verbose: Whether to print optimization progress (default: False).
    """
    
    def __init__(
        self,
        learning_rate: float = 0.005,  # MD typically needs smaller LR
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
        
        # Store optimization history for debugging
        self.last_histories: List[jnp.ndarray] = []
        self.last_final_energies: List[float] = []
        self.last_best_restart_idx: int = -1
    
    @property
    def name(self) -> str:
        return f"JAX-MD(lr={self.config.learning_rate},restarts={self.config.num_restarts})"
    
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
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Mirror Descent.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
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
            return 0.0
        if np.sum(adjacency_matrix) == 0:  # No edges
            # Handle graph with no edges - populate dummy optimization details
            dummy_history = jnp.array([0.0])
            self.last_histories = [dummy_history] * self.config.num_restarts
            self.last_final_energies = [0.0] * self.config.num_restarts
            self.last_best_restart_idx = 0
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
                return 0.0
            
            # Run multi-restart optimization
            best_x, best_energy, all_histories, all_final_energies = run_multi_restart_optimization(
                poly_indices=poly_indices,
                poly_coefficients=poly_coefficients,
                num_vars=n,
                config=self.config,
                algorithm="md",
                seed=42  # Fixed seed for reproducibility
            )
            
            # Store results for debugging
            self.last_histories = all_histories
            self.last_final_energies = all_final_energies
            self.last_best_restart_idx = all_final_energies.index(max(all_final_energies))
            
            if self.config.verbose:
                print(f"MD Oracle: Best energy = {best_energy:.6f} from restart {self.last_best_restart_idx}")
                print(f"MD Oracle: Final solution sum = {float(jnp.sum(best_x)):.6f}")
                print(f"MD Oracle: Solution sparsity = {float(jnp.sum(best_x > 1e-6))}/{n}")
            
            return float(best_energy)
            
        except Exception as e:
            raise OracleError(f"JAX Mirror Descent optimization failed: {e}")
    
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
            color = 'orange' if i == self.last_best_restart_idx else 'blue'
            
            ax1.plot(history, alpha=alpha, linewidth=linewidth, color=color,
                    label='Best' if i == self.last_best_restart_idx else None)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.set_title(f'MD Convergence Curves ({len(self.last_histories)} restarts)')
        ax1.grid(True, alpha=0.3)
        if self.last_best_restart_idx >= 0:
            ax1.legend()
        
        # Plot final energy distribution
        ax2.hist(self.last_final_energies, bins=min(10, len(self.last_final_energies)), 
                alpha=0.7, edgecolor='black', color='orange')
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
    
    def compare_with_oracle(self, other_oracle, graph, save_path: Optional[str] = None):
        """
        Compare convergence behavior with another oracle on the same graph.
        
        Args:
            other_oracle: Another oracle to compare against.
            graph: NetworkX graph to test on.
            save_path: Path to save comparison plot.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Matplotlib or NetworkX not available for comparison plotting")
            return
        
        # Run both oracles
        self_result = self.get_omega(graph)
        other_result = other_oracle.get_omega(graph)
        
        # Get histories if available
        self_histories = self.get_convergence_histories()
        other_histories = getattr(other_oracle, 'get_convergence_histories', lambda: [])()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot convergence curves
        if self_histories:
            for hist in self_histories:
                axes[0, 0].plot(hist, alpha=0.3, color='orange')
            if self_histories:
                axes[0, 0].plot(self_histories[self.last_best_restart_idx], 
                               color='orange', linewidth=2, label=f'{self.name} Best')
        
        if other_histories:
            for hist in other_histories:
                axes[0, 0].plot(hist, alpha=0.3, color='blue')
            if hasattr(other_oracle, 'last_best_restart_idx'):
                axes[0, 0].plot(other_histories[other_oracle.last_best_restart_idx], 
                               color='blue', linewidth=2, label=f'{other_oracle.name} Best')
        
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].set_title('Convergence Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot final energy distributions
        if self_histories:
            axes[0, 1].hist(self.last_final_energies, alpha=0.5, color='orange', 
                           label=f'{self.name}', bins=10)
        if other_histories and hasattr(other_oracle, 'last_final_energies'):
            axes[0, 1].hist(other_oracle.last_final_energies, alpha=0.5, color='blue', 
                           label=f'{other_oracle.name}', bins=10)
        
        axes[0, 1].set_xlabel('Final Energy')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Final Energy Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot graph structure
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, ax=axes[1, 0], with_labels=True, node_color='lightblue',
                node_size=500, font_size=10)
        axes[1, 0].set_title(f'Graph Structure ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)')
        
        # Results summary
        axes[1, 1].text(0.1, 0.8, f'{self.name}:', fontsize=12, fontweight='bold', color='orange')
        axes[1, 1].text(0.1, 0.7, f'  Omega: {self_result}', fontsize=11)
        if self_histories:
            axes[1, 1].text(0.1, 0.6, f'  Best Energy: {max(self.last_final_energies):.6f}', fontsize=11)
            axes[1, 1].text(0.1, 0.5, f'  Energy Std: {np.std(self.last_final_energies):.6f}', fontsize=11)
        
        axes[1, 1].text(0.1, 0.3, f'{other_oracle.name}:', fontsize=12, fontweight='bold', color='blue')
        axes[1, 1].text(0.1, 0.2, f'  Omega: {other_result}', fontsize=11)
        if other_histories and hasattr(other_oracle, 'last_final_energies'):
            axes[1, 1].text(0.1, 0.1, f'  Best Energy: {max(other_oracle.last_final_energies):.6f}', fontsize=11)
            axes[1, 1].text(0.1, 0.0, f'  Energy Std: {np.std(other_oracle.last_final_energies):.6f}', fontsize=11)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Results Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()