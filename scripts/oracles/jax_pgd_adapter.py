"""
JAX-PGD oracle adapter for maximal clique finding.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from .base import OracleAdapter, OracleConfig

# Try to import JAX and motzkinstraus modules
try:
    from motzkinstraus.jax_optimizers import (
        JAXOptimizerConfig, 
        adjacency_to_polynomial, 
        run_projected_gradient_descent,
        sample_dirichlet
    )
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    JAXOptimizerConfig = None
    adjacency_to_polynomial = None
    run_projected_gradient_descent = None
    sample_dirichlet = None


class JAXPGDConfig(OracleConfig):
    """Configuration for JAX Projected Gradient Descent oracle."""
    
    def __init__(
        self,
        num_restarts: int = 50,
        learning_rate: float = 0.01,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
        min_iterations: int = 50,
        dirichlet_alpha: float = 1.0
    ):
        self.num_restarts = num_restarts
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.dirichlet_alpha = dirichlet_alpha
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_restarts': self.num_restarts,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'min_iterations': self.min_iterations,
            'dirichlet_alpha': self.dirichlet_alpha
        }
    
    def validate(self) -> bool:
        if not (1 <= self.num_restarts <= 1000):
            raise ValueError("num_restarts must be between 1 and 1000")
        if not (1e-6 <= self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be between 1e-6 and 1.0")
        if not (10 <= self.max_iterations <= 50000):
            raise ValueError("max_iterations must be between 10 and 50000")
        if not (1e-12 <= self.tolerance <= 1e-3):
            raise ValueError("tolerance must be between 1e-12 and 1e-3")
        if not (1 <= self.min_iterations <= self.max_iterations):
            raise ValueError("min_iterations must be between 1 and max_iterations")
        if not (0.1 <= self.dirichlet_alpha <= 10.0):
            raise ValueError("dirichlet_alpha must be between 0.1 and 10.0")
        return True


class JAXPGDAdapter(OracleAdapter):
    """Oracle adapter for JAX Projected Gradient Descent solver."""
    
    def __init__(self, config: JAXPGDConfig, verbose: bool = False, enable_refinement: bool = True):
        super().__init__(config, verbose, enable_refinement)
        # Store optimization details for analysis
        self.last_histories: List[jnp.ndarray] = []
        self.last_final_energies: List[float] = []
        self.last_best_restart_idx: int = -1
        self.last_solutions: List[np.ndarray] = []
    
    @property
    def name(self) -> str:
        return f"JAX-PGD(restarts={self.config.num_restarts},lr={self.config.learning_rate})"
    
    @property
    def is_available(self) -> bool:
        return JAX_AVAILABLE
    
    def _validate_dependencies(self) -> None:
        if not self.is_available:
            raise ImportError(
                "JAX and motzkinstraus modules are not available. "
                "Make sure you're in the correct virtual environment."
            )
    
    def find_maximal_cliques(
        self, 
        graph: nx.Graph, 
        support_threshold: float = 1e-5
    ) -> List[Set[int]]:
        """
        Find maximal cliques using JAX Projected Gradient Descent.
        
        Args:
            graph: NetworkX graph to analyze
            support_threshold: Threshold for extracting support from solutions
            
        Returns:
            List of sets, each containing vertices of a maximal clique
        """
        if graph.number_of_nodes() == 0:
            return []
        
        if self.verbose:
            print(f"JAX-PGD: Finding maximal cliques in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"JAX-PGD: Using {self.config.num_restarts} restarts with threshold {support_threshold}")
        
        # Set up adjacency matrix and polynomial representation
        node_list = list(graph.nodes())  # Get ordered list for index mapping
        adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)
        
        # Convert to polynomial format for optimization
        poly_indices, poly_coefficients = adjacency_to_polynomial(adj_matrix)
        
        if len(poly_indices) == 0:
            if self.verbose:
                print("JAX-PGD: No polynomial terms - empty graph or no edges")
            return []
        
        # Create configuration for optimization
        jax_config = JAXOptimizerConfig(
            learning_rate=self.config.learning_rate,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
            min_iterations=self.config.min_iterations,
            num_restarts=self.config.num_restarts,
            dirichlet_alpha=self.config.dirichlet_alpha,
            verbose=False  # Disable verbose for individual runs
        )
        
        # Generate the same initializations that run_multi_restart_optimization would use
        key = jax.random.PRNGKey(42)
        alpha = jnp.ones(len(node_list)) * self.config.dirichlet_alpha
        key, subkey = jax.random.split(key)
        initial_states = sample_dirichlet(subkey, alpha, sample_shape=(self.config.num_restarts,))
        
        # Run individual optimizations to get all final solution vectors
        all_final_solutions = []
        all_histories = []
        all_final_energies = []
        
        for i in range(self.config.num_restarts):
            try:
                final_x, energy_history = run_projected_gradient_descent(
                    poly_indices=poly_indices,
                    poly_coefficients=poly_coefficients,
                    num_vars=len(node_list),
                    config=jax_config,
                    x_init=initial_states[i],
                    seed=42 + i
                )
                all_final_solutions.append(np.array(final_x))
                all_histories.append(energy_history)
                all_final_energies.append(float(energy_history[-1]))
            except Exception as e:
                if self.verbose:
                    print(f"JAX-PGD: Warning: Restart {i+1} failed: {e}")
                continue
        
        # Store results for analysis
        self.last_solutions = all_final_solutions
        self.last_histories = all_histories
        self.last_final_energies = all_final_energies
        self.last_best_restart_idx = all_final_energies.index(max(all_final_energies)) if all_final_energies else -1
        
        if self.verbose:
            print(f"JAX-PGD: Successfully collected {len(all_final_solutions)} solution vectors")
        
        # Extract candidate cliques from all restart solutions
        maximal_cliques = set()  # Use set of frozensets for deduplication
        
        for i, solution_vector in enumerate(all_final_solutions):
            if self.verbose:
                print(f"JAX-PGD: Processing solution {i+1}/{len(all_final_solutions)}")
                print(f"  Solution sum: {np.sum(solution_vector):.6f}")
                print(f"  Solution max: {np.max(solution_vector):.6f}")
                print(f"  Non-zero entries: {np.sum(solution_vector > support_threshold)}")
            
            # Extract support (candidate clique vertices)
            support_indices = self.extract_support(solution_vector, support_threshold)
            
            if not support_indices:
                if self.verbose:
                    print(f"  No support found (all values below threshold {support_threshold})")
                continue
            
            # Map indices back to actual node IDs
            candidate_clique = {node_list[idx] for idx in support_indices if idx < len(node_list)}
            
            if self.verbose:
                print(f"  Candidate clique: {sorted(candidate_clique)} (size: {len(candidate_clique)})")
            
            # Verify it's actually a clique
            if self.verify_clique(graph, candidate_clique):
                # Pure solution: check if it's maximal
                if self.verify_maximal_clique(graph, candidate_clique):
                    # Add to results (using frozenset for hashing/deduplication)
                    clique_frozen = frozenset(candidate_clique)
                    if clique_frozen not in maximal_cliques:
                        maximal_cliques.add(clique_frozen)
                        if self.verbose:
                            print(f"  Found maximal clique: {sorted(candidate_clique)}")
                else:
                    if self.verbose:
                        print(f"  Valid clique but not maximal - skipping")
            else:
                # Superposition solution: attempt refinement if enabled
                if self.enable_refinement and len(support_indices) > 1:  # Only refine non-trivial supports
                    if self.verbose:
                        print(f"  Not a valid clique - attempting superposition refinement")
                    
                    try:
                        refined_cliques = self.refine_superposition_solution(
                            graph, solution_vector, support_indices
                        )
                        
                        for refined_clique in refined_cliques:
                            # Verify refined clique is maximal
                            if self.verify_maximal_clique(graph, refined_clique):
                                clique_frozen = frozenset(refined_clique)
                                if clique_frozen not in maximal_cliques:
                                    maximal_cliques.add(clique_frozen)
                                    if self.verbose:
                                        print(f"  Refined to maximal clique: {sorted(refined_clique)}")
                        
                        if self.verbose:
                            print(f"  Refinement yielded {len(refined_cliques)} cliques")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"  Refinement failed: {e}")
                else:
                    if self.verbose:
                        refinement_status = "disabled" if not self.enable_refinement else "trivial support"
                        print(f"  Not a valid clique - skipping refinement ({refinement_status})")
        
        if self.verbose:
            print(f"JAX-PGD: Found {len(maximal_cliques)} unique maximal cliques")
        
        # Convert back to list of sets
        return [set(clique) for clique in maximal_cliques]
    
    def get_optimization_details(self) -> Dict[str, Any]:
        """Get detailed information about the last optimization run."""
        if not self.last_histories:
            return {"message": "No optimization run yet"}
        
        return {
            "oracle_type": "jax-pgd",
            "num_restarts": len(self.last_histories),
            "best_restart_idx": self.last_best_restart_idx,
            "best_energy": max(self.last_final_energies) if self.last_final_energies else 0.0,
            "worst_energy": min(self.last_final_energies) if self.last_final_energies else 0.0,
            "energy_std": float(np.std(self.last_final_energies)) if self.last_final_energies else 0.0,
            "energy_range": (max(self.last_final_energies) - min(self.last_final_energies)) if self.last_final_energies else 0.0,
            "convergence_iterations": [len(hist) for hist in self.last_histories],
            "config": self.config.to_dict(),
            "solutions": len(self.last_solutions)
        }