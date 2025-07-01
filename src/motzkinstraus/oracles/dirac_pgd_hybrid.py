"""
Hybrid Dirac/PGD oracle that combines Dirac's global search with JAX PGD's precision refinement.

This oracle provides the best of both worlds:
- Dirac-3 continuous cloud solver for good initial solutions
- JAX Projected Gradient Descent for high-precision refinement
- NetworkX exact solver for small graphs
"""

import numpy as np
import networkx as nx
import logging
from typing import Optional
from .base import Oracle
from .dirac import DiracOracle, DIRAC_AVAILABLE
from .jax_pgd import ProjectedGradientDescentOracle
from ..exceptions import OracleError, SolverUnavailableError

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class DiracPGDHybridOracle(Oracle):
    """
    Hybrid oracle that combines Dirac-3 global search with JAX PGD precision refinement.
    
    Algorithm:
    1. For graphs ≤ nx_threshold nodes: Use NetworkX exact algorithm
    2. For graphs > nx_threshold nodes:
       a. Run Dirac-3 to get initial solution vector
       b. Initialize JAX PGD from Dirac solution
       c. Run high-precision PGD refinement
       d. Fallback to PGD multi-restart if Dirac fails
    
    Args:
        nx_threshold: Node count threshold for NetworkX vs. hybrid approach (default: 35).
        dirac_num_samples: Number of samples for Dirac solver (default: 100).
        dirac_relax_schedule: Relaxation schedule for Dirac solver (default: 2).
        dirac_solution_precision: Solution precision for Dirac solver (default: 0.001).
        dirac_sum_constraint: Constraint for solution variables sum in Dirac solver (default: 1).
        dirac_mean_photon_number: Optional mean photon number override for Dirac solver (default: None).
        dirac_quantum_fluctuation_coefficient: Optional quantum fluctuation coefficient override for Dirac solver (default: None).
        pgd_tolerance: High-precision tolerance for PGD refinement (default: 1e-7).
        pgd_max_iterations: Maximum iterations for PGD refinement (default: 5000).
        pgd_learning_rate: Learning rate for PGD refinement (default: 0.01).
        fallback_num_restarts: Number of restarts for PGD fallback mode (default: 20).
        verbose: Whether to print detailed progress information (default: False).
        save_raw_data: Whether to save the raw response from Dirac solver (default: False).
        raw_data_path: Directory path to save raw data files (default: 'data/' in project root).
    """
    
    def __init__(
        self,
        nx_threshold: int = 35,
        dirac_num_samples: int = 100,
        dirac_relax_schedule: int = 2,
        dirac_solution_precision: float = 0.001,
        dirac_sum_constraint: int = 1,
        dirac_mean_photon_number: Optional[float] = None,
        dirac_quantum_fluctuation_coefficient: Optional[int] = None,
        pgd_tolerance: float = 1e-7,
        pgd_max_iterations: int = 5000,
        pgd_learning_rate: float = 0.01,
        fallback_num_restarts: int = 20,
        verbose: bool = False,
        save_raw_data: bool = False,
        raw_data_path: Optional[str] = None
    ):
        super().__init__()
        
        self.nx_threshold = nx_threshold
        self.verbose = verbose
        
        # Check availability of required solvers
        if not JAX_AVAILABLE:
            raise SolverUnavailableError(
                "JAX is not available. Please install with: pip install jax jaxlib"
            )
        
        # Dirac oracle for initial solution (may not be available)
        self.dirac_available = DIRAC_AVAILABLE
        if self.dirac_available:
            try:
                self.dirac_oracle = DiracOracle(
                    num_samples=dirac_num_samples,
                    relax_schedule=dirac_relax_schedule,
                    solution_precision=dirac_solution_precision,
                    sum_constraint=dirac_sum_constraint,
                    mean_photon_number=dirac_mean_photon_number,
                    quantum_fluctuation_coefficient=dirac_quantum_fluctuation_coefficient,
                    save_raw_data=save_raw_data,
                    raw_data_path=raw_data_path
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Dirac oracle initialization failed: {e}")
                self.dirac_available = False
                self.dirac_oracle = None
        else:
            self.dirac_oracle = None
        
        # PGD oracle for refinement (single-start mode)
        from ..jax_optimizers import JAXOptimizerConfig
        self.pgd_config_refinement = JAXOptimizerConfig(
            learning_rate=pgd_learning_rate,
            max_iterations=pgd_max_iterations,
            tolerance=pgd_tolerance,
            min_iterations=10,
            num_restarts=1,  # Single run for refinement
            verbose=verbose
        )
        
        self.pgd_oracle_refinement = ProjectedGradientDescentOracle(
            learning_rate=pgd_learning_rate,
            max_iterations=pgd_max_iterations,
            tolerance=pgd_tolerance,
            num_restarts=1,
            verbose=verbose
        )
        
        # PGD oracle for fallback (multi-restart mode)
        self.pgd_config_fallback = JAXOptimizerConfig(
            learning_rate=pgd_learning_rate,
            max_iterations=pgd_max_iterations,
            tolerance=pgd_tolerance,
            min_iterations=50,
            num_restarts=fallback_num_restarts,
            verbose=verbose
        )
        
        self.pgd_oracle_fallback = ProjectedGradientDescentOracle(
            learning_rate=pgd_learning_rate,
            max_iterations=pgd_max_iterations,
            tolerance=pgd_tolerance,
            num_restarts=fallback_num_restarts,
            verbose=verbose
        )
        
        # Store information about the last solve
        self.last_solve_info = {}
    
    @property
    def name(self) -> str:
        dirac_status = "✓" if self.dirac_available else "✗"
        return f"Hybrid(NX≤{self.nx_threshold}, Dirac{dirac_status}+PGD>{self.nx_threshold})"
    
    @property
    def is_available(self) -> bool:
        # Always available since JAX PGD can handle all cases
        return JAX_AVAILABLE
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using the hybrid approach.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If all solvers fail.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            self.last_solve_info = {"method": "trivial", "n": 0}
            return 0.0
        
        # Choose solver based on graph size
        if n <= self.nx_threshold:
            return self._solve_with_networkx_exact(adjacency_matrix)
        else:
            return self._solve_with_hybrid_approach(adjacency_matrix)
    
    def _solve_with_networkx_exact(self, adjacency_matrix: np.ndarray) -> float:
        """Solve using NetworkX exact algorithm for small graphs."""
        n = adjacency_matrix.shape[0]
        
        if self.verbose:
            print(f"Using NetworkX exact solver for {n}-node graph (≤{self.nx_threshold} threshold)")
        
        try:
            # Convert to NetworkX graph and find maximum clique
            G = nx.from_numpy_array(adjacency_matrix)
            
            if G.number_of_edges() == 0:
                # Graph with no edges has omega = 1
                omega = 1
            else:
                # Find the maximum clique size
                max_clique = max(nx.find_cliques(G), key=len, default=[])
                omega = len(max_clique)
            
            if omega == 0:
                optimal_value = 0.0
            else:
                # Calculate Motzkin-Straus optimal value: 0.5 * (1 - 1/ω)
                optimal_value = 0.5 * (1.0 - 1.0 / omega)
            
            self.last_solve_info = {
                "method": "networkx_exact",
                "n": n,
                "omega": omega,
                "optimal_value": optimal_value
            }
            
            if self.verbose:
                print(f"NetworkX exact: ω = {omega}, optimal value = {optimal_value:.8f}")
            
            return float(optimal_value)
            
        except Exception as e:
            raise OracleError(f"NetworkX exact solver failed: {str(e)}")
    
    def _solve_with_hybrid_approach(self, adjacency_matrix: np.ndarray) -> float:
        """Solve using hybrid Dirac + PGD approach for larger graphs."""
        n = adjacency_matrix.shape[0]
        
        if self.verbose:
            print(f"Using hybrid Dirac+PGD approach for {n}-node graph (>{self.nx_threshold} threshold)")
        
        dirac_solution = None
        dirac_objective = None
        
        # Phase 1: Try Dirac solver for initial solution
        if self.dirac_available:
            try:
                if self.verbose:
                    print("Phase 1: Running Dirac-3 oracle for initial solution...")
                
                dirac_objective, dirac_solution = self.dirac_oracle._solve_and_get_vector(adjacency_matrix)
                
                if self.verbose:
                    print(f"Dirac solver: objective = {dirac_objective:.8f}")
                    print(f"Dirac solution: sum = {np.sum(dirac_solution):.6f}, sparsity = {np.sum(dirac_solution > 1e-6)}/{n}")
                
            except Exception as e:
                if self.verbose:
                    print(f"Dirac solver failed: {e}")
                dirac_solution = None
                dirac_objective = None
        
        # Phase 2: PGD refinement or fallback
        if dirac_solution is not None:
            # PGD refinement from Dirac solution
            try:
                if self.verbose:
                    print(f"Phase 2: Running PGD refinement from Dirac solution (tolerance={self.pgd_config_refinement.tolerance})")
                
                pgd_objective = self.pgd_oracle_refinement.solve_quadratic_program(
                    adjacency_matrix, 
                    x0=dirac_solution
                )
                
                self.last_solve_info = {
                    "method": "dirac_pgd_refinement",
                    "n": n,
                    "dirac_objective": dirac_objective,
                    "pgd_objective": pgd_objective,
                    "improvement": pgd_objective - dirac_objective
                }
                
                if self.verbose:
                    improvement = pgd_objective - dirac_objective
                    print(f"PGD refinement: objective = {pgd_objective:.8f} (improvement: {improvement:+.8f})")
                
                return float(pgd_objective)
                
            except Exception as e:
                if self.verbose:
                    print(f"PGD refinement failed: {e}")
                    print("Returning Dirac solution as fallback")
                
                self.last_solve_info = {
                    "method": "dirac_only",
                    "n": n,
                    "dirac_objective": dirac_objective,
                    "pgd_failed": True,
                    "error": str(e)
                }
                
                return float(dirac_objective)
        
        else:
            # PGD multi-restart fallback (Dirac failed or unavailable)
            try:
                if self.verbose:
                    dirac_status = "unavailable" if not self.dirac_available else "failed"
                    print(f"Phase 2: Dirac {dirac_status}, using PGD multi-restart fallback...")
                
                pgd_objective = self.pgd_oracle_fallback.solve_quadratic_program(adjacency_matrix)
                
                self.last_solve_info = {
                    "method": "pgd_fallback",
                    "n": n,
                    "dirac_available": self.dirac_available,
                    "pgd_objective": pgd_objective
                }
                
                if self.verbose:
                    print(f"PGD fallback: objective = {pgd_objective:.8f}")
                
                return float(pgd_objective)
                
            except Exception as e:
                raise OracleError(f"All solvers failed. Final PGD fallback error: {str(e)}")
    
    def get_solver_info(self, graph_size: int) -> dict:
        """Get information about which solver would be used for a given graph size."""
        if graph_size <= self.nx_threshold:
            return {
                'solver': 'NetworkX exact',
                'reason': f'Graph size {graph_size} ≤ threshold {self.nx_threshold}',
                'available': True,
                'expected_performance': 'Fast, deterministic'
            }
        else:
            solver_name = "Dirac+PGD hybrid" if self.dirac_available else "PGD only"
            return {
                'solver': solver_name,
                'reason': f'Graph size {graph_size} > threshold {self.nx_threshold}',
                'available': True,
                'dirac_available': self.dirac_available,
                'expected_performance': 'High precision with global + local optimization',
                'pgd_tolerance': self.pgd_config_refinement.tolerance
            }
    
    def get_last_solve_info(self) -> dict:
        """Get detailed information about the last solve operation."""
        return self.last_solve_info.copy()