"""
Hybrid Dirac/NetworkX exact solver that automatically chooses the best approach based on graph size.
"""

import numpy as np
import networkx as nx
from typing import Optional
from .base import Oracle
from ..exceptions import OracleError, SolverUnavailableError


class DiracNetworkXHybridOracle(Oracle):
    """
    Hybrid oracle that automatically selects between NetworkX exact and Dirac-3 based on graph size.
    
    For graphs with ≤35 nodes: Uses NetworkX exact algorithm (fast, deterministic)
    For graphs with >35 nodes: Uses Dirac-3 continuous cloud solver (handles larger graphs)
    
    This provides the best of both worlds: speed for small graphs and scalability for large graphs.
    
    Args:
        threshold_nodes: Node count threshold for switching solvers (default: 35).
        num_samples: Number of samples for Dirac solver (default: 10).
        relax_schedule: Relaxation schedule for Dirac solver (default: 2).
        solution_precision: Solution precision for Dirac solver (default: 0.001).
        sum_constraint: Constraint for solution variables sum (default: 1).
        mean_photon_number: Optional mean photon number override (default: None).
        quantum_fluctuation_coefficient: Optional quantum fluctuation coefficient override (default: None).
    """
    
    def __init__(self, threshold_nodes: int = 35, num_samples: int = 10, 
                 relax_schedule: int = 2, solution_precision: float = 0.001,
                 sum_constraint: int = 1, mean_photon_number: Optional[float] = None,
                 quantum_fluctuation_coefficient: Optional[int] = None):
        super().__init__()
        
        self.threshold_nodes = threshold_nodes
        self.num_samples = num_samples
        self.relax_schedule = relax_schedule
        self.solution_precision = solution_precision
        self.sum_constraint = sum_constraint
        self.mean_photon_number = mean_photon_number
        self.quantum_fluctuation_coefficient = quantum_fluctuation_coefficient
        
        # Import and check availability of Dirac solver
        try:
            from .dirac import DiracOracle, DIRAC_AVAILABLE
            self.dirac_available = DIRAC_AVAILABLE
            self.DiracOracle = DiracOracle
        except ImportError:
            self.dirac_available = False
            self.DiracOracle = None
            
        # NetworkX is always available
        self.networkx_available = True
    
    @property
    def name(self) -> str:
        return f"Hybrid(NX≤{self.threshold_nodes}, Dirac>{self.threshold_nodes})"
    
    @property
    def is_available(self) -> bool:
        # Always available since NetworkX is always available for small graphs
        return True
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using the appropriate solver.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If both solvers fail or are unavailable.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return 0.0
        
        if n <= self.threshold_nodes:
            # Use NetworkX exact for small graphs
            if self.verbose_oracle_calls:
                print(f"Using NetworkX exact solver for {n}-node graph (≤{self.threshold_nodes} threshold)")
            return self._solve_with_networkx_exact(adjacency_matrix)
        else:
            # Use Dirac solver for large graphs
            if not self.dirac_available:
                raise OracleError(
                    f"Graph has {n} nodes (>{self.threshold_nodes}), requiring Dirac solver, "
                    "but Dirac is not available. Install qci-client and eqc-models."
                )
            
            if self.verbose_oracle_calls:
                print(f"Using Dirac-3 cloud solver for {n}-node graph (>{self.threshold_nodes} threshold)")
            return self._solve_with_dirac(adjacency_matrix)
    
    def _solve_with_networkx_exact(self, adjacency_matrix: np.ndarray) -> float:
        """Solve using NetworkX exact algorithm by finding the clique number."""
        try:
            # The oracle's job is to find the clique number of the graph it is given (G).
            # The calling algorithm is responsible for passing the complement if it needs an MIS size.
            # This method should NOT take a complement itself.
            G = nx.from_numpy_array(adjacency_matrix)
            
            # Handle edge cases for omega calculation
            if G.number_of_nodes() == 0:
                omega = 0
            elif G.number_of_edges() == 0:
                # A graph with no edges has a max clique size of 1 (a single node)
                omega = 1
            else:
                # Find the maximum clique in the graph G
                max_clique = max(nx.find_cliques(G), key=len, default=[])
                omega = len(max_clique)

            # If the graph is empty or has omega=0, the M-S formula is undefined. Return 0.
            if omega == 0:
                return 0.0
            
            # Calculate Motzkin-Straus optimal value: 0.5 * (1 - 1/ω)
            optimal_value = 0.5 * (1.0 - 1.0 / omega)
            
            if self.verbose_oracle_calls:
                print(f"NetworkX exact found ω = {omega}, optimal value = {optimal_value:.6f}")
            
            return float(optimal_value)
            
        except Exception as e:
            raise OracleError(f"NetworkX exact solver failed: {str(e)}")
    
    def _solve_with_dirac(self, adjacency_matrix: np.ndarray) -> float:
        """Solve using Dirac-3 continuous cloud solver."""
        try:
            # Create Dirac oracle with configured parameters
            dirac_oracle = self.DiracOracle(
                num_samples=self.num_samples,
                relax_schedule=self.relax_schedule,
                solution_precision=self.solution_precision,
                sum_constraint=self.sum_constraint,
                mean_photon_number=self.mean_photon_number,
                quantum_fluctuation_coefficient=self.quantum_fluctuation_coefficient
            )
            
            # Use Dirac solver
            result = dirac_oracle.solve_quadratic_program(adjacency_matrix)
            
            if self.verbose_oracle_calls:
                print(f"Dirac-3 solver returned optimal value = {result:.6f}")
            
            return result
            
        except Exception as e:
            raise OracleError(f"Dirac-3 solver failed: {str(e)}")
    
    def solve_mis(self, graph: nx.Graph) -> set:
        """
        Constructively finds the Maximum Independent Set.
        
        This method directly returns the MIS node set when possible, avoiding
        the inefficient search-to-decision wrapper for exact solvers.
        
        Args:
            graph: The NetworkX graph to solve.
            
        Returns:
            Set of node IDs representing the maximum independent set.
            
        Raises:
            NotImplementedError: If the graph is too large and requires Dirac solver
                                (which is not constructive).
        """
        n = graph.number_of_nodes()
        
        if n == 0:
            return set()
        
        if n <= self.threshold_nodes:
            # Use NetworkX exact for small graphs - this is constructive
            if self.verbose_oracle_calls:
                print(f"Using NetworkX exact constructive solver for {n}-node graph (≤{self.threshold_nodes} threshold)")
            
            self.call_count = 1  # This is one complete operation
            
            # Get complement graph
            G_complement = nx.complement(graph)
            
            # Find maximum clique in complement (= maximum independent set in original)
            if G_complement.number_of_edges() == 0:
                # Complement has no edges, so each node is an independent clique of size 1
                # Return any single node as the MIS
                return {list(graph.nodes())[0]} if graph.number_of_nodes() > 0 else set()
            else:
                # Find maximum clique in complement
                cliques = list(nx.find_cliques(G_complement))
                if not cliques:
                    return set()
                
                max_clique = max(cliques, key=len)
                return set(max_clique)
        else:
            # The Dirac-3 part is not constructive, so we raise an error
            # The calling code will catch this and fall back to the search wrapper
            raise NotImplementedError(
                f"Graph has {n} nodes (>{self.threshold_nodes}), requiring Dirac solver, "
                "which is not a constructive solver. Use search-to-decision wrapper instead."
            )
    
    def get_solver_info(self, graph_size: int) -> dict:
        """Get information about which solver would be used for a given graph size."""
        if graph_size <= self.threshold_nodes:
            return {
                'solver': 'NetworkX exact',
                'reason': f'Graph size {graph_size} ≤ threshold {self.threshold_nodes}',
                'available': self.networkx_available,
                'expected_performance': 'Fast, deterministic'
            }
        else:
            return {
                'solver': 'Dirac-3 cloud',
                'reason': f'Graph size {graph_size} > threshold {self.threshold_nodes}',
                'available': self.dirac_available,
                'expected_performance': 'Slower due to cloud API, but scalable',
                'parameters': {
                    'num_samples': self.num_samples,
                    'relax_schedule': self.relax_schedule,
                    'solution_precision': self.solution_precision,
                    'sum_constraint': self.sum_constraint,
                    'mean_photon_number': self.mean_photon_number,
                    'quantum_fluctuation_coefficient': self.quantum_fluctuation_coefficient
                }
            }