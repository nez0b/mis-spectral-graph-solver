"""
Dirac-3 oracle adapter for maximal clique finding.
"""

import sys
import os
import json
import time
import tempfile
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import Counter

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from .base import OracleAdapter, OracleConfig

# Try to import QCI client and related functions
try:
    import qci_client as qc
    QCI_AVAILABLE = True
    print("QCI client available for Dirac oracle")
except ImportError:
    QCI_AVAILABLE = False
    qc = None

# Import functions from graph_to_omega.py
try:
    scripts_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to scripts/
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    from graph_to_omega import (
        qplib_to_polynomial_file,
        submit_to_dirac,
        extract_best_energy,
        energy_to_omega
    )
    GRAPH_TO_OMEGA_AVAILABLE = True
except ImportError as e:
    GRAPH_TO_OMEGA_AVAILABLE = False
    qplib_to_polynomial_file = None
    submit_to_dirac = None
    extract_best_energy = None
    energy_to_omega = None


class DiracConfig(OracleConfig):
    """Configuration for Dirac-3 oracle."""
    
    def __init__(
        self,
        num_samples: int = 100,
        relaxation_schedule: int = 2,
        solution_precision: Optional[float] = None,
        sum_constraint: int = 1,
        mean_photon_number: Optional[float] = None,
        quantum_fluctuation_coefficient: Optional[int] = None,
        save_raw_data: bool = False,
        job_timeout: int = 300
    ):
        self.num_samples = num_samples
        self.relaxation_schedule = relaxation_schedule
        self.solution_precision = solution_precision
        self.sum_constraint = sum_constraint
        self.mean_photon_number = mean_photon_number
        self.quantum_fluctuation_coefficient = quantum_fluctuation_coefficient
        self.save_raw_data = save_raw_data
        self.job_timeout = job_timeout
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_samples': self.num_samples,
            'relaxation_schedule': self.relaxation_schedule,
            'solution_precision': self.solution_precision,
            'sum_constraint': self.sum_constraint,
            'mean_photon_number': self.mean_photon_number,
            'quantum_fluctuation_coefficient': self.quantum_fluctuation_coefficient,
            'save_raw_data': self.save_raw_data,
            'job_timeout': self.job_timeout
        }
    
    def validate(self) -> bool:
        if not (1 <= self.num_samples <= 1000):
            raise ValueError("num_samples must be between 1 and 1000")
        if not (1 <= self.relaxation_schedule <= 4):
            raise ValueError("relaxation_schedule must be between 1 and 4")
        if not (1 <= self.sum_constraint <= 10000):
            raise ValueError("sum_constraint must be between 1 and 10000")
        if self.solution_precision is not None and not (1e-6 <= self.solution_precision <= 1.0):
            raise ValueError("solution_precision must be between 1e-6 and 1.0")
        if self.mean_photon_number is not None and not (0.0000667 <= self.mean_photon_number <= 0.0066666):
            raise ValueError("mean_photon_number must be between 0.0000667 and 0.0066666")
        if self.quantum_fluctuation_coefficient is not None and not (1 <= self.quantum_fluctuation_coefficient <= 100):
            raise ValueError("quantum_fluctuation_coefficient must be between 1 and 100")
        if not (30 <= self.job_timeout <= 1800):
            raise ValueError("job_timeout must be between 30 and 1800 seconds")
        return True


class DiracAdapter(OracleAdapter):
    """Oracle adapter for Dirac-3 quantum annealing solver."""
    
    def __init__(self, config: DiracConfig, verbose: bool = False, enable_refinement: bool = True):
        super().__init__(config, verbose, enable_refinement)
        # Store optimization details for analysis
        self.last_job_response: Optional[Dict[str, Any]] = None
        self.last_energies: List[float] = []
        self.last_solutions: List[np.ndarray] = []
        self.last_best_energy: float = 0.0
        self.last_best_solution: Optional[np.ndarray] = None
        self.last_omega: float = 0.0
    
    @property
    def name(self) -> str:
        return f"Dirac-3(samples={self.config.num_samples},schedule={self.config.relaxation_schedule})"
    
    @property
    def is_available(self) -> bool:
        return QCI_AVAILABLE and GRAPH_TO_OMEGA_AVAILABLE
    
    def _validate_dependencies(self) -> None:
        if not QCI_AVAILABLE:
            raise ImportError(
                "qci_client not available. Please install: pip install qci-client"
            )
        if not GRAPH_TO_OMEGA_AVAILABLE:
            raise ImportError(
                "graph_to_omega functions not available. Check graph_to_omega.py import."
            )
        
        # Test connection to QCI
        try:
            client = qc.QciClient()
            allocations = client.get_allocations()
            if "dirac" not in allocations.get("allocations", {}):
                raise RuntimeError(
                    "Dirac solver allocation not available. Check your QCI account."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to QCI services: {e}")
    
    def _graph_to_qplib_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Convert NetworkX graph to QPLIB data format.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            QPLIB data dictionary with poly_indices, poly_coefficients, sum_constraint
        """
        # Get adjacency matrix
        node_list = list(graph.nodes())
        adj_matrix = nx.to_numpy_array(graph, nodelist=node_list)
        n = adj_matrix.shape[0]
        
        poly_indices = []
        poly_coefficients = []
        
        # Add quadratic terms: 0.5 * x^T * A * x = 0.5 * sum_{i,j} A[i,j] * x_i * x_j
        # Process upper triangular part to ensure ascending indices for QCI API compliance
        # Since adjacency matrix is symmetric: A[i,j] = A[j,i], we can process only upper triangle
        for i in range(n):
            for j in range(i, n):  # j starts from i to ensure i <= j
                if adj_matrix[i, j] != 0:
                    if i == j:
                        # Diagonal term: 0.5 * A[i,i] * x_i^2
                        poly_indices.append([i + 1, i + 1])  # 1-based indexing
                        poly_coefficients.append(0.5 * adj_matrix[i, j])
                    else:
                        # Off-diagonal term: A[i,j] * x_i * x_j 
                        # Full coefficient since we're only counting each unique pair once
                        poly_indices.append([i + 1, j + 1])  # Guaranteed i+1 <= j+1
                        poly_coefficients.append(adj_matrix[i, j])
        
        return {
            'poly_indices': poly_indices,
            'poly_coefficients': poly_coefficients,
            'sum_constraint': self.config.sum_constraint
        }
    
    def _extract_cliques_from_dirac_response(
        self, 
        graph: nx.Graph,
        solutions: List[np.ndarray], 
        support_threshold: float = 1e-5
    ) -> List[Set[int]]:
        """
        Extract maximal cliques from Dirac solution vectors.
        
        Args:
            graph: Original graph
            solutions: List of solution vectors from Dirac
            support_threshold: Threshold for support extraction
            
        Returns:
            List of maximal cliques found
        """
        maximal_cliques = set()  # Use set of frozensets for deduplication
        node_list = list(graph.nodes())
        
        for i, solution_vector in enumerate(solutions):
            if self.verbose:
                print(f"Dirac: Processing solution {i+1}/{len(solutions)}")
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
                support_indices = self.extract_support(solution_vector, support_threshold)
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
        
        return [set(clique) for clique in maximal_cliques]
    
    def find_maximal_cliques(
        self, 
        graph: nx.Graph, 
        support_threshold: float = 1e-5
    ) -> List[Set[int]]:
        """
        Find maximal cliques using Dirac-3 quantum annealing.
        
        Args:
            graph: NetworkX graph to analyze
            support_threshold: Threshold for extracting support from solutions
            
        Returns:
            List of sets, each containing vertices of a maximal clique
        """
        if graph.number_of_nodes() == 0:
            return []
        
        if self.verbose:
            print(f"Dirac: Finding maximal cliques in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"Dirac: Using {self.config.num_samples} samples with schedule {self.config.relaxation_schedule}")
        
        # Convert graph to QPLIB format
        qplib_data = self._graph_to_qplib_data(graph)
        
        if not qplib_data['poly_indices']:
            if self.verbose:
                print("Dirac: No polynomial terms - empty graph or no edges")
            return []
        
        # Transform to QCI polynomial file format
        file_name = f"clique_graph_{graph.number_of_nodes()}n_{graph.number_of_edges()}e"
        polynomial_file = qplib_to_polynomial_file(qplib_data, file_name)
        
        # Submit job to Dirac-3
        job_name = f"clique_finder_{int(time.time())}"
        
        try:
            job_response = submit_to_dirac(
                polynomial_file=polynomial_file,
                job_name=job_name,
                num_samples=self.config.num_samples,
                relaxation_schedule=self.config.relaxation_schedule,
                solution_precision=self.config.solution_precision,
                sum_constraint=self.config.sum_constraint,
                wait=True,
                job_tags=['maximal_clique', 'motzkin_straus']
            )
            
            # Store job response for analysis
            self.last_job_response = job_response
            
            # Extract energies and solutions
            best_energy, all_energies, best_solution = extract_best_energy(job_response)
            
            # Store results for analysis
            self.last_best_energy = best_energy
            self.last_energies = all_energies
            self.last_best_solution = best_solution
            self.last_omega = energy_to_omega(best_energy)
            
            # Extract all solution vectors from job response
            results = job_response.get('results', {})
            solutions = results.get('solutions', [])
            self.last_solutions = [np.array(sol) for sol in solutions]
            
            if self.verbose:
                print(f"Dirac: Best energy: {best_energy:.6f}")
                print(f"Dirac: Theoretical omega: {self.last_omega:.3f}")
                print(f"Dirac: Total solutions: {len(solutions)}")
                print(f"Dirac: Energy range: [{min(all_energies):.6f}, {max(all_energies):.6f}]")
            
            # Extract cliques from all solution vectors
            maximal_cliques = self._extract_cliques_from_dirac_response(
                graph, self.last_solutions, support_threshold
            )
            
            if self.verbose:
                print(f"Dirac: Found {len(maximal_cliques)} unique maximal cliques")
            
            return maximal_cliques
            
        except Exception as e:
            if self.verbose:
                print(f"Dirac: Error during optimization: {e}")
            raise RuntimeError(f"Dirac optimization failed: {e}")
    
    def get_optimization_details(self) -> Dict[str, Any]:
        """Get detailed information about the last Dirac optimization run."""
        if not self.last_job_response:
            return {"message": "No Dirac optimization run yet"}
        
        details = {
            "oracle_type": "dirac-3",
            "job_status": self.last_job_response.get('status', 'unknown'),
            "best_energy": self.last_best_energy,
            "theoretical_omega": self.last_omega,
            "num_solutions": len(self.last_solutions),
            "energy_statistics": {
                "min": min(self.last_energies) if self.last_energies else 0.0,
                "max": max(self.last_energies) if self.last_energies else 0.0,
                "mean": np.mean(self.last_energies) if self.last_energies else 0.0,
                "std": np.std(self.last_energies) if self.last_energies else 0.0
            },
            "config": self.config.to_dict()
        }
        
        # Add job-specific information if available
        if 'job_id' in self.last_job_response:
            details['job_id'] = self.last_job_response['job_id']
        
        return details