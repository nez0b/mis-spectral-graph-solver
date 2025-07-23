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
    
    def __init__(self, config: DiracConfig, verbose: bool = False, enable_refinement: bool = False):
        # Note: Refinement is disabled by default as requested - Dirac should find optimal solutions directly
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
    
    def construct_gibbons_matrix(self, graph: nx.Graph, weights: Dict[int, float]) -> Tuple[np.ndarray, List[int]]:
        """
        Construct the Gibbons B matrix for the weighted Motzkin-Straus theorem.
        
        For a weighted MAXIMUM CLIQUE problem on graph G, the Gibbons matrix B is:
        - B[i,i] = 1/w[i] (diagonal elements)  
        - B[i,j] = 0 for ADJACENT vertices (they can both be in the clique)
        - B[i,j] = (1/w[i] + 1/w[j])/2 for NON-ADJACENT vertices (constraint violation)
        
        The optimization problem is: min{x^T B x | e^T x = 1, x ≥ 0}
        
        Args:
            graph: NetworkX graph (original graph for max weight clique)
            weights: Dictionary mapping node IDs to weights
            
        Returns:
            Tuple of (B_matrix, node_list) where B_matrix is the Gibbons matrix
        """
        node_list = list(graph.nodes())
        n = len(node_list)
        
        if n == 0:
            return np.array([]), []
        
        # Initialize B matrix
        B = np.zeros((n, n), dtype=np.float64)
        
        # Set diagonal elements: B[i,i] = 1/w[i]
        for i, node in enumerate(node_list):
            weight = weights.get(node, 1.0)
            if weight <= 0:
                raise ValueError(f"Weight for node {node} must be positive, got {weight}")
            B[i, i] = 1.0 / weight
        
        # Set off-diagonal elements according to Gibbons' Theorem 5 for MAX CLIQUE:
        # B[i,j] = 0 for ADJACENT vertices (both can be selected in clique)
        # B[i,j] = (1/w[i] + 1/w[j])/2 for NON-ADJACENT vertices (constraint)
        
        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                if i != j:
                    if graph.has_edge(node_i, node_j):
                        # Adjacent vertices: both can be in clique, no constraint
                        B[i, j] = 0.0
                    else:
                        # Non-adjacent vertices: cannot both be in clique, add constraint
                        weight_i = weights.get(node_i, 1.0)
                        weight_j = weights.get(node_j, 1.0)
                        B[i, j] = (1.0/weight_i + 1.0/weight_j) / 2.0
        
        return B, node_list

    def _graph_to_qplib_data(self, graph: nx.Graph, weights: Dict[int, float]) -> Dict[str, Any]:
        """
        Convert NetworkX graph to QPLIB data format using Gibbons B matrix.
        
        Args:
            graph: NetworkX graph (original graph for max weight clique)
            weights: Dictionary mapping node IDs to weights
            
        Returns:
            QPLIB data dictionary with poly_indices, poly_coefficients, sum_constraint
        """
        # Construct Gibbons B matrix
        B_matrix, node_list = self.construct_gibbons_matrix(graph, weights)
        n = B_matrix.shape[0]
        
        # Debug: Save B matrix and graph info for inspection
        if self.verbose:
            self._save_matrix_debug_info(B_matrix, node_list, graph, weights)
        
        if n == 0:
            return {
                'poly_indices': [],
                'poly_coefficients': [],
                'sum_constraint': self.config.sum_constraint
            }
        
        poly_indices = []
        poly_coefficients = []
        
        # Add quadratic terms: x^T * B * x = sum_{i,j} B[i,j] * x_i * x_j
        # Process upper triangular part to ensure ascending indices for QCI API compliance
        for i in range(n):
            for j in range(i, n):  # j starts from i to ensure i <= j
                if B_matrix[i, j] != 0:
                    if i == j:
                        # Diagonal term: B[i,i] * x_i^2
                        poly_indices.append([i + 1, i + 1])  # 1-based indexing
                        poly_coefficients.append(B_matrix[i, j])
                    else:
                        # Off-diagonal term: B[i,j] * x_i * x_j + B[j,i] * x_j * x_i = 2 * B[i,j] * x_i * x_j
                        # Since B is symmetric, we double the coefficient for off-diagonal terms
                        poly_indices.append([i + 1, j + 1])  # Guaranteed i+1 <= j+1
                        poly_coefficients.append(2.0 * B_matrix[i, j])
        
        return {
            'poly_indices': poly_indices,
            'poly_coefficients': poly_coefficients,
            'sum_constraint': self.config.sum_constraint
        }
    
    def _save_matrix_debug_info(self, B_matrix: np.ndarray, node_list: List[int], 
                               graph: nx.Graph, weights: Dict[int, float]) -> None:
        """
        Save B matrix and related information for debugging.
        
        Args:
            B_matrix: The Gibbons B matrix
            node_list: List of node IDs
            graph: Original graph
            weights: Vertex weights
        """
        import json
        import time
        from pathlib import Path
        
        try:
            # Create debug directory
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Calculate theoretical optimal for complete graphs
            if graph.number_of_edges() == graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2:
                # Complete graph - optimal clique is all vertices
                total_weight = sum(weights.values())
                theoretical_optimal = 1.0 / total_weight
                graph_type = "complete"
            else:
                theoretical_optimal = "unknown"
                graph_type = "general"
            
            debug_info = {
                "timestamp": timestamp,
                "graph_info": {
                    "nodes": list(graph.nodes()),
                    "edges": list(graph.edges()),
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "graph_type": graph_type
                },
                "weights": weights,
                "node_list": node_list,
                "B_matrix": B_matrix.tolist(),
                "B_matrix_shape": B_matrix.shape,
                "theoretical_optimal": theoretical_optimal,
                "config": self.config.to_dict()
            }
            
            # Save to JSON file
            filename = debug_dir / f"dirac_matrix_debug_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(debug_info, f, indent=2)
            
            print(f"Debug: Saved B matrix debug info to {filename}")
            print(f"Debug: Graph type: {graph_type}, nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
            print(f"Debug: B matrix diagonal: {np.diag(B_matrix)}")
            if theoretical_optimal != "unknown":
                print(f"Debug: Theoretical optimal objective: {theoretical_optimal:.6f}")
            
        except Exception as e:
            print(f"Warning: Failed to save matrix debug info: {e}")
    
    def _validate_energy_values(self, best_energy: float, all_energies: List[float], 
                               graph: nx.Graph, weights: Dict[int, float]) -> None:
        """
        Validate energy values against theoretical expectations.
        
        Args:
            best_energy: Best (lowest) energy from Dirac solver
            all_energies: All energy values from solutions
            graph: Original graph
            weights: Vertex weights
        """
        if self.verbose:
            print(f"Energy Validation:")
            print(f"  Best energy: {best_energy:.8f}")
            print(f"  Energy std dev: {np.std(all_energies):.8f}")
            
            # For complete graphs, we can calculate the theoretical optimum
            if graph.number_of_edges() == graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2:
                total_weight = sum(weights.values())
                theoretical_optimal = 1.0 / total_weight
                
                print(f"  Theoretical optimal (complete graph): {theoretical_optimal:.8f}")
                print(f"  Energy vs theoretical: {abs(best_energy - theoretical_optimal):.8f}")
                
                # Check if energy is close to theoretical
                if abs(best_energy - theoretical_optimal) < 0.01:
                    print(f"  PASS: Energy matches theoretical expectation!")
                else:
                    print(f"  WARNING: Energy differs significantly from theoretical!")
                    
            else:
                print(f"  Note: General graph - no theoretical optimum calculated")
            
            # Check for negative energies (could indicate coefficient negation issues)
            if best_energy < 0:
                print(f"  WARNING: Negative energy detected - possible coefficient negation issue!")
            
            # Check energy range
            energy_range = max(all_energies) - min(all_energies)
            print(f"  Energy range: {energy_range:.8f}")
    
    def _print_detailed_support_analysis(
        self, 
        solution_vector: np.ndarray, 
        node_list: List[int], 
        support_threshold: float,
        weights: Optional[Dict[int, float]] = None,
        solution_index: int = 0
    ) -> None:
        """
        Print detailed analysis of solution vector support values for threshold determination.
        
        Args:
            solution_vector: Solution vector from Dirac
            node_list: List of node IDs corresponding to vector indices
            support_threshold: Current threshold for support extraction
            weights: Optional node weights for analysis
            solution_index: Index of this solution for labeling
        """
        if not self.verbose or len(solution_vector) == 0:
            return
            
        print(f"  Detailed support analysis for solution {solution_index + 1}:")
        
        # Show full solution vector with node labels
        vector_str = "    Full vector: ["
        for i, (node_id, value) in enumerate(zip(node_list, solution_vector)):
            if i > 0:
                vector_str += ", "
            vector_str += f"x{node_id}={value:.6f}"
            if i >= 10 and len(solution_vector) > 12:  # Truncate very long vectors
                vector_str += f", ... ({len(solution_vector) - 11} more)"
                break
        vector_str += "]"
        print(vector_str)
        
        # Analyze values above and below threshold
        above_threshold = {}
        below_threshold = {}
        
        for i, (node_id, value) in enumerate(zip(node_list, solution_vector)):
            if value > support_threshold:
                above_threshold[node_id] = value
            else:
                below_threshold[node_id] = value
        
        # Print above threshold values (support)
        if above_threshold:
            print(f"    Above threshold ({support_threshold}): {{", end="")
            sorted_above = sorted(above_threshold.items(), key=lambda x: x[1], reverse=True)
            above_parts = []
            for node_id, value in sorted_above:
                above_parts.append(f"{node_id}: {value:.6f}")
            print(", ".join(above_parts) + "}")
        else:
            print(f"    Above threshold ({support_threshold}): (none)")
            
        # Print below threshold values
        if below_threshold and len(below_threshold) <= 10:  # Only show if reasonable number
            print(f"    Below threshold: {{", end="")
            sorted_below = sorted(below_threshold.items(), key=lambda x: x[1], reverse=True)
            below_parts = []
            for node_id, value in sorted_below:
                below_parts.append(f"{node_id}: {value:.6f}")
            print(", ".join(below_parts) + "}")
        elif below_threshold:
            max_below = max(below_threshold.values())
            min_below = min(below_threshold.values())
            print(f"    Below threshold: {len(below_threshold)} values (range: {min_below:.6f} to {max_below:.6f})")
        
        # Weight analysis if weights provided
        if weights and above_threshold:
            print("    Weight analysis for support nodes:")
            total_weight = 0
            for node_id in above_threshold:
                weight = weights.get(node_id, 1.0)
                value = above_threshold[node_id]
                total_weight += weight
                print(f"      node{node_id}: weight={weight:.3f}, x={value:.6f}, weighted_x={weight*value:.6f}")
            print(f"      Total support weight: {total_weight:.3f}")
        
        # Threshold sensitivity analysis
        test_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        print("    Threshold sensitivity:")
        for test_thresh in test_thresholds:
            support_count = np.sum(solution_vector > test_thresh)
            marker = " <-- current" if abs(test_thresh - support_threshold) < 1e-9 else ""
            print(f"      {test_thresh}: {support_count} nodes{marker}")
    
    def analyze_threshold_sensitivity(
        self,
        solutions: List[np.ndarray],
        node_list: List[int],
        graph: nx.Graph,
        weights: Optional[Dict[int, float]] = None,
        test_thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze how different threshold values affect support extraction and clique finding.
        
        Args:
            solutions: List of solution vectors from Dirac
            node_list: List of node IDs corresponding to vector indices
            graph: Original graph for clique validation
            weights: Optional node weights
            test_thresholds: List of thresholds to test (default: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
            
        Returns:
            Dictionary with threshold analysis results
        """
        if test_thresholds is None:
            test_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        
        if not solutions:
            return {"error": "No solutions provided for analysis"}
        
        print("Threshold Sensitivity Analysis")
        print("=" * 50)
        
        analysis_results = {}
        
        for threshold in test_thresholds:
            print(f"\nAnalyzing threshold: {threshold}")
            print("-" * 30)
            
            # Extract cliques for this threshold
            found_cliques_at_threshold = set()
            total_support_nodes = 0
            valid_solutions = 0
            
            for i, solution_vector in enumerate(solutions):
                # Extract support at this threshold
                support_indices = set(np.where(solution_vector > threshold)[0].tolist())
                total_support_nodes += len(support_indices)
                
                if support_indices:
                    # Map to node IDs
                    candidate_clique = {node_list[idx] for idx in support_indices if idx < len(node_list)}
                    
                    # Verify if it's a valid clique
                    if self.verify_clique(graph, candidate_clique):
                        found_cliques_at_threshold.add(frozenset(candidate_clique))
                        valid_solutions += 1
            
            # Calculate statistics
            avg_support_size = total_support_nodes / len(solutions) if solutions else 0
            unique_cliques = len(found_cliques_at_threshold)
            success_rate = valid_solutions / len(solutions) if solutions else 0
            
            # Calculate weights of found cliques
            clique_weights = []
            if weights:
                for clique in found_cliques_at_threshold:
                    clique_weight = sum(weights.get(node, 1.0) for node in clique)
                    clique_weights.append(clique_weight)
            
            print(f"  Average support size: {avg_support_size:.2f} nodes")
            print(f"  Valid solutions: {valid_solutions}/{len(solutions)} ({success_rate:.1%})")
            print(f"  Unique valid cliques: {unique_cliques}")
            
            if clique_weights:
                max_weight = max(clique_weights)
                min_weight = min(clique_weights)
                avg_weight = sum(clique_weights) / len(clique_weights)
                print(f"  Clique weight range: {min_weight:.3f} to {max_weight:.3f} (avg: {avg_weight:.3f})")
            
            # Store results
            analysis_results[threshold] = {
                'avg_support_size': avg_support_size,
                'valid_solutions': valid_solutions,
                'success_rate': success_rate,
                'unique_cliques': unique_cliques,
                'clique_weights': clique_weights,
                'found_cliques': [set(clique) for clique in found_cliques_at_threshold]
            }
        
        # Provide recommendations
        print(f"\nThreshold Recommendations")
        print("=" * 30)
        
        # Find threshold with highest success rate
        best_success_threshold = max(analysis_results.keys(), 
                                   key=lambda t: analysis_results[t]['success_rate'])
        
        # Find threshold with most unique cliques
        best_diversity_threshold = max(analysis_results.keys(), 
                                     key=lambda t: analysis_results[t]['unique_cliques'])
        
        print(f"Best success rate: {best_success_threshold} "
              f"({analysis_results[best_success_threshold]['success_rate']:.1%} valid solutions)")
        print(f"Best diversity: {best_diversity_threshold} "
              f"({analysis_results[best_diversity_threshold]['unique_cliques']} unique cliques)")
        
        if weights:
            # Find threshold with best maximum weight
            best_weight_threshold = None
            best_max_weight = -1
            
            for threshold, results in analysis_results.items():
                if results['clique_weights']:
                    max_weight = max(results['clique_weights'])
                    if max_weight > best_max_weight:
                        best_max_weight = max_weight
                        best_weight_threshold = threshold
            
            if best_weight_threshold:
                print(f"Best max weight: {best_weight_threshold} "
                      f"(max clique weight: {best_max_weight:.3f})")
        
        analysis_results['recommendations'] = {
            'best_success_rate': best_success_threshold,
            'best_diversity': best_diversity_threshold,
            'best_weight': best_weight_threshold if weights else None
        }
        
        return analysis_results
    
    def _qplib_to_polynomial_file_gibbons(self, qplib_data: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """
        Transform QPLIB data to QCI polynomial file format for Gibbons minimization problems.
        
        CRITICAL: Unlike the standard qplib_to_polynomial_file, this function does NOT negate
        coefficients because Gibbons' weighted Motzkin-Straus theorem is already a MINIMIZATION
        problem: min{x^T B x | e^T x = 1, x ≥ 0}
        
        Args:
            qplib_data: QPLIB data dictionary with Gibbons B matrix coefficients
            file_name: Name for the polynomial file
            
        Returns:
            QCI polynomial file configuration dictionary with original coefficients
            
        Raises:
            ValueError: If QPLIB data is invalid
        """
        from collections import Counter
        
        try:
            poly_indices = qplib_data['poly_indices']
            poly_coefficients = qplib_data['poly_coefficients']
            
            if len(poly_indices) != len(poly_coefficients):
                raise ValueError("poly_indices and poly_coefficients must have same length")
            
            # Calculate number of variables and degrees
            all_indices = np.array(poly_indices).flatten()
            if len(all_indices) == 0:
                raise ValueError("Empty polynomial data")
            
            ind_dict = Counter(all_indices.tolist())
            num_vars = int(max(all_indices)) if len(all_indices) > 0 else 0
            max_degree = len(poly_indices[0]) if len(poly_indices) > 0 else 2
            min_degree = max_degree
            
            if self.verbose:
                print(f"Gibbons polynomial: {num_vars} variables, degree {min_degree}-{max_degree}")
                print(f"Coefficient range: [{min(poly_coefficients):.6f}, {max(poly_coefficients):.6f}]")
            
            # Transform to QCI format WITHOUT negating coefficients (Gibbons is already minimization)
            data = []
            for idx, val in zip(poly_indices, poly_coefficients):
                # Convert indices to native Python ints and coefficients to native Python floats
                if isinstance(idx, (list, tuple)):
                    idx_converted = [int(i) for i in idx]
                else:
                    idx_converted = int(idx)
                
                val_converted = float(val)  # NO NEGATION for Gibbons minimization
                data.append({"idx": idx_converted, "val": val_converted})
            
            # Create QCI polynomial file configuration
            file_config = {
                "file_name": file_name,
                "file_config": {
                    "polynomial": {
                        "num_variables": int(num_vars),
                        "min_degree": int(min_degree),
                        "max_degree": int(max_degree),
                        "data": data
                    }
                }
            }
            
            if self.verbose:
                print(f"Created Gibbons QCI polynomial file config with {len(data)} terms (NO negation)")
            
            return file_config
            
        except Exception as e:
            raise ValueError(f"Failed to convert QPLIB data to polynomial file: {e}")
    
    def _extract_cliques_from_dirac_response(
        self, 
        graph: nx.Graph,
        solutions: List[np.ndarray], 
        support_threshold: float = 1e-5,
        weights: Optional[Dict[int, float]] = None
    ) -> List[Set[int]]:
        """
        Extract maximal cliques from Dirac solution vectors.
        
        Args:
            graph: Original graph
            solutions: List of solution vectors from Dirac
            support_threshold: Threshold for support extraction
            weights: Optional node weights for detailed analysis
            
        Returns:
            List of valid cliques found
        """
        found_cliques = set()  # Use set of frozensets for deduplication
        node_list = list(graph.nodes())
        
        for i, solution_vector in enumerate(solutions):
            if self.verbose:
                print(f"Dirac: Processing solution {i+1}/{len(solutions)}")
                print(f"  Solution sum: {np.sum(solution_vector):.6f}")
                print(f"  Solution max: {np.max(solution_vector):.6f}")
                print(f"  Non-zero entries: {np.sum(solution_vector > support_threshold)}")
            
            # Print detailed support analysis for debugging
            self._print_detailed_support_analysis(
                solution_vector, node_list, support_threshold, weights, i
            )
            
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
                # Add all valid cliques (not just maximal ones) for maximum weight clique problems
                clique_frozen = frozenset(candidate_clique)
                if clique_frozen not in found_cliques:
                    found_cliques.add(clique_frozen)
                    if self.verbose:
                        is_maximal = self.verify_maximal_clique(graph, candidate_clique)
                        maximal_status = "maximal" if is_maximal else "non-maximal"
                        print(f"  Found valid clique ({maximal_status}): {sorted(candidate_clique)}")
            else:
                # Invalid clique - Dirac should find optimal solutions directly without refinement
                if self.verbose:
                    print(f"  Not a valid clique - skipping (refinement disabled, Dirac should find optimal solutions)")
                    print(f"  Candidate was: {sorted(candidate_clique)} (size: {len(candidate_clique)})")
        
        return [set(clique) for clique in found_cliques]
    
    def find_maximal_cliques(
        self, 
        graph: nx.Graph, 
        support_threshold: float = 1e-5,
        weights: Optional[Dict[int, float]] = None
    ) -> List[Set[int]]:
        """
        Find cliques using Dirac-3 quantum annealing with weighted Motzkin-Straus formulation.
        
        Note: Returns ALL valid cliques found (both maximal and non-maximal) since for 
        maximum weight clique problems, a smaller clique might have higher total weight.
        
        Args:
            graph: NetworkX graph to analyze (complement graph for MWIS)
            support_threshold: Threshold for extracting support from solutions
            weights: Dictionary mapping node IDs to weights (default: uniform weights of 1.0)
            
        Returns:
            List of sets, each containing vertices of a valid clique
        """
        if graph.number_of_nodes() == 0:
            return []
        
        # Default to uniform weights if not provided
        if weights is None:
            weights = {node: 1.0 for node in graph.nodes()}
        
        if self.verbose:
            print(f"Dirac: Finding maximal cliques in graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"Dirac: Using {self.config.num_samples} samples with schedule {self.config.relaxation_schedule}")
            print(f"Dirac: Using weighted Gibbons B matrix formulation with {len(weights)} weighted nodes")
        
        # Convert graph to QPLIB format using Gibbons B matrix
        qplib_data = self._graph_to_qplib_data(graph, weights)
        
        if not qplib_data['poly_indices']:
            if self.verbose:
                print("Dirac: No polynomial terms - empty graph or no edges")
            return []
        
        # Transform to QCI polynomial file format
        file_name = f"clique_graph_{graph.number_of_nodes()}n_{graph.number_of_edges()}e"
        polynomial_file = self._qplib_to_polynomial_file_gibbons(qplib_data, file_name)
        
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
            
            # Validate energy values before support extraction
            self._validate_energy_values(best_energy, all_energies, graph, weights)
            
            if self.verbose:
                print(f"Dirac: Best energy: {best_energy:.6f}")
                print(f"Dirac: Theoretical omega: {self.last_omega:.3f}")
                print(f"Dirac: Total solutions: {len(solutions)}")
                print(f"Dirac: Energy range: [{min(all_energies):.6f}, {max(all_energies):.6f}]")
            
            # Extract cliques from all solution vectors
            found_cliques = self._extract_cliques_from_dirac_response(
                graph, self.last_solutions, support_threshold, weights
            )
            
            if self.verbose:
                print(f"Dirac: Found {len(found_cliques)} unique valid cliques")
            
            return found_cliques
            
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