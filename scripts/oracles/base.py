"""
Base classes for oracle adapters used in maximal clique finding.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any, Optional
import networkx as nx
import numpy as np


class OracleConfig(ABC):
    """Base class for oracle-specific configurations."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration parameters."""
        pass


class OracleAdapter(ABC):
    """
    Abstract base class for oracle adapters.
    
    Oracle adapters provide a unified interface for different solvers
    while handling solver-specific implementation details.
    """
    
    def __init__(self, config: OracleConfig, verbose: bool = False, enable_refinement: bool = True):
        self.config = config
        self.verbose = verbose
        self.enable_refinement = enable_refinement
        self._validate_dependencies()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the oracle adapter."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the oracle's dependencies are available."""
        pass
    
    @abstractmethod
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        pass
    
    @abstractmethod
    def find_maximal_cliques(
        self, 
        graph: nx.Graph,
        support_threshold: float = 1e-5
    ) -> List[Set[int]]:
        """
        Find maximal cliques in the given graph.
        
        Args:
            graph: NetworkX graph to analyze
            support_threshold: Threshold for extracting support from solutions
            
        Returns:
            List of sets, each containing vertices of a maximal clique
        """
        pass
    
    @abstractmethod
    def get_optimization_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the last optimization run.
        
        Returns:
            Dictionary with solver-specific optimization statistics
        """
        pass
    
    def verify_clique(self, graph: nx.Graph, candidate_clique: Set[int]) -> bool:
        """
        Verify that a candidate set forms a valid clique.
        
        Args:
            graph: The input graph
            candidate_clique: Set of vertices to verify
            
        Returns:
            True if the set forms a clique, False otherwise
        """
        if not candidate_clique:
            return False
        
        # Check all pairs of vertices in the candidate clique
        for u in candidate_clique:
            for v in candidate_clique:
                if u != v and not graph.has_edge(u, v):
                    return False
        return True
    
    def verify_maximal_clique(self, graph: nx.Graph, candidate_clique: Set[int]) -> bool:
        """
        Verify that a clique is maximal (cannot be extended).
        
        Args:
            graph: The input graph
            candidate_clique: Set of vertices to check for maximality
            
        Returns:
            True if the clique is maximal, False otherwise
        """
        if not candidate_clique:
            return False
        
        # First verify it's actually a clique
        if not self.verify_clique(graph, candidate_clique):
            return False
        
        # Check if we can add any vertex to extend the clique
        for vertex in graph.nodes():
            if vertex not in candidate_clique:
                # Check if this vertex is connected to ALL vertices in the current clique
                is_connected_to_all = all(graph.has_edge(vertex, clique_member) 
                                        for clique_member in candidate_clique)
                
                if is_connected_to_all:
                    # We can extend the clique, so it's not maximal
                    return False
        
        # No vertex can be added, so it's maximal
        return True
    
    def extract_support(self, solution_vector: np.ndarray, threshold: float = 1e-5) -> Set[int]:
        """
        Extract support (non-zero indices) from solution vector.
        
        Args:
            solution_vector: Solution vector from optimization
            threshold: Minimum value to consider as "non-zero"
            
        Returns:
            Set of vertex indices where x_i > threshold
        """
        if len(solution_vector) == 0:
            return set()
        
        # Find indices where solution values exceed threshold
        support_indices = np.where(solution_vector > threshold)[0]
        return set(support_indices.tolist())
    
    def refine_superposition_solution(
        self, 
        graph: nx.Graph, 
        solution_vector: np.ndarray, 
        support: Set[int]
    ) -> List[Set[int]]:
        """
        Refine a superposition solution into constituent maximal cliques using Greedy Clique Peeling.
        
        This method implements the "Greedy Clique Peeling" algorithm to extract multiple
        maximal cliques from a single superposition solution. When the support size is
        larger than the expected clique size (indicating a superposition of multiple cliques),
        this method uses the solution weights to iteratively extract cliques.
        
        Algorithm:
        1. Use solution weights to select seed vertices (highest weight first)
        2. Greedily grow clique by adding highest-weighted neighbors
        3. Verify maximality and store found clique
        4. Remove clique vertices from working set
        5. Repeat until all vertices processed
        
        Args:
            graph: NetworkX graph being analyzed
            solution_vector: Solution vector from optimization (contains weights)
            support: Set of vertices with non-zero weights (support of solution)
            
        Returns:
            List of maximal cliques extracted from the superposition
        """
        if not support or len(support) <= 1:
            return []
        
        found_cliques = []
        node_list = list(graph.nodes())
        
        # Create weighted support dictionary: {vertex: weight}
        weighted_support = {}
        for vertex_idx in support:
            if vertex_idx < len(node_list):
                vertex = node_list[vertex_idx]
                weighted_support[vertex] = solution_vector[vertex_idx]
        
        if self.verbose:
            print(f"  Refining superposition with {len(weighted_support)} vertices")
        
        # Greedy Clique Peeling: iteratively extract cliques
        iteration = 0
        while weighted_support and iteration < 10:  # Safety limit
            iteration += 1
            
            if self.verbose:
                print(f"    Iteration {iteration}: {len(weighted_support)} vertices remaining")
            
            # 1. Seed Selection: pick highest-weighted vertex
            seed_vertex = max(weighted_support, key=weighted_support.get)
            current_clique = {seed_vertex}
            
            if self.verbose:
                print(f"    Seed vertex: {seed_vertex} (weight: {weighted_support[seed_vertex]:.4f})")
            
            # 2. Clique Growth: greedily add highest-weighted common neighbors
            while True:
                # Find vertices that are neighbors of ALL vertices in current clique
                candidates = set(weighted_support.keys()) - current_clique
                common_neighbors = candidates.copy()
                
                for clique_vertex in current_clique:
                    neighbors_of_vertex = set(graph.neighbors(clique_vertex)) & candidates
                    common_neighbors &= neighbors_of_vertex
                
                if not common_neighbors:
                    break  # No more vertices can be added
                
                # Select highest-weighted common neighbor
                best_candidate = max(common_neighbors, key=weighted_support.get)
                current_clique.add(best_candidate)
                
                if self.verbose:
                    print(f"      Added vertex {best_candidate} (weight: {weighted_support[best_candidate]:.4f})")
            
            # 3. Verify clique is valid and maximal
            if self.verify_clique(graph, current_clique):
                if self.verify_maximal_clique(graph, current_clique):
                    found_cliques.append(current_clique.copy())
                    if self.verbose:
                        print(f"      Found maximal clique: {sorted(current_clique)} (size: {len(current_clique)})")
                else:
                    if self.verbose:
                        print(f"      Found valid clique but not maximal: {sorted(current_clique)}")
                    # Still add it - it might become maximal after other vertices are removed
                    found_cliques.append(current_clique.copy())
            else:
                if self.verbose:
                    print(f"      Invalid clique detected: {sorted(current_clique)}")
            
            # 4. Remove processed vertices from working set
            for vertex in current_clique:
                if vertex in weighted_support:
                    del weighted_support[vertex]
        
        if self.verbose:
            print(f"    Refinement complete: extracted {len(found_cliques)} cliques")
        
        return found_cliques