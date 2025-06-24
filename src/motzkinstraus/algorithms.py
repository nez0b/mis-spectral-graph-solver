"""
Algorithms for finding Maximum Independent Sets using oracles.
"""

import networkx as nx
from typing import Set, Callable
from itertools import combinations
from .oracles.base import Oracle


def find_mis_with_oracle(graph: nx.Graph, oracle: Oracle, verbose: bool = False) -> tuple[Set[int], int]:
    """
    Find a Maximum Independent Set using an oracle-based search algorithm.
    
    This algorithm uses the relationship alpha(G) = omega(G_complement) where:
    - alpha(G) is the independence number (size of maximum independent set)
    - omega(G_complement) is the clique number of the complement graph
    
    The algorithm iteratively builds the MIS by testing each vertex using
    the oracle to guide the search decisions.
    
    Args:
        graph: The input networkx graph.
        oracle: An Oracle instance that can determine clique numbers.
        verbose: Whether to print progress information.
        
    Returns:
        A tuple containing:
        - A set of node IDs representing a maximum independent set
        - The number of oracle calls made
    """
    # Step 1: Find target MIS size using alpha(G) = omega(G_complement)
    # Reset oracle call counter
    oracle.call_count = 0
    
    G_complement = nx.complement(graph)
    k_target = oracle.get_omega(G_complement)
    
    if verbose:
        print(f"Oracle reports MIS size (alpha) for the graph is: {k_target}")
    
    # Initialize the search
    mis_nodes = set()
    G_current = graph.copy()
    
    # Iterate through nodes to build the MIS
    for v in list(graph.nodes()):
        # Skip if node was already removed as a neighbor of a selected node
        if v not in G_current.nodes():
            continue
        
        # Create test graph by removing v and its neighbors
        # This represents the remaining problem if we choose v
        neighbors_of_v = list(G_current.neighbors(v))
        G_test = G_current.copy()
        G_test.remove_nodes_from([v] + neighbors_of_v)
        
        # Ask oracle for MIS size of remaining subgraph
        k_test = oracle.get_omega(nx.complement(G_test))
        
        # Check if choosing v leads to a valid maximum independent set
        if 1 + k_test == k_target:
            if verbose:
                print(f"  - Testing node {v}... If chosen, need MIS size {k_test} from rest.")
                print(f"    -> 1 + {k_test} == {k_target}. Viable. Committing to {v}.")
            
            # Commit to this node
            mis_nodes.add(v)
            G_current = G_test
            k_target -= 1
        else:
            if verbose:
                print(f"  - Testing node {v}... If chosen, need MIS size {k_test} from rest.")
                print(f"    -> 1 + {k_test} != {k_target}. Not optimal. Discarding {v}.")
    
    return mis_nodes, oracle.call_count


def find_mis_brute_force(graph: nx.Graph) -> Set[int]:
    """
    Find a Maximum Independent Set using brute force enumeration.
    
    This function checks all possible combinations of vertices to find
    the largest independent set. It's only practical for small graphs
    but serves as a ground truth for testing.
    
    Args:
        graph: The input networkx graph.
        
    Returns:
        A set of node IDs representing a maximum independent set.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    
    if n == 0:
        return set()
    
    # Try all combinations from largest to smallest
    for k in range(n, 0, -1):
        for combo in combinations(nodes, k):
            # Check if this combination forms an independent set
            is_independent = True
            for u, v in combinations(combo, 2):
                if graph.has_edge(u, v):
                    is_independent = False
                    break
            
            if is_independent:
                return set(combo)  # First (largest) independent set found
    
    return set()


def verify_independent_set(graph: nx.Graph, node_set: Set[int]) -> bool:
    """
    Verify that a set of nodes forms an independent set.
    
    Args:
        graph: The input networkx graph.
        node_set: Set of node IDs to verify.
        
    Returns:
        True if the node set is independent, False otherwise.
    """
    # Check that no two nodes in the set are connected
    for u in node_set:
        for v in node_set:
            if u != v and graph.has_edge(u, v):
                return False
    return True