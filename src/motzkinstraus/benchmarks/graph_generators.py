"""
Graph generators for benchmarking MIS algorithms across diverse graph types.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Iterator, Optional
from enum import Enum
from dataclasses import dataclass


class GraphType(Enum):
    """Types of graphs for benchmarking."""
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"
    RANDOM_PARTITION = "random_partition"
    PATH = "path"
    CYCLE = "cycle"
    COMPLETE = "complete"
    STAR = "star"
    GRID_2D = "grid_2d"


@dataclass
class ScalingConfig:
    """Configuration for graph size scaling."""
    small_range: Tuple[int, int] = (10, 30)
    medium_range: Tuple[int, int] = (50, 200)
    large_range: Tuple[int, int] = (500, 1000)
    step_size: int = 10


def generate_erdos_renyi_graphs(
    sizes: List[int], 
    probabilities: List[float] = [0.1, 0.3, 0.5, 0.7],
    seed: int = 42
) -> Iterator[Tuple[nx.Graph, str]]:
    """
    Generate Erdos-Renyi random graphs with various sizes and edge probabilities.
    
    Args:
        sizes: List of graph sizes (number of nodes).
        probabilities: List of edge probabilities.
        seed: Random seed for reproducibility.
        
    Yields:
        Tuple of (graph, description).
    """
    rng = np.random.RandomState(seed)
    
    for n in sizes:
        for p in probabilities:
            # Use different seed for each graph
            graph_seed = rng.randint(0, 100000)
            G = nx.erdos_renyi_graph(n, p, seed=graph_seed)
            desc = f"ER_n{n}_p{p:.1f}"
            yield G, desc


def generate_barabasi_albert_graphs(
    sizes: List[int],
    attachment_params: List[int] = [1, 2, 3, 5],
    seed: int = 42
) -> Iterator[Tuple[nx.Graph, str]]:
    """
    Generate Barabasi-Albert scale-free graphs.
    
    Args:
        sizes: List of graph sizes (number of nodes).
        attachment_params: List of m parameters (edges to attach from new node).
        seed: Random seed for reproducibility.
        
    Yields:
        Tuple of (graph, description).
    """
    rng = np.random.RandomState(seed)
    
    for n in sizes:
        for m in attachment_params:
            if m >= n:
                continue  # Skip invalid parameters
            
            graph_seed = rng.randint(0, 100000)
            G = nx.barabasi_albert_graph(n, m, seed=graph_seed)
            desc = f"BA_n{n}_m{m}"
            yield G, desc


def generate_community_graphs(
    sizes: List[int],
    num_communities: List[int] = [2, 3, 4],
    p_in: float = 0.7,
    p_out: float = 0.1,
    seed: int = 42
) -> Iterator[Tuple[nx.Graph, str]]:
    """
    Generate graphs with community structure.
    
    Args:
        sizes: List of graph sizes (number of nodes).
        num_communities: List of number of communities.
        p_in: Probability of edges within communities.
        p_out: Probability of edges between communities.
        seed: Random seed for reproducibility.
        
    Yields:
        Tuple of (graph, description).
    """
    rng = np.random.RandomState(seed)
    
    for n in sizes:
        for k in num_communities:
            if k > n:
                continue
                
            # Create community sizes (roughly equal)
            base_size = n // k
            sizes_list = [base_size] * k
            # Distribute remaining nodes
            for i in range(n % k):
                sizes_list[i] += 1
                
            graph_seed = rng.randint(0, 100000)
            G = nx.random_partition_graph(sizes_list, p_in, p_out, seed=graph_seed)
            desc = f"Community_n{n}_k{k}_pin{p_in}_pout{p_out}"
            yield G, desc


def generate_structured_graphs(
    sizes: List[int]
) -> Iterator[Tuple[nx.Graph, str]]:
    """
    Generate structured graphs (path, cycle, complete, star, grid).
    
    Args:
        sizes: List of graph sizes.
        
    Yields:
        Tuple of (graph, description).
    """
    for n in sizes:
        # Path graph
        G_path = nx.path_graph(n)
        yield G_path, f"Path_n{n}"
        
        # Cycle graph  
        if n >= 3:
            G_cycle = nx.cycle_graph(n)
            yield G_cycle, f"Cycle_n{n}"
        
        # Complete graph (only for small sizes due to density)
        if n <= 20:
            G_complete = nx.complete_graph(n)
            yield G_complete, f"Complete_n{n}"
        
        # Star graph
        if n >= 4:
            G_star = nx.star_graph(n - 1)  # n-1 spokes + 1 center = n nodes
            yield G_star, f"Star_n{n}"
        
        # 2D Grid (approximately square)
        if n >= 4:
            side = int(np.sqrt(n))
            if side * side <= n < (side + 1) * (side + 1):
                # Try to get close to n nodes
                if abs(side * side - n) <= abs((side + 1) * side - n):
                    rows, cols = side, side
                else:
                    rows, cols = side + 1, side
                    
                if rows * cols >= n:  # Adjust if we overshot
                    G_grid = nx.grid_2d_graph(rows, cols)
                    # Convert to simple graph with integer node labels
                    G_grid = nx.convert_node_labels_to_integers(G_grid)
                    yield G_grid, f"Grid2D_n{G_grid.number_of_nodes()}"


def generate_test_graphs(
    config: ScalingConfig,
    graph_types: List[GraphType] = None,
    seed: int = 42
) -> Iterator[Tuple[nx.Graph, str, str]]:
    """
    Generate comprehensive test suite of graphs for benchmarking.
    
    Args:
        config: Scaling configuration for graph sizes.
        graph_types: List of graph types to generate (None for all).
        seed: Random seed for reproducibility.
        
    Yields:
        Tuple of (graph, description, size_category).
    """
    if graph_types is None:
        graph_types = list(GraphType)
    
    # Define size ranges
    small_sizes = list(range(config.small_range[0], config.small_range[1] + 1, config.step_size))
    medium_sizes = list(range(config.medium_range[0], config.medium_range[1] + 1, config.step_size * 2))
    large_sizes = list(range(config.large_range[0], config.large_range[1] + 1, config.step_size * 5))
    
    size_categories = [
        (small_sizes, "small"),
        (medium_sizes, "medium"), 
        (large_sizes, "large")
    ]
    
    for sizes, category in size_categories:
        if not sizes:
            continue
            
        for graph_type in graph_types:
            if graph_type == GraphType.ERDOS_RENYI:
                for G, desc in generate_erdos_renyi_graphs(sizes, seed=seed):
                    yield G, desc, category
                    
            elif graph_type == GraphType.BARABASI_ALBERT:
                for G, desc in generate_barabasi_albert_graphs(sizes, seed=seed):
                    yield G, desc, category
                    
            elif graph_type == GraphType.RANDOM_PARTITION:
                for G, desc in generate_community_graphs(sizes, seed=seed):
                    yield G, desc, category
                    
            elif graph_type in [GraphType.PATH, GraphType.CYCLE, GraphType.COMPLETE, 
                               GraphType.STAR, GraphType.GRID_2D]:
                for G, desc in generate_structured_graphs(sizes):
                    if (graph_type.value.upper() in desc.upper() or 
                        (graph_type == GraphType.GRID_2D and "Grid2D" in desc)):
                        yield G, desc, category


def create_small_test_graphs() -> List[Tuple[nx.Graph, str]]:
    """
    Create a small set of test graphs for validation and debugging.
    
    Returns:
        List of (graph, description) tuples.
    """
    graphs = []
    
    # Basic test graphs
    graphs.extend([
        (nx.path_graph(5), "Path_5"),
        (nx.cycle_graph(6), "Cycle_6"), 
        (nx.complete_graph(4), "Complete_4"),
        (nx.star_graph(6), "Star_7"),
        (nx.erdos_renyi_graph(8, 0.3, seed=42), "ER_8_p0.3"),
        (nx.barabasi_albert_graph(10, 2, seed=42), "BA_10_m2")
    ])
    
    # Empty and trivial graphs
    graphs.extend([
        (nx.empty_graph(5), "Empty_5"),
        (nx.trivial_graph(), "Trivial_1"),
        (nx.path_graph(1), "Single_1"),
        (nx.path_graph(2), "Edge_2")
    ])
    
    return graphs


if __name__ == "__main__":
    # Example usage
    config = ScalingConfig(
        small_range=(10, 20),
        medium_range=(30, 50), 
        large_range=(100, 200),
        step_size=5
    )
    
    # Generate a subset of graphs for testing
    test_types = [GraphType.ERDOS_RENYI, GraphType.PATH, GraphType.CYCLE]
    
    print("Generated test graphs:")
    count = 0
    for G, desc, category in generate_test_graphs(config, test_types):
        print(f"{category:6s} | {desc:20s} | n={G.number_of_nodes():3d}, m={G.number_of_edges():3d}")
        count += 1
        if count >= 20:  # Limit output
            break