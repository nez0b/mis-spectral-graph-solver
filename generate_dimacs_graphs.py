#!/usr/bin/env python3
"""
Generate random graphs using NetworkX and save them in DIMACS format.

This script generates Erdős-Rényi random graphs with various parameters
and saves them to the DIMACS/ folder for testing clique solvers.

USAGE:
    python generate_dimacs_graphs.py [OPTIONS]

EXAMPLES:
    # Generate all graph types (default behavior)
    python generate_dimacs_graphs.py
    
    # Generate only Erdős-Rényi random graphs
    python generate_dimacs_graphs.py --erdos-only
    
    # Generate only special graph types (complete, cycle, etc.)
    python generate_dimacs_graphs.py --others-only
    
    # Generate a custom Erdős-Rényi graph
    python generate_dimacs_graphs.py --custom 15 0.4 42 my_test_graph

COMMAND LINE ARGUMENTS:
    --erdos-only          Generate only Erdős-Rényi graphs with predefined parameters
                         Creates 10 graphs with various sizes (10-30 nodes) and 
                         edge probabilities (0.3-0.7)
    
    --others-only         Generate only special graph types including:
                         - Complete graphs (K5, K6)
                         - Cycle graphs (8-cycle, 12-cycle) 
                         - Path graphs (10 nodes)
                         - Wheel graphs
                         - Grid graphs (4x4)
                         - Complete bipartite graphs
                         - Petersen graph
                         - Barabási-Albert graphs
    
    --custom N P SEED NAME   Generate a custom Erdős-Rényi graph with:
                            N = number of nodes
                            P = edge probability (0.0 to 1.0)
                            SEED = random seed for reproducibility
                            NAME = output filename (without .dimacs extension)
    
    -h, --help           Show help message and exit

OUTPUT:
    All graphs are saved to the DIMACS/ directory with descriptive filenames.
    Each file includes:
    - Comment lines with graph description and properties
    - Problem definition line: "p edge <nodes> <edges>"
    - Edge definitions: "e <node1> <node2>" (1-based indexing)

INTEGRATION:
    Generated DIMACS files can be used with:
    python demo_dimacs_clique_solvers.py DIMACS/<filename>.dimacs
"""

import os
import networkx as nx
from pathlib import Path
import argparse


def write_dimacs_graph(graph, filename, description="Generated graph"):
    """
    Write a NetworkX graph to DIMACS format file.
    
    DIMACS format:
    - Lines starting with 'c' are comments
    - Line starting with 'p edge n m' defines problem with n nodes and m edges
    - Lines starting with 'e u v' define edges between nodes u and v
    
    Args:
        graph (nx.Graph): NetworkX graph to write
        filename (str): Output filename
        description (str): Description for comment line
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    with open(filename, 'w') as f:
        # Write comment with description
        f.write(f"c {description}\n")
        f.write(f"c Generated using NetworkX\n")
        f.write(f"c Nodes: {num_nodes}, Edges: {num_edges}\n")
        
        # Write problem definition
        f.write(f"p edge {num_nodes} {num_edges}\n")
        
        # Write edges (ensure 1-based indexing for DIMACS)
        for u, v in graph.edges():
            # Convert to 1-based indexing if needed
            u_dimacs = u + 1 if min(graph.nodes()) == 0 else u
            v_dimacs = v + 1 if min(graph.nodes()) == 0 else v
            f.write(f"e {u_dimacs} {v_dimacs}\n")


def generate_erdos_renyi_graphs():
    """Generate a collection of Erdős-Rényi random graphs with different parameters."""
    
    # Create DIMACS directory if it doesn't exist
    dimacs_dir = Path("DIMACS")
    dimacs_dir.mkdir(exist_ok=True)
    
    # Define different graph parameters to generate
    graph_configs = [
        # Small graphs for quick testing
        {"n": 10, "p": 0.3, "seed": 42, "name": "erdos_renyi_10_p03_seed42"},
        {"n": 10, "p": 0.5, "seed": 42, "name": "erdos_renyi_10_p05_seed42"},
        {"n": 10, "p": 0.7, "seed": 42, "name": "erdos_renyi_10_p07_seed42"},
        
        # Medium graphs
        {"n": 15, "p": 0.3, "seed": 123, "name": "erdos_renyi_15_p03_seed123"},
        {"n": 15, "p": 0.5, "seed": 123, "name": "erdos_renyi_15_p05_seed123"},
        {"n": 15, "p": 0.7, "seed": 123, "name": "erdos_renyi_15_p07_seed123"},
        
        # Larger graphs for performance testing
        {"n": 20, "p": 0.3, "seed": 456, "name": "erdos_renyi_20_p03_seed456"},
        {"n": 20, "p": 0.5, "seed": 456, "name": "erdos_renyi_20_p05_seed456"},
        {"n": 25, "p": 0.4, "seed": 789, "name": "erdos_renyi_25_p04_seed789"},
        {"n": 30, "p": 0.3, "seed": 999, "name": "erdos_renyi_30_p03_seed999"},
    ]
    
    print("🎲 Generating Erdős-Rényi random graphs...")
    print("=" * 60)
    
    for config in graph_configs:
        n = config["n"]
        p = config["p"]
        seed = config["seed"]
        name = config["name"]
        
        # Generate graph
        graph = nx.erdos_renyi_graph(n, p, seed=seed)
        
        # Calculate some properties
        num_edges = graph.number_of_edges()
        density = num_edges / (n * (n - 1) / 2)
        max_clique_upper_bound = nx.graph_clique_number(graph) if n <= 15 else "N/A"
        
        # Create filename
        filename = dimacs_dir / f"{name}.dimacs"
        
        # Create description
        description = f"Erdős-Rényi G({n}, {p}) with seed={seed}"
        
        # Write to DIMACS format
        write_dimacs_graph(graph, filename, description)
        
        # Print summary
        print(f"✅ {name}")
        print(f"   File: {filename}")
        print(f"   Nodes: {n}, Edges: {num_edges}, Density: {density:.3f}")
        if max_clique_upper_bound != "N/A":
            print(f"   Max clique size: {max_clique_upper_bound}")
        print()
    
    print(f"🎉 Generated {len(graph_configs)} graphs in DIMACS/ folder")


def generate_other_graph_types():
    """Generate other interesting graph types for testing."""
    
    dimacs_dir = Path("DIMACS")
    dimacs_dir.mkdir(exist_ok=True)
    
    print("\n🏗️  Generating other graph types...")
    print("=" * 60)
    
    other_graphs = [
        # Complete graphs (known clique sizes)
        {"graph": nx.complete_graph(5), "name": "complete_k5", "desc": "Complete graph K5 (clique size = 5)"},
        {"graph": nx.complete_graph(6), "name": "complete_k6", "desc": "Complete graph K6 (clique size = 6)"},
        
        # Cycle graphs
        {"graph": nx.cycle_graph(8), "name": "cycle_8", "desc": "8-cycle graph (clique size = 2)"},
        {"graph": nx.cycle_graph(12), "name": "cycle_12", "desc": "12-cycle graph (clique size = 2)"},
        
        # Path graphs
        {"graph": nx.path_graph(10), "name": "path_10", "desc": "Path graph with 10 nodes (clique size = 2)"},
        
        # Wheel graphs
        {"graph": nx.wheel_graph(8), "name": "wheel_8", "desc": "Wheel graph with 9 nodes total (clique size = 3)"},
        
        # Grid graphs
        {"graph": nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4)), "name": "grid_4x4", "desc": "4x4 grid graph (clique size = 2)"},
        
        # Complete bipartite
        {"graph": nx.complete_bipartite_graph(4, 5), "name": "complete_bipartite_4_5", "desc": "Complete bipartite K(4,5) (clique size = 2)"},
        
        # Petersen graph
        {"graph": nx.petersen_graph(), "name": "petersen", "desc": "Petersen graph (clique size = 2)"},
        
        # Barabási-Albert
        {"graph": nx.barabasi_albert_graph(15, 3, seed=42), "name": "barabasi_albert_15_3", "desc": "Barabási-Albert graph (15 nodes, m=3)"},
    ]
    
    for graph_info in other_graphs:
        graph = graph_info["graph"]
        name = graph_info["name"]
        description = graph_info["desc"]
        
        filename = dimacs_dir / f"{name}.dimacs"
        write_dimacs_graph(graph, filename, description)
        
        # Calculate properties
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        density = m / (n * (n - 1) / 2) if n > 1 else 0
        
        print(f"✅ {name}")
        print(f"   File: {filename}")
        print(f"   Nodes: {n}, Edges: {m}, Density: {density:.3f}")
        print(f"   Description: {description}")
        print()
    
    print(f"🎉 Generated {len(other_graphs)} additional graphs")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate graphs in DIMACS format")
    parser.add_argument("--erdos-only", action="store_true", 
                       help="Generate only Erdős-Rényi graphs")
    parser.add_argument("--others-only", action="store_true", 
                       help="Generate only other graph types")
    parser.add_argument("--custom", nargs=4, metavar=("N", "P", "SEED", "NAME"),
                       help="Generate custom Erdős-Rényi graph: N P SEED NAME")
    
    args = parser.parse_args()
    
    print("🎯 DIMACS Graph Generator")
    print("=" * 60)
    
    if args.custom:
        # Generate custom graph
        n, p, seed, name = int(args.custom[0]), float(args.custom[1]), int(args.custom[2]), args.custom[3]
        
        dimacs_dir = Path("DIMACS")
        dimacs_dir.mkdir(exist_ok=True)
        
        graph = nx.erdos_renyi_graph(n, p, seed=seed)
        filename = dimacs_dir / f"{name}.dimacs"
        description = f"Custom Erdős-Rényi G({n}, {p}) with seed={seed}"
        
        write_dimacs_graph(graph, filename, description)
        
        print(f"✅ Generated custom graph: {name}")
        print(f"   File: {filename}")
        print(f"   Nodes: {n}, Edges: {graph.number_of_edges()}")
        
    elif args.erdos_only:
        generate_erdos_renyi_graphs()
    elif args.others_only:
        generate_other_graph_types()
    else:
        # Generate both types
        generate_erdos_renyi_graphs()
        generate_other_graph_types()
    
    print("\n📁 All files saved to DIMACS/ directory")
    print("🔧 Use with demo_dimacs_clique_solvers.py for testing")


if __name__ == "__main__":
    main()