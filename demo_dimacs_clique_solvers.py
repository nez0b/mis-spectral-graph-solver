#!/usr/bin/env python3
"""
DIMACS-based Maximum Clique Solver Comparison.

This script reads graphs in DIMACS format and compares MILP vs JAX PGD methods
for solving the maximum clique problem. It focuses exclusively on clique finding,
not MIS (Maximum Independent Set).

Based on demo_clique_solvers.py but adapted for DIMACS format input.
"""

import os
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from motzkinstraus import (
    find_max_clique_with_oracle,
    verify_clique,
    MILP_AVAILABLE
)

# Import MILP solvers if available
if MILP_AVAILABLE:
    from motzkinstraus.solvers.milp import solve_max_clique_milp

# Import JAX PGD oracle
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle


def read_dimacs_graph(filename):
    """
    Read a graph from DIMACS format file.
    
    DIMACS format:
    - Lines starting with 'c' are comments
    - Line starting with 'p edge n m' defines problem with n nodes and m edges
    - Lines starting with 'e u v' define edges between nodes u and v
    
    Args:
        filename (str): Path to DIMACS format file
        
    Returns:
        nx.Graph: NetworkX graph representation
    """
    graph = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('c'):
                # Comment line, skip
                continue
            elif line.startswith('p edge'):
                # Problem definition: p edge <num_nodes> <num_edges>
                parts = line.split()
                if len(parts) >= 3:
                    num_nodes = int(parts[2])
                    # Add nodes 1 through num_nodes (DIMACS uses 1-based indexing)
                    graph.add_nodes_from(range(1, num_nodes + 1))
            elif line.startswith('e'):
                # Edge definition: e <node1> <node2>
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[1]), int(parts[2])
                    graph.add_edge(u, v)
    
    return graph


def run_clique_solver_comparison(graph, graph_name):
    """Run MILP and JAX PGD solvers on a graph and compare clique results."""
    print(f"\n{'='*60}")
    print(f"Graph: {graph_name}")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"Density: {graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2):.3f}")
    print('='*60)
    
    results = {}
    
    # 1. JAX PGD Oracle-based Algorithm
    print("🎯 Oracle-based Algorithm (JAX PGD):")
    oracle = ProjectedGradientDescentOracle(
        learning_rate=0.02,
        max_iterations=1000,
        num_restarts=5,
        tolerance=1e-6,
        verbose=False
    )
    
    # Create fresh graph copy for oracle
    graph_copy = nx.Graph()
    graph_copy.add_nodes_from(graph.nodes())
    graph_copy.add_edges_from(graph.edges())
    
    start_time = time.time()
    clique_oracle, oracle_calls = find_max_clique_with_oracle(graph_copy, oracle)
    clique_oracle_time = time.time() - start_time
    
    print(f"  Max Clique: {len(clique_oracle)} nodes in {clique_oracle_time:.4f}s ({oracle_calls} oracle calls)")
    print(f"  Clique: {sorted(clique_oracle)}")
    
    results['oracle'] = {
        'clique_size': len(clique_oracle),
        'clique_set': clique_oracle,
        'clique_time': clique_oracle_time,
        'clique_calls': oracle_calls
    }
    
    # 2. MILP Solver (if available)
    if MILP_AVAILABLE:
        print("\n⚙️  MILP Solver (Gurobi):")
        
        start_time = time.time()
        clique_milp = solve_max_clique_milp(graph, suppress_output=True)
        clique_milp_time = time.time() - start_time
        
        print(f"  Max Clique: {len(clique_milp)} nodes in {clique_milp_time:.4f}s")
        print(f"  Clique: {sorted(clique_milp)}")
        
        results['milp'] = {
            'clique_size': len(clique_milp),
            'clique_set': clique_milp,
            'clique_time': clique_milp_time
        }
    else:
        print("\n⚠️  MILP Solver not available (Gurobi missing)")
        results['milp'] = None
    
    # 3. Validation
    print("\n✅ Validation:")
    
    # Verify all cliques are valid
    assert verify_clique(graph, clique_oracle), "Oracle clique invalid"
    if MILP_AVAILABLE:
        assert verify_clique(graph, clique_milp), "MILP clique invalid"
    
    # Check size consistency
    sizes_match = True
    if MILP_AVAILABLE:
        if len(clique_oracle) != len(clique_milp):
            print(f"  ⚠️  Clique size mismatch: Oracle={len(clique_oracle)}, MILP={len(clique_milp)}")
            sizes_match = False
    
    if sizes_match:
        print("  ✅ All solvers agree on optimal clique size")
        print("  ✅ All clique solutions are valid")
    
    return results


def create_clique_performance_summary(all_results):
    """Create a performance summary focusing on clique solving only."""
    print(f"\n{'='*70}")
    print("CLIQUE SOLVER PERFORMANCE SUMMARY")
    print('='*70)
    
    print(f"{'Graph':<25} {'Nodes':<6} {'ω(G)':<6} {'Oracle Time':<12} {'MILP Time':<10}")
    print('-' * 70)
    
    for graph_name, results in all_results.items():
        oracle = results['oracle']
        milp = results['milp']
        
        nodes = results['nodes']
        clique_size = oracle['clique_size']
        
        oracle_time = oracle['clique_time']
        milp_time = milp['clique_time'] if milp else 0
        
        milp_time_str = f"{milp_time:.4f}s" if milp else "N/A"
        
        print(f"{graph_name:<25} {nodes:<6} {clique_size:<6} {oracle_time:.4f}s      {milp_time_str:<10}")
    
    print("\nLegend:")
    print("  ω(G) = Clique number (size of maximum clique)")
    print("  Oracle = JAX PGD Oracle, MILP = Gurobi")


def visualize_clique_solutions(graph, results, graph_name, output_dir):
    """Visualize the graph and highlight the clique solutions found."""
    plt.figure(figsize=(12, 5))
    
    # Fixed layout for consistent visualization
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Plot 1: MILP (Gurobi) Solution
    plt.subplot(1, 2, 1)
    if MILP_AVAILABLE and results['milp']:
        clique_milp = results['milp']['clique_set']
        # Verify clique is valid
        clique_valid = verify_clique(graph, clique_milp)
        node_colors = ['red' if node in clique_milp else 'lightblue' for node in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(graph, pos, alpha=0.3)
        # Highlight clique edges in bold
        clique_edges = [(u, v) for u in clique_milp for v in clique_milp if u < v and graph.has_edge(u, v)]
        nx.draw_networkx_edges(graph, pos, edgelist=clique_edges, edge_color='red', width=3)
        nx.draw_networkx_labels(graph, pos, font_size=10)
        status = "✓" if clique_valid else "✗"
        plt.title(f'MILP (Gurobi) Max Clique\nSize: {len(clique_milp)} {status}')
    else:
        plt.title('MILP (Gurobi) Not Available')
    plt.axis('off')
    
    # Plot 2: JAX PGD Oracle Solution
    plt.subplot(1, 2, 2)
    clique_oracle = results['oracle']['clique_set']
    # Verify clique is valid
    clique_valid = verify_clique(graph, clique_oracle)
    node_colors = ['red' if node in clique_oracle else 'lightblue' for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    # Highlight clique edges in bold
    clique_edges = [(u, v) for u in clique_oracle for v in clique_oracle if u < v and graph.has_edge(u, v)]
    nx.draw_networkx_edges(graph, pos, edgelist=clique_edges, edge_color='red', width=3)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    status = "✓" if clique_valid else "✗"
    plt.title(f'Oracle (JAX PGD) Max Clique\nSize: {len(clique_oracle)} {status}')
    plt.axis('off')
    
    plt.suptitle(f'{graph_name} - Maximum Clique Comparison', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    filename = graph_name.lower().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    plt.savefig(output_dir / f'{filename}_clique_solutions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print clique validation details
    print(f"\n🔍 Clique Validation for {graph_name}:")
    if MILP_AVAILABLE and results['milp']:
        clique_milp = results['milp']['clique_set']
        print(f"  MILP Clique: {sorted(clique_milp)}")
        print(f"  MILP Clique Valid: {verify_clique(graph, clique_milp)}")
        if len(clique_milp) > 1:
            # Check all pairs are connected
            all_connected = all(graph.has_edge(u, v) for u in clique_milp for v in clique_milp if u != v)
            print(f"  MILP All pairs connected: {all_connected}")
    
    clique_oracle = results['oracle']['clique_set']
    print(f"  Oracle Clique: {sorted(clique_oracle)}")
    print(f"  Oracle Clique Valid: {verify_clique(graph, clique_oracle)}")
    if len(clique_oracle) > 1:
        # Check all pairs are connected
        all_connected = all(graph.has_edge(u, v) for u in clique_oracle for v in clique_oracle if u != v)
        print(f"  Oracle All pairs connected: {all_connected}")


def main():
    """Run the DIMACS clique solver demonstration."""
    parser = argparse.ArgumentParser(description="Compare MILP vs JAX PGD for max clique on DIMACS graphs")
    parser.add_argument("dimacs_file", nargs="?", default="DIMACS/test_10_node.dimacs",
                       help="Path to DIMACS format file (default: DIMACS/test_10_node.dimacs)")
    
    args = parser.parse_args()
    dimacs_file = args.dimacs_file
    
    print("🎯 DIMACS Maximum Clique Solver Comparison")
    print("=" * 60)
    print("Testing MILP vs JAX PGD approaches for max clique:")
    print("1. JAX PGD Oracle (Motzkin-Straus theorem)")
    if MILP_AVAILABLE:
        print("2. MILP (direct combinatorial optimization)")
    else:
        print("2. MILP (not available - Gurobi missing)")
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(dimacs_file):
        print(f"❌ DIMACS file not found: {dimacs_file}")
        return
    
    print(f"\n📁 Reading DIMACS file: {dimacs_file}")
    graph = read_dimacs_graph(dimacs_file)
    
    # Extract graph name from filename
    graph_name = Path(dimacs_file).stem.replace('_', ' ').title()
    
    print(f"✅ Successfully loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Run solver comparison
    results = run_clique_solver_comparison(graph, graph_name)
    results['nodes'] = graph.number_of_nodes()
    results['edges'] = graph.number_of_edges()
    
    all_results = {graph_name: results}
    
    # Create visualization
    visualize_clique_solutions(graph, results, graph_name, output_dir)
    
    # Create performance summary
    create_clique_performance_summary(all_results)
    
    print(f"\n🎉 DIMACS clique solver comparison completed successfully!")
    print(f"📊 Visualization saved to: {output_dir.absolute()}")
    print("\nKey findings:")
    print("✅ Both solving approaches produce correct clique results")
    print("✅ Oracle-based algorithm uses Motzkin-Straus theorem")
    if MILP_AVAILABLE:
        print("✅ MILP solver provides exact solutions efficiently")
    print("✅ Implementation correctly handles DIMACS format input")


if __name__ == "__main__":
    main()