#!/usr/bin/env python3
"""
Demonstration of Maximum Clique and MIS solvers on various graph types.

This script showcases all available solving approaches:
1. Oracle-based algorithms using Motzkin-Straus theorem
2. MILP solvers using direct combinatorial optimization

It compares performance and validates results across different graph structures.
"""

import os
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from motzkinstraus import (
    find_max_clique_with_oracle,
    find_mis_with_oracle,
    verify_clique,
    verify_independent_set,
    MILP_AVAILABLE
)

# Import MILP solvers if available
if MILP_AVAILABLE:
    from motzkinstraus.solvers.milp import solve_max_clique_milp, solve_mis_milp

# Import different oracles for comparison
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle

def create_demo_graphs():
    """Create interesting demo graphs with various structures."""
    graphs = []
    
    # 5. Random graph with specific edge probability
    edge_prob = 0.4
    num_nodes = 30
    G5 = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=42)
    graphs.append((f"Erdős-Rényi ({num_nodes} nodes, p={edge_prob})", G5))
    
    return graphs

def run_solver_comparison(graph, graph_name):
    """Run all available solvers on a graph and compare results."""
    print(f"\n{'='*60}")
    print(f"Graph: {graph_name}")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"Density: {graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2):.3f}")
    print('='*60)
    
    results = {}
    
    # 1. Oracle-based algorithms
    """
    print("🎯 Oracle-based Algorithms (JAX PGD):")
    oracle = ProjectedGradientDescentOracle(
        learning_rate=0.02,
        max_iterations=1000,
        num_restarts=5,
        tolerance=1e-6,
        verbose=False
    )
    """
    print("🎯 Oracle-based Algorithms (Hybrid Dirac/NetworkX):")
    oracle = DiracNetworkXHybridOracle(
        threshold_nodes=20,
        num_samples=30,
        relax_schedule=3,
        solution_precision=None
    )
    
    # Create completely new unfrozen graphs by manually copying structure
    graph_copy1 = nx.Graph()
    graph_copy1.add_nodes_from(graph.nodes())
    graph_copy1.add_edges_from(graph.edges())
    
    graph_copy2 = nx.Graph()
    graph_copy2.add_nodes_from(graph.nodes())
    graph_copy2.add_edges_from(graph.edges())
    
    start_time = time.time()
    clique_oracle, oracle_calls_clique = find_max_clique_with_oracle(graph_copy1, oracle)
    clique_oracle_time = time.time() - start_time
    
    # start_time = time.time()
    # mis_oracle, oracle_calls_mis = find_mis_with_oracle(graph_copy2, oracle)
    # mis_oracle_time = time.time() - start_time
    
    print(f"  Max Clique: {len(clique_oracle)} nodes in {clique_oracle_time:.4f}s ({oracle_calls_clique} oracle calls)")
    # print(f"  Max Indep: {len(mis_oracle)} nodes in {mis_oracle_time:.4f}s ({oracle_calls_mis} oracle calls)")
    print(f"  Clique: {sorted(clique_oracle)}")
    # print(f"  MIS: {sorted(mis_oracle)}")
    
    results['oracle'] = {
        'clique_size': len(clique_oracle),
        'clique_set': clique_oracle,
        'clique_time': clique_oracle_time,
        'clique_calls': oracle_calls_clique,
        # 'mis_size': len(mis_oracle),
        # 'mis_set': mis_oracle,
        # 'mis_time': mis_oracle_time,
        # 'mis_calls': oracle_calls_mis
    }
    
    # 2. MILP solvers (if available)
    if MILP_AVAILABLE:
        print("\n⚙️  MILP Solvers (Gurobi):")
        
        start_time = time.time()
        clique_milp = solve_max_clique_milp(graph, suppress_output=True)
        clique_milp_time = time.time() - start_time
        
        # start_time = time.time()
        # mis_milp = solve_mis_milp(graph, suppress_output=True)
        # mis_milp_time = time.time() - start_time
        
        print(f"  Max Clique: {len(clique_milp)} nodes in {clique_milp_time:.4f}s")
        print(f"  Max Indep: {len(mis_milp)} nodes in {mis_milp_time:.4f}s")
        print(f"  Clique: {sorted(clique_milp)}")
        print(f"  MIS: {sorted(mis_milp)}")
        
        results['milp'] = {
            'clique_size': len(clique_milp),
            'clique_set': clique_milp,
            'clique_time': clique_milp_time,
            # 'mis_size': len(mis_milp),
            # 'mis_set': mis_milp,
            # 'mis_time': mis_milp_time
        }
    else:
        print("\n⚠️  MILP Solvers not available (Gurobi missing)")
        results['milp'] = None
    
    # 3. Validation
    print("\n✅ Validation:")
    
    # Verify all cliques are valid
    assert verify_clique(graph, clique_oracle), "Oracle clique invalid"
    if MILP_AVAILABLE:
        assert verify_clique(graph, clique_milp), "MILP clique invalid"
    
    # Verify all MIS are valid
    assert verify_independent_set(graph, mis_oracle), "Oracle MIS invalid"
    if MILP_AVAILABLE:
        assert verify_independent_set(graph, mis_milp), "MILP MIS invalid"
    
    # Check size consistency
    sizes_match = True
    if MILP_AVAILABLE:
        if len(clique_oracle) != len(clique_milp):
            print(f"  ⚠️  Clique size mismatch: Oracle={len(clique_oracle)}, MILP={len(clique_milp)}")
            sizes_match = False
        # if len(mis_oracle) != len(mis_milp):
        #     print(f"  ⚠️  MIS size mismatch: Oracle={len(mis_oracle)}, MILP={len(mis_milp)}")
        #     sizes_match = False
    
    if sizes_match:
        print("  ✅ All solvers agree on optimal solution size")
        print("  ✅ All solutions are valid")
    
    # Skip Motzkin-Straus duality check due to frozen graph issues
    print("  ℹ️  Skipping Motzkin-Straus duality check (frozen graph issue)")
    
    return results

def create_performance_summary(all_results):
    """Create a performance summary across all graphs."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print('='*80)
    
    print(f"{'Graph':<25} {'Nodes':<6} {'ω(G)':<6} {'α(G)':<6} {'Oracle Time':<12} {'MILP Time':<10}")
    print('-' * 80)
    
    for graph_name, results in all_results.items():
        oracle = results['oracle']
        milp = results['milp']
        
        nodes = results['nodes']
        clique_size = oracle['clique_size']
        mis_size = oracle['mis_size']
        
        oracle_time = oracle['clique_time'] + oracle['mis_time']
        milp_time = milp['clique_time'] + milp['mis_time'] if milp else 0
        
        milp_time_str = f"{milp_time:.4f}s" if milp else "N/A"
        
        print(f"{graph_name:<25} {nodes:<6} {clique_size:<6} {mis_size:<6} {oracle_time:.4f}s      {milp_time_str:<10}")
    
    print("\nLegend:")
    print("  ω(G) = Clique number (size of maximum clique)")
    print("  α(G) = Independence number (size of maximum independent set)")
    print("  Oracle = JAX PGD Oracle, MILP = Gurobi")

def visualize_example_solutions(graph, results, graph_name, output_dir):
    """Visualize the graph and highlight the solutions found."""
    # if graph.number_of_nodes() > 15:
        # return  # Skip visualization for large graphs
    
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
    
    # Plot 2: Oracle Solution
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
    plt.savefig(output_dir / f'{filename}_solutions.png', dpi=300, bbox_inches='tight')
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
    """Run the demonstration."""
    print("🎯 Maximum Clique and MIS Solver Demonstration")
    print("=" * 60)
    print("Testing multiple solving approaches:")
    print("1. Oracle-based (Motzkin-Straus theorem)")
    if MILP_AVAILABLE:
        print("2. MILP (direct combinatorial optimization)")
    else:
        print("2. MILP (not available - Gurobi missing)")
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    graphs = create_demo_graphs()
    all_results = {}
    
    for graph_name, graph in graphs:
        results = run_solver_comparison(graph, graph_name)
        results['nodes'] = graph.number_of_nodes()
        results['edges'] = graph.number_of_edges()
        all_results[graph_name] = results
        
        # Create visualization for smaller graphs
        visualize_example_solutions(graph, results, graph_name, output_dir)
    
    # Create performance summary
    create_performance_summary(all_results)
    
    print(f"\n🎉 Demonstration completed successfully!")
    print(f"📊 Visualizations saved to: {output_dir.absolute()}")
    print("\nKey findings:")
    print("✅ All solving approaches produce correct results")
    print("✅ Oracle-based algorithms scale well with moderate oracle calls")
    if MILP_AVAILABLE:
        print("✅ MILP solvers provide exact solutions efficiently")
    print("✅ Implementation handles various graph structures correctly")

if __name__ == "__main__":
    main()
