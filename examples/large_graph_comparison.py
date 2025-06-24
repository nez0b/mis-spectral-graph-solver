"""
Large graph comparison example for Motzkin-Straus MIS solver.

This example demonstrates the solver on a 12-node graph, comparing Gurobi 
and Dirac-3 performance with oracle call counting and dual-panel visualization.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles.gurobi import GurobiOracle
from motzkinstraus.oracles.dirac import DiracOracle
from motzkinstraus.exceptions import SolverUnavailableError


def create_test_graph_12_nodes():
    """Create an interesting 12-node test graph."""
    # Create a random graph with controlled properties
    np.random.seed(42)  # For reproducibility
    
    # Start with a cycle and add some extra edges to make it interesting
    G = nx.cycle_graph(12)
    
    # Add some chord edges to create a more complex structure
    chords = [(0, 6), (2, 8), (4, 10), (1, 7), (3, 9), (5, 11)]
    G.add_edges_from(chords)
    
    # Add a few more random edges
    additional_edges = [(0, 3), (2, 5), (7, 10), (8, 11)]
    G.add_edges_from(additional_edges)
    
    print(f"Created 12-node test graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G


def create_dual_panel_visualization(graph, gurobi_result, dirac_result, save_path):
    """
    Create a dual-panel visualization comparing Gurobi and Dirac results.
    
    Args:
        graph: The input graph
        gurobi_result: Tuple of (mis_set, oracle_calls) from Gurobi
        dirac_result: Tuple of (mis_set, oracle_calls) from Dirac
        save_path: Where to save the visualization
    """
    gurobi_mis, gurobi_calls = gurobi_result
    dirac_mis, dirac_calls = dirac_result
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use consistent layout for both panels
    pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)
    
    # Panel 1: Gurobi results
    node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                         for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
            node_size=800, edge_color='gray', width=1.5, font_size=12, 
            font_weight='bold', font_color='white')
    
    ax1.set_title(f'Gurobi Oracle Result\nMIS Size: {len(gurobi_mis)}, Oracle Calls: {gurobi_calls}', 
                  fontsize=14, fontweight='bold')
    ax1.text(0.02, 0.98, f'Oracle Calls: {gurobi_calls}', transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Dirac results
    node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                        for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
            node_size=800, edge_color='gray', width=1.5, font_size=12, 
            font_weight='bold', font_color='white')
    
    ax2.set_title(f'Dirac-3 Oracle Result\nMIS Size: {len(dirac_mis)}, Oracle Calls: {dirac_calls}', 
                  fontsize=14, fontweight='bold')
    ax2.text(0.02, 0.98, f'Oracle Calls: {dirac_calls}', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle('Motzkin-Straus MIS Solver Comparison (12-Node Graph)', 
                 fontsize=16, fontweight='bold')
    
    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set')
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
    fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()


def run_large_graph_test():
    """Run the complete large graph test."""
    print("=" * 70)
    print("MOTZKIN-STRAUS MIS SOLVER - LARGE GRAPH COMPARISON")
    print("=" * 70)
    
    # Create test graph
    G = create_test_graph_12_nodes()
    
    # Get ground truth using brute force (might be slow)
    print("\n1. Computing ground truth using brute force...")
    start_time = time.time()
    try:
        ground_truth = find_mis_brute_force(G)
        brute_force_time = time.time() - start_time
        print(f"   ✓ Ground truth MIS: {ground_truth}")
        print(f"   ✓ Size: {len(ground_truth)}")
        print(f"   ✓ Time: {brute_force_time:.3f} seconds")
    except Exception as e:
        print(f"   ⚠️  Brute force too slow/failed: {e}")
        ground_truth = None
        brute_force_time = float('inf')
    
    results = {}
    
    # Test Gurobi Oracle
    print("\n2. Testing Gurobi Oracle...")
    try:
        gurobi_oracle = GurobiOracle(suppress_output=True)
        print("   ✓ Gurobi oracle initialized")
        
        start_time = time.time()
        gurobi_mis, gurobi_calls = find_mis_with_oracle(G, gurobi_oracle, verbose=False)
        gurobi_time = time.time() - start_time
        
        gurobi_valid = verify_independent_set(G, gurobi_mis)
        
        results['Gurobi'] = {
            'mis': gurobi_mis,
            'calls': gurobi_calls,
            'time': gurobi_time,
            'valid': gurobi_valid,
            'success': True
        }
        
        print(f"   ✓ MIS found: {gurobi_mis}")
        print(f"   ✓ Size: {len(gurobi_mis)}")
        print(f"   ✓ Oracle calls: {gurobi_calls}")
        print(f"   ✓ Time: {gurobi_time:.3f} seconds")
        print(f"   ✓ Valid independent set: {gurobi_valid}")
        
    except SolverUnavailableError as e:
        print(f"   ✗ Gurobi not available: {e}")
        results['Gurobi'] = {'success': False}
    except Exception as e:
        print(f"   ✗ Gurobi failed: {e}")
        results['Gurobi'] = {'success': False}
    
    # Test Dirac Oracle
    print("\n3. Testing Dirac-3 Oracle...")
    try:
        dirac_oracle = DiracOracle(num_samples=100, relax_schedule=2)
        print("   ✓ Dirac-3 oracle initialized")
        print("   ⏳ This may take 2-5 minutes due to cloud processing...")
        
        start_time = time.time()
        dirac_mis, dirac_calls = find_mis_with_oracle(G, dirac_oracle, verbose=False)
        dirac_time = time.time() - start_time
        
        dirac_valid = verify_independent_set(G, dirac_mis)
        
        results['Dirac-3'] = {
            'mis': dirac_mis,
            'calls': dirac_calls,
            'time': dirac_time,
            'valid': dirac_valid,
            'success': True
        }
        
        print(f"   ✓ MIS found: {dirac_mis}")
        print(f"   ✓ Size: {len(dirac_mis)}")
        print(f"   ✓ Oracle calls: {dirac_calls}")
        print(f"   ✓ Time: {dirac_time:.3f} seconds")
        print(f"   ✓ Valid independent set: {dirac_valid}")
        
    except SolverUnavailableError as e:
        print(f"   ✗ Dirac-3 not available: {e}")
        results['Dirac-3'] = {'success': False}
    except Exception as e:
        print(f"   ✗ Dirac-3 failed: {e}")
        results['Dirac-3'] = {'success': False}
    
    # Analysis and Comparison
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) >= 2:
        print("\n📊 Performance Comparison:")
        
        # Compare solutions
        gurobi_result = results.get('Gurobi', {})
        dirac_result = results.get('Dirac-3', {})
        
        if gurobi_result.get('success') and dirac_result.get('success'):
            print(f"   Solution sizes: Gurobi={len(gurobi_result['mis'])}, "
                  f"Dirac-3={len(dirac_result['mis'])}")
            print(f"   Oracle calls: Gurobi={gurobi_result['calls']}, "
                  f"Dirac-3={dirac_result['calls']}")
            print(f"   Times: Gurobi={gurobi_result['time']:.3f}s, "
                  f"Dirac-3={dirac_result['time']:.3f}s")
            
            if ground_truth:
                gurobi_optimal = len(gurobi_result['mis']) == len(ground_truth)
                dirac_optimal = len(dirac_result['mis']) == len(ground_truth)
                print(f"   Optimal solutions: Gurobi={gurobi_optimal}, Dirac-3={dirac_optimal}")
            
            # Create visualization
            print("\n4. Creating dual-panel visualization...")
            save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 
                                   'large_graph_comparison.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            create_dual_panel_visualization(
                G, 
                (gurobi_result['mis'], gurobi_result['calls']),
                (dirac_result['mis'], dirac_result['calls']),
                save_path
            )
    
    print("\n✨ Large graph comparison complete!")
    
    return results


if __name__ == "__main__":
    results = run_large_graph_test()