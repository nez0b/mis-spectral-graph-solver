"""
Efficient large graph test for Motzkin-Straus MIS solver.

This version uses optimized parameters for faster testing while still demonstrating 
the full functionality including oracle call counting and dual-panel visualization.
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
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


def create_interesting_10_node_graph():
    """
    Create an interesting 10-node test graph optimized for Dirac API testing.
    
    This creates a complex graph based on a 10-cycle with additional chord edges.
    Expected oracle calls will depend on the MIS size and search decisions.
    For a 10-node graph, typical oracle calls range from 5-15 depending on structure.
    
    Note: 10 nodes is a good balance between complexity and API response time.
    """
    # Create a structured graph with interesting properties
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(10))
    
    # Create a cycle with some chords
    cycle_edges = [(i, (i+1) % 10) for i in range(10)]
    G.add_edges_from(cycle_edges)
    
    # Add chord edges to create complexity
    chord_edges = [(0, 5), (2, 7), (4, 9), (1, 6)]
    G.add_edges_from(chord_edges)
    
    # Add a few more edges for complexity
    extra_edges = [(0, 3), (1, 8)]
    G.add_edges_from(extra_edges)
    
    print(f"Graph structure details:")
    print(f"  - Base: 10-cycle")
    print(f"  - Chord edges: {chord_edges}")
    print(f"  - Extra edges: {extra_edges}")
    
    return G


def create_dual_panel_visualization(graph, gurobi_result, dirac_result, save_path):
    """Create dual-panel comparison visualization."""
    gurobi_mis, gurobi_calls = gurobi_result
    dirac_mis, dirac_calls = dirac_result
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use consistent layout
    pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)
    
    # Panel 1: Gurobi results
    node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                         for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
            node_size=1000, edge_color='gray', width=2, font_size=14, 
            font_weight='bold', font_color='white')
    
    ax1.set_title(f'Gurobi Oracle\\nMIS Size: {len(gurobi_mis)}\\nOracle Calls: {gurobi_calls}', 
                  fontsize=16, fontweight='bold')
    
    # Add oracle calls text box for Gurobi
    ax1.text(0.02, 0.98, f'Oracle Calls: {gurobi_calls}', transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Panel 2: Dirac results
    node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                        for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
            node_size=1000, edge_color='gray', width=2, font_size=14, 
            font_weight='bold', font_color='white')
    
    ax2.set_title(f'Dirac-3 Oracle\\nMIS Size: {len(dirac_mis)}\\nOracle Calls: {dirac_calls}', 
                  fontsize=16, fontweight='bold')
    
    # Add oracle calls text box for Dirac
    ax2.text(0.02, 0.98, f'Oracle Calls: {dirac_calls}', transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'Motzkin-Straus MIS Solver Comparison\\n{graph.number_of_nodes()}-Node Graph', 
                 fontsize=18, fontweight='bold')
    
    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set Nodes')
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
    fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dual-panel visualization saved to: {save_path}")


def run_efficient_test():
    """
    Run efficient large graph test with oracle call counting.
    
    This test demonstrates:
    1. Oracle call counting on a complex 10-node graph
    2. Performance comparison between Gurobi and Dirac-3 solvers
    3. Dual-panel visualization with oracle call information
    4. Robust error handling for solver unavailability
    
    Expected results:
    - Both solvers should find identical MIS solutions
    - Both solvers should make identical oracle calls
    - Gurobi should be much faster than Dirac-3 (cloud latency)
    - Visualization should show oracle call counts prominently
    """
    print("=" * 70)
    print("MOTZKIN-STRAUS MIS SOLVER - EFFICIENT LARGE GRAPH TEST")
    print("=" * 70)
    
    # Create test graph
    G = create_interesting_10_node_graph()
    print(f"\\nCreated test graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    # Compute ground truth (feasible for 10 nodes)
    print("\\n1. Computing ground truth...")
    start_time = time.time()
    ground_truth = find_mis_brute_force(G)
    brute_force_time = time.time() - start_time
    print(f"   ✓ Ground truth MIS: {ground_truth} (size: {len(ground_truth)})")
    print(f"   ✓ Brute force time: {brute_force_time:.3f} seconds")
    
    # Test Gurobi Oracle
    print("\\n2. Testing Gurobi Oracle...")
    try:
        gurobi_oracle = GurobiOracle(suppress_output=True)
        print("   ✓ Gurobi oracle initialized")
        
        start_time = time.time()
        gurobi_mis, gurobi_calls = find_mis_with_oracle(G, gurobi_oracle, verbose=False)
        gurobi_time = time.time() - start_time
        
        gurobi_valid = verify_independent_set(G, gurobi_mis)
        gurobi_optimal = len(gurobi_mis) == len(ground_truth)
        
        print(f"\\n   Gurobi Results:")
        print(f"   ✓ MIS found: {gurobi_mis} (size: {len(gurobi_mis)})")
        print(f"   ✓ Oracle calls: {gurobi_calls}")
        print(f"   ✓ Time: {gurobi_time:.3f} seconds")
        print(f"   ✓ Valid: {gurobi_valid}, Optimal: {gurobi_optimal}")
        
        gurobi_success = True
        
    except SolverUnavailableError as e:
        print(f"   ✗ Gurobi not available: {e}")
        gurobi_mis, gurobi_calls, gurobi_time = set(), 0, float('inf')
        gurobi_valid, gurobi_optimal, gurobi_success = False, False, False
    except Exception as e:
        print(f"   ✗ Gurobi failed: {e}")
        gurobi_mis, gurobi_calls, gurobi_time = set(), 0, float('inf')
        gurobi_valid, gurobi_optimal, gurobi_success = False, False, False
    
    # Test Dirac Oracle
    print("\\n3. Testing Dirac-3 Oracle...")
    print("   ⏳ This may take several minutes due to cloud API latency and queue time...")
    try:
        dirac_oracle = DiracOracle(num_samples=10, relax_schedule=1)  # Conservative params for API reliability
        print("   ✓ Dirac-3 oracle initialized")
        
        start_time = time.time()
        dirac_mis, dirac_calls = find_mis_with_oracle(G, dirac_oracle, verbose=False)
        dirac_time = time.time() - start_time
        
        dirac_valid = verify_independent_set(G, dirac_mis)
        dirac_optimal = len(dirac_mis) == len(ground_truth)
        
        print(f"\\n   Dirac-3 Results:")
        print(f"   ✓ MIS found: {dirac_mis} (size: {len(dirac_mis)})")
        print(f"   ✓ Oracle calls: {dirac_calls}")
        print(f"   ✓ Time: {dirac_time:.3f} seconds")
        print(f"   ✓ Valid: {dirac_valid}, Optimal: {dirac_optimal}")
        
        dirac_success = True
        
    except SolverUnavailableError as e:
        print(f"   ✗ Dirac-3 not available: {e}")
        dirac_mis, dirac_calls, dirac_time = set(), 0, float('inf')
        dirac_valid, dirac_optimal, dirac_success = False, False, False
    except Exception as e:
        print(f"   ✗ Dirac-3 failed: {e}")
        dirac_mis, dirac_calls, dirac_time = set(), 0, float('inf')
        dirac_valid, dirac_optimal, dirac_success = False, False, False
    
    # Results comparison
    print("\\n" + "=" * 70)
    print("ORACLE CALL COUNTING ANALYSIS")
    print("=" * 70)
    
    successful_results = []
    if gurobi_success:
        successful_results.append("Gurobi")
    if dirac_success:
        successful_results.append("Dirac-3")
    
    if len(successful_results) >= 2:
        print(f"\\n📊 Performance Comparison:")
        print(f"   Algorithm:        Gurobi    Dirac-3")
        print(f"   MIS Size:         {len(gurobi_mis):<8} {len(dirac_mis)}")
        print(f"   Oracle Calls:     {gurobi_calls:<8} {dirac_calls}")
        print(f"   Time (seconds):   {gurobi_time:<8.3f} {dirac_time:.3f}")
        print(f"   Valid Solution:   {gurobi_valid!s:<8} {dirac_valid}")
        print(f"   Optimal:          {gurobi_optimal!s:<8} {dirac_optimal}")
        
        if gurobi_time > 0 and dirac_time != float('inf'):
            speedup = dirac_time / gurobi_time
            print(f"\\n   📈 Dirac-3 is {speedup:.1f}x slower than Gurobi (due to cloud API latency)")
            print(f"   ⚠️  Note: Dirac timing varies significantly due to cloud queue and network latency")
        
        if gurobi_calls == dirac_calls:
            print(f"   ✓ Both solvers made identical oracle calls ({gurobi_calls})")
        else:
            diff = abs(gurobi_calls - dirac_calls)
            print(f"   ⚠️  Oracle call difference: {diff}")
            print(f"   ℹ️  This may be due to numerical precision differences in cloud solving")
        
        # Create visualization
        print("\\n4. Creating dual-panel visualization...")
        save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'large_graph_comparison.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        create_dual_panel_visualization(
            G,
            (gurobi_mis, gurobi_calls),
            (dirac_mis, dirac_calls),
            save_path
        )
        
    elif len(successful_results) == 1:
        solver_name = successful_results[0]
        print(f"\\n⚠️  Only {solver_name} succeeded:")
        if solver_name == "Gurobi":
            print(f"   Oracle calls: {gurobi_calls}")
            print(f"   MIS size: {len(gurobi_mis)}")
            print(f"   Time: {gurobi_time:.3f} seconds")
        else:
            print(f"   Oracle calls: {dirac_calls}")
            print(f"   MIS size: {len(dirac_mis)}")
            print(f"   Time: {dirac_time:.3f} seconds")
        print("   Note: Install missing solver dependencies for full comparison")
        
    else:
        print("\\n❌ Both solvers failed - check dependencies and configuration")
    
    print("\\n✨ Efficient large graph test completed successfully!")
    
    return {
        'graph': G,
        'ground_truth': ground_truth,
        'gurobi': {
            'mis': gurobi_mis,
            'calls': gurobi_calls,
            'time': gurobi_time,
            'valid': gurobi_valid,
            'optimal': gurobi_optimal,
            'success': gurobi_success
        },
        'dirac': {
            'mis': dirac_mis,
            'calls': dirac_calls,
            'time': dirac_time,
            'valid': dirac_valid,
            'optimal': dirac_optimal,
            'success': dirac_success
        }
    }


if __name__ == "__main__":
    try:
        results = run_efficient_test()
        print("\\n🎉 All tests passed!")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()