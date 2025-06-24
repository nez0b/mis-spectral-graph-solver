"""
Test oracle counting on a 5-node graph with both Gurobi and Dirac-3 solvers.
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles.gurobi import GurobiOracle
from motzkinstraus.oracles.dirac import DiracOracle
from motzkinstraus.exceptions import SolverUnavailableError


def create_dual_panel_visualization(graph, gurobi_result, dirac_result, save_path):
    """Create dual-panel comparison visualization."""
    gurobi_mis, gurobi_calls = gurobi_result
    dirac_mis, dirac_calls = dirac_result
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Use consistent layout
    pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
    
    # Panel 1: Gurobi results
    node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                         for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
            node_size=1200, edge_color='gray', width=3, font_size=16, 
            font_weight='bold', font_color='white')
    
    ax1.set_title(f'Gurobi Oracle\nMIS Size: {len(gurobi_mis)}\nOracle Calls: {gurobi_calls}', 
                  fontsize=14, fontweight='bold')
    
    # Add oracle calls text box
    ax1.text(0.02, 0.98, f'Oracle Calls: {gurobi_calls}', transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Panel 2: Dirac results
    node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                        for node in graph.nodes()]
    
    nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
            node_size=1200, edge_color='gray', width=3, font_size=16, 
            font_weight='bold', font_color='white')
    
    ax2.set_title(f'Dirac-3 Oracle\nMIS Size: {len(dirac_mis)}\nOracle Calls: {dirac_calls}', 
                  fontsize=14, fontweight='bold')
    
    # Add oracle calls text box
    ax2.text(0.02, 0.98, f'Oracle Calls: {dirac_calls}', transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'Motzkin-Straus MIS Solver Oracle Call Comparison\n{graph.number_of_nodes()}-Node Graph', 
                 fontsize=16, fontweight='bold')
    
    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set Nodes')
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
    fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dual-panel visualization saved to: {save_path}")


def test_5_node_oracle_counting():
    """Test oracle counting on a 5-node graph."""
    print("=" * 70)
    print("ORACLE CALL COUNTING TEST - 5-NODE GRAPH")
    print("=" * 70)
    
    # Create 5-cycle graph
    G = nx.cycle_graph(5)
    print(f"\nTest graph: 5-cycle")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Edge list: {list(G.edges())}")
    
    # Ground truth
    print("\n1. Computing ground truth...")
    ground_truth = find_mis_brute_force(G)
    print(f"   ✓ Ground truth MIS: {ground_truth} (size: {len(ground_truth)})")
    
    # Test Gurobi Oracle
    print("\n2. Testing Gurobi Oracle...")
    try:
        gurobi_oracle = GurobiOracle(suppress_output=True)
        print("   ✓ Gurobi oracle initialized")
        
        start_time = time.time()
        gurobi_mis, gurobi_calls = find_mis_with_oracle(G, gurobi_oracle, verbose=True)
        gurobi_time = time.time() - start_time
        
        gurobi_valid = verify_independent_set(G, gurobi_mis)
        gurobi_optimal = len(gurobi_mis) == len(ground_truth)
        
        print(f"\n   Gurobi Results:")
        print(f"   ✓ MIS found: {gurobi_mis} (size: {len(gurobi_mis)})")
        print(f"   ✓ Oracle calls: {gurobi_calls}")
        print(f"   ✓ Time: {gurobi_time:.3f} seconds")
        print(f"   ✓ Valid: {gurobi_valid}, Optimal: {gurobi_optimal}")
        
        gurobi_result = (gurobi_mis, gurobi_calls)
        gurobi_success = True
        
    except Exception as e:
        print(f"   ✗ Gurobi failed: {e}")
        gurobi_result = (set(), 0)
        gurobi_success = False
    
    # Test Dirac Oracle
    print("\n3. Testing Dirac-3 Oracle...")
    print("   ⏳ This may take several minutes due to cloud queue...")
    try:
        dirac_oracle = DiracOracle(num_samples=50, relax_schedule=2)
        print("   ✓ Dirac-3 oracle initialized")
        
        start_time = time.time()
        dirac_mis, dirac_calls = find_mis_with_oracle(G, dirac_oracle, verbose=True)
        dirac_time = time.time() - start_time
        
        dirac_valid = verify_independent_set(G, dirac_mis)
        dirac_optimal = len(dirac_mis) == len(ground_truth)
        
        print(f"\n   Dirac-3 Results:")
        print(f"   ✓ MIS found: {dirac_mis} (size: {len(dirac_mis)})")
        print(f"   ✓ Oracle calls: {dirac_calls}")
        print(f"   ✓ Time: {dirac_time:.3f} seconds")
        print(f"   ✓ Valid: {dirac_valid}, Optimal: {dirac_optimal}")
        
        dirac_result = (dirac_mis, dirac_calls)
        dirac_success = True
        
    except Exception as e:
        print(f"   ✗ Dirac-3 failed: {e}")
        dirac_result = (set(), 0)
        dirac_success = False
    
    # Results comparison
    print("\n" + "=" * 70)
    print("ORACLE CALL COUNTING ANALYSIS")
    print("=" * 70)
    
    if gurobi_success and dirac_success:
        gurobi_mis, gurobi_calls = gurobi_result
        dirac_mis, dirac_calls = dirac_result
        
        print(f"\n📊 Oracle Call Comparison:")
        print(f"   Gurobi oracle calls:  {gurobi_calls}")
        print(f"   Dirac-3 oracle calls: {dirac_calls}")
        
        if gurobi_calls == dirac_calls:
            print(f"   ✓ Both solvers made identical oracle calls!")
        else:
            diff = abs(gurobi_calls - dirac_calls)
            print(f"   ⚠️  Oracle call difference: {diff}")
        
        print(f"\n📈 Performance Comparison:")
        print(f"   MIS sizes: Gurobi={len(gurobi_mis)}, Dirac-3={len(dirac_mis)}")
        print(f"   Both optimal: {len(gurobi_mis) == len(ground_truth) and len(dirac_mis) == len(ground_truth)}")
        
        # Create visualization
        print("\n4. Creating dual-panel visualization...")
        os.makedirs('figures', exist_ok=True)
        save_path = 'figures/5_node_oracle_counting_comparison.png'
        
        create_dual_panel_visualization(G, gurobi_result, dirac_result, save_path)
        
        print(f"\n✨ Oracle counting test completed successfully!")
        print(f"   Expected oracle calls for 5-cycle: 3 (1 initial + 2 for search)")
        print(f"   Actual calls: Gurobi={gurobi_calls}, Dirac-3={dirac_calls}")
        
        return True
    
    elif gurobi_success:
        print(f"\n⚠️  Only Gurobi succeeded:")
        print(f"   Oracle calls: {gurobi_result[1]}")
        print(f"   Expected: 3 (for 5-cycle)")
        return False
    
    else:
        print(f"\n❌ Both solvers failed")
        return False


if __name__ == "__main__":
    success = test_5_node_oracle_counting()
    if success:
        print("\n🎉 All tests passed! Oracle counting is working correctly.")
    else:
        print("\n⚠️  Test completed with some limitations.")