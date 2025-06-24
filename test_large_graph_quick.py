"""
Quick test of the large graph functionality without brute force.
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
import matplotlib.pyplot as plt
import os
from motzkinstraus.algorithms import find_mis_with_oracle, verify_independent_set
from motzkinstraus.oracles.gurobi import GurobiOracle
from motzkinstraus.oracles.dirac import DiracOracle

def create_dual_panel_visualization(graph, gurobi_result, dirac_result, save_path):
    """Create a dual-panel visualization."""
    gurobi_mis, gurobi_calls = gurobi_result
    dirac_mis, dirac_calls = dirac_result
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)
    
    # Panel 1: Gurobi
    node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                         for node in graph.nodes()]
    nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
            node_size=800, edge_color='gray', width=1.5, font_size=12, 
            font_weight='bold', font_color='white')
    ax1.set_title(f'Gurobi Oracle Result\\nMIS Size: {len(gurobi_mis)}, Oracle Calls: {gurobi_calls}', 
                  fontsize=14, fontweight='bold')
    ax1.text(0.02, 0.98, f'Oracle Calls: {gurobi_calls}', transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: Dirac
    node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                        for node in graph.nodes()]
    nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
            node_size=800, edge_color='gray', width=1.5, font_size=12, 
            font_weight='bold', font_color='white')
    ax2.set_title(f'Dirac-3 Oracle Result\\nMIS Size: {len(dirac_mis)}, Oracle Calls: {dirac_calls}', 
                  fontsize=14, fontweight='bold')
    ax2.text(0.02, 0.98, f'Oracle Calls: {dirac_calls}', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Motzkin-Straus MIS Solver Comparison', fontsize=16, fontweight='bold')
    
    # Legend
    red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set')
    blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
    fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

# Test with 8-node graph (manageable size)
G = nx.cycle_graph(8)
G.add_edges_from([(0, 4), (2, 6)])  # Add some chords

print(f"Testing on {G.number_of_nodes()}-node graph with {G.number_of_edges()} edges")

# Test Gurobi
gurobi_oracle = GurobiOracle(suppress_output=True)
gurobi_mis, gurobi_calls = find_mis_with_oracle(G, gurobi_oracle)
print(f"Gurobi: MIS={gurobi_mis}, size={len(gurobi_mis)}, calls={gurobi_calls}")

# Test Dirac (quick version)
print("Testing Dirac (this may take 30-60 seconds)...")
dirac_oracle = DiracOracle(num_samples=50, relax_schedule=1)
dirac_mis, dirac_calls = find_mis_with_oracle(G, dirac_oracle)
print(f"Dirac-3: MIS={dirac_mis}, size={len(dirac_mis)}, calls={dirac_calls}")

# Verify both are valid
gurobi_valid = verify_independent_set(G, gurobi_mis)
dirac_valid = verify_independent_set(G, dirac_mis)
print(f"Valid solutions: Gurobi={gurobi_valid}, Dirac={dirac_valid}")

# Create visualization
os.makedirs('figures', exist_ok=True)
create_dual_panel_visualization(G, 
                               (gurobi_mis, gurobi_calls),
                               (dirac_mis, dirac_calls),
                               'figures/test_comparison.png')

print("Test completed successfully!")