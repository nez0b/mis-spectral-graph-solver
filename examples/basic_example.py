"""
Basic example demonstrating the Motzkin-Straus MIS solver.

This example shows how to use the package to find Maximum Independent Sets
using the oracle-based algorithm with different solvers.
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles import get_available_oracles
from motzkinstraus.exceptions import SolverUnavailableError


def visualize_mis_solution(graph, mis_nodes, title="Graph with MIS highlighted", save_path=None):
    """Visualize a graph with the MIS nodes highlighted."""
    pos = nx.spring_layout(graph, seed=42)
    
    node_colors = []
    for node in graph.nodes():
        if node in mis_nodes:
            node_colors.append('lightblue')  # MIS nodes
        else:
            node_colors.append('lightgray')  # Other nodes
    
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, 
            node_size=800, edge_color='gray', width=1.5, 
            font_size=12, font_weight='bold')
    plt.title(title, size=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def test_on_cycle_graph():
    """Test the algorithm on a 5-cycle graph."""
    print("=== Testing on 5-Cycle Graph ===")
    
    # Create 5-cycle: 0-1-2-3-4-0
    G = nx.cycle_graph(5)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Edges: {list(G.edges())}")
    
    # Find MIS using brute force (ground truth)
    brute_force_mis = find_mis_brute_force(G)
    print(f"\nBrute force MIS: {brute_force_mis} (size: {len(brute_force_mis)})")
    
    # Test with available oracles
    available_oracles = get_available_oracles()
    print(f"\nAvailable oracles: {[oracle.__name__ for oracle in available_oracles]}")
    
    for oracle_class in available_oracles:
        try:
            oracle_name = oracle_class.__name__.replace('Oracle', '')
            print(f"\n--- Testing with {oracle_name} oracle ---")
            
            if oracle_name == "Gurobi":
                oracle = oracle_class(suppress_output=True)
            elif oracle_name == "Dirac":
                oracle = oracle_class(num_samples=50, relax_schedule=1)
            else:
                oracle = oracle_class()
            
            # Find MIS using oracle
            oracle_mis = find_mis_with_oracle(G, oracle, verbose=True)
            
            # Verify results
            is_independent = verify_independent_set(G, oracle_mis)
            correct_size = len(oracle_mis) == len(brute_force_mis)
            
            print(f"\n{oracle_name} results:")
            print(f"  MIS found: {oracle_mis} (size: {len(oracle_mis)})")
            print(f"  Is independent set: {is_independent}")
            print(f"  Correct size: {correct_size}")
            
            if is_independent and correct_size:
                print(f"  ✓ {oracle_name} oracle succeeded!")
                
                # Save visualization
                save_path = f"../figures/cycle_graph_{oracle_name.lower()}_mis.png"
                visualize_mis_solution(G, oracle_mis, 
                                     f"5-Cycle MIS using {oracle_name} Oracle",
                                     save_path)
            else:
                print(f"  ✗ {oracle_name} oracle failed!")
                
        except SolverUnavailableError as e:
            print(f"  ⚠️  {oracle_name} solver not available: {e}")
        except Exception as e:
            print(f"  ✗ {oracle_name} oracle failed with error: {e}")


def test_on_petersen_graph():
    """Test the algorithm on the Petersen graph."""
    print("\n" + "="*50)
    print("=== Testing on Petersen Graph ===")
    
    # Create Petersen graph
    G = nx.petersen_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Find MIS using brute force (ground truth) - only if small enough
    if G.number_of_nodes() <= 10:
        brute_force_mis = find_mis_brute_force(G)
        print(f"Brute force MIS size: {len(brute_force_mis)}")
    else:
        print("Graph too large for brute force, skipping ground truth")
        brute_force_mis = None
    
    # Test with available oracles
    available_oracles = get_available_oracles()
    
    for oracle_class in available_oracles:
        try:
            oracle_name = oracle_class.__name__.replace('Oracle', '')
            print(f"\n--- Testing with {oracle_name} oracle ---")
            
            if oracle_name == "Gurobi":
                oracle = oracle_class(suppress_output=True)
            elif oracle_name == "Dirac":
                oracle = oracle_class(num_samples=100, relax_schedule=2)
            else:
                oracle = oracle_class()
            
            # Find MIS using oracle
            oracle_mis = find_mis_with_oracle(G, oracle, verbose=False)
            
            # Verify results
            is_independent = verify_independent_set(G, oracle_mis)
            
            print(f"\n{oracle_name} results:")
            print(f"  MIS found: {oracle_mis} (size: {len(oracle_mis)})")
            print(f"  Is independent set: {is_independent}")
            
            if brute_force_mis:
                correct_size = len(oracle_mis) == len(brute_force_mis)
                print(f"  Correct size: {correct_size}")
                
                if is_independent and correct_size:
                    print(f"  ✓ {oracle_name} oracle succeeded!")
                else:
                    print(f"  ✗ {oracle_name} oracle failed!")
            else:
                if is_independent:
                    print(f"  ✓ {oracle_name} oracle found valid independent set!")
                else:
                    print(f"  ✗ {oracle_name} oracle failed!")
            
            # Save visualization
            save_path = f"../figures/petersen_graph_{oracle_name.lower()}_mis.png"
            visualize_mis_solution(G, oracle_mis, 
                                 f"Petersen Graph MIS using {oracle_name} Oracle",
                                 save_path)
                
        except SolverUnavailableError as e:
            print(f"  ⚠️  {oracle_name} solver not available: {e}")
        except Exception as e:
            print(f"  ✗ {oracle_name} oracle failed with error: {e}")


def main():
    """Run the basic examples."""
    print("Motzkin-Straus Maximum Independent Set Solver")
    print("=" * 50)
    
    # Ensure figures directory exists
    os.makedirs("../figures", exist_ok=True)
    
    # Run tests
    test_on_cycle_graph()
    test_on_petersen_graph()
    
    print("\n" + "="*50)
    print("Examples complete! Check the figures/ directory for visualizations.")


if __name__ == "__main__":
    main()