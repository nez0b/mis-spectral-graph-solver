"""
Comprehensive solver comparison example.

This example compares different oracle implementations on various graph types
to demonstrate their performance and accuracy characteristics.
"""

import networkx as nx
import numpy as np
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles import get_available_oracles
from motzkinstraus.exceptions import SolverUnavailableError


def create_test_graphs():
    """Create a collection of test graphs with known properties."""
    graphs = {
        "Triangle (K3)": nx.complete_graph(3),
        "4-Cycle": nx.cycle_graph(4),
        "5-Cycle": nx.cycle_graph(5),
        "Path-5": nx.path_graph(5),
        "Complete-4 (K4)": nx.complete_graph(4),
        "Star-5": nx.star_graph(5),
        "Petersen": nx.petersen_graph(),
    }
    
    # Add expected MIS sizes (if known)
    expected_mis_sizes = {
        "Triangle (K3)": 1,  # Any single vertex
        "4-Cycle": 2,        # Opposite vertices
        "5-Cycle": 2,        # Non-adjacent vertices
        "Path-5": 3,         # Every other vertex
        "Complete-4 (K4)": 1, # Any single vertex
        "Star-5": 5,         # All leaf nodes
        "Petersen": 4,       # Known result
    }
    
    return graphs, expected_mis_sizes


def run_solver_comparison():
    """Compare all available solvers on test graphs."""
    print("=== Solver Comparison on Test Graphs ===")
    
    # Get test graphs
    graphs, expected_sizes = create_test_graphs()
    
    # Get available oracles
    available_oracles = get_available_oracles()
    
    if not available_oracles:
        print("No oracles available for testing!")
        return
    
    print(f"Available solvers: {[cls.__name__ for cls in available_oracles]}")
    print()
    
    # Initialize oracles
    oracles = {}
    for oracle_class in available_oracles:
        try:
            oracle_name = oracle_class.__name__.replace('Oracle', '')
            
            if oracle_name == "Gurobi":
                oracle = oracle_class(suppress_output=True)
            elif oracle_name == "Dirac":
                oracle = oracle_class(num_samples=100, relax_schedule=2)
            else:
                oracle = oracle_class()
            
            oracles[oracle_name] = oracle
            print(f"✓ {oracle_name} oracle initialized")
            
        except SolverUnavailableError as e:
            print(f"✗ {oracle_name} oracle not available: {e}")
        except Exception as e:
            print(f"✗ {oracle_name} oracle initialization failed: {e}")
    
    if not oracles:
        print("No oracles could be initialized!")
        return
    
    print()
    
    # Test each graph
    results = {}
    
    for graph_name, graph in graphs.items():
        print(f"--- Testing {graph_name} ---")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        # Get ground truth if graph is small enough
        ground_truth = None
        if graph.number_of_nodes() <= 12:  # Limit for brute force
            try:
                start_time = time.time()
                ground_truth = find_mis_brute_force(graph)
                brute_force_time = time.time() - start_time
                print(f"Brute force MIS: size {len(ground_truth)} (time: {brute_force_time:.3f}s)")
            except Exception as e:
                print(f"Brute force failed: {e}")
        else:
            print("Graph too large for brute force")
        
        expected_size = expected_sizes.get(graph_name)
        if expected_size:
            print(f"Expected MIS size: {expected_size}")
        
        # Test each oracle
        graph_results = {}
        
        for oracle_name, oracle in oracles.items():
            try:
                print(f"\nTesting {oracle_name}...")
                
                start_time = time.time()
                mis_result = find_mis_with_oracle(graph, oracle, verbose=False)
                solve_time = time.time() - start_time
                
                # Verify result
                is_valid = verify_independent_set(graph, mis_result)
                size_correct = (not expected_size) or (len(mis_result) == expected_size)
                brute_match = (not ground_truth) or (len(mis_result) == len(ground_truth))
                
                graph_results[oracle_name] = {
                    'mis': mis_result,
                    'size': len(mis_result),
                    'time': solve_time,
                    'valid': is_valid,
                    'size_correct': size_correct,
                    'brute_match': brute_match,
                    'success': is_valid and size_correct and brute_match
                }
                
                status = "✓" if graph_results[oracle_name]['success'] else "✗"
                print(f"  {status} MIS size: {len(mis_result)}, Time: {solve_time:.3f}s, Valid: {is_valid}")
                
            except Exception as e:
                print(f"  ✗ {oracle_name} failed: {e}")
                graph_results[oracle_name] = {
                    'mis': set(),
                    'size': 0,
                    'time': float('inf'),
                    'valid': False,
                    'size_correct': False,
                    'brute_match': False,
                    'success': False
                }
        
        results[graph_name] = graph_results
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Success rate by solver
    print("\nSuccess rate by solver:")
    for oracle_name in oracles.keys():
        successes = sum(1 for graph_results in results.values() 
                       if graph_results.get(oracle_name, {}).get('success', False))
        total = len(results)
        success_rate = successes / total * 100 if total > 0 else 0
        print(f"  {oracle_name}: {successes}/{total} ({success_rate:.1f}%)")
    
    # Average solve times
    print("\nAverage solve times:")
    for oracle_name in oracles.keys():
        times = [graph_results.get(oracle_name, {}).get('time', float('inf')) 
                for graph_results in results.values()]
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = np.mean(valid_times)
            print(f"  {oracle_name}: {avg_time:.3f}s")
        else:
            print(f"  {oracle_name}: N/A (no successful runs)")
    
    # Detailed results table
    print("\nDetailed results:")
    print(f"{'Graph':<15} {'Solver':<10} {'Size':<6} {'Time':<8} {'Valid':<7} {'Success'}")
    print("-" * 60)
    
    for graph_name, graph_results in results.items():
        for oracle_name, result in graph_results.items():
            size = result['size']
            time_str = f"{result['time']:.3f}s" if result['time'] != float('inf') else "FAIL"
            valid = "Yes" if result['valid'] else "No"
            success = "✓" if result['success'] else "✗"
            
            print(f"{graph_name:<15} {oracle_name:<10} {size:<6} {time_str:<8} {valid:<7} {success}")


if __name__ == "__main__":
    run_solver_comparison()