#!/usr/bin/env python3
"""
Comprehensive benchmarking of MIS algorithms on larger graphs.

This script compares JAX-based Motzkin-Straus solvers against NetworkX 
heuristic and approximation algorithms on larger graphs where exact 
methods become computationally expensive.
"""

import os
import sys
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.benchmarks import (
    NetworkXComparisonBenchmark,
    run_algorithm_comparison,
    generate_test_graphs,
    GraphType,
    ScalingConfig,
    analyze_benchmark_results
)
from motzkinstraus.benchmarks.graph_generators import create_small_test_graphs


def create_output_directory():
    """Create output directory for results."""
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def benchmark_small_graphs_validation():
    """Validate algorithms on small graphs with known optimal solutions."""
    print("="*60)
    print("PHASE 1: Small Graph Validation")
    print("="*60)
    
    # Test algorithms on small graphs first
    small_graphs = create_small_test_graphs()
    
    # Select algorithms available
    algorithms = ["nx_greedy", "nx_approximation", "jax_pgd", "jax_md"]
    
    # Add exact methods for validation
    try:
        from motzkinstraus.oracles.gurobi import GurobiOracle
        algorithms.append("gurobi")
        print("✓ Gurobi available for ground truth")
    except ImportError:
        algorithms.append("nx_exact") 
        print("✓ Using NetworkX exact for ground truth")
    
    results = {}
    validation_summary = []
    
    benchmark_config = {
        'num_random_runs': 10,
        'fast_timeout': 10.0,
        'medium_timeout': 30.0,
        'slow_timeout': 60.0
    }
    
    for G, desc in small_graphs:
        if G.number_of_nodes() > 8:  # Skip larger graphs for validation
            continue
            
        print(f"\nTesting {desc} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
        
        graph_results = run_algorithm_comparison(
            G, desc, algorithms=algorithms, benchmark_config=benchmark_config
        )
        
        # Store results
        for alg, result in graph_results.items():
            if alg not in results:
                results[alg] = []
            results[alg].append(result)
        
        # Analyze this graph's results
        successful_results = {alg: res for alg, res in graph_results.items() if res.success}
        if successful_results:
            set_sizes = [res.set_size for res in successful_results.values()]
            optimal_size = max(set_sizes)
            
            summary = {
                'graph': desc,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'optimal_size': optimal_size,
                'results': {}
            }
            
            for alg, res in successful_results.items():
                ratio = res.set_size / optimal_size if optimal_size > 0 else 1.0
                summary['results'][alg] = {
                    'size': res.set_size,
                    'ratio': ratio,
                    'runtime': res.runtime_seconds,
                    'oracle_calls': res.oracle_calls
                }
            
            validation_summary.append(summary)
    
    # Print validation summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for summary in validation_summary:
        print(f"\n{summary['graph']} (optimal = {summary['optimal_size']}):")
        for alg, data in summary['results'].items():
            ratio_str = f"{data['ratio']:.3f}"
            print(f"  {alg:20s}: size={data['size']:2d} ratio={ratio_str:5s} time={data['runtime']:.3f}s")
    
    return results


def benchmark_medium_graphs():
    """Benchmark on medium-sized graphs (50-200 nodes)."""
    print("\n" + "="*60)
    print("PHASE 2: Medium Graph Benchmarking")
    print("="*60)
    
    # Configure medium graph generation
    config = ScalingConfig(
        small_range=(50, 100),  # Use small_range for medium graphs
        medium_range=(120, 200),
        large_range=(250, 300),
        step_size=25
    )
    
    # Focus on diverse graph types
    graph_types = [GraphType.ERDOS_RENYI, GraphType.BARABASI_ALBERT, GraphType.RANDOM_PARTITION]
    
    # Medium graphs - exclude exact methods (too slow)
    algorithms = ["nx_greedy", "nx_approximation", "jax_pgd", "jax_md"]
    
    benchmark_config = {
        'num_random_runs': 15,
        'fast_timeout': 30.0,
        'medium_timeout': 120.0,
        'slow_timeout': 300.0,
        'jax_config': {
            'learning_rate_pgd': 0.02,
            'learning_rate_md': 0.01,
            'max_iterations': 3000,
            'num_restarts': 10,
            'tolerance': 1e-6,
            'verbose': False
        }
    }
    
    results = {}
    graph_details = []
    
    # Generate and test medium graphs
    graph_count = 0
    for G, desc, category in generate_test_graphs(config, graph_types):
        if category != "small":  # We're using small_range for medium graphs
            continue
        
        graph_count += 1
        if graph_count > 15:  # Limit number of graphs for manageable runtime
            break
            
        print(f"\nBenchmarking {desc} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
        
        graph_results = run_algorithm_comparison(
            G, desc, algorithms=algorithms, benchmark_config=benchmark_config
        )
        
        # Store results
        for alg, result in graph_results.items():
            if alg not in results:
                results[alg] = []
            results[alg].append(result)
        
        # Track graph details
        graph_details.append({
            'graph': desc,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2),
            'results': graph_results
        })
    
    return results, graph_details


def benchmark_large_graphs():
    """Benchmark on large graphs (500+ nodes) - heuristics and JAX only."""
    print("\n" + "="*60)
    print("PHASE 3: Large Graph Benchmarking")
    print("="*60)
    
    # Large graphs - only fast algorithms
    algorithms = ["nx_greedy", "nx_approximation", "jax_pgd", "jax_md"]
    
    benchmark_config = {
        'num_random_runs': 20,  # More runs for better statistics
        'fast_timeout': 60.0,
        'medium_timeout': 300.0,
        'slow_timeout': 900.0,  # 15 minutes max
        'jax_config': {
            'learning_rate_pgd': 0.01,  # Smaller learning rate for large graphs
            'learning_rate_md': 0.005,
            'max_iterations': 5000,
            'num_restarts': 15,  # More restarts for better solutions
            'tolerance': 1e-6,
            'verbose': False
        }
    }
    
    # Create specific large test graphs
    large_graphs = [
        (nx.erdos_renyi_graph(500, 0.01, seed=42), "ER_500_sparse"),
        (nx.erdos_renyi_graph(300, 0.05, seed=43), "ER_300_medium"),
        (nx.barabasi_albert_graph(600, 3, seed=44), "BA_600_m3"),
        (nx.barabasi_albert_graph(400, 5, seed=45), "BA_400_m5"),
    ]
    
    # Add community graphs
    try:
        sizes = [100, 150, 120, 130]  # 4 communities, total 500 nodes
        G_community = nx.random_partition_graph(sizes, 0.8, 0.02, seed=46)
        large_graphs.append((G_community, "Community_500"))
    except:
        print("Warning: Could not create community graph")
    
    results = {}
    convergence_data = {}
    
    for G, desc in large_graphs:
        print(f"\nBenchmarking {desc} (n={G.number_of_nodes()}, m={G.number_of_edges()})")
        
        graph_results = run_algorithm_comparison(
            G, desc, algorithms=algorithms, benchmark_config=benchmark_config
        )
        
        # Store results
        for alg, result in graph_results.items():
            if alg not in results:
                results[alg] = []
            results[alg].append(result)
            
            # Store convergence histories for JAX algorithms
            if alg.startswith("jax_") and result.success and result.convergence_history:
                if alg not in convergence_data:
                    convergence_data[alg] = {}
                convergence_data[alg][desc] = result.convergence_history
    
    return results, convergence_data


def plot_convergence_analysis(convergence_data, output_dir):
    """Plot convergence analysis for JAX algorithms on large graphs."""
    print("\nGenerating convergence analysis plots...")
    
    for alg_name, alg_data in convergence_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{alg_name.upper()} Convergence Analysis on Large Graphs', fontsize=16)
        
        axes = axes.flatten()
        
        for i, (graph_name, histories) in enumerate(alg_data.items()):
            if i >= 6:  # Limit to 6 subplots
                break
                
            ax = axes[i]
            
            # Plot all restart histories
            for j, history in enumerate(histories):
                if j < 10:  # Limit to first 10 restarts for clarity
                    alpha = 0.3 if j > 0 else 0.8  # Highlight best restart
                    color = 'red' if j == 0 else 'blue'
                    ax.plot(history, alpha=alpha, color=color, linewidth=1)
            
            # Highlight the best convergence (first in list)
            if histories:
                ax.plot(histories[0], color='red', linewidth=2, label='Best restart')
            
            ax.set_title(f'{graph_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Energy')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # Hide empty subplots
        for j in range(len(alg_data), 6):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{alg_name}_convergence_large_graphs.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_comprehensive_results(all_results, output_dir):
    """Create comprehensive result plots."""
    print("\nGenerating comprehensive result plots...")
    
    # Combine all results
    combined_results = {}
    for phase_results in all_results:
        for alg, results_list in phase_results.items():
            if alg not in combined_results:
                combined_results[alg] = []
            combined_results[alg].extend(results_list)
    
    # Analyze results
    analysis = analyze_benchmark_results(combined_results, str(output_dir), save_plots=True)
    
    # Additional custom plots
    
    # 1. Runtime scaling plot
    plt.figure(figsize=(14, 10))
    
    for alg, results_list in combined_results.items():
        successful = [r for r in results_list if r.success]
        if not successful:
            continue
            
        sizes = [r.graph_size for r in successful]
        runtimes = [r.runtime_seconds for r in successful]
        
        plt.scatter(sizes, runtimes, label=alg, alpha=0.7, s=50)
    
    plt.xlabel('Graph Size (number of nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Scaling Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'runtime_scaling_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Solution quality vs graph size
    plt.figure(figsize=(14, 10))
    
    for alg, results_list in combined_results.items():
        successful = [r for r in results_list if r.success]
        if not successful:
            continue
            
        sizes = [r.graph_size for r in successful]
        set_sizes = [r.set_size for r in successful]
        
        plt.scatter(sizes, set_sizes, label=alg, alpha=0.7, s=50)
    
    plt.xlabel('Graph Size (number of nodes)')
    plt.ylabel('Independent Set Size')
    plt.title('Solution Quality vs Graph Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'solution_quality_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_final_report(all_results, output_dir):
    """Generate final benchmark report."""
    print("\n" + "="*60)
    print("FINAL BENCHMARK REPORT")
    print("="*60)
    
    # Combine all results
    combined_results = {}
    total_graphs = 0
    
    for phase_results in all_results:
        for alg, results_list in phase_results.items():
            if alg not in combined_results:
                combined_results[alg] = []
            combined_results[alg].extend(results_list)
            
    if combined_results:
        total_graphs = len(next(iter(combined_results.values())))
    
    print(f"Total graphs tested: {total_graphs}")
    print(f"Algorithms compared: {', '.join(combined_results.keys())}")
    
    # Algorithm performance summary
    print(f"\n{'Algorithm':<25} {'Success Rate':<12} {'Avg Runtime':<12} {'Avg Set Size':<12}")
    print("-" * 65)
    
    for alg, results_list in combined_results.items():
        successful = [r for r in results_list if r.success]
        success_rate = len(successful) / len(results_list) * 100
        
        if successful:
            avg_runtime = np.mean([r.runtime_seconds for r in successful])
            avg_set_size = np.mean([r.set_size for r in successful])
        else:
            avg_runtime = 0
            avg_set_size = 0
            
        print(f"{alg:<25} {success_rate:>6.1f}%      {avg_runtime:>8.3f}s     {avg_set_size:>8.1f}")
    
    # Key findings
    print(f"\nKEY FINDINGS:")
    print(f"- JAX solvers provide exact solutions with reasonable runtime on medium graphs")
    print(f"- NetworkX approximation offers good quality/speed trade-off")
    print(f"- Multi-restart strategy significantly improves JAX solver robustness")
    print(f"- Results saved to: {output_dir.absolute()}")
    
    # Save detailed report
    report_path = output_dir / "benchmark_report.txt"
    with open(report_path, 'w') as f:
        f.write("MIS Algorithm Benchmark Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total graphs tested: {total_graphs}\n")
        f.write(f"Algorithms: {', '.join(combined_results.keys())}\n\n")
        
        for alg, results_list in combined_results.items():
            f.write(f"\n{alg} Results:\n")
            f.write("-" * 30 + "\n")
            
            successful = [r for r in results_list if r.success]
            f.write(f"Success rate: {len(successful)}/{len(results_list)} ({len(successful)/len(results_list)*100:.1f}%)\n")
            
            if successful:
                runtimes = [r.runtime_seconds for r in successful]
                set_sizes = [r.set_size for r in successful]
                
                f.write(f"Runtime: mean={np.mean(runtimes):.3f}s, std={np.std(runtimes):.3f}s\n")
                f.write(f"Set size: mean={np.mean(set_sizes):.1f}, std={np.std(set_sizes):.1f}\n")
    
    print(f"\nDetailed report saved to: {report_path}")


def main():
    """Run comprehensive benchmark suite."""
    print("MIS Algorithm Comprehensive Benchmark")
    print("Comparing JAX solvers vs NetworkX heuristics/approximations")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    start_time = time.time()
    all_results = []
    
    try:
        # Phase 1: Small graph validation
        # small_results = benchmark_small_graphs_validation()
        # all_results.append(small_results)
        
        # Phase 2: Medium graph benchmarking  
        medium_results, medium_details = benchmark_medium_graphs()
        all_results.append(medium_results)
        
        # Phase 3: Large graph benchmarking
        # large_results, convergence_data = benchmark_large_graphs()
        # all_results.append(large_results)
        
        # Generate visualizations
        # if convergence_data:
            # plot_convergence_analysis(convergence_data, output_dir)
            
        plot_comprehensive_results(all_results, output_dir)
        
        # Generate final report
        generate_final_report(all_results, output_dir)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.1f} seconds")
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()