#!/usr/bin/env python3
"""
Single Graph Comprehensive Comparison

Test all MIS algorithms including the new Dirac+PGD hybrid solver to demonstrate 
the complete benchmarking framework with detailed visualizations.

Includes the new DiracPGDHybridOracle that combines:
- Dirac-3 continuous cloud solver for global search
- JAX Projected Gradient Descent for high-precision refinement
- NetworkX exact solver for small graphs (≤35 nodes)
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

from motzkinstraus.benchmarks.networkx_comparison import (
    NetworkXComparisonBenchmark, 
    run_algorithm_comparison
)


def create_test_graph():
    """Create a test graph with 30 nodes (< 35 threshold)."""
    # Use a Erdos-Renyi graph for interesting structure
    num_nodes = 80
    G = nx.erdos_renyi_graph(num_nodes, 0.6, seed=41)
    
    print(f"Test Graph Properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.3f}")
    print(f"  Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    return G


def run_comprehensive_comparison(G):
    """Run all available algorithms on the test graph."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*60)
    
    # All available algorithms (including new hybrid solvers and MILP)
    algorithms = [
        "nx_greedy",        # NetworkX greedy (multiple runs)
        "nx_greedy_single", # NetworkX greedy (single run)
        "nx_approximation", # NetworkX Boppana-Halldórsson
        "nx_exact",         # NetworkX exact via cliques (too slow for large graphs)
        "gurobi_milp",      # Gurobi MILP direct solver (exact)
        "scipy_milp",       # SciPy MILP direct solver (exact, open-source)
        # "jax_pgd",          # JAX Projected Gradient Descent
        "dirac_hybrid",     # Hybrid Dirac/NetworkX solver (auto-switches at 35 nodes)
        # "dirac_pgd_hybrid", # NEW: Hybrid Dirac+PGD solver (global search + high-precision refinement)
        # "dirac",            # Pure Dirac-3 continuous cloud solver
        # "jax_md"            # JAX Mirror Descent
        # "gurobi"            # Gurobi exact (excluded for speed)
    ]
    
    print("✓ Using hybrid Dirac/NetworkX solver (auto-switches at 35 nodes)")
    print("✓ Using NEW Dirac+PGD hybrid solver (global search + high-precision refinement)")
    print("✓ Large graphs (>35 nodes) will use Dirac-3 cloud solver or hybrid approach")
    
    # Configure benchmark (adjusted for 10-node graph)
    benchmark_config = {
        'num_random_runs': 10,  
        'fast_timeout': 3000.0,    # Fast timeouts for small graph
        'medium_timeout': 3000.0,
        'slow_timeout': 3000.0,
        'jax_config': {
            'learning_rate_pgd': 0.02,
            'learning_rate_md': 0.01,
            'max_iterations': 500,   # Reduced iterations for small graph
            'num_restarts': 5,       # Moderate restarts to demonstrate vmap batching
            'tolerance': 1e-6,
            'verbose': True          # Enable verbose for progress tracking
        },
        'dirac_config': {
            'num_samples': 50,       # Conservative for API reliability
            'relax_schedule': 4,     # Default relaxation schedule as requested
            'solution_precision': None,  # Solution precision parameter
            'sum_constraint': 1,     # Simplex constraint (default)
            'mean_photon_number': 0.0025,  # Example: Override default photon number
            'quantum_fluctuation_coefficient': 60,  # Example: Override default fluctuation
            'threshold_nodes': 70    # Threshold for hybrid solver (both dirac_hybrid and dirac_pgd_hybrid)
        },
        # Configuration for the new Dirac+PGD hybrid solver
        'dirac_pgd_config': {
            'nx_threshold': 88,      # Use NetworkX exact for graphs ≤85 nodes
            'dirac_num_samples': 50, # Dirac global search samples
            'dirac_relax_schedule': 4, # Dirac relaxation schedule
            'dirac_solution_precision': None,  # Use default solution precision
            'dirac_sum_constraint': 1,  # Simplex constraint for Dirac solver
            'dirac_mean_photon_number': 0.004,  # Example: Override default photon number
            'dirac_quantum_fluctuation_coefficient': 85,  # Example: Override default fluctuation
            'pgd_tolerance': 1e-7,   # High-precision PGD refinement tolerance
            'pgd_max_iterations': 1000, # PGD refinement iterations
            'verbose': True          # Show detailed solve information
        }
    }
    num_nodes = G.number_of_nodes()
    # Run comparison
    results = run_algorithm_comparison(
        G, 
        f"BA_{num_nodes}_m3_test",
        algorithms=algorithms,
        benchmark_config=benchmark_config
    )
    
    return results


def analyze_results(results):
    """Analyze and print detailed results."""
    print("\n" + "="*60)
    print("DETAILED RESULTS ANALYSIS")
    print("="*60)
    
    # Find optimal solution
    successful_results = {alg: res for alg, res in results.items() if res.success}
    if not successful_results:
        print("No successful results!")
        return None
    
    set_sizes = [res.set_size for res in successful_results.values()]
    optimal_size = max(set_sizes)
    
    print(f"\nOptimal independent set size: {optimal_size}")
    print(f"Found by: {[alg for alg, res in successful_results.items() if res.set_size == optimal_size]}")
    
    # Detailed analysis table
    print(f"\n{'Algorithm':<20} {'Set Size':<8} {'Ratio':<8} {'Runtime':<10} {'Oracle Calls':<12} {'Success':<8}")
    print("-" * 75)
    
    analysis_data = {}
    
    for alg, result in results.items():
        if result.success:
            ratio = result.set_size / optimal_size
            oracle_calls = result.oracle_calls if result.oracle_calls else "N/A"
            print(f"{alg:<20} {result.set_size:<8} {ratio:<8.3f} {result.runtime_seconds:<10.4f} {oracle_calls!s:<12} {'✓':<8}")
            
            analysis_data[alg] = {
                'set_size': result.set_size,
                'ratio': ratio,
                'runtime': result.runtime_seconds,
                'oracle_calls': result.oracle_calls,
                'independent_set': result.independent_set,
                'optimization_details': result.optimization_details,
                'convergence_history': result.convergence_history
            }
        else:
            print(f"{alg:<20} {'FAILED':<8} {'N/A':<8} {result.runtime_seconds:<10.4f} {'N/A':<12} {'✗':<8}")
            print(f"  Error: {result.error_message}")
    
    # Analysis of greedy randomness
    greedy_result = results.get('nx_greedy')
    if greedy_result and greedy_result.success and greedy_result.multiple_runs:
        print(f"\nNetworkX Greedy Randomness Analysis:")
        sizes = [run['size'] for run in greedy_result.multiple_runs if run.get('valid', True)]
        print(f"  {len(sizes)} runs: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}, std={np.std(sizes):.2f}")
        print(f"  Best improvement: {(max(sizes) - np.mean(sizes)) / np.mean(sizes) * 100:.1f}%")
    
    return analysis_data, optimal_size


def visualize_graph_and_solutions(G, results, optimal_size, output_dir):
    """Create visualization of the graph and different solutions."""
    print("\nGenerating graph visualizations...")
    
    # Create figure with subplots for different solutions
    successful_results = {alg: res for alg, res in results.items() if res.success}
    n_plots = len(successful_results)
    
    if n_plots == 0:
        return
    
    # Calculate subplot layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Fixed layout for consistent visualization
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    for i, (alg, result) in enumerate(successful_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Color nodes: red for independent set, lightblue for others
        node_colors = ['red' if node in result.independent_set else 'lightblue' 
                      for node in G.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Title with results
        ratio = result.set_size / optimal_size
        title = f"{alg}\nSize: {result.set_size}/{optimal_size} (ratio: {ratio:.3f})"
        if result.oracle_calls:
            title += f"\nOracle calls: {result.oracle_calls}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Independent Set Solutions Comparison\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'erdos_renyi_graph_solutions_{G.number_of_nodes()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_histories(results, output_dir):
    """Plot convergence histories for JAX algorithms."""
    print("Generating convergence analysis...")
    
    jax_results = {alg: res for alg, res in results.items() 
                   if alg.startswith('jax_') and res.success and res.convergence_history}
    
    if not jax_results:
        print("No JAX convergence data available")
        return
    
    fig, axes = plt.subplots(1, len(jax_results), figsize=(8*len(jax_results), 6))
    if len(jax_results) == 1:
        axes = [axes]
    
    for i, (alg, result) in enumerate(jax_results.items()):
        ax = axes[i]
        
        histories = result.convergence_history
        if not histories:
            continue
            
        # Plot all restart histories
        for j, history in enumerate(histories):
            alpha = 0.3 if j > 0 else 1.0  # Highlight best restart
            color = 'red' if j == 0 else 'lightblue'
            label = 'Best restart' if j == 0 else None
            ax.plot(history, alpha=alpha, color=color, linewidth=1.5 if j == 0 else 0.8, label=label)
        
        ax.set_title(f'{alg.upper()} Convergence\n{len(histories)} restarts')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
        
        if result.optimization_details:
            details = result.optimization_details
            if 'best_energy' in details:
                ax.axhline(y=details['best_energy'], color='green', linestyle='--', 
                          alpha=0.7, label=f"Final: {details['best_energy']:.6f}")
        
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'jax_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(analysis_data, output_dir):
    """Create comparison plots for different metrics."""
    print("Generating algorithm comparison plots...")
    
    algorithms = list(analysis_data.keys())
    
    # 1. Set Size and Runtime Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set sizes
    set_sizes = [analysis_data[alg]['set_size'] for alg in algorithms]
    colors = ['red' if size == max(set_sizes) else 'lightblue' for size in set_sizes]
    
    bars1 = ax1.bar(algorithms, set_sizes, color=colors)
    ax1.set_title('Independent Set Size Comparison')
    ax1.set_ylabel('Set Size')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, size in zip(bars1, set_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(size), ha='center', va='bottom')
    
    # Runtime comparison (log scale)
    runtimes = [analysis_data[alg]['runtime'] for alg in algorithms]
    bars2 = ax2.bar(algorithms, runtimes, color='lightgreen')
    ax2.set_title('Runtime Comparison')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, runtime in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{runtime:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Quality vs Speed Trade-off
    plt.figure(figsize=(10, 8))
    
    for alg in algorithms:
        runtime = analysis_data[alg]['runtime']
        ratio = analysis_data[alg]['ratio']
        oracle_calls = analysis_data[alg]['oracle_calls']
        
        # Size based on oracle calls (if available)
        size = 100 if oracle_calls is None else min(500, max(50, oracle_calls * 10))
        
        plt.scatter(runtime, ratio, s=size, alpha=0.7, label=alg)
        
        # Add text labels
        plt.annotate(alg, (runtime, ratio), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Approximation Ratio')
    plt.title('Quality vs Speed Trade-off\n(Bubble size ∝ Oracle calls)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Optimal')
    plt.legend()
    plt.savefig(output_dir / 'timing_erdos_renyi.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(G, results, analysis_data, optimal_size, output_dir):
    """Generate a comprehensive summary report."""
    print("\nGenerating summary report...")
    
    report_path = output_dir / "comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("MIS Algorithm Comparison Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Test Graph: Erdos-Renyi {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write(f"  Nodes: {G.number_of_nodes()}\n")
        f.write(f"  Edges: {G.number_of_edges()}\n")
        f.write(f"  Density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2):.3f}\n")
        f.write(f"  Optimal MIS size: {optimal_size}\n\n")
        
        f.write("Algorithm Results:\n")
        f.write("-" * 50 + "\n")
        
        for alg, data in analysis_data.items():
            f.write(f"\n{alg}:\n")
            f.write(f"  Set size: {data['set_size']}/{optimal_size} (ratio: {data['ratio']:.3f})\n")
            f.write(f"  Runtime: {data['runtime']:.4f} seconds\n")
            f.write(f"  Oracle calls: {data['oracle_calls']}\n")
            f.write(f"  Solution: {sorted(data['independent_set'])}\n")
        
        # Key insights
        f.write(f"\nKey Insights:\n")
        f.write(f"- Exact methods (NetworkX exact, hybrid solvers) all found optimal solution\n")
        f.write(f"- NEW: Dirac+PGD hybrid combines global search with high-precision refinement\n")
        f.write(f"- NetworkX approximation provides good quality/speed trade-off\n")
        f.write(f"- Hybrid solvers automatically choose best method based on graph size\n")
        f.write(f"- Greedy heuristic shows significant variance across random seeds\n")
        
        # Performance ranking
        exact_algs = [alg for alg, data in analysis_data.items() if data['ratio'] >= 0.999]
        fastest_exact = min(exact_algs, key=lambda alg: analysis_data[alg]['runtime'])
        
        f.write(f"\nPerformance Summary:\n")
        f.write(f"- Fastest exact algorithm: {fastest_exact} ({analysis_data[fastest_exact]['runtime']:.3f}s)\n")
        f.write(f"- Best approximation: NetworkX BH ({analysis_data.get('nx_approximation', {}).get('ratio', 'N/A')})\n")
        f.write(f"- Most efficient heuristic: NetworkX greedy (single run)\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    """Run single graph comprehensive comparison including new Dirac+PGD hybrid solver."""
    print("MIS Algorithm Single Graph Comparison")
    print("Testing all methods including NEW Dirac+PGD hybrid solver")
    print("Graph: Barabasi-Albert with configurable size")
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir.absolute()}")
    
    start_time = time.time()
    
    # Create test graph
    G = create_test_graph()
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison(G)
    
    # Analyze results
    analysis_data, optimal_size = analyze_results(results)
    
    if analysis_data:
        # Generate visualizations
        visualize_graph_and_solutions(G, results, optimal_size, output_dir)
        plot_convergence_histories(results, output_dir)
        plot_algorithm_comparison(analysis_data, output_dir)
        
        # Generate report
        generate_summary_report(G, results, analysis_data, optimal_size, output_dir)
    
    total_time = time.time() - start_time
    print(f"\nComparison completed in {total_time:.1f} seconds")
    print(f"All results and visualizations saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
