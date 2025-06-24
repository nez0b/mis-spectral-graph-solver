"""
Statistical analysis and result processing for MIS algorithm benchmarks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings

from .networkx_comparison import BenchmarkResult


@dataclass
class StatisticalSummary:
    """Statistical summary for a set of benchmark results."""
    algorithm_name: str
    num_graphs: int
    
    # Set size statistics
    mean_set_size: float
    median_set_size: float
    std_set_size: float
    min_set_size: int
    max_set_size: int
    
    # Runtime statistics  
    mean_runtime: float
    median_runtime: float
    std_runtime: float
    min_runtime: float
    max_runtime: float
    
    # Success rate
    success_rate: float
    timeout_rate: float
    error_rate: float
    
    # Approximation quality (if ground truth available)
    mean_approx_ratio: Optional[float] = None
    median_approx_ratio: Optional[float] = None
    std_approx_ratio: Optional[float] = None


def compute_approximation_ratios(
    results: Dict[str, List[BenchmarkResult]],
    ground_truth_algorithm: str = "gurobi"
) -> Dict[str, List[float]]:
    """
    Compute approximation ratios for all algorithms relative to ground truth.
    
    Args:
        results: Dictionary mapping algorithm names to lists of BenchmarkResults.
        ground_truth_algorithm: Name of algorithm providing ground truth.
        
    Returns:
        Dictionary mapping algorithm names to lists of approximation ratios.
    """
    if ground_truth_algorithm not in results:
        print(f"Warning: Ground truth algorithm '{ground_truth_algorithm}' not found")
        return {}
    
    ground_truth = results[ground_truth_algorithm]
    ratios = {}
    
    for alg_name, alg_results in results.items():
        if alg_name == ground_truth_algorithm:
            continue
            
        alg_ratios = []
        
        for i, result in enumerate(alg_results):
            if i < len(ground_truth) and ground_truth[i].success and result.success:
                optimal_size = ground_truth[i].set_size
                found_size = result.set_size
                
                if optimal_size > 0:
                    ratio = found_size / optimal_size
                    alg_ratios.append(ratio)
                else:
                    # Handle edge case where optimal is 0
                    alg_ratios.append(1.0 if found_size == 0 else 0.0)
            else:
                # Missing or failed result
                alg_ratios.append(np.nan)
                
        ratios[alg_name] = alg_ratios
        
    return ratios


def statistical_summary(
    results: List[BenchmarkResult],
    algorithm_name: str,
    ground_truth_sizes: Optional[List[int]] = None
) -> StatisticalSummary:
    """
    Compute statistical summary for a single algorithm's results.
    
    Args:
        results: List of BenchmarkResults for the algorithm.
        algorithm_name: Name of the algorithm.
        ground_truth_sizes: Optional list of optimal set sizes for approximation ratios.
        
    Returns:
        StatisticalSummary object.
    """
    if not results:
        return StatisticalSummary(
            algorithm_name=algorithm_name,
            num_graphs=0,
            mean_set_size=0.0, median_set_size=0.0, std_set_size=0.0,
            min_set_size=0, max_set_size=0,
            mean_runtime=0.0, median_runtime=0.0, std_runtime=0.0,
            min_runtime=0.0, max_runtime=0.0,
            success_rate=0.0, timeout_rate=0.0, error_rate=0.0
        )
    
    num_graphs = len(results)
    successful_results = [r for r in results if r.success]
    
    # Set size statistics
    if successful_results:
        set_sizes = [r.set_size for r in successful_results]
        mean_set_size = np.mean(set_sizes)
        median_set_size = np.median(set_sizes)
        std_set_size = np.std(set_sizes)
        min_set_size = min(set_sizes)
        max_set_size = max(set_sizes)
    else:
        mean_set_size = median_set_size = std_set_size = 0.0
        min_set_size = max_set_size = 0
    
    # Runtime statistics (include all results)
    runtimes = [r.runtime_seconds for r in results]
    mean_runtime = np.mean(runtimes)
    median_runtime = np.median(runtimes)
    std_runtime = np.std(runtimes)
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)
    
    # Success rates
    success_count = len(successful_results)
    timeout_count = sum(1 for r in results if r.timeout)
    error_count = num_graphs - success_count - timeout_count
    
    success_rate = success_count / num_graphs
    timeout_rate = timeout_count / num_graphs
    error_rate = error_count / num_graphs
    
    # Approximation ratios (if ground truth provided)
    mean_approx_ratio = median_approx_ratio = std_approx_ratio = None
    if ground_truth_sizes and len(ground_truth_sizes) == len(successful_results):
        ratios = []
        for result, optimal_size in zip(successful_results, ground_truth_sizes):
            if optimal_size > 0:
                ratios.append(result.set_size / optimal_size)
            else:
                ratios.append(1.0 if result.set_size == 0 else 0.0)
                
        if ratios:
            mean_approx_ratio = np.mean(ratios)
            median_approx_ratio = np.median(ratios)
            std_approx_ratio = np.std(ratios)
    
    return StatisticalSummary(
        algorithm_name=algorithm_name,
        num_graphs=num_graphs,
        mean_set_size=mean_set_size,
        median_set_size=median_set_size,
        std_set_size=std_set_size,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
        mean_runtime=mean_runtime,
        median_runtime=median_runtime,
        std_runtime=std_runtime,
        min_runtime=min_runtime,
        max_runtime=max_runtime,
        success_rate=success_rate,
        timeout_rate=timeout_rate,
        error_rate=error_rate,
        mean_approx_ratio=mean_approx_ratio,
        median_approx_ratio=median_approx_ratio,
        std_approx_ratio=std_approx_ratio
    )


def analyze_randomized_algorithm(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    Analyze results from randomized algorithms with multiple runs per graph.
    
    Args:
        results: List of BenchmarkResults where each may have multiple_runs data.
        
    Returns:
        Dictionary with detailed analysis of randomization effects.
    """
    analysis = {
        'graphs_analyzed': 0,
        'total_runs': 0,
        'variance_statistics': {},
        'best_vs_average': {},
        'seed_effects': []
    }
    
    all_variances = []
    all_ranges = []
    best_improvements = []
    
    for result in results:
        if not result.multiple_runs:
            continue
            
        runs = result.multiple_runs
        if len(runs) < 2:
            continue
            
        analysis['graphs_analyzed'] += 1
        analysis['total_runs'] += len(runs)
        
        # Extract set sizes and runtimes
        sizes = [run['size'] for run in runs if run.get('valid', True)]
        runtimes = [run['runtime'] for run in runs]
        
        if len(sizes) < 2:
            continue
            
        # Variance analysis
        size_variance = np.var(sizes)
        size_range = max(sizes) - min(sizes)
        all_variances.append(size_variance)
        all_ranges.append(size_range)
        
        # Best vs average improvement
        best_size = max(sizes)
        avg_size = np.mean(sizes)
        if avg_size > 0:
            improvement = (best_size - avg_size) / avg_size
            best_improvements.append(improvement)
        
        # Seed effect analysis (if enough data)
        if len(sizes) >= 10:
            # Simple correlation between seed and performance
            seeds = [run['seed'] for run in runs if run.get('valid', True)]
            if len(seeds) == len(sizes):
                correlation, p_value = stats.pearsonr(seeds, sizes)
                analysis['seed_effects'].append({
                    'graph': result.graph_description,
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    # Aggregate variance statistics
    if all_variances:
        analysis['variance_statistics'] = {
            'mean_variance': np.mean(all_variances),
            'median_variance': np.median(all_variances),
            'mean_range': np.mean(all_ranges),
            'median_range': np.median(all_ranges),
            'max_range': max(all_ranges)
        }
    
    # Best vs average statistics
    if best_improvements:
        analysis['best_vs_average'] = {
            'mean_improvement': np.mean(best_improvements),
            'median_improvement': np.median(best_improvements),
            'max_improvement': max(best_improvements),
            'percent_with_improvement': sum(1 for x in best_improvements if x > 0) / len(best_improvements) * 100
        }
    
    return analysis


def create_comparison_dataframe(
    results: Dict[str, List[BenchmarkResult]],
    graph_descriptions: List[str] = None
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from benchmark results for easy analysis.
    
    Args:
        results: Dictionary mapping algorithm names to lists of BenchmarkResults.
        graph_descriptions: Optional list of graph descriptions.
        
    Returns:
        DataFrame with columns: algorithm, graph_id, graph_desc, graph_size, 
                               graph_edges, set_size, runtime, success, etc.
    """
    rows = []
    
    for alg_name, alg_results in results.items():
        for i, result in enumerate(alg_results):
            graph_desc = result.graph_description
            if graph_descriptions and i < len(graph_descriptions):
                graph_desc = graph_descriptions[i]
                
            row = {
                'algorithm': alg_name,
                'graph_id': i,
                'graph_description': graph_desc,
                'graph_size': result.graph_size,
                'graph_edges': result.graph_edges,
                'set_size': result.set_size,
                'runtime_seconds': result.runtime_seconds,
                'success': result.success,
                'timeout': result.timeout,
                'oracle_calls': result.oracle_calls,
                'error_message': result.error_message
            }
            
            # Add algorithm-specific data
            if result.multiple_runs:
                # For randomized algorithms, add statistics
                sizes = [run['size'] for run in result.multiple_runs if run.get('valid', True)]
                if sizes:
                    row.update({
                        'num_runs': len(sizes),
                        'mean_size': np.mean(sizes),
                        'std_size': np.std(sizes),
                        'min_size': min(sizes),
                        'max_size': max(sizes)
                    })
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def analyze_benchmark_results(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str = "figures",
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of benchmark results with plots and statistics.
    
    Args:
        results: Dictionary mapping algorithm names to lists of BenchmarkResults.
        output_dir: Directory to save plots.
        save_plots: Whether to save plots to files.
        
    Returns:
        Dictionary containing all analysis results.
    """
    analysis = {
        'summary_statistics': {},
        'approximation_ratios': {},
        'randomized_analysis': {},
        'dataframe': None
    }
    
    # Create DataFrame for easier analysis
    df = create_comparison_dataframe(results)
    analysis['dataframe'] = df
    
    # Compute summary statistics for each algorithm
    for alg_name, alg_results in results.items():
        summary = statistical_summary(alg_results, alg_name)
        analysis['summary_statistics'][alg_name] = summary
    
    # Compute approximation ratios if ground truth is available
    if 'gurobi' in results:
        ratios = compute_approximation_ratios(results, 'gurobi')
        analysis['approximation_ratios'] = ratios
    elif 'nx_exact' in results:
        ratios = compute_approximation_ratios(results, 'nx_exact')
        analysis['approximation_ratios'] = ratios
    
    # Analyze randomized algorithms
    for alg_name, alg_results in results.items():
        if any(r.multiple_runs for r in alg_results):
            rand_analysis = analyze_randomized_algorithm(alg_results)
            analysis['randomized_analysis'][alg_name] = rand_analysis
    
    # Generate plots if requested
    if save_plots:
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot 1: Runtime vs Graph Size
            plot_runtime_vs_size(df, save_path=f"{output_dir}/runtime_vs_size.png")
            
            # Plot 2: Set Size vs Graph Size  
            plot_setsize_vs_size(df, save_path=f"{output_dir}/setsize_vs_size.png")
            
            # Plot 3: Approximation Ratios (if available)
            if analysis['approximation_ratios']:
                plot_approximation_ratios(
                    analysis['approximation_ratios'],
                    save_path=f"{output_dir}/approximation_ratios.png"
                )
            
            # Plot 4: Success Rates
            plot_success_rates(analysis['summary_statistics'], save_path=f"{output_dir}/success_rates.png")
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return analysis


def plot_runtime_vs_size(df: pd.DataFrame, save_path: str = None):
    """Plot runtime vs graph size for all algorithms."""
    plt.figure(figsize=(12, 8))
    
    for alg in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == alg]
        successful = alg_data[alg_data['success'] == True]
        
        if len(successful) > 0:
            plt.scatter(successful['graph_size'], successful['runtime_seconds'], 
                       label=alg, alpha=0.7, s=50)
    
    plt.xlabel('Graph Size (number of nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Size')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_setsize_vs_size(df: pd.DataFrame, save_path: str = None):
    """Plot independent set size vs graph size for all algorithms."""
    plt.figure(figsize=(12, 8))
    
    for alg in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == alg]
        successful = alg_data[alg_data['success'] == True]
        
        if len(successful) > 0:
            plt.scatter(successful['graph_size'], successful['set_size'],
                       label=alg, alpha=0.7, s=50)
    
    plt.xlabel('Graph Size (number of nodes)')
    plt.ylabel('Independent Set Size')
    plt.title('Independent Set Size vs Graph Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_approximation_ratios(ratios: Dict[str, List[float]], save_path: str = None):
    """Plot approximation ratios for all algorithms."""
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    data = []
    labels = []
    
    for alg_name, alg_ratios in ratios.items():
        # Remove NaN values
        clean_ratios = [r for r in alg_ratios if not np.isnan(r)]
        if clean_ratios:
            data.append(clean_ratios)
            labels.append(alg_name)
    
    if data:
        plt.boxplot(data, labels=labels)
        plt.ylabel('Approximation Ratio (Found Size / Optimal Size)')
        plt.title('Approximation Quality Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Optimal')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_success_rates(summaries: Dict[str, StatisticalSummary], save_path: str = None):
    """Plot success rates for all algorithms."""
    algorithms = list(summaries.keys())
    success_rates = [summaries[alg].success_rate * 100 for alg in algorithms]
    timeout_rates = [summaries[alg].timeout_rate * 100 for alg in algorithms]
    error_rates = [summaries[alg].error_rate * 100 for alg in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, success_rates, width, label='Success', color='green', alpha=0.7)
    ax.bar(x, timeout_rates, width, label='Timeout', color='orange', alpha=0.7)
    ax.bar(x + width, error_rates, width, label='Error', color='red', alpha=0.7)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Percentage')
    ax.set_title('Algorithm Success/Failure Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing analysis framework...")
    
    # This would normally come from actual benchmark runs
    dummy_results = {
        'nx_greedy': [
            BenchmarkResult(
                algorithm_name="NetworkX_Maximal_Greedy",
                graph_description="test_graph",
                graph_size=10,
                graph_edges=15,
                independent_set=[0, 2, 4, 6, 8],
                set_size=5,
                runtime_seconds=0.001
            )
        ]
    }
    
    df = create_comparison_dataframe(dummy_results)
    print(df)