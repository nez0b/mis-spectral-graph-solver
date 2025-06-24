"""
Visualization and debugging tools for Motzkin-Straus optimization analysis.
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Dict, Any, Tuple
import os


def plot_oracle_comparison(
    oracles_results: Dict[str, Dict[str, Any]],
    graph: nx.Graph,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (20, 12)
):
    """
    Create comprehensive comparison plot for multiple oracles.
    
    Args:
        oracles_results: Dict mapping oracle names to their results.
        graph: The graph that was tested.
        save_path: Path to save the plot.
        show_plot: Whether to display the plot.
        figsize: Figure size tuple.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    n_oracles = len(oracles_results)
    if n_oracles == 0:
        print("No oracle results to plot")
        return
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_oracles + 1, figure=fig, height_ratios=[2, 1.5, 1])
    
    # Colors for different oracles
    colors = ['red', 'orange', 'blue', 'green', 'purple', 'brown', 'pink', 'gray']
    
    oracle_names = list(oracles_results.keys())
    
    # Main convergence plot (top row, spanning all columns)
    ax_conv = fig.add_subplot(gs[0, :])
    
    for i, (oracle_name, results) in enumerate(oracles_results.items()):
        color = colors[i % len(colors)]
        
        if 'convergence_histories' in results:
            histories = results['convergence_histories']
            for j, history in enumerate(histories):
                alpha = 0.8 if j == results.get('best_restart_idx', 0) else 0.2
                linewidth = 2 if j == results.get('best_restart_idx', 0) else 0.5
                ax_conv.plot(history, color=color, alpha=alpha, linewidth=linewidth)
            
            # Highlight best curve
            if results.get('best_restart_idx', 0) < len(histories):
                best_history = histories[results['best_restart_idx']]
                ax_conv.plot(best_history, color=color, linewidth=3, 
                           label=f"{oracle_name} (best)", alpha=0.9)
    
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Energy')
    ax_conv.set_title(f'Convergence Comparison - {graph.number_of_nodes()}-node Graph')
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    
    # Individual oracle analysis (middle row)
    for i, (oracle_name, results) in enumerate(oracles_results.items()):
        ax = fig.add_subplot(gs[1, i])
        color = colors[i % len(colors)]
        
        if 'final_energies' in results:
            final_energies = results['final_energies']
            ax.hist(final_energies, bins=min(10, len(final_energies)), 
                   alpha=0.7, color=color, edgecolor='black')
            ax.axvline(max(final_energies), color='red', linestyle='--', 
                      label=f'Best: {max(final_energies):.4f}')
            ax.set_xlabel('Final Energy')
            ax.set_ylabel('Count')
            ax.set_title(f'{oracle_name}\nEnergy Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No history\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{oracle_name}')
    
    # Graph structure (bottom left)
    ax_graph = fig.add_subplot(gs[2, 0])
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, ax=ax_graph, with_labels=True, node_color='lightblue',
            node_size=300, font_size=8, edge_color='gray')
    ax_graph.set_title(f'Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
    
    # Results summary (bottom right)
    ax_summary = fig.add_subplot(gs[2, 1:])
    ax_summary.axis('off')
    
    # Create summary table
    summary_text = "Oracle Comparison Summary:\\n\\n"
    summary_text += f"{'Oracle':<15} {'Omega':<6} {'Best Energy':<12} {'Std Dev':<10} {'Restarts':<8}\\n"
    summary_text += "-" * 60 + "\\n"
    
    for oracle_name, results in oracles_results.items():
        omega = results.get('omega', 'N/A')
        best_energy = f"{max(results.get('final_energies', [0])):.4f}" if 'final_energies' in results else 'N/A'
        std_dev = f"{np.std(results.get('final_energies', [0])):.4f}" if 'final_energies' in results else 'N/A'
        restarts = len(results.get('final_energies', [])) if 'final_energies' in results else 'N/A'
        
        summary_text += f"{oracle_name:<15} {omega:<6} {best_energy:<12} {std_dev:<10} {restarts:<8}\\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_convergence_analysis(
    oracle,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot detailed convergence analysis for a single oracle.
    
    Args:
        oracle: Oracle with convergence history.
        title_suffix: Additional text for plot title.
        save_path: Path to save the plot.
        show_plot: Whether to display the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    if not hasattr(oracle, 'get_convergence_histories'):
        print("Oracle does not support convergence history")
        return
    
    histories = oracle.get_convergence_histories()
    if not histories:
        print("No convergence history available")
        return
    
    details = oracle.get_optimization_details()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{oracle.name} Convergence Analysis {title_suffix}', fontsize=16)
    
    # 1. All convergence curves
    for i, history in enumerate(histories):
        alpha = 0.8 if i == details.get('best_restart_idx', 0) else 0.3
        linewidth = 2 if i == details.get('best_restart_idx', 0) else 1
        color = 'red' if i == details.get('best_restart_idx', 0) else 'blue'
        axes[0, 0].plot(history, alpha=alpha, linewidth=linewidth, color=color)
    
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title(f'Convergence Curves ({len(histories)} restarts)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Best vs worst comparison
    if len(histories) > 1:
        final_energies = [hist[-1] for hist in histories]
        best_idx = np.argmax(final_energies)
        worst_idx = np.argmin(final_energies)
        
        axes[0, 1].plot(histories[best_idx], 'g-', linewidth=2, label=f'Best ({final_energies[best_idx]:.4f})')
        axes[0, 1].plot(histories[worst_idx], 'r--', linewidth=2, label=f'Worst ({final_energies[worst_idx]:.4f})')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].set_title('Best vs Worst Restart')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Only one restart', ha='center', va='center', 
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Best vs Worst Restart')
    
    # 3. Final energy distribution
    if len(histories) > 1:
        final_energies = [hist[-1] for hist in histories]
        axes[0, 2].hist(final_energies, bins=min(10, len(final_energies)), 
                       alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(max(final_energies), color='red', linestyle='--', 
                          label=f'Best: {max(final_energies):.4f}')
        axes[0, 2].set_xlabel('Final Energy')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Final Energy Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Only one restart', ha='center', va='center', 
                       transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Final Energy Distribution')
    
    # 4. Convergence speed analysis
    convergence_iters = details.get('convergence_iterations', [])
    if convergence_iters:
        axes[1, 0].bar(range(len(convergence_iters)), convergence_iters, alpha=0.7)
        axes[1, 0].set_xlabel('Restart Index')
        axes[1, 0].set_ylabel('Iterations to Convergence')
        axes[1, 0].set_title('Convergence Speed per Restart')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No convergence\\ndata available', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Convergence Speed')
    
    # 5. Energy improvement over iterations (average)
    if histories:
        max_len = max(len(hist) for hist in histories)
        avg_energy = np.zeros(max_len)
        count = np.zeros(max_len)
        
        for hist in histories:
            for i, energy in enumerate(hist):
                avg_energy[i] += energy
                count[i] += 1
        
        avg_energy = avg_energy / np.maximum(count, 1)
        axes[1, 1].plot(avg_energy, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Average Energy')
        axes[1, 1].set_title('Average Energy Progress')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Configuration summary
    axes[1, 2].axis('off')
    config_text = "Configuration:\\n\\n"
    if 'config' in details:
        config = details['config']
        config_text += f"Learning Rate: {config.get('learning_rate', 'N/A')}\\n"
        config_text += f"Max Iterations: {config.get('max_iterations', 'N/A')}\\n"
        config_text += f"Tolerance: {config.get('tolerance', 'N/A')}\\n"
        config_text += f"Num Restarts: {config.get('num_restarts', 'N/A')}\\n"
        config_text += f"Dirichlet Alpha: {config.get('dirichlet_alpha', 'N/A')}\\n\\n"
    
    config_text += "Results:\\n\\n"
    config_text += f"Best Energy: {details.get('best_energy', 'N/A'):.6f}\\n"
    config_text += f"Energy Range: {details.get('energy_range', 'N/A'):.6f}\\n"
    config_text += f"Energy Std: {details.get('energy_std', 'N/A'):.6f}\\n"
    
    axes[1, 2].text(0.05, 0.95, config_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_oracle_benchmark_report(
    test_results: Dict[str, Dict[str, Any]],
    save_dir: str = "figures",
    report_name: str = "oracle_benchmark_report"
):
    """
    Create a comprehensive benchmark report for oracle performance.
    
    Args:
        test_results: Dict mapping test names to oracle results.
        save_dir: Directory to save report files.
        report_name: Base name for report files.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for report generation")
        return
    
    # Generate plots for each test
    for test_name, oracles_results in test_results.items():
        plot_path = os.path.join(save_dir, f"{report_name}_{test_name.replace(' ', '_')}.png")
        
        # Create a dummy graph for visualization (you might want to pass the actual graph)
        if 'graph' in oracles_results:
            graph = oracles_results['graph']
            del oracles_results['graph']  # Remove for plotting
        else:
            graph = nx.cycle_graph(5)  # Default graph
        
        plot_oracle_comparison(
            oracles_results, 
            graph, 
            save_path=plot_path, 
            show_plot=False
        )
    
    # Generate summary report
    report_path = os.path.join(save_dir, f"{report_name}_summary.txt")
    with open(report_path, 'w') as f:
        f.write("Oracle Benchmark Report\\n")
        f.write("=" * 50 + "\\n\\n")
        
        for test_name, oracles_results in test_results.items():
            f.write(f"Test: {test_name}\\n")
            f.write("-" * 30 + "\\n")
            
            for oracle_name, results in oracles_results.items():
                if oracle_name == 'graph':
                    continue
                    
                f.write(f"  {oracle_name}:\\n")
                f.write(f"    Omega: {results.get('omega', 'N/A')}\\n")
                
                if 'final_energies' in results:
                    energies = results['final_energies']
                    f.write(f"    Best Energy: {max(energies):.6f}\\n")
                    f.write(f"    Worst Energy: {min(energies):.6f}\\n")
                    f.write(f"    Energy Std: {np.std(energies):.6f}\\n")
                    f.write(f"    Restarts: {len(energies)}\\n")
                
                if 'convergence_iterations' in results:
                    conv_iters = results['convergence_iterations']
                    f.write(f"    Avg Convergence: {np.mean(conv_iters):.1f} iterations\\n")
                
                f.write("\\n")
            f.write("\\n")
    
    print(f"Benchmark report saved to {save_dir}/")
    print(f"Summary: {report_path}")


def plot_parameter_sensitivity(
    graph: nx.Graph,
    oracle_class,
    parameter_ranges: Dict[str, List],
    base_config: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot sensitivity analysis for oracle parameters.
    
    Args:
        graph: Test graph.
        oracle_class: Oracle class to test.
        parameter_ranges: Dict mapping parameter names to value ranges.
        base_config: Base configuration for other parameters.
        save_path: Path to save the plot.
        show_plot: Whether to display the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    n_params = len(parameter_ranges)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, param_values) in enumerate(parameter_ranges.items()):
        results = []
        
        for param_value in param_values:
            config = base_config.copy()
            config[param_name] = param_value
            
            try:
                oracle = oracle_class(**config)
                omega = oracle.get_omega(graph)
                details = oracle.get_optimization_details()
                
                results.append({
                    'param_value': param_value,
                    'omega': omega,
                    'best_energy': details.get('best_energy', 0),
                    'energy_std': details.get('energy_std', 0)
                })
            except Exception as e:
                print(f"Failed for {param_name}={param_value}: {e}")
                results.append({
                    'param_value': param_value,
                    'omega': 0,
                    'best_energy': 0,
                    'energy_std': 0
                })
        
        # Plot results
        param_vals = [r['param_value'] for r in results]
        energies = [r['best_energy'] for r in results]
        stds = [r['energy_std'] for r in results]
        
        axes[i].errorbar(param_vals, energies, yerr=stds, marker='o', capsize=5)
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Best Energy')
        axes[i].set_title(f'Sensitivity to {param_name}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()