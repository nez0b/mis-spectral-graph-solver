#!/usr/bin/env python3
"""
Gumbel Noise Injection Experiment for Motzkin-Straus Optimization.

This experiment tests whether Gumbel noise injection (Perturb & MAP method)
can help projected gradient descent escape local optima in the non-convex
Motzkin-Straus quadratic program.

The experiment compares three conditions:
1. Gumbel noise injection
2. Gaussian noise baseline  
3. Multi-start control (no noise, multiple random initializations)

Usage:
    python experiments/gumbel_noise_experiment.py
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import existing JAX PGD oracle
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

# Import our noise utilities
from noise_utils import (
    generate_gumbel_noise, 
    generate_gaussian_noise, 
    perturb_adjacency_matrix,
    compute_noise_scale,
    validate_noise_parameters
)


class GumbelNoiseExperiment:
    """
    Experiment class for testing Gumbel noise injection on Motzkin-Straus optimization.
    """
    
    def __init__(self, 
                 num_trials: int = 100,
                 pgd_iterations: int = 500,
                 learning_rate: float = 0.02,
                 tolerance: float = 1e-6,
                 verbose: bool = True):
        """
        Initialize the experiment.
        
        Args:
            num_trials: Number of optimization trials per condition
            pgd_iterations: Number of PGD iterations per trial
            learning_rate: PGD learning rate
            tolerance: Convergence tolerance
            verbose: Whether to print progress information
        """
        self.num_trials = num_trials
        self.pgd_iterations = pgd_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Create PGD oracle
        self.pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=learning_rate,
            max_iterations=pgd_iterations,
            tolerance=tolerance,
            verbose=False  # Suppress per-trial output
        )
        
        # Results storage
        self.results = {}
        
    def create_test_graphs(self) -> Dict[str, nx.Graph]:
        """Create test graphs with known maximum clique sizes."""
        graphs = {}
        
        # Triangle graph (3-clique)
        # triangle = nx.complete_graph(3)
        # graphs['triangle'] = triangle
        
        # Complete graph K4 (4-clique)
        # complete_4 = nx.complete_graph(4)
        # graphs['complete_4'] = complete_4
        
        # Erdős-Rényi graph (realistic complexity)
        np.random.seed(42)  # For reproducibility
        # erdos_renyi = nx.erdos_renyi_graph(10, 0.7, seed=42)
        # graphs['erdos_renyi_10_p07'] = erdos_renyi

        erdos_renyi = nx.erdos_renyi_graph(20, 0.7, seed=42)
        graphs['erdos_renyi_20_p07'] = erdos_renyi
        
        return graphs
        
    def compute_theoretical_maximum(self, graph: nx.Graph) -> float:
        """
        Compute theoretical maximum for Motzkin-Straus objective.
        
        For a k-clique, the maximum value is 0.5 * (1 - 1/k).
        """
        # Find maximum clique size using NetworkX (for small graphs)
        if graph.number_of_nodes() <= 20:
            max_clique_size = nx.graph_clique_number(graph)
            theoretical_max = 0.5 * (1 - 1/max_clique_size)
            return theoretical_max, max_clique_size
        else:
            return None, None
    
    def run_condition_trials(self, 
                           adjacency_matrix: jnp.ndarray,
                           condition: str,
                           noise_scale: float = 0.1) -> List[Dict[str, Any]]:
        """
        Run multiple trials for a single experimental condition.
        
        Args:
            adjacency_matrix: Original adjacency matrix
            condition: 'gumbel', 'gaussian', or 'multistart'
            noise_scale: Scale parameter for noise injection
            
        Returns:
            List of trial results
        """
        trial_results = []
        n = adjacency_matrix.shape[0]
        
        # Set up random keys
        key = jax.random.PRNGKey(42)
        
        for trial in range(self.num_trials):
            if self.verbose and trial % 20 == 0:
                print(f"  Running trial {trial+1}/{self.num_trials} for {condition}")
            
            key, trial_key = jax.random.split(key)
            
            # Generate problem instance based on condition
            if condition == 'gumbel':
                # --- PERTURB STEP ---
                # Add Gumbel noise to the adjacency matrix
                key, noise_key = jax.random.split(trial_key)
                gumbel_noise = generate_gumbel_noise(noise_key, adjacency_matrix.shape, noise_scale)
                A_perturbed = perturb_adjacency_matrix(adjacency_matrix, gumbel_noise)
                
            elif condition == 'gaussian':
                # Add Gaussian noise for baseline comparison
                key, noise_key = jax.random.split(trial_key)
                gaussian_noise = generate_gaussian_noise(noise_key, adjacency_matrix.shape, noise_scale)
                A_perturbed = perturb_adjacency_matrix(adjacency_matrix, gaussian_noise)
                
            elif condition == 'multistart':
                # No noise, use original matrix
                A_perturbed = adjacency_matrix
                
            else:
                raise ValueError(f"Unknown condition: {condition}")
            
            # Generate unique random key for this trial's optimization
            key, optimization_key = jax.random.split(trial_key)
            
            # Run PGD optimization
            start_time = time.time()
            try:
                # Get final solution from PGD with unique random key
                energy_value = self.pgd_oracle.solve_quadratic_program(A_perturbed, key=optimization_key)
                
                # Get the final x solution to evaluate on original matrix
                x_final = self.pgd_oracle.x_current  # Access final solution
                
                # --- MAP STEP ---
                # Always evaluate on the ORIGINAL matrix for fair comparison
                energy_original = float(0.5 * x_final.T @ adjacency_matrix @ x_final)
                
                runtime = time.time() - start_time
                success = True
                error_msg = ""
                
            except Exception as e:
                energy_original = 0.0
                x_final = jnp.zeros(n)
                runtime = time.time() - start_time
                success = False
                error_msg = str(e)
            
            # Store trial result
            trial_result = {
                'trial': trial,
                'condition': condition,
                'energy_original': energy_original,
                'x_final': x_final.tolist(),
                'runtime': runtime,
                'success': success,
                'error': error_msg,
                'noise_scale': noise_scale
            }
            
            trial_results.append(trial_result)
        
        return trial_results
    
    def run_experiment(self, graph: nx.Graph, graph_name: str, noise_scales: List[float] = None) -> Dict[str, Any]:
        """
        Run the complete experiment on a single graph.
        
        Args:
            graph: NetworkX graph to test
            graph_name: Name for saving results
            noise_scales: List of noise scales to test
            
        Returns:
            Complete experiment results
        """
        if noise_scales is None:
            noise_scales = [0.01, 0.1, 1.0]
        
        print(f"\n{'='*60}")
        print(f"Running Gumbel Noise Experiment: {graph_name}")
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Convert to adjacency matrix
        adjacency_matrix = jnp.array(nx.to_numpy_array(graph), dtype=float)
        
        # Compute theoretical maximum if possible
        theoretical_max, max_clique_size = self.compute_theoretical_maximum(graph)
        if theoretical_max is not None:
            print(f"Theoretical maximum: {theoretical_max:.6f} (k={max_clique_size})")
        
        print('='*60)
        
        # Store experiment metadata
        experiment_results = {
            'graph_name': graph_name,
            'graph_info': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'max_clique_size': max_clique_size,
                'theoretical_max': theoretical_max
            },
            'experiment_config': {
                'num_trials': self.num_trials,
                'pgd_iterations': self.pgd_iterations,
                'learning_rate': self.learning_rate,
                'tolerance': self.tolerance
            },
            'conditions': {},
            'noise_scales_tested': noise_scales
        }
        
        # Test different noise scales
        for noise_scale in noise_scales:
            scale_suffix = f"_scale_{noise_scale:.2f}"
            
            # Validate noise scale
            if self.verbose:
                print(f"\nTesting noise scale: {noise_scale:.3f}")
                validate_noise_parameters(adjacency_matrix, noise_scale, verbose=True)
            
            # Run Gumbel noise condition
            print(f"\nCondition 1: Gumbel Noise (scale={noise_scale:.3f})")
            gumbel_results = self.run_condition_trials(adjacency_matrix, 'gumbel', noise_scale)
            experiment_results['conditions'][f'gumbel{scale_suffix}'] = gumbel_results
            
            # Run Gaussian noise condition  
            print(f"\nCondition 2: Gaussian Noise (scale={noise_scale:.3f})")
            gaussian_results = self.run_condition_trials(adjacency_matrix, 'gaussian', noise_scale)
            experiment_results['conditions'][f'gaussian{scale_suffix}'] = gaussian_results
        
        # Run multi-start control condition (once, independent of noise scale)
        print(f"\nCondition 3: Multi-Start Control (no noise)")
        multistart_results = self.run_condition_trials(adjacency_matrix, 'multistart', 0.0)
        experiment_results['conditions']['multistart'] = multistart_results
        
        return experiment_results
    
    def analyze_results(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and compute statistics."""
        analysis = {
            'condition_stats': {},
            'best_solutions': {},
            'statistical_tests': {}
        }
        
        for condition_name, trials in experiment_results['conditions'].items():
            energies = [trial['energy_original'] for trial in trials if trial['success']]
            
            if energies:
                stats = {
                    'count': len(energies),
                    'mean': float(np.mean(energies)),
                    'std': float(np.std(energies)),
                    'min': float(np.min(energies)),
                    'max': float(np.max(energies)),
                    'median': float(np.median(energies)),
                    'q25': float(np.percentile(energies, 25)),
                    'q75': float(np.percentile(energies, 75))
                }
                
                analysis['condition_stats'][condition_name] = stats
                
                # Find best solution
                best_idx = np.argmax(energies)
                best_trial = [t for t in trials if t['success']][best_idx]
                analysis['best_solutions'][condition_name] = {
                    'energy': best_trial['energy_original'],
                    'x_solution': best_trial['x_final'],
                    'trial_index': best_trial['trial']
                }
            else:
                analysis['condition_stats'][condition_name] = {'count': 0, 'error': 'All trials failed'}
        
        return analysis
    
    def plot_histograms(self, experiment_results: Dict[str, Any], save_path: str = None) -> None:
        """Create histogram plots comparing energy distributions."""
        # Extract energies for each condition
        condition_energies = {}
        for condition_name, trials in experiment_results['conditions'].items():
            energies = [trial['energy_original'] for trial in trials if trial['success']]
            if energies:
                condition_energies[condition_name] = energies
        
        if not condition_energies:
            print("No successful trials to plot")
            return
        
        # Create subplot for each condition
        n_conditions = len(condition_energies)
        fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 5))
        if n_conditions == 1:
            axes = [axes]
        
        # Plot histograms
        for i, (condition, energies) in enumerate(condition_energies.items()):
            ax = axes[i]
            
            # Create histogram
            n_bins = min(20, len(energies) // 3) if len(energies) > 10 else 10
            counts, bins, patches = ax.hist(energies, bins=n_bins, alpha=0.7, edgecolor='black')
            
            # Add vertical line for best energy and min energy
            best_energy = max(energies)
            min_energy = min(energies)
            ax.axvline(x=best_energy, color='red', linestyle='--', linewidth=2,
                      label=f'Best: {best_energy:.6f}')
            ax.axvline(x=min_energy, color='orange', linestyle=':', linewidth=2,
                      label=f'Min: {min_energy:.6f}')
            
            # Add theoretical maximum if available
            theoretical_max = experiment_results['graph_info'].get('theoretical_max')
            if theoretical_max is not None:
                energy_diff = theoretical_max - best_energy
                ax.axvline(x=theoretical_max, color='green', linestyle='-', linewidth=2,
                          label=f'Theoretical: {theoretical_max:.6f}')
                # Add energy difference to legend
                ax.plot([], [], ' ', label=f'Gap: {energy_diff:.6f}')
            
            # Formatting
            ax.set_xlabel('Energy (x^T * A * x)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{condition.title()}\nSamples: {len(energies)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved histogram to: {save_path}")
        
        plt.show()
    
    def save_results(self, experiment_results: Dict[str, Any], analysis: Dict[str, Any], 
                    output_dir: str = "experiments/results") -> None:
        """Save experiment results and analysis to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        graph_name = experiment_results['graph_name']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / "data" / f"gumbel_experiment_{graph_name}_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_results': experiment_results,
                'analysis': analysis,
                'timestamp': timestamp
            }, f, indent=2)
        
        if self.verbose:
            print(f"Saved detailed results to: {results_file}")
        
        # Create and save histogram
        histogram_file = output_path / "histograms" / f"gumbel_histogram_{graph_name}_{timestamp}.png"
        histogram_file.parent.mkdir(exist_ok=True)
        self.plot_histograms(experiment_results, str(histogram_file))
        
        return str(results_file), str(histogram_file)


def main():
    """Main function to run the Gumbel noise experiment."""
    parser = argparse.ArgumentParser(
        description="Run Gumbel noise injection experiment on Motzkin-Straus optimization"
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per condition")
    parser.add_argument("--iterations", type=int, default=100, help="PGD iterations per trial")
    parser.add_argument("--lr", type=float, default=0.02, help="PGD learning rate")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.01, 0.1, 1.0], 
                       help="Noise scales to test")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = GumbelNoiseExperiment(
        num_trials=args.trials,
        pgd_iterations=args.iterations,
        learning_rate=args.lr,
        tolerance=args.tolerance,
        verbose=not args.quiet
    )
    
    # Create test graphs
    test_graphs = experiment.create_test_graphs()
    
    print("🎯 Gumbel Noise Injection Experiment for Motzkin-Straus Optimization")
    print("="*80)
    print(f"Configuration:")
    print(f"  Trials per condition: {args.trials}")
    print(f"  PGD iterations: {args.iterations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Noise scales: {args.scales}")
    print(f"  Test graphs: {list(test_graphs.keys())}")
    
    # Run experiments on all test graphs
    all_results = {}
    
    for graph_name, graph in test_graphs.items():
        try:
            # Run experiment
            experiment_results = experiment.run_experiment(graph, graph_name, args.scales)
            
            # Analyze results
            analysis = experiment.analyze_results(experiment_results)
            
            # Save results
            results_file, histogram_file = experiment.save_results(experiment_results, analysis)
            
            # Store for summary
            all_results[graph_name] = {
                'experiment_results': experiment_results,
                'analysis': analysis,
                'files': {'results': results_file, 'histogram': histogram_file}
            }
            
            # Print summary
            print(f"\n📊 SUMMARY for {graph_name}:")
            print("-" * 50)
            for condition, stats in analysis['condition_stats'].items():
                if 'error' not in stats:
                    best_energy = analysis['best_solutions'][condition]['energy']
                    print(f"{condition:20s}: max={best_energy:.6f}, mean={stats['mean']:.6f} ± {stats['std']:.6f}")
                else:
                    print(f"{condition:20s}: {stats['error']}")
            
        except Exception as e:
            print(f"❌ Error running experiment on {graph_name}: {e}")
            continue
    
    print(f"\n✅ Experiment completed! Results saved to experiments/results/")
    print(f"📈 Histograms saved to experiments/results/histograms/")
    
    return all_results


if __name__ == "__main__":
    results = main()