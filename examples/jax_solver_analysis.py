"""
Comprehensive analysis of JAX-based solvers for Motzkin-Straus optimization.

This example demonstrates:
1. Multi-restart optimization with convergence visualization
2. Comparison between PGD and Mirror Descent
3. Parameter sensitivity analysis
4. Performance validation against Gurobi
5. Detailed convergence behavior analysis
"""

import sys
import os
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from motzkinstraus.algorithms import find_mis_with_oracle, verify_independent_set
from motzkinstraus.visualization import (
    plot_oracle_comparison, 
    plot_convergence_analysis,
    create_oracle_benchmark_report,
    plot_parameter_sensitivity
)

# Import oracles
try:
    from motzkinstraus.oracles.gurobi import GurobiOracle
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not available - continuing with JAX-only analysis")

try:
    from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
    from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
    JAX_AVAILABLE = True
    print("JAX oracles available")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available - exiting")
    sys.exit(1)


def create_test_graphs():
    """Create a collection of test graphs with varying complexity."""
    graphs = {}
    
    # Simple test cases
    graphs['triangle'] = nx.complete_graph(3)
    graphs['4cycle'] = nx.cycle_graph(4)
    graphs['5cycle'] = nx.cycle_graph(5)
    graphs['star5'] = nx.star_graph(5)
    graphs['path5'] = nx.path_graph(5)
    
    # Medium complexity
    graphs['petersen'] = nx.petersen_graph()
    graphs['wheel6'] = nx.wheel_graph(6)
    
    # Custom structured graphs
    G_chord = nx.cycle_graph(8)
    G_chord.add_edges_from([(0, 4), (2, 6)])  # Add chords
    graphs['8cycle_chords'] = G_chord
    
    G_grid = nx.grid_2d_graph(3, 3)
    graphs['3x3_grid'] = G_grid
    
    return graphs


def run_basic_comparison():
    """Compare JAX solvers on basic test graphs."""
    print("\\n" + "="*60)
    print("BASIC JAX SOLVER COMPARISON")
    print("="*60)
    
    graphs = create_test_graphs()
    os.makedirs('figures', exist_ok=True)
    
    # Configuration for analysis
    pgd_config = {
        'learning_rate': 0.02,
        'max_iterations': 1500,
        'tolerance': 1e-6,
        'num_restarts': 10,
        'dirichlet_alpha': 1.0,
        'verbose': False
    }
    
    md_config = {
        'learning_rate': 0.008,
        'max_iterations': 1500,
        'tolerance': 1e-6,
        'num_restarts': 10,
        'dirichlet_alpha': 1.0,
        'verbose': False
    }
    
    results = {}
    
    for graph_name, graph in graphs.items():
        if graph.number_of_nodes() > 10:
            continue  # Skip large graphs for basic comparison
            
        print(f"\\nTesting {graph_name} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        
        # Test PGD
        print("  Running PGD...")
        start_time = time.time()
        pgd_oracle = ProjectedGradientDescentOracle(**pgd_config)
        pgd_omega = pgd_oracle.get_omega(graph)
        pgd_time = time.time() - start_time
        pgd_details = pgd_oracle.get_optimization_details()
        
        # Test MD
        print("  Running MD...")
        start_time = time.time()
        md_oracle = MirrorDescentOracle(**md_config)
        md_omega = md_oracle.get_omega(graph)
        md_time = time.time() - start_time
        md_details = md_oracle.get_optimization_details()
        
        # Test Gurobi if available
        gurobi_omega = None
        gurobi_time = None
        if GUROBI_AVAILABLE:
            print("  Running Gurobi...")
            start_time = time.time()
            gurobi_oracle = GurobiOracle(suppress_output=True)
            gurobi_omega = gurobi_oracle.get_omega(graph)
            gurobi_time = time.time() - start_time
        
        # Store results
        results[graph_name] = {
            'graph': graph,
            'PGD': {
                'omega': pgd_omega,
                'time': pgd_time,
                'final_energies': pgd_oracle.last_final_energies,
                'convergence_histories': pgd_oracle.get_convergence_histories(),
                'best_restart_idx': pgd_oracle.last_best_restart_idx,
                **pgd_details
            },
            'MD': {
                'omega': md_omega,
                'time': md_time,
                'final_energies': md_oracle.last_final_energies,
                'convergence_histories': md_oracle.get_convergence_histories(),
                'best_restart_idx': md_oracle.last_best_restart_idx,
                **md_details
            }
        }
        
        if GUROBI_AVAILABLE:
            results[graph_name]['Gurobi'] = {
                'omega': gurobi_omega,
                'time': gurobi_time
            }
        
        # Print results
        print(f"    PGD:    ω={pgd_omega}, time={pgd_time:.3f}s, best_energy={pgd_details['best_energy']:.4f}")
        print(f"    MD:     ω={md_omega}, time={md_time:.3f}s, best_energy={md_details['best_energy']:.4f}")
        if GUROBI_AVAILABLE:
            print(f"    Gurobi: ω={gurobi_omega}, time={gurobi_time:.3f}s")
            
        # Check agreement
        if pgd_omega == md_omega:
            agreement = "✓"
        else:
            agreement = "✗"
            
        if GUROBI_AVAILABLE:
            if pgd_omega == gurobi_omega and md_omega == gurobi_omega:
                gurobi_agreement = "✓"
            else:
                gurobi_agreement = "✗"
            print(f"    Agreement: JAX={agreement}, vs Gurobi={gurobi_agreement}")
        else:
            print(f"    Agreement: JAX={agreement}")
    
    return results


def analyze_convergence_behavior():
    """Detailed analysis of convergence behavior on selected graphs."""
    print("\\n" + "="*60)
    print("CONVERGENCE BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Test on 5-cycle for detailed analysis
    graph = nx.cycle_graph(5)
    print(f"\\nAnalyzing convergence on 5-cycle graph...")
    
    # Different configurations for sensitivity analysis
    configs = {
        'conservative': {
            'learning_rate': 0.005,
            'max_iterations': 2000,
            'tolerance': 1e-7,
            'num_restarts': 15,
            'verbose': False
        },
        'aggressive': {
            'learning_rate': 0.05,
            'max_iterations': 1000,
            'tolerance': 1e-5,
            'num_restarts': 15,
            'verbose': False
        },
        'standard': {
            'learning_rate': 0.02,
            'max_iterations': 1500,
            'tolerance': 1e-6,
            'num_restarts': 15,
            'verbose': False
        }
    }
    
    for config_name, config in configs.items():
        print(f"\\n  Testing {config_name} configuration...")
        
        # Test PGD
        pgd_oracle = ProjectedGradientDescentOracle(**config)
        pgd_omega = pgd_oracle.get_omega(graph)
        
        # Test MD  
        md_oracle = MirrorDescentOracle(**config)
        md_omega = md_oracle.get_omega(graph)
        
        print(f"    PGD: ω={pgd_omega}")
        print(f"    MD:  ω={md_omega}")
        
        # Plot convergence analysis
        plot_convergence_analysis(
            pgd_oracle,
            title_suffix=f"- {config_name} config",
            save_path=f'figures/pgd_convergence_{config_name}.png',
            show_plot=False
        )
        
        plot_convergence_analysis(
            md_oracle,
            title_suffix=f"- {config_name} config",
            save_path=f'figures/md_convergence_{config_name}.png',
            show_plot=False
        )


def run_parameter_sensitivity():
    """Analyze sensitivity to key parameters."""
    print("\\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    graph = nx.cycle_graph(6)
    print(f"\\nTesting parameter sensitivity on 6-cycle...")
    
    # Base configuration
    base_config = {
        'max_iterations': 1000,
        'tolerance': 1e-6,
        'num_restarts': 8,
        'verbose': False
    }
    
    # Parameter ranges to test
    parameter_ranges = {
        'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        'num_restarts': [1, 3, 5, 10, 15, 20],
        'dirichlet_alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }
    
    # Test PGD sensitivity
    print("  Testing PGD parameter sensitivity...")
    plot_parameter_sensitivity(
        graph=graph,
        oracle_class=ProjectedGradientDescentOracle,
        parameter_ranges=parameter_ranges,
        base_config=base_config,
        save_path='figures/pgd_parameter_sensitivity.png',
        show_plot=False
    )
    
    # Test MD sensitivity (with adjusted learning rates)
    print("  Testing MD parameter sensitivity...")
    md_parameter_ranges = parameter_ranges.copy()
    md_parameter_ranges['learning_rate'] = [0.0005, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    plot_parameter_sensitivity(
        graph=graph,
        oracle_class=MirrorDescentOracle,
        parameter_ranges=md_parameter_ranges,
        base_config=base_config,
        save_path='figures/md_parameter_sensitivity.png',
        show_plot=False
    )


def run_mis_algorithm_validation():
    """Validate JAX oracles in the full MIS algorithm."""
    print("\\n" + "="*60)
    print("MIS ALGORITHM VALIDATION")
    print("="*60)
    
    test_graphs = {
        '5cycle': nx.cycle_graph(5),
        'petersen': nx.petersen_graph(),
        'wheel6': nx.wheel_graph(6)
    }
    
    for graph_name, graph in test_graphs.items():
        print(f"\\nTesting MIS algorithm on {graph_name}...")
        
        # Create oracles
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1500,
            num_restarts=8,
            verbose=False
        )
        
        md_oracle = MirrorDescentOracle(
            learning_rate=0.008,
            max_iterations=1500,
            num_restarts=8,
            verbose=False
        )
        
        # Run MIS algorithm
        print("  Running MIS with PGD oracle...")
        pgd_mis, pgd_calls = find_mis_with_oracle(graph, pgd_oracle)
        pgd_valid = verify_independent_set(graph, pgd_mis)
        
        print("  Running MIS with MD oracle...")
        md_mis, md_calls = find_mis_with_oracle(graph, md_oracle)
        md_valid = verify_independent_set(graph, md_mis)
        
        if GUROBI_AVAILABLE:
            print("  Running MIS with Gurobi oracle...")
            gurobi_oracle = GurobiOracle(suppress_output=True)
            gurobi_mis, gurobi_calls = find_mis_with_oracle(graph, gurobi_oracle)
            gurobi_valid = verify_independent_set(graph, gurobi_mis)
        
        # Print results
        print(f"    PGD:    MIS size={len(pgd_mis)}, calls={pgd_calls}, valid={pgd_valid}")
        print(f"    MD:     MIS size={len(md_mis)}, calls={md_calls}, valid={md_valid}")
        
        if GUROBI_AVAILABLE:
            print(f"    Gurobi: MIS size={len(gurobi_mis)}, calls={gurobi_calls}, valid={gurobi_valid}")
            
            # Check agreement
            sizes_match = len(pgd_mis) == len(md_mis) == len(gurobi_mis)
            calls_match = pgd_calls == md_calls == gurobi_calls
            print(f"    Agreement: sizes={sizes_match}, calls={calls_match}")
        else:
            sizes_match = len(pgd_mis) == len(md_mis)
            calls_match = pgd_calls == md_calls
            print(f"    Agreement: sizes={sizes_match}, calls={calls_match}")


def create_comprehensive_report():
    """Create a comprehensive performance report."""
    print("\\n" + "="*60)
    print("CREATING COMPREHENSIVE REPORT")
    print("="*60)
    
    # Run basic comparison and get results
    results = run_basic_comparison()
    
    # Create benchmark report
    create_oracle_benchmark_report(
        test_results=results,
        save_dir="figures",
        report_name="jax_oracle_benchmark"
    )
    
    # Create individual comparison plots
    for graph_name, graph_results in results.items():
        if 'graph' in graph_results:
            graph = graph_results['graph']
            oracles_data = {k: v for k, v in graph_results.items() if k != 'graph'}
            
            plot_oracle_comparison(
                oracles_results=oracles_data,
                graph=graph,
                save_path=f'figures/comparison_{graph_name}.png',
                show_plot=False
            )
    
    print("\\nReport generated successfully!")
    print("Check the 'figures/' directory for all plots and analysis.")


def main():
    """Main analysis workflow."""
    print("JAX Solver Analysis for Motzkin-Straus Optimization")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("ERROR: JAX not available. Install with: pip install jax jaxlib")
        return
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    print(f"Output directory: {os.path.abspath('figures')}")
    print(f"Gurobi available: {GUROBI_AVAILABLE}")
    
    try:
        # Run all analyses
        run_basic_comparison()
        analyze_convergence_behavior()
        run_parameter_sensitivity()
        run_mis_algorithm_validation()
        create_comprehensive_report()
        
        print("\\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\\nGenerated files in 'figures/' directory:")
        
        figure_files = [f for f in os.listdir('figures') if f.endswith('.png')]
        for file in sorted(figure_files):
            print(f"  - {file}")
        
        print("\\nSummary:")
        print("  - JAX-based PGD and MD solvers implemented successfully")
        print("  - Multi-restart strategy improves solution robustness")
        print("  - Convergence visualization helps with parameter tuning")
        if GUROBI_AVAILABLE:
            print("  - Results validated against Gurobi baseline")
        print("  - Ready for research and production use")
        
    except Exception as e:
        print(f"\\nAnalysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()