#!/usr/bin/env python3
"""
Compute maximum clique size (ω) using various solvers.

This script reads graphs in DIMACS format and computes the maximum clique size
using multiple solver approaches including oracle-based methods (Motzkin-Straus)
and direct MILP formulations.

Similar to examples/test_er.py but focused on ω computation only.
Uses existing oracle get_omega() methods and MILP functions.

USAGE:
    python examples/get_omega.py DIMACS/graph.dimacs [OPTIONS]

EXAMPLES:
    # Basic usage with all configured solvers
    python examples/get_omega.py DIMACS/complete_k5.dimacs
    
    # Quiet mode (suppress detailed output)
    python examples/get_omega.py DIMACS/erdos_renyi_15_p05.dimacs --quiet
    
    # JSON output format
    python examples/get_omega.py DIMACS/petersen.dimacs --format json
"""

import os
import sys
import time
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import matplotlib for histogram plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import DIMACS reading utility
from motzkinstraus.io import read_dimacs_graph

# Import omega solver classes
from motzkinstraus.solvers.omega import create_omega_solvers


# =============================================================================
# SOLVER CONFIGURATION (Following examples/test_er.py pattern)
# =============================================================================

# List of solvers to run (comment out to disable)
OMEGA_SOLVERS = [
    "jax_pgd",           # JAX Projected Gradient Descent Oracle
    # "jax_mirror",        # JAX Mirror Descent Oracle  
    # "gurobi_oracle",     # Gurobi Oracle (Motzkin-Straus)
    "gurobi_milp",       # Gurobi MILP (Direct combinatorial)
    "scipy_milp",        # SciPy MILP (Direct combinatorial)
    # "networkx_exact",    # NetworkX exact (small graphs only)
    # "dirac_oracle",      # Dirac Oracle
    # "dirac_pgd_hybrid",  # Dirac-PGD Hybrid Oracle (best of both worlds)
]

# Solver configuration parameters
OMEGA_CONFIG = {
    'timeout_per_solver': 120.0,
    'plot_histograms': True,          # Whether to plot energy histograms for Dirac solver
    
    # JAX Oracle configurations
    'jax_config': {
        'learning_rate_pgd': 0.02,        # PGD learning rate
        'learning_rate_md': 0.01,         # Mirror Descent learning rate  
        'max_iterations': 500,           # Maximum optimization iterations
        'num_restarts': 5,                # Number of random restarts
        'tolerance': 1e-6,                # Convergence tolerance
        'verbose': False                  # Suppress optimization output
    },
    
    # Gurobi configurations
    'gurobi_config': {
        'suppress_output': False           # Suppress Gurobi console output
    },
    
    # NetworkX configuration
    'networkx_config': {
        'max_nodes': 150                   # Size limit for NetworkX exact solver
    },
    
    # Dirac configuration
    'dirac_config': {
        'num_samples': 100,                # Number of quantum annealing samples
        'relax_schedule': 2,              # Relaxation schedule parameter
        'solution_precision': None,       # Solution precision
        'sum_constraint': 1,              # Simplex constraint (default)
        'mean_photon_number': None,     # Example: Override default photon number
        'quantum_fluctuation_coefficient': 1,  # Example: Override default fluctuation
        'save_raw_data': True,            # Save raw response from Dirac solver
        'raw_data_path': 'data'           # Directory to save raw data files
    },
    
    # Dirac-PGD Hybrid configuration
    'dirac_pgd_hybrid_config': {
        'nx_threshold': 15,               # Use NetworkX for graphs ≤ 15 nodes
        'dirac_num_samples': 50,          # Number of Dirac samples for initialization
        'dirac_relax_schedule': 3,        # Dirac relaxation schedule
        'dirac_solution_precision': 0.001,  # Dirac solution precision
        'dirac_sum_constraint': 1,        # Simplex constraint for Dirac solver
        'dirac_mean_photon_number': 0.003,  # Example: Override default photon number
        'dirac_quantum_fluctuation_coefficient': 40,  # Example: Override default fluctuation
        'pgd_tolerance': 1e-8,            # High-precision PGD tolerance
        'pgd_max_iterations': 3000,       # Maximum PGD refinement iterations
        'pgd_learning_rate': 0.015,       # PGD learning rate
        'fallback_num_restarts': 15,      # PGD restarts if Dirac fails
        'verbose': True                   # Show detailed output for debugging
    }
}


# =============================================================================
# OMEGA COMPUTATION FUNCTIONS
# =============================================================================


# =============================================================================
# MAIN COMPUTATION FUNCTIONS
# =============================================================================



def run_omega_computation(graph: nx.Graph, graph_name: str, quiet: bool = False) -> Dict[str, Tuple[int, float, bool, str]]:
    """Run all configured solvers on a graph and compare ω results."""
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Graph: {graph_name}")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        if graph.number_of_nodes() > 1:
            density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2)
            print(f"Density: {density:.3f}")
        print('='*60)
    
    results = {}
    solvers = create_omega_solvers(OMEGA_SOLVERS, OMEGA_CONFIG)
    
    for solver in solvers:
        if not solver.is_available():
            if not quiet:
                print(f"🔴 {solver.name}: SKIPPED (not available)")
            results[solver.name] = (0, 0.0, False, "Not available")
            continue
        
        if not quiet:
            print(f"🔵 {solver.name}: Computing ω...")
        
        omega, runtime, success, error = solver.compute_omega(graph)
        
        if success:
            if not quiet:
                print(f"   ✅ ω = {omega} in {runtime:.3f}s")
        else:
            if not quiet:
                print(f"   ❌ FAILED: {error}")
        
        results[solver.name] = (omega, runtime, success, error)
    
    return results


def plot_energy_histogram(graph_name: str, data_dir: str = "data", show_plot: bool = True) -> bool:
    """
    Plot histogram of energies from the most recent Dirac solver response.
    
    Args:
        graph_name: Name of the graph for plot title.
        data_dir: Directory containing saved Dirac response files.
        show_plot: Whether to display the plot immediately.
        
    Returns:
        True if histogram was successfully created, False otherwise.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, cannot plot histogram")
        return False
    
    try:
        # Find the most recent Dirac response file
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: Data directory {data_dir} does not exist")
            return False
        
        json_files = list(data_path.glob("dirac_response_*.json"))
        if not json_files:
            print(f"Warning: No Dirac response files found in {data_dir}")
            return False
        
        # Get the most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📊 Plotting histogram from: {latest_file.name}")
        
        # Load the JSON data
        with open(latest_file, 'r') as f:
            response = json.load(f)
        
        # Extract energies from the response
        if "results" in response and "energies" in response["results"]:
            energies = response["results"]["energies"]
            
            if not energies:
                print("Warning: No energies found in response")
                return False
            
            # Create the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(energies, bins=20, edgecolor='black', alpha=0.7)
            
            # Find the best (minimum) energy for the vertical line
            best_energy = min(energies)
            plt.axvline(x=best_energy, color='red', linestyle='--', linewidth=2, 
                       label=f'Best Energy: {best_energy:.6f}')
            
            # Labels and title
            plt.xlabel('Energy')
            plt.ylabel('Frequency')
            plt.title(f'{graph_name}: Energy Distribution from Dirac Solver\n'
                     f'Samples: {len(energies)}, Best: {best_energy:.6f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_file = data_path / f"energy_histogram_{graph_name.replace(' ', '_')}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"💾 Saved histogram to: {plot_file}")
            
            # Show the plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return True
        else:
            print("Warning: No energies found in Dirac response structure")
            return False
            
    except Exception as e:
        print(f"Error creating energy histogram: {e}")
        return False


def create_omega_summary(results: Dict[str, Tuple[int, float, bool, str]], format_type: str = "table") -> str:
    """Create summary of ω computation results."""
    if format_type == "json":
        json_results = {}
        for solver_name, (omega, runtime, success, error) in results.items():
            json_results[solver_name] = {
                "omega": omega,
                "runtime_seconds": runtime,
                "success": success,
                "error_message": error if not success else ""
            }
        return json.dumps(json_results, indent=2)
    
    elif format_type == "csv":
        lines = ["Solver,Omega,Runtime_Seconds,Status,Notes"]
        exact_solvers = ["Gurobi MILP", "SciPy MILP", "NetworkX Exact"]
        
        for solver_name, (omega, runtime, success, error) in results.items():
            status = "SUCCESS" if success else "FAILED"
            notes = "Exact" if solver_name in exact_solvers else "Oracle-based"
            if not success:
                notes = error.replace(",", ";")  # Escape commas for CSV
            lines.append(f"{solver_name},{omega},{runtime:.4f},{status},{notes}")
        
        return "\n".join(lines)
    
    else:  # table format
        output = []
        output.append(f"\n{'='*70}")
        output.append("ω COMPUTATION RESULTS SUMMARY")
        output.append('='*70)
        
        output.append(f"{'Solver':<20} {'ω':<4} {'Runtime(s)':<12} {'Status':<10} {'Notes':<15}")
        output.append('-' * 70)
        
        exact_solvers = ["Gurobi MILP", "SciPy MILP", "NetworkX Exact"]
        
        for solver_name, (omega, runtime, success, error) in results.items():
            status = "SUCCESS" if success else "FAILED"
            
            if not success:
                notes = error[:15] + "..." if len(error) > 15 else error
            elif solver_name in exact_solvers:
                notes = "Exact"
            elif solver_name == "Dirac-PGD Hybrid" and error:
                # Show hybrid method details
                notes = error[:15] + "..." if len(error) > 15 else error
            else:
                notes = "Oracle-based"
            
            output.append(f"{solver_name:<20} {omega:<4} {runtime:<12.4f} {status:<10} {notes:<15}")
        
        # Validation
        exact_results = [(name, result) for name, result in results.items() 
                        if name in exact_solvers and result[2]]  # success = True
        
        if len(exact_results) > 1:
            omega_values = [result[0] for _, result in exact_results]
            if len(set(omega_values)) == 1:
                output.append(f"\n✅ Validation: All exact solvers agree (ω = {omega_values[0]})")
            else:
                output.append(f"\n⚠️  Warning: Exact solvers disagree: {omega_values}")
        elif len(exact_results) == 1:
            output.append(f"\n✅ Validation: Single exact solver result (ω = {exact_results[0][1][0]})")
        else:
            output.append(f"\n⚠️  Warning: No exact solver results available")
        
        return "\n".join(output)


def main():
    """Main function following examples/test_er.py pattern."""
    parser = argparse.ArgumentParser(
        description="Compute maximum clique size (ω) using multiple solvers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/get_omega.py DIMACS/complete_k5.dimacs
  python examples/get_omega.py DIMACS/erdos_renyi_15_p05.dimacs --quiet
  python examples/get_omega.py DIMACS/petersen.dimacs --format json
        """
    )
    parser.add_argument("dimacs_file", help="Path to DIMACS format file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--no-plot", action="store_true", help="Disable energy histogram plotting")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🎯 Maximum Clique Size (ω) Computation")
        print("=" * 60)
        print("Using multiple solver approaches:")
        
        # List configured solvers
        for solver in OMEGA_SOLVERS:
            print(f"  • {solver}")
    
    # Check DIMACS file exists
    if not os.path.exists(args.dimacs_file):
        print(f"❌ DIMACS file not found: {args.dimacs_file}")
        return 1
    
    try:
        # Read DIMACS file
        if not args.quiet:
            print(f"\n📁 Reading DIMACS file: {args.dimacs_file}")
        
        graph = read_dimacs_graph(args.dimacs_file)
        graph_name = Path(args.dimacs_file).stem.replace('_', ' ').title()
        
        if not args.quiet:
            print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Run computation
        results = run_omega_computation(graph, graph_name, quiet=args.quiet)
        
        # Create and display summary
        summary = create_omega_summary(results, format_type=args.format)
        print(summary)
        
        # Plot energy histogram if Dirac solver was used and plotting is enabled
        if (OMEGA_CONFIG.get('plot_histograms', False) and not args.no_plot 
            and "dirac_oracle" in OMEGA_SOLVERS):
            dirac_results = results.get("Dirac Oracle", (0, 0.0, False, ""))
            if dirac_results[2]:  # If Dirac solver was successful
                if not args.quiet:
                    print(f"\n📊 Creating energy histogram...")
                plot_energy_histogram(graph_name, 
                                    data_dir=OMEGA_CONFIG['dirac_config'].get('raw_data_path', 'data'),
                                    show_plot=not args.quiet)
        
        if not args.quiet:
            print(f"\n🎉 ω computation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
