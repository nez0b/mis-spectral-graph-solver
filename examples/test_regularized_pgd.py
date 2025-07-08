#!/usr/bin/env python3
"""
Test Regularized PGD Oracles for Maximum Clique Size (ω) Computation

This script tests various PGD-based solvers including the newly implemented
regularized versions. It focuses on gradient-based optimization methods
without requiring Gurobi or SciPy MILP dependencies.

Key features:
- JAX Projected Gradient Descent Oracle
- JAX Mirror Descent Oracle  
- Regularized JAX PGD Oracle (NEW)
- NetworkX exact solver for validation
- Comparative analysis between regularized and standard methods

USAGE:
    python examples/test_regularized_pgd.py DIMACS/graph.dimacs [OPTIONS]

EXAMPLES:
    # Test all PGD methods on triangle graph
    python examples/test_regularized_pgd.py DIMACS/triangle.dimacs
    
    # Quiet mode with JSON output
    python examples/test_regularized_pgd.py DIMACS/erdos_renyi_15_p03_seed123.dimacs --quiet --format json
    
    # Test with custom regularization parameter
    python examples/test_regularized_pgd.py DIMACS/triangle.dimacs --regularization-c 1.0
    
    # Compare regularized vs unregularized performance
    python examples/test_regularized_pgd.py DIMACS/cycle_12.clq --compare-regularization
"""

import os
import sys
import time
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import DIMACS reading utility
from motzkinstraus.io import read_dimacs_graph

# Import omega solver classes - only PGD-based methods
from motzkinstraus.solvers.omega import create_omega_solvers


# =============================================================================
# SOLVER CONFIGURATION - PGD FOCUSED
# =============================================================================

# List of PGD-based solvers to run
PGD_OMEGA_SOLVERS = [
    "jax_pgd",                # Standard JAX Projected Gradient Descent Oracle
    "jax_pgd_regularized",    # NEW: Regularized JAX PGD Oracle
    # "jax_mirror",             # JAX Mirror Descent Oracle  
    "networkx_exact",         # NetworkX exact (for validation on small graphs)
]

# Solver configuration parameters - optimized for PGD methods
PGD_OMEGA_CONFIG = {
    'timeout_per_solver': 60.0,
    
    # JAX Oracle configurations - tuned for better convergence
    'jax_config': {
        'learning_rate_pgd': 0.02,        # PGD learning rate
        'learning_rate_md': 0.01,         # Mirror Descent learning rate  
        'max_iterations': 1000,           # Increased for better convergence
        'num_restarts': 10,               # More restarts for robustness
        'tolerance': 1e-6,                # Convergence tolerance
        'verbose': False,                 # Suppress detailed optimization output
        'regularization_c': 0.3           # Default regularization parameter (avoid c=0.5 for polynomial methods)
    },
    
    # NetworkX configuration
    'networkx_config': {
        'max_nodes': 20                   # Size limit for NetworkX exact solver
    }
}


# =============================================================================
# PGD OMEGA COMPUTATION FUNCTIONS
# =============================================================================

def run_pgd_omega_computation(graph: nx.Graph, graph_name: str, 
                             regularization_c: float = 0.3,
                             quiet: bool = False) -> Dict[str, Tuple[int, float, bool, str]]:
    """Run all configured PGD solvers on a graph and compare ω results."""
    if not quiet:
        print(f"\n{'='*70}")
        print(f"Graph: {graph_name}")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        if graph.number_of_nodes() > 1:
            density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2)
            print(f"Density: {density:.3f}")
        print(f"Regularization parameter c: {regularization_c}")
        print('='*70)
    
    # Update regularization parameter in config
    config = PGD_OMEGA_CONFIG.copy()
    config['jax_config']['regularization_c'] = regularization_c
    
    results = {}
    solvers = create_omega_solvers(PGD_OMEGA_SOLVERS, config)
    
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
                if error:  # Additional info for regularized methods
                    print(f"      ℹ️  {error}")
        else:
            if not quiet:
                print(f"   ❌ FAILED: {error}")
        
        results[solver.name] = (omega, runtime, success, error)
    
    return results


def compare_regularization_effect(graph: nx.Graph, graph_name: str, 
                                c_values: List[float] = [0.0, 0.3, 0.5, 1.0],
                                quiet: bool = False) -> Dict[str, Dict]:
    """Compare regularized vs unregularized performance across different c values."""
    if not quiet:
        print(f"\n{'='*70}")
        print(f"REGULARIZATION COMPARISON: {graph_name}")
        print(f"Testing c values: {c_values}")
        print('='*70)
    
    # Import oracle classes for direct comparison
    try:
        from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
        from motzkinstraus.oracles.jax_pgd_regularized import create_regularized_jax_pgd_oracle
        
        comparison_results = {}
        
        # Test unregularized oracle
        unreg_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=1000,
            num_restarts=20,
            verbose=False
        )
        
        start_time = time.time()
        unreg_omega = unreg_oracle.get_omega(graph)
        unreg_runtime = time.time() - start_time
        
        comparison_results['unregularized'] = {
            'omega': unreg_omega,
            'runtime': unreg_runtime,
            'c_value': 0.0
        }
        
        if not quiet:
            print(f"🔵 Unregularized: ω = {unreg_omega} in {unreg_runtime:.3f}s")
        
        # Test regularized oracle with different c values
        for c in c_values:
            reg_oracle = create_regularized_jax_pgd_oracle(
                c=c,
                learning_rate=0.02,
                max_iterations=1000,
                num_restarts=20,
                verbose=False
            )
            
            start_time = time.time()
            reg_omega = reg_oracle.get_omega(graph)
            reg_runtime = time.time() - start_time
            
            comparison_results[f'regularized_c_{c}'] = {
                'omega': reg_omega,
                'runtime': reg_runtime,
                'c_value': c
            }
            
            if not quiet:
                consistency = "✅" if reg_omega == unreg_omega else "⚠️"
                print(f"🔵 Regularized (c={c}): ω = {reg_omega} in {reg_runtime:.3f}s {consistency}")
        
        return comparison_results
        
    except ImportError as e:
        if not quiet:
            print(f"❌ Could not import regularized oracle: {e}")
        return {}


def create_pgd_summary(results: Dict[str, Tuple[int, float, bool, str]], 
                      format_type: str = "table") -> str:
    """Create summary of PGD ω computation results."""
    if format_type == "json":
        json_results = {}
        for solver_name, (omega, runtime, success, error) in results.items():
            json_results[solver_name] = {
                "omega": omega,
                "runtime_seconds": runtime,
                "success": success,
                "error_message": error if not success else "",
                "method_type": "exact" if "NetworkX" in solver_name else "pgd_oracle"
            }
        return json.dumps(json_results, indent=2)
    
    elif format_type == "csv":
        lines = ["Solver,Omega,Runtime_Seconds,Status,Method_Type,Notes"]
        
        for solver_name, (omega, runtime, success, error) in results.items():
            status = "SUCCESS" if success else "FAILED"
            method_type = "exact" if "NetworkX" in solver_name else "pgd_oracle"
            notes = error.replace(",", ";") if error else ""  # Escape commas for CSV
            lines.append(f"{solver_name},{omega},{runtime:.4f},{status},{method_type},{notes}")
        
        return "\n".join(lines)
    
    else:  # table format
        output = []
        output.append(f"\n{'='*80}")
        output.append("PGD-BASED ω COMPUTATION RESULTS SUMMARY")
        output.append('='*80)
        
        output.append(f"{'Solver':<25} {'ω':<4} {'Runtime(s)':<12} {'Status':<10} {'Notes':<20}")
        output.append('-' * 80)
        
        exact_solvers = ["NetworkX Exact"]
        regularized_solvers = ["Regularized JAX PGD Oracle"]
        
        for solver_name, (omega, runtime, success, error) in results.items():
            status = "SUCCESS" if success else "FAILED"
            
            if not success:
                notes = error[:20] + "..." if len(error) > 20 else error
            elif solver_name in exact_solvers:
                notes = "Exact"
            elif solver_name in regularized_solvers:
                notes = "Regularized PGD"
            else:
                notes = "Standard PGD"
            
            output.append(f"{solver_name:<25} {omega:<4} {runtime:<12.4f} {status:<10} {notes:<20}")
        
        # Validation section
        exact_results = [(name, result) for name, result in results.items() 
                        if name in exact_solvers and result[2]]  # success = True
        
        pgd_results = [(name, result) for name, result in results.items() 
                      if name not in exact_solvers and result[2]]  # success = True
        
        if exact_results and pgd_results:
            exact_omega = exact_results[0][1][0]
            pgd_omegas = [result[0] for _, result in pgd_results]
            correct_pgd = [omega for omega in pgd_omegas if omega == exact_omega]
            
            if len(correct_pgd) == len(pgd_omegas):
                output.append(f"\n✅ Validation: All PGD methods agree with exact result (ω = {exact_omega})")
            else:
                output.append(f"\n⚠️  Validation: {len(correct_pgd)}/{len(pgd_omegas)} PGD methods agree with exact (ω = {exact_omega})")
                output.append(f"   PGD results: {pgd_omegas}")
        elif exact_results:
            output.append(f"\n✅ Validation: Exact solver result (ω = {exact_results[0][1][0]})")
        elif pgd_results:
            pgd_omegas = [result[0] for _, result in pgd_results]
            if len(set(pgd_omegas)) == 1:
                output.append(f"\n✅ Consistency: All PGD methods agree (ω = {pgd_omegas[0]})")
            else:
                output.append(f"\n⚠️  Inconsistency: PGD methods disagree: {pgd_omegas}")
        else:
            output.append(f"\n⚠️  Warning: No successful solver results")
        
        return "\n".join(output)


def main():
    """Main function for PGD-based omega computation testing."""
    parser = argparse.ArgumentParser(
        description="Test PGD-based methods for maximum clique size (ω) computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/test_regularized_pgd.py DIMACS/triangle.dimacs
  python examples/test_regularized_pgd.py DIMACS/erdos_renyi_15_p03_seed123.dimacs --quiet
  python examples/test_regularized_pgd.py DIMACS/cycle_12.clq --format json
  python examples/test_regularized_pgd.py DIMACS/triangle.dimacs --regularization-c 1.0
  python examples/test_regularized_pgd.py DIMACS/cycle_12.clq --compare-regularization
        """
    )
    parser.add_argument("dimacs_file", help="Path to DIMACS format file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--regularization-c", type=float, default=0.3,
                       help="Regularization parameter for identity regularization, must be in [0, 1] (default: 0.3)")
    parser.add_argument("--compare-regularization", action="store_true",
                       help="Compare regularized vs unregularized performance across different c values")
    
    args = parser.parse_args()
    
    # Validate regularization parameter
    if args.regularization_c < 0 or args.regularization_c > 1:
        print(f"❌ Error: --regularization-c must be in [0, 1], got {args.regularization_c}")
        return 1
    
    if not args.quiet:
        print("🎯 PGD-Based Maximum Clique Size (ω) Computation")
        print("=" * 70)
        print("Testing PGD and gradient-based optimization methods:")
        
        # List configured solvers
        for solver in PGD_OMEGA_SOLVERS:
            emoji = "🆕" if "regularized" in solver else "🔵"
            print(f"  {emoji} {solver}")
    
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
        
        # Run standard computation
        results = run_pgd_omega_computation(
            graph, graph_name, 
            regularization_c=args.regularization_c,
            quiet=args.quiet
        )
        
        # Create and display summary
        summary = create_pgd_summary(results, format_type=args.format)
        print(summary)
        
        # Run regularization comparison if requested
        if args.compare_regularization:
            c_values = [0.0, 0.3, 0.5, 1.0]
            comparison = compare_regularization_effect(
                graph, graph_name, c_values, quiet=args.quiet
            )
            
            if comparison and not args.quiet:
                print(f"\n{'='*70}")
                print("REGULARIZATION COMPARISON SUMMARY")
                print('='*70)
                
                for method_name, data in comparison.items():
                    c_val = data['c_value']
                    omega = data['omega']
                    runtime = data['runtime']
                    
                    c_display = "none" if c_val == 0.0 else f"c={c_val}"
                    print(f"{method_name:<25} {c_display:<8} ω={omega:<3} runtime={runtime:.3f}s")
        
        if not args.quiet:
            print(f"\n🎉 PGD ω computation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())