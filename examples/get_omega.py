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

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import existing infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from demo_dimacs_clique_solvers import read_dimacs_graph

# Import motzkin-straus package components
from motzkinstraus import (
    GUROBI_AVAILABLE, 
    SCIPY_MILP_AVAILABLE
)

# Import MILP solver functions  
try:
    from motzkinstraus.solvers.milp import get_clique_number_milp
except ImportError:
    get_clique_number_milp = None

try:
    from motzkinstraus.solvers.scipy_milp import get_clique_number_scipy
except ImportError:
    get_clique_number_scipy = None

# Import oracle classes
try:
    from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
    JAX_PGD_AVAILABLE = True
except ImportError:
    ProjectedGradientDescentOracle = None
    JAX_PGD_AVAILABLE = False

try:
    from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
    JAX_MIRROR_AVAILABLE = True
except ImportError:
    MirrorDescentOracle = None
    JAX_MIRROR_AVAILABLE = False

try:
    from motzkinstraus.oracles.gurobi import GurobiOracle
    GUROBI_ORACLE_AVAILABLE = GUROBI_AVAILABLE
except ImportError:
    GurobiOracle = None
    GUROBI_ORACLE_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DiracOracle = None
    DIRAC_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle
    DIRAC_PGD_HYBRID_AVAILABLE = True
except ImportError:
    DiracPGDHybridOracle = None
    DIRAC_PGD_HYBRID_AVAILABLE = False


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
    "networkx_exact",    # NetworkX exact (small graphs only)
    "dirac_oracle",      # Dirac Oracle
    "dirac_pgd_hybrid",  # Dirac-PGD Hybrid Oracle (best of both worlds)
]

# Solver configuration parameters
OMEGA_CONFIG = {
    'timeout_per_solver': 120.0,
    
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
        'suppress_output': True           # Suppress Gurobi console output
    },
    
    # NetworkX configuration
    'networkx_config': {
        'max_nodes': 20                   # Size limit for NetworkX exact solver
    },
    
    # Dirac configuration
    'dirac_config': {
        'num_samples': 30,                # Number of quantum annealing samples
        'relax_schedule': 2,              # Relaxation schedule parameter
        'solution_precision': None       # Solution precision
    },
    
    # Dirac-PGD Hybrid configuration
    'dirac_pgd_hybrid_config': {
        'nx_threshold': 15,               # Use NetworkX for graphs ≤ 15 nodes
        'dirac_num_samples': 50,          # Number of Dirac samples for initialization
        'dirac_relax_schedule': 3,        # Dirac relaxation schedule
        'dirac_solution_precision': 0.001,  # Dirac solution precision
        'pgd_tolerance': 1e-8,            # High-precision PGD tolerance
        'pgd_max_iterations': 3000,       # Maximum PGD refinement iterations
        'pgd_learning_rate': 0.015,       # PGD learning rate
        'fallback_num_restarts': 15,      # PGD restarts if Dirac fails
        'verbose': True                   # Show detailed output for debugging
    }
}


# =============================================================================
# SOLVER WRAPPER CLASSES
# =============================================================================

class OmegaSolver:
    """Base class for omega computation solvers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        """
        Compute the maximum clique size (ω) for a graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Tuple of (omega, runtime_seconds, success, error_message)
        """
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if solver dependencies are available."""
        return True


class JAXPGDOmegaSolver(OmegaSolver):
    """JAX Projected Gradient Descent Oracle solver."""
    
    def __init__(self, config: Dict):
        super().__init__("JAX PGD Oracle")
        if not JAX_PGD_AVAILABLE:
            self.oracle = None
        else:
            self.oracle = ProjectedGradientDescentOracle(
                learning_rate=config['learning_rate_pgd'],
                max_iterations=config['max_iterations'],
                num_restarts=config['num_restarts'],
                tolerance=config['tolerance'],
                verbose=config['verbose']
            )
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "JAX PGD Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return JAX_PGD_AVAILABLE and self.oracle is not None and self.oracle.is_available


class JAXMirrorOmegaSolver(OmegaSolver):
    """JAX Mirror Descent Oracle solver."""
    
    def __init__(self, config: Dict):
        super().__init__("JAX Mirror Oracle")
        if not JAX_MIRROR_AVAILABLE:
            self.oracle = None
        else:
            self.oracle = MirrorDescentOracle(
                learning_rate=config['learning_rate_md'],
                max_iterations=config['max_iterations'],
                num_restarts=config['num_restarts'],
                tolerance=config['tolerance'],
                verbose=config['verbose']
            )
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "JAX Mirror Descent Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return JAX_MIRROR_AVAILABLE and self.oracle is not None and self.oracle.is_available


class GurobiOracleOmegaSolver(OmegaSolver):
    """Gurobi Oracle (Motzkin-Straus) solver."""
    
    def __init__(self, config: Dict):
        super().__init__("Gurobi Oracle")
        if not GUROBI_ORACLE_AVAILABLE:
            self.oracle = None
        else:
            self.oracle = GurobiOracle(suppress_output=config['suppress_output'])
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Gurobi Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return GUROBI_ORACLE_AVAILABLE and self.oracle is not None and self.oracle.is_available


class GurobiMILPOmegaSolver(OmegaSolver):
    """Gurobi MILP (direct combinatorial) solver."""
    
    def __init__(self, config: Dict):
        super().__init__("Gurobi MILP")
        self.suppress_output = config['suppress_output']
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Gurobi MILP not available"
        
        start_time = time.time()
        try:
            omega = get_clique_number_milp(graph, suppress_output=self.suppress_output)
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return GUROBI_AVAILABLE and get_clique_number_milp is not None


class ScipyMILPOmegaSolver(OmegaSolver):
    """SciPy MILP (direct combinatorial) solver."""
    
    def __init__(self, config: Dict):
        super().__init__("SciPy MILP")
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "SciPy MILP not available"
        
        start_time = time.time()
        try:
            omega = get_clique_number_scipy(graph)
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return SCIPY_MILP_AVAILABLE and get_clique_number_scipy is not None


class NetworkXOmegaSolver(OmegaSolver):
    """NetworkX exact solver (for small graphs)."""
    
    def __init__(self, config: Dict):
        super().__init__("NetworkX Exact")
        self.max_nodes = config['max_nodes']
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if graph.number_of_nodes() > self.max_nodes:
            return 0, 0.0, False, f"Graph too large ({graph.number_of_nodes()} > {self.max_nodes} nodes)"
        
        start_time = time.time()
        try:
            omega = nx.graph_clique_number(graph)  # Built-in NetworkX function
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return True  # NetworkX is always available


class DiracOracleOmegaSolver(OmegaSolver):
    """Dirac Oracle (quantum annealing) solver."""
    
    def __init__(self, config: Dict):
        super().__init__("Dirac Oracle")
        if not DIRAC_AVAILABLE:
            self.oracle = None
        else:
            try:
                self.oracle = DiracOracle(
                    num_samples=config['num_samples'],
                    relax_schedule=config['relax_schedule'],
                    solution_precision=config['solution_precision']
                )
            except Exception:
                self.oracle = None
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Dirac Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return DIRAC_AVAILABLE and self.oracle is not None and self.oracle.is_available


class DiracPGDHybridOmegaSolver(OmegaSolver):
    """Dirac-PGD Hybrid Oracle solver (best of both worlds)."""
    
    def __init__(self, config: Dict):
        super().__init__("Dirac-PGD Hybrid")
        if not DIRAC_PGD_HYBRID_AVAILABLE:
            self.oracle = None
        else:
            try:
                self.oracle = DiracPGDHybridOracle(
                    nx_threshold=config['nx_threshold'],
                    dirac_num_samples=config['dirac_num_samples'],
                    dirac_relax_schedule=config['dirac_relax_schedule'],
                    dirac_solution_precision=config['dirac_solution_precision'],
                    pgd_tolerance=config['pgd_tolerance'],
                    pgd_max_iterations=config['pgd_max_iterations'],
                    pgd_learning_rate=config['pgd_learning_rate'],
                    fallback_num_restarts=config['fallback_num_restarts'],
                    verbose=config['verbose']
                )
            except Exception:
                self.oracle = None
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Dirac-PGD Hybrid Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            
            # Get additional solve info for detailed reporting
            solve_info = self.oracle.get_last_solve_info()
            method = solve_info.get('method', 'unknown')
            
            # Create informative message about which method was used
            if method == 'networkx_exact':
                info_msg = f"Used NetworkX exact (≤{self.oracle.nx_threshold} nodes)"
            elif method == 'dirac_pgd_refinement':
                improvement = solve_info.get('improvement', 0)
                info_msg = f"Used Dirac+PGD refinement (improvement: {improvement:+.2e})"
            elif method == 'dirac_only':
                info_msg = "Used Dirac only (PGD refinement failed)"
            elif method == 'pgd_fallback':
                dirac_available = solve_info.get('dirac_available', False)
                dirac_status = "unavailable" if not dirac_available else "failed"
                info_msg = f"Used PGD fallback (Dirac {dirac_status})"
            else:
                info_msg = f"Used method: {method}"
            
            return omega, runtime, True, info_msg
            
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return DIRAC_PGD_HYBRID_AVAILABLE and self.oracle is not None and self.oracle.is_available


# =============================================================================
# MAIN COMPUTATION FUNCTIONS
# =============================================================================

def create_solvers() -> List[OmegaSolver]:
    """Create solver instances based on configuration."""
    solvers = []
    
    if "jax_pgd" in OMEGA_SOLVERS:
        solvers.append(JAXPGDOmegaSolver(OMEGA_CONFIG['jax_config']))
    
    if "jax_mirror" in OMEGA_SOLVERS:
        solvers.append(JAXMirrorOmegaSolver(OMEGA_CONFIG['jax_config']))
    
    if "gurobi_oracle" in OMEGA_SOLVERS:
        solvers.append(GurobiOracleOmegaSolver(OMEGA_CONFIG['gurobi_config']))
    
    if "gurobi_milp" in OMEGA_SOLVERS:
        solvers.append(GurobiMILPOmegaSolver(OMEGA_CONFIG['gurobi_config']))
    
    if "scipy_milp" in OMEGA_SOLVERS:
        solvers.append(ScipyMILPOmegaSolver({}))
    
    if "networkx_exact" in OMEGA_SOLVERS:
        solvers.append(NetworkXOmegaSolver(OMEGA_CONFIG['networkx_config']))
    
    if "dirac_oracle" in OMEGA_SOLVERS:
        solvers.append(DiracOracleOmegaSolver(OMEGA_CONFIG['dirac_config']))
    
    if "dirac_pgd_hybrid" in OMEGA_SOLVERS:
        solvers.append(DiracPGDHybridOmegaSolver(OMEGA_CONFIG['dirac_pgd_hybrid_config']))
    
    return solvers


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
    solvers = create_solvers()
    
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
        
        if not args.quiet:
            print(f"\n🎉 ω computation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())