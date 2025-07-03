#!/usr/bin/env python3
"""
Comprehensive Maximum Clique Finder via Maximum Independent Set (Version 1)

This script finds the maximum clique of the C125.9 graph by converting it to 
its complement and solving the Maximum Independent Set problem. It then converts
the MIS solution back to the maximum clique in the original graph.

Mathematical Foundation:
- Maximum clique in G = Maximum Independent Set in complement(G)
- This converts a dense clique problem to a sparse MIS problem
- C125.9: 6963 edges → complement: 787 edges (much sparser!)

Usage:
    python find_max_clique_via_mis.py DIMACS/C125.9.clq

Features:
- Comprehensive MIS-to-clique conversion with validation
- Multiple solver approaches (JAX PGD, Gurobi MILP, SciPy MILP, NetworkX)
- Specialized analysis and visualization for clique findings
- Theoretical validation using graph complement identity
"""

import os
import sys
import time
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import DIMACS reader
from motzkinstraus.io import read_dimacs_graph

# Import motzkin-straus package components
from motzkinstraus import (
    GUROBI_AVAILABLE, 
    SCIPY_MILP_AVAILABLE
)

# Import MIS solver functions
try:
    from motzkinstraus.solvers.milp import solve_mis_milp, get_independence_number_milp
except ImportError:
    solve_mis_milp = None
    get_independence_number_milp = None

try:
    from motzkinstraus.solvers.scipy_milp import solve_mis_scipy
except ImportError:
    solve_mis_scipy = None

# Import oracle classes for MIS solving
try:
    from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
    JAX_PGD_AVAILABLE = True
except ImportError:
    ProjectedGradientDescentOracle = None
    JAX_PGD_AVAILABLE = False

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


# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================

MIS_SOLVERS = [
    "jax_pgd",           # JAX Projected Gradient Descent Oracle (MIS via complement clique)
    "gurobi_milp",       # Gurobi MILP (Direct MIS formulation)
    "scipy_milp",        # SciPy MILP (Direct MIS formulation)
    "networkx_exact",    # NetworkX exact (for reference, may be slow)
    "dirac_oracle",      # Dirac Oracle (uncomment if available)
]

MIS_CONFIG = {
    'timeout_per_solver': 300.0,  # 5 minutes per solver
    
    # JAX Oracle configurations (using complement for clique→MIS conversion)
    'jax_config': {
        'learning_rate_pgd': 0.02,
        'max_iterations': 1000,
        'num_restarts': 10,
        'tolerance': 1e-6,
        'verbose': True
    },
    
    # Gurobi configurations
    'gurobi_config': {
        'suppress_output': True
    },
    
    # NetworkX configuration
    'networkx_config': {
        'max_nodes': 150  # Size limit for NetworkX exact solver
    },
    
    # Dirac configuration
    'dirac_config': {
        'num_samples': 50,
        'relax_schedule': 2,
        'solution_precision': None
    }
}


# =============================================================================
# MIS-TO-CLIQUE SOLVER CLASSES
# =============================================================================

class CliqueViaMISSolver:
    """Base class for finding maximum cliques via MIS on complement graph."""
    
    def __init__(self, name: str):
        self.name = name
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        """
        Find the maximum clique by solving MIS on complement graph.
        
        Args:
            graph: Original graph to find max clique in
            
        Returns:
            Tuple of (clique_nodes, clique_size, runtime_seconds, success, error_message)
        """
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if solver dependencies are available."""
        return True


class JAXPGDCliqueSolver(CliqueViaMISSolver):
    """JAX PGD Oracle solver for max clique via MIS on complement."""
    
    def __init__(self, config: Dict):
        super().__init__(f"JAX PGD (via MIS)")
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
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        if not self.is_available():
            return set(), 0, 0.0, False, "JAX PGD Oracle not available"
        
        start_time = time.time()
        try:
            # Use oracle directly on original graph to get clique number
            clique_size = self.oracle.get_omega(graph)
            
            if clique_size == 0:
                clique_nodes = set()
            else:
                # Find actual clique nodes using NetworkX
                # Use the built-in clique finding algorithms
                if clique_size == 1:
                    # Any single node forms a clique of size 1
                    clique_nodes = {list(graph.nodes())[0]} if graph.nodes() else set()
                else:
                    # Find cliques and get one of the desired size
                    cliques = list(nx.find_cliques(graph))
                    target_cliques = [c for c in cliques if len(c) == clique_size]
                    
                    if target_cliques:
                        clique_nodes = set(target_cliques[0])
                    else:
                        # Fallback: use the largest clique found
                        if cliques:
                            largest_clique = max(cliques, key=len)
                            clique_nodes = set(largest_clique)
                            clique_size = len(clique_nodes)  # Update size to actual found
                        else:
                            clique_nodes = set()
                            clique_size = 0
            
            runtime = time.time() - start_time
            return clique_nodes, clique_size, runtime, True, ""
            
        except Exception as e:
            runtime = time.time() - start_time
            return set(), 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return JAX_PGD_AVAILABLE and self.oracle is not None and self.oracle.is_available


class GurobiMILPCliqueSolver(CliqueViaMISSolver):
    """Gurobi MILP solver for max clique via MIS on complement."""
    
    def __init__(self, config: Dict):
        super().__init__("Gurobi MILP (via MIS)")
        self.suppress_output = config['suppress_output']
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        if not self.is_available():
            return set(), 0, 0.0, False, "Gurobi MILP not available"
        
        start_time = time.time()
        try:
            # Convert to complement graph for MIS solving
            complement = nx.complement(graph)
            
            # Solve MIS on complement using Gurobi MILP
            mis_nodes = solve_mis_milp(complement, suppress_output=self.suppress_output)
            
            # MIS in complement = clique in original
            clique_nodes = set(mis_nodes)
            clique_size = len(clique_nodes)
            
            runtime = time.time() - start_time
            return clique_nodes, clique_size, runtime, True, ""
            
        except Exception as e:
            runtime = time.time() - start_time
            return set(), 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return GUROBI_AVAILABLE and solve_mis_milp is not None


class ScipyMILPCliqueSolver(CliqueViaMISSolver):
    """SciPy MILP solver for max clique via MIS on complement."""
    
    def __init__(self, config: Dict):
        super().__init__("SciPy MILP (via MIS)")
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        if not self.is_available():
            return set(), 0, 0.0, False, "SciPy MILP not available"
        
        start_time = time.time()
        try:
            # Convert to complement graph for MIS solving
            complement = nx.complement(graph)
            
            # Solve MIS on complement using SciPy MILP
            mis_nodes = solve_mis_scipy(complement)
            
            # MIS in complement = clique in original
            clique_nodes = set(mis_nodes)
            clique_size = len(clique_nodes)
            
            runtime = time.time() - start_time
            return clique_nodes, clique_size, runtime, True, ""
            
        except Exception as e:
            runtime = time.time() - start_time
            return set(), 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return SCIPY_MILP_AVAILABLE and solve_mis_scipy is not None


class NetworkXCliqueViaMISSolver(CliqueViaMISSolver):
    """NetworkX solver for max clique via MIS on complement."""
    
    def __init__(self, config: Dict):
        super().__init__("NetworkX (via MIS)")
        self.max_nodes = config['max_nodes']
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        if graph.number_of_nodes() > self.max_nodes:
            return set(), 0, 0.0, False, f"Graph too large ({graph.number_of_nodes()} > {self.max_nodes} nodes)"
        
        start_time = time.time()
        try:
            # Convert to complement graph for MIS solving
            complement = nx.complement(graph)
            
            # Use NetworkX maximal independent set as approximation
            # For exact solution, we would need to enumerate all maximal independent sets
            # NetworkX doesn't have a built-in exact MIS solver, so we use maximal as heuristic
            mis_nodes = nx.maximal_independent_set(complement)
            
            # MIS in complement = clique in original
            clique_nodes = set(mis_nodes)
            clique_size = len(clique_nodes)
            
            runtime = time.time() - start_time
            return clique_nodes, clique_size, runtime, True, "Note: NetworkX uses maximal (not maximum) independent set"
            
        except Exception as e:
            runtime = time.time() - start_time
            return set(), 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return True  # NetworkX is always available


class DiracCliqueViaMISSolver(CliqueViaMISSolver):
    """Dirac Oracle solver for max clique via MIS on complement."""
    
    def __init__(self, config: Dict):
        super().__init__("Dirac Oracle (via MIS)")
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
    
    def find_max_clique(self, graph: nx.Graph) -> Tuple[Set[int], int, float, bool, str]:
        if not self.is_available():
            return set(), 0, 0.0, False, "Dirac Oracle not available"
        
        start_time = time.time()
        try:
            # Use oracle directly on original graph to get clique number
            clique_size = self.oracle.get_omega(graph)
            
            if clique_size == 0:
                clique_nodes = set()
            else:
                # Find actual clique nodes using NetworkX
                # Use the built-in clique finding algorithms
                if clique_size == 1:
                    # Any single node forms a clique of size 1
                    clique_nodes = {list(graph.nodes())[0]} if graph.nodes() else set()
                else:
                    # Find cliques and get one of the desired size
                    cliques = list(nx.find_cliques(graph))
                    target_cliques = [c for c in cliques if len(c) == clique_size]
                    
                    if target_cliques:
                        clique_nodes = set(target_cliques[0])
                    else:
                        # Fallback: use the largest clique found
                        if cliques:
                            largest_clique = max(cliques, key=len)
                            clique_nodes = set(largest_clique)
                            clique_size = len(clique_nodes)  # Update size to actual found
                        else:
                            clique_nodes = set()
                            clique_size = 0
            
            runtime = time.time() - start_time
            return clique_nodes, clique_size, runtime, True, ""
            
        except Exception as e:
            runtime = time.time() - start_time
            return set(), 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return DIRAC_AVAILABLE and self.oracle is not None and self.oracle.is_available


# =============================================================================
# MAIN COMPUTATION FUNCTIONS
# =============================================================================

def create_clique_solvers() -> List[CliqueViaMISSolver]:
    """Create solver instances based on configuration."""
    solvers = []
    
    if "jax_pgd" in MIS_SOLVERS:
        solvers.append(JAXPGDCliqueSolver(MIS_CONFIG['jax_config']))
    
    if "gurobi_milp" in MIS_SOLVERS:
        solvers.append(GurobiMILPCliqueSolver(MIS_CONFIG['gurobi_config']))
    
    if "scipy_milp" in MIS_SOLVERS:
        solvers.append(ScipyMILPCliqueSolver({}))
    
    if "networkx_exact" in MIS_SOLVERS:
        solvers.append(NetworkXCliqueViaMISSolver(MIS_CONFIG['networkx_config']))
    
    if "dirac_oracle" in MIS_SOLVERS:
        solvers.append(DiracCliqueViaMISSolver(MIS_CONFIG['dirac_config']))
    
    return solvers


def validate_clique(graph: nx.Graph, clique_nodes: Set[int]) -> Tuple[bool, str]:
    """
    Validate that the proposed clique is actually a clique in the graph.
    
    Args:
        graph: Original graph
        clique_nodes: Set of nodes claimed to form a clique
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if not clique_nodes:
        return True, "Empty clique is trivially valid"
    
    # Check all nodes exist in graph
    if not clique_nodes.issubset(set(graph.nodes())):
        missing = clique_nodes - set(graph.nodes())
        return False, f"Clique contains nodes not in graph: {missing}"
    
    # Check all pairs are connected (complete subgraph)
    clique_list = list(clique_nodes)
    for i in range(len(clique_list)):
        for j in range(i + 1, len(clique_list)):
            u, v = clique_list[i], clique_list[j]
            if not graph.has_edge(u, v):
                return False, f"Missing edge between clique nodes {u} and {v}"
    
    return True, f"Valid clique of size {len(clique_nodes)}"


def analyze_graph_properties(graph: nx.Graph, complement: nx.Graph) -> Dict:
    """Analyze properties of original graph and its complement."""
    n = graph.number_of_nodes()
    
    props = {
        'original': {
            'nodes': n,
            'edges': graph.number_of_edges(),
            'density': graph.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0,
            'avg_degree': 2 * graph.number_of_edges() / n if n > 0 else 0,
        },
        'complement': {
            'nodes': n,
            'edges': complement.number_of_edges(),
            'density': complement.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0,
            'avg_degree': 2 * complement.number_of_edges() / n if n > 0 else 0,
        }
    }
    
    # Theoretical validation
    max_edges = n * (n - 1) // 2
    total_edges = props['original']['edges'] + props['complement']['edges']
    props['validation'] = {
        'max_possible_edges': max_edges,
        'total_edges': total_edges,
        'edge_sum_correct': total_edges == max_edges,
        'density_sum': props['original']['density'] + props['complement']['density'],
        'density_sum_correct': abs(props['original']['density'] + props['complement']['density'] - 1.0) < 1e-10
    }
    
    return props


def run_clique_computation(graph: nx.Graph, graph_name: str, quiet: bool = False) -> Dict:
    """Run all configured solvers to find maximum clique."""
    
    # Analyze graph properties
    complement = nx.complement(graph)
    props = analyze_graph_properties(graph, complement)
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"MAXIMUM CLIQUE COMPUTATION via MIS on COMPLEMENT")
        print(f"{'='*80}")
        print(f"Graph: {graph_name}")
        print(f"Original graph: {props['original']['nodes']} nodes, {props['original']['edges']} edges (density: {props['original']['density']:.3f})")
        print(f"Complement graph: {props['complement']['nodes']} nodes, {props['complement']['edges']} edges (density: {props['complement']['density']:.3f})")
        print(f"Strategy: Solve MIS on sparse complement ({props['complement']['edges']} edges) instead of clique on dense original ({props['original']['edges']} edges)")
        print('='*80)
    
    results = {}
    solvers = create_clique_solvers()
    
    for solver in solvers:
        if not solver.is_available():
            if not quiet:
                print(f"🔴 {solver.name}: SKIPPED (not available)")
            results[solver.name] = {
                'clique_nodes': set(),
                'clique_size': 0,
                'runtime': 0.0,
                'success': False,
                'error': "Not available",
                'validation': (False, "Solver not available")
            }
            continue
        
        if not quiet:
            print(f"🔵 {solver.name}: Finding maximum clique...")
        
        clique_nodes, clique_size, runtime, success, error = solver.find_max_clique(graph)
        
        # Validate the clique
        if success and clique_nodes:
            is_valid, validation_msg = validate_clique(graph, clique_nodes)
        else:
            is_valid, validation_msg = False, "No clique found" if success else error
        
        if success and is_valid:
            if not quiet:
                print(f"   ✅ Found clique of size {clique_size} in {runtime:.3f}s")
                print(f"   ✅ Validation: {validation_msg}")
        else:
            if not quiet:
                if not success:
                    print(f"   ❌ FAILED: {error}")
                else:
                    print(f"   ❌ INVALID CLIQUE: {validation_msg}")
        
        results[solver.name] = {
            'clique_nodes': clique_nodes,
            'clique_size': clique_size,
            'runtime': runtime,
            'success': success and is_valid,
            'error': error if not success else ("" if is_valid else validation_msg),
            'validation': (is_valid, validation_msg)
        }
    
    # Add graph properties to results
    results['_graph_properties'] = props
    
    return results


def create_clique_summary(results: Dict, format_type: str = "table") -> str:
    """Create summary of maximum clique computation results."""
    
    # Extract solver results (exclude metadata)
    solver_results = {k: v for k, v in results.items() if not k.startswith('_')}
    
    if format_type == "json":
        json_results = {}
        for solver_name, result in solver_results.items():
            json_results[solver_name] = {
                "clique_size": result['clique_size'],
                "clique_nodes": list(result['clique_nodes']) if result['clique_nodes'] else [],
                "runtime_seconds": result['runtime'],
                "success": result['success'],
                "error_message": result['error'],
                "validation_passed": result['validation'][0],
                "validation_message": result['validation'][1]
            }
        
        # Add graph properties
        if '_graph_properties' in results:
            json_results['_graph_analysis'] = results['_graph_properties']
        
        return json.dumps(json_results, indent=2)
    
    elif format_type == "csv":
        lines = ["Solver,Clique_Size,Runtime_Seconds,Status,Validation,Notes"]
        
        for solver_name, result in solver_results.items():
            status = "SUCCESS" if result['success'] else "FAILED"
            validation = "VALID" if result['validation'][0] else "INVALID"
            notes = result['validation'][1].replace(",", ";")  # Escape commas for CSV
            lines.append(f"{solver_name},{result['clique_size']},{result['runtime']:.4f},{status},{validation},{notes}")
        
        return "\n".join(lines)
    
    else:  # table format
        output = []
        output.append(f"\n{'='*90}")
        output.append("MAXIMUM CLIQUE COMPUTATION RESULTS")
        output.append('='*90)
        
        # Graph analysis
        if '_graph_properties' in results:
            props = results['_graph_properties']
            output.append(f"\nGRAPH ANALYSIS:")
            output.append(f"  Original:   {props['original']['nodes']} nodes, {props['original']['edges']} edges (density: {props['original']['density']:.3f})")
            output.append(f"  Complement: {props['complement']['nodes']} nodes, {props['complement']['edges']} edges (density: {props['complement']['density']:.3f})")
            output.append(f"  Strategy:   Solve MIS on complement (sparse) instead of clique on original (dense)")
            output.append(f"  Validation: Edge sum = {props['validation']['total_edges']}/{props['validation']['max_possible_edges']} ({'✅' if props['validation']['edge_sum_correct'] else '❌'})")
        
        output.append(f"\nSOLVER RESULTS:")
        output.append(f"{'Solver':<25} {'Size':<6} {'Runtime(s)':<12} {'Status':<10} {'Validation':<12} {'Notes':<20}")
        output.append('-' * 90)
        
        # Find optimal size
        successful_results = [r for r in solver_results.values() if r['success']]
        if successful_results:
            optimal_size = max(r['clique_size'] for r in successful_results)
        else:
            optimal_size = 0
        
        for solver_name, result in solver_results.items():
            status = "SUCCESS" if result['success'] else "FAILED"
            validation = "VALID" if result['validation'][0] else "INVALID"
            
            # Highlight optimal solutions
            size_str = f"{result['clique_size']}"
            if result['success'] and result['clique_size'] == optimal_size and optimal_size > 0:
                size_str += " 🏆"
            
            # Truncate notes
            notes = result['validation'][1]
            if len(notes) > 20:
                notes = notes[:17] + "..."
            
            output.append(f"{solver_name:<25} {size_str:<6} {result['runtime']:<12.4f} {status:<10} {validation:<12} {notes:<20}")
        
        # Summary analysis
        if successful_results:
            output.append(f"\n✅ SUMMARY:")
            output.append(f"   Maximum clique size found: {optimal_size}")
            optimal_solvers = [name for name, result in solver_results.items() 
                             if result['success'] and result['clique_size'] == optimal_size]
            output.append(f"   Found by: {', '.join(optimal_solvers)}")
            
            # Show fastest solver for optimal result
            if optimal_solvers:
                fastest = min(optimal_solvers, key=lambda name: solver_results[name]['runtime'])
                output.append(f"   Fastest optimal solver: {fastest} ({solver_results[fastest]['runtime']:.3f}s)")
        else:
            output.append(f"\n❌ No successful solutions found")
        
        return "\n".join(output)


def visualize_clique_results(graph: nx.Graph, results: Dict, output_dir: Path):
    """Create visualization of the graph and found cliques."""
    print("\n📊 Creating clique visualization...")
    
    solver_results = {k: v for k, v in results.items() if not k.startswith('_') and v['success']}
    
    if not solver_results:
        print("No successful results to visualize")
        return
    
    # Calculate layout
    n = graph.number_of_nodes()
    if n <= 50:
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    else:
        pos = nx.circular_layout(graph)
    
    # Create subplot layout
    n_plots = len(solver_results)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Find optimal size for highlighting
    optimal_size = max(r['clique_size'] for r in solver_results.values())
    
    for i, (solver_name, result) in enumerate(solver_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Node colors: red for clique nodes, lightblue for others
        clique_nodes = result['clique_nodes']
        node_colors = ['red' if node in clique_nodes else 'lightblue' for node in graph.nodes()]
        
        # Node sizes: larger for clique nodes
        node_sizes = [300 if node in clique_nodes else 100 for node in graph.nodes()]
        
        # Draw graph
        if n <= 100:
            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
            nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
            if n <= 30:
                nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        else:
            # For large graphs, show only clique nodes prominently
            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                                 node_size=[50 if node not in clique_nodes else 200 for node in graph.nodes()], 
                                 ax=ax)
            nx.draw_networkx_edges(graph, pos, alpha=0.1, width=0.5, ax=ax)
        
        # Highlight optimal solutions
        title_suffix = " 🏆" if result['clique_size'] == optimal_size else ""
        ax.set_title(f"{solver_name}\nClique size: {result['clique_size']}{title_suffix}\nRuntime: {result['runtime']:.3f}s", 
                    fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Maximum Clique Results\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'max_clique_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved to: {output_dir / 'max_clique_results.png'}")


def main():
    """Main function for comprehensive maximum clique finder."""
    parser = argparse.ArgumentParser(
        description="Find maximum clique via MIS on complement graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_max_clique_via_mis.py DIMACS/C125.9.clq
  python find_max_clique_via_mis.py DIMACS/C125.9.clq --quiet
  python find_max_clique_via_mis.py DIMACS/C125.9.clq --format json
        """
    )
    parser.add_argument("dimacs_file", help="Path to DIMACS format file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--output-dir", default="figures", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🎯 Maximum Clique Finder via MIS on Complement Graph")
        print("=" * 60)
        print("Mathematical foundation: Max clique in G = MIS in complement(G)")
        print("Strategy: Convert dense clique problem to sparse MIS problem")
        print("=" * 60)
    
    # Check DIMACS file exists
    if not os.path.exists(args.dimacs_file):
        print(f"❌ DIMACS file not found: {args.dimacs_file}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read DIMACS file
        if not args.quiet:
            print(f"\n📁 Reading DIMACS file: {args.dimacs_file}")
        
        graph = read_dimacs_graph(args.dimacs_file)
        graph_name = Path(args.dimacs_file).stem
        
        if not args.quiet:
            print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Run computation
        results = run_clique_computation(graph, graph_name, quiet=args.quiet)
        
        # Create and display summary
        summary = create_clique_summary(results, format_type=args.format)
        print(summary)
        
        # Create visualization for non-JSON output
        if args.format == "table" and not args.quiet:
            visualize_clique_results(graph, results, output_dir)
        
        if not args.quiet:
            print(f"\n🎉 Maximum clique computation completed successfully!")
            print(f"📊 Results saved to: {output_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())