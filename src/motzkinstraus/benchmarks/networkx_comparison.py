"""
Core benchmarking framework for comparing Motzkin-Straus solvers with NetworkX algorithms.
"""

import time
import traceback
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import TimeoutError
import signal

# Import our solvers
from ..oracles.base import Oracle
from ..oracles.jax_pgd import ProjectedGradientDescentOracle
from ..oracles.jax_mirror import MirrorDescentOracle

try:
    from ..oracles.gurobi import GurobiOracle
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from ..oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False

try:
    from ..oracles.dirac_hybrid import DiracNetworkXHybridOracle
    DIRAC_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_HYBRID_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from running a single algorithm on a single graph."""
    algorithm_name: str
    graph_description: str
    graph_size: int
    graph_edges: int
    
    # Core results
    independent_set: List[int]
    set_size: int
    runtime_seconds: float
    
    # Algorithm-specific details
    oracle_calls: Optional[int] = None
    convergence_history: Optional[List[float]] = None
    optimization_details: Optional[Dict] = None
    
    # Error handling
    success: bool = True
    error_message: Optional[str] = None
    timeout: bool = False
    
    # Statistical info (for randomized algorithms)
    seed_used: Optional[int] = None
    multiple_runs: Optional[List[Dict]] = None  # For algorithms run multiple times


@dataclass 
class NetworkXComparisonBenchmark:
    """Main benchmarking class for comparing different MIS algorithms."""
    
    # Timeout settings (in seconds)
    fast_timeout: float = 10.0      # For heuristics
    medium_timeout: float = 300.0   # For approximation algorithms  
    slow_timeout: float = 1800.0    # For exact algorithms
    
    # Number of runs for randomized algorithms
    num_random_runs: int = 20
    
    # Algorithm configurations
    jax_config: Dict = field(default_factory=lambda: {
        'learning_rate_pgd': 0.02,
        'learning_rate_md': 0.01,
        'max_iterations': 2000,
        'num_restarts': 10,
        'tolerance': 1e-6,
        'verbose': False
    })
    
    def __init__(self, **kwargs):
        """Initialize benchmark with custom settings."""
        # Initialize default values
        if not hasattr(self, 'jax_config'):
            self.jax_config = {
                'learning_rate_pgd': 0.02,
                'learning_rate_md': 0.01,
                'max_iterations': 2000,
                'num_restarts': 10,
                'tolerance': 1e-6,
                'verbose': False
            }
        
        if not hasattr(self, 'dirac_config'):
            self.dirac_config = {
                'num_samples': 10,
                'relax_schedule': 2,
                'solution_precision': 0.001,
                'threshold_nodes': 35  # For hybrid solver
            }
        
        # Apply custom settings
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutError("Algorithm timed out")
    
    def _run_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Run a function with a timeout."""
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _verify_independent_set(self, graph: nx.Graph, node_set: List[int]) -> bool:
        """Verify that a set of nodes forms a valid independent set."""
        if not node_set:
            return True
            
        # Check all nodes are in graph
        if not all(node in graph.nodes for node in node_set):
            return False
            
        # Check no edges between nodes in the set
        for i, u in enumerate(node_set):
            for v in node_set[i+1:]:
                if graph.has_edge(u, v):
                    return False
        return True
    
    def run_networkx_maximal_greedy(
        self, 
        graph: nx.Graph, 
        multiple_seeds: bool = True
    ) -> BenchmarkResult:
        """Run NetworkX maximal independent set (greedy heuristic)."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        try:
            if multiple_seeds:
                # Run multiple times with different seeds
                runs = []
                best_size = 0
                best_set = []
                total_time = 0
                
                for seed in range(self.num_random_runs):
                    start_time = time.time()
                    independent_set = self._run_with_timeout(
                        nx.maximal_independent_set, 
                        self.fast_timeout,
                        graph, 
                        seed=seed
                    )
                    runtime = time.time() - start_time
                    total_time += runtime
                    
                    set_list = list(independent_set)
                    size = len(set_list)
                    
                    runs.append({
                        'seed': seed,
                        'set': set_list,
                        'size': size, 
                        'runtime': runtime,
                        'valid': self._verify_independent_set(graph, set_list)
                    })
                    
                    if size > best_size:
                        best_size = size
                        best_set = set_list
                
                return BenchmarkResult(
                    algorithm_name="NetworkX_Maximal_Greedy",
                    graph_description=graph_desc,
                    graph_size=graph.number_of_nodes(),
                    graph_edges=graph.number_of_edges(),
                    independent_set=best_set,
                    set_size=best_size,
                    runtime_seconds=total_time,
                    multiple_runs=runs
                )
            else:
                # Single run with fixed seed
                start_time = time.time()
                independent_set = self._run_with_timeout(
                    nx.maximal_independent_set,
                    self.fast_timeout, 
                    graph,
                    seed=42
                )
                runtime = time.time() - start_time
                
                set_list = list(independent_set)
                
                return BenchmarkResult(
                    algorithm_name="NetworkX_Maximal_Greedy_Single",
                    graph_description=graph_desc,
                    graph_size=graph.number_of_nodes(),
                    graph_edges=graph.number_of_edges(),
                    independent_set=set_list,
                    set_size=len(set_list),
                    runtime_seconds=runtime,
                    seed_used=42
                )
                
        except TimeoutError:
            alg_name = "NetworkX_Maximal_Greedy_Single" if not multiple_seeds else "NetworkX_Maximal_Greedy"
            return BenchmarkResult(
                algorithm_name=alg_name,
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.fast_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            alg_name = "NetworkX_Maximal_Greedy_Single" if not multiple_seeds else "NetworkX_Maximal_Greedy"
            return BenchmarkResult(
                algorithm_name=alg_name,
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_networkx_approximation(self, graph: nx.Graph) -> BenchmarkResult:
        """Run NetworkX Boppana-Halldórsson approximation algorithm."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        try:
            start_time = time.time()
            independent_set = self._run_with_timeout(
                nx.algorithms.approximation.clique.maximum_independent_set,
                self.medium_timeout,
                graph
            )
            runtime = time.time() - start_time
            
            set_list = list(independent_set)
            
            return BenchmarkResult(
                algorithm_name="NetworkX_BH_Approximation",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=set_list,
                set_size=len(set_list),
                runtime_seconds=runtime
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name="NetworkX_BH_Approximation", 
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.medium_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name="NetworkX_BH_Approximation",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_networkx_exact(self, graph: nx.Graph) -> BenchmarkResult:
        """Run exact NetworkX algorithm using complement graph + clique finding."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        try:
            start_time = time.time()
            
            # Find maximum clique in complement graph
            complement = self._run_with_timeout(nx.complement, self.slow_timeout / 2, graph)
            cliques = self._run_with_timeout(
                nx.find_cliques, 
                self.slow_timeout / 2,
                complement
            )
            max_clique = max(cliques, key=len)
            
            runtime = time.time() - start_time
            independent_set = list(max_clique)
            
            return BenchmarkResult(
                algorithm_name="NetworkX_Exact_Clique",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=independent_set,
                set_size=len(independent_set),
                runtime_seconds=runtime
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name="NetworkX_Exact_Clique",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.slow_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name="NetworkX_Exact_Clique",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_jax_oracle(self, graph: nx.Graph, algorithm: str = "pgd") -> BenchmarkResult:
        """Run JAX-based oracle (PGD or Mirror Descent)."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        try:
            # Create oracle
            if algorithm.lower() == "pgd":
                oracle = ProjectedGradientDescentOracle(
                    learning_rate=self.jax_config['learning_rate_pgd'],
                    max_iterations=self.jax_config['max_iterations'],
                    num_restarts=self.jax_config['num_restarts'],
                    tolerance=self.jax_config['tolerance'],
                    verbose=self.jax_config['verbose']
                )
                alg_name = "JAX_PGD"
            elif algorithm.lower() == "md" or algorithm.lower() == "mirror":
                oracle = MirrorDescentOracle(
                    learning_rate=self.jax_config['learning_rate_md'],
                    max_iterations=self.jax_config['max_iterations'],
                    num_restarts=self.jax_config['num_restarts'],
                    tolerance=self.jax_config['tolerance'],
                    verbose=self.jax_config['verbose']
                )
                alg_name = "JAX_MirrorDescent"
            else:
                raise ValueError(f"Unknown JAX algorithm: {algorithm}")
            
            # Import MIS algorithm
            from ..algorithms import find_mis_with_oracle
            
            start_time = time.time()
            independent_set, oracle_calls = self._run_with_timeout(
                find_mis_with_oracle,
                self.slow_timeout,
                graph,
                oracle
            )
            runtime = time.time() - start_time
            
            # Get optimization details
            opt_details = oracle.get_optimization_details()
            convergence_histories = oracle.get_convergence_histories() if hasattr(oracle, 'get_convergence_histories') else None
            
            return BenchmarkResult(
                algorithm_name=alg_name,
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=list(independent_set),
                set_size=len(independent_set),
                runtime_seconds=runtime,
                oracle_calls=oracle_calls,
                convergence_history=convergence_histories,
                optimization_details=opt_details
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name=alg_name if 'alg_name' in locals() else f"JAX_{algorithm}",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.slow_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name=alg_name if 'alg_name' in locals() else f"JAX_{algorithm}",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def run_gurobi_oracle(self, graph: nx.Graph) -> BenchmarkResult:
        """Run Gurobi-based oracle for ground truth (if available)."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        if not GUROBI_AVAILABLE:
            return BenchmarkResult(
                algorithm_name="Gurobi_Exact",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message="Gurobi not available"
            )
        
        try:
            oracle = GurobiOracle(suppress_output=True)
            from ..algorithms import find_mis_with_oracle
            
            start_time = time.time()
            independent_set, oracle_calls = self._run_with_timeout(
                find_mis_with_oracle,
                self.slow_timeout,
                graph,
                oracle
            )
            runtime = time.time() - start_time
            
            return BenchmarkResult(
                algorithm_name="Gurobi_Exact",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=list(independent_set),
                set_size=len(independent_set),
                runtime_seconds=runtime,
                oracle_calls=oracle_calls
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name="Gurobi_Exact",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.slow_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name="Gurobi_Exact",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )

    def run_dirac_oracle(self, graph: nx.Graph) -> BenchmarkResult:
        """Run Dirac-3 continuous cloud solver oracle."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        if not DIRAC_AVAILABLE:
            return BenchmarkResult(
                algorithm_name="Dirac-3",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message="Dirac solver not available (install qci-client and eqc-models)"
            )
        
        try:
            # Use configuration from benchmark_config if available
            dirac_config = getattr(self, 'dirac_config', {})
            num_samples = dirac_config.get('num_samples', 10)
            relax_schedule = dirac_config.get('relax_schedule', 2)
            solution_precision = dirac_config.get('solution_precision', 0.001)
            oracle = DiracOracle(
                num_samples=num_samples, 
                relax_schedule=relax_schedule,
                solution_precision=solution_precision
            )
            from ..algorithms import find_mis_with_oracle
            
            start_time = time.time()
            independent_set, oracle_calls = self._run_with_timeout(
                find_mis_with_oracle,
                self.slow_timeout,  # Use slow timeout for cloud API
                graph,
                oracle
            )
            runtime = time.time() - start_time
            
            return BenchmarkResult(
                algorithm_name="Dirac-3",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=list(independent_set),
                set_size=len(independent_set),
                runtime_seconds=runtime,
                oracle_calls=oracle_calls
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name="Dirac-3",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.slow_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name="Dirac-3",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )

    def run_dirac_hybrid_oracle(self, graph: nx.Graph) -> BenchmarkResult:
        """Run Dirac/NetworkX hybrid solver that automatically chooses based on graph size."""
        graph_desc = f"Graph_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
        
        if not DIRAC_HYBRID_AVAILABLE:
            return BenchmarkResult(
                algorithm_name="Dirac-Hybrid",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message="Dirac hybrid solver not available"
            )
        
        try:
            # Use configuration from benchmark_config if available
            dirac_config = getattr(self, 'dirac_config', {})
            threshold_nodes = dirac_config.get('threshold_nodes', 35)
            num_samples = dirac_config.get('num_samples', 10)
            relax_schedule = dirac_config.get('relax_schedule', 2)
            solution_precision = dirac_config.get('solution_precision', 0.001)
            
            oracle = DiracNetworkXHybridOracle(
                threshold_nodes=threshold_nodes,
                num_samples=num_samples,
                relax_schedule=relax_schedule,
                solution_precision=solution_precision
            )
            
            # Enable verbose mode if configured
            oracle.verbose_oracle_calls = getattr(self, 'verbose', False)
            
            from ..algorithms import find_mis_with_oracle
            
            # Get solver info for logging
            solver_info = oracle.get_solver_info(graph.number_of_nodes())
            print(f"  Using {solver_info['solver']} for {graph.number_of_nodes()}-node graph")
            
            start_time = time.time()
            
            # Strategy: Use constructive solver if available, otherwise fall back to search
            if hasattr(oracle, 'solve_mis'):
                try:
                    # Attempt direct constructive solving
                    oracle.call_count = 0  # Reset counter
                    independent_set = oracle.solve_mis(graph)
                    oracle_calls = oracle.call_count
                    print(f"    → Used direct constructive solver")
                except NotImplementedError:
                    # Fallback to search-to-decision for large graphs
                    print(f"    → Constructive method not available, using search-to-decision")
                    independent_set, oracle_calls = self._run_with_timeout(
                        find_mis_with_oracle,
                        self.slow_timeout,
                        graph,
                        oracle
                    )
            else:
                # Pure oracle - use search wrapper
                print(f"    → Using search-to-decision wrapper")
                independent_set, oracle_calls = self._run_with_timeout(
                    find_mis_with_oracle,
                    self.slow_timeout,
                    graph,
                    oracle
                )
            
            runtime = time.time() - start_time
            
            return BenchmarkResult(
                algorithm_name="Dirac-Hybrid",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=list(independent_set),
                set_size=len(independent_set),
                runtime_seconds=runtime,
                oracle_calls=oracle_calls,
                optimization_details={'solver_used': solver_info['solver'], 'threshold': threshold_nodes}
            )
            
        except TimeoutError:
            return BenchmarkResult(
                algorithm_name="Dirac-Hybrid",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=self.slow_timeout,
                success=False,
                timeout=True,
                error_message="Timeout exceeded"
            )
        except Exception as e:
            return BenchmarkResult(
                algorithm_name="Dirac-Hybrid",
                graph_description=graph_desc,
                graph_size=graph.number_of_nodes(),
                graph_edges=graph.number_of_edges(),
                independent_set=[],
                set_size=0,
                runtime_seconds=0.0,
                success=False,
                error_message=str(e)
            )


def run_algorithm_comparison(
    graph: nx.Graph,
    graph_description: str = "",
    algorithms: List[str] = None,
    benchmark_config: Dict = None
) -> Dict[str, BenchmarkResult]:
    """
    Run comparison of all specified algorithms on a single graph.
    
    Args:
        graph: NetworkX graph to analyze.
        graph_description: Description of the graph.
        algorithms: List of algorithms to run. Options:
            - "nx_greedy": NetworkX maximal greedy (multiple runs)
            - "nx_greedy_single": NetworkX maximal greedy (single run)  
            - "nx_approximation": NetworkX Boppana-Halldórsson approximation
            - "nx_exact": NetworkX exact via complement + cliques
            - "jax_pgd": JAX Projected Gradient Descent
            - "jax_md": JAX Mirror Descent
            - "gurobi": Gurobi exact (if available)
            - "dirac": Dirac-3 continuous cloud solver (if available)
            - "dirac_hybrid": Hybrid Dirac/NetworkX solver (auto-switches based on graph size)
        benchmark_config: Configuration dict for benchmark settings.
        
    Returns:
        Dictionary mapping algorithm names to BenchmarkResult objects.
    """
    if algorithms is None:
        algorithms = ["nx_greedy", "nx_approximation", "jax_pgd", "jax_md"]  # Note: Add "dirac" manually if needed
        
    # Initialize benchmark
    config = benchmark_config or {}
    benchmark = NetworkXComparisonBenchmark(**config)
    
    results = {}
    
    for alg in algorithms:
        print(f"Running {alg} on {graph_description}...")
        
        try:
            if alg == "nx_greedy":
                result = benchmark.run_networkx_maximal_greedy(graph, multiple_seeds=True)
            elif alg == "nx_greedy_single":
                result = benchmark.run_networkx_maximal_greedy(graph, multiple_seeds=False)
            elif alg == "nx_approximation":
                result = benchmark.run_networkx_approximation(graph)
            elif alg == "nx_exact":
                result = benchmark.run_networkx_exact(graph)
            elif alg == "jax_pgd":
                result = benchmark.run_jax_oracle(graph, "pgd")
            elif alg == "jax_md" or alg == "jax_mirror":
                result = benchmark.run_jax_oracle(graph, "md")
            elif alg == "gurobi":
                result = benchmark.run_gurobi_oracle(graph)
            elif alg == "dirac":
                result = benchmark.run_dirac_oracle(graph)
            elif alg == "dirac_hybrid":
                result = benchmark.run_dirac_hybrid_oracle(graph)
            else:
                print(f"Unknown algorithm: {alg}")
                continue
                
            results[alg] = result
            
            if result.success:
                print(f"  ✓ {alg}: Set size = {result.set_size}, Runtime = {result.runtime_seconds:.3f}s")
            else:
                print(f"  ✗ {alg}: {result.error_message}")
                
        except Exception as e:
            print(f"  ✗ {alg}: Unexpected error - {str(e)}")
            
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing NetworkX comparison framework...")
    
    # Create test graph
    G = nx.cycle_graph(8)
    
    # Run comparison
    results = run_algorithm_comparison(
        G, 
        "test_cycle_8",
        algorithms=["nx_greedy", "nx_approximation", "jax_pgd"],
        benchmark_config={'num_random_runs': 5}
    )
    
    print("\nResults:")
    for alg, result in results.items():
        print(f"{alg}: size={result.set_size}, time={result.runtime_seconds:.3f}s, success={result.success}")