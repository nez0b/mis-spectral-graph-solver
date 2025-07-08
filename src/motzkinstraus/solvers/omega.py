"""
Omega computation solvers for maximum clique size calculation.

This module provides various solver implementations for computing the maximum
clique size (ω) of graphs using different approaches including oracle-based
methods (Motzkin-Straus) and direct MILP formulations.
"""

import time
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import availability flags and components
from .. import GUROBI_AVAILABLE, SCIPY_MILP_AVAILABLE

# Import MILP solver functions  
try:
    from .milp import get_clique_number_milp
except ImportError:
    get_clique_number_milp = None

try:
    from .scipy_milp import get_clique_number_scipy
except ImportError:
    get_clique_number_scipy = None

# Import oracle classes
try:
    from ..oracles.jax_pgd import ProjectedGradientDescentOracle
    JAX_PGD_AVAILABLE = True
except ImportError:
    ProjectedGradientDescentOracle = None
    JAX_PGD_AVAILABLE = False

try:
    from ..oracles.jax_pgd_regularized import RegularizedJAXPGDOracle
    JAX_PGD_REGULARIZED_AVAILABLE = True
except ImportError:
    RegularizedJAXPGDOracle = None
    JAX_PGD_REGULARIZED_AVAILABLE = False

try:
    from ..oracles.jax_mirror import MirrorDescentOracle
    JAX_MIRROR_AVAILABLE = True
except ImportError:
    MirrorDescentOracle = None
    JAX_MIRROR_AVAILABLE = False

try:
    from ..oracles.gurobi import GurobiOracle
    GUROBI_ORACLE_AVAILABLE = GUROBI_AVAILABLE
except ImportError:
    GurobiOracle = None
    GUROBI_ORACLE_AVAILABLE = False

try:
    from ..oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DiracOracle = None
    DIRAC_AVAILABLE = False

try:
    from ..oracles.dirac_pgd_hybrid import DiracPGDHybridOracle
    DIRAC_PGD_HYBRID_AVAILABLE = True
except ImportError:
    DiracPGDHybridOracle = None
    DIRAC_PGD_HYBRID_AVAILABLE = False


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


class RegularizedJAXPGDOmegaSolver(OmegaSolver):
    """Regularized JAX Projected Gradient Descent Oracle solver."""
    
    def __init__(self, config: Dict):
        super().__init__("Regularized JAX PGD Oracle")
        self.regularization_c = config.get('regularization_c', 0.5)
        if not JAX_PGD_REGULARIZED_AVAILABLE:
            self.oracle = None
        else:
            from ..oracles.regularized_base import IdentityRegularization
            regularization_function = IdentityRegularization(c=self.regularization_c)
            self.oracle = RegularizedJAXPGDOracle(
                regularization_function=regularization_function,
                learning_rate=config['learning_rate_pgd'],
                max_iterations=config['max_iterations'],
                num_restarts=config['num_restarts'],
                tolerance=config['tolerance'],
                verbose=config['verbose']
            )
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Regularized JAX PGD Oracle not available"
        
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)
            runtime = time.time() - start_time
            return omega, runtime, True, f"Regularized with c={self.regularization_c}"
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def is_available(self) -> bool:
        return JAX_PGD_REGULARIZED_AVAILABLE and self.oracle is not None and self.oracle.is_available


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
        self.config = config  # Store full config for batch parameters
        if not DIRAC_AVAILABLE:
            self.oracle = None
        else:
            try:
                self.oracle = DiracOracle(
                    num_samples=config['num_samples'],
                    relax_schedule=config['relax_schedule'],
                    solution_precision=config['solution_precision'],
                    sum_constraint=config['sum_constraint'],
                    mean_photon_number=config['mean_photon_number'],
                    quantum_fluctuation_coefficient=config['quantum_fluctuation_coefficient'],
                    save_raw_data=config.get('save_raw_data', False),
                    raw_data_path=config.get('raw_data_path', 'data')
                )
            except Exception:
                self.oracle = None
    
    def compute_omega(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        if not self.is_available():
            return 0, 0.0, False, "Dirac Oracle not available"
        
        # Check for batch mode
        num_batches = self.config.get('num_batch_runs', 1)
        batch_size = self.config.get('batch_size', 100)
        
        if num_batches > 1:
            return self._compute_omega_batch_mode(graph, num_batches, batch_size)
        else:
            return self._compute_omega_single_mode(graph)
    
    def _compute_omega_single_mode(self, graph: nx.Graph) -> Tuple[int, float, bool, str]:
        """Single batch mode (original behavior)."""
        start_time = time.time()
        try:
            omega = self.oracle.get_omega(graph)  # Use existing method!
            runtime = time.time() - start_time
            return omega, runtime, True, ""
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, str(e)
    
    def _compute_omega_batch_mode(self, graph: nx.Graph, num_batches: int, batch_size: int) -> Tuple[int, float, bool, str]:
        """Batch mode: collect multiple API calls and combine results."""
        start_time = time.time()
        batch_delay = self.config.get('batch_delay', 1.0)
        
        print(f"🔄 Starting batch collection: {num_batches} batches of {batch_size} samples each...")
        
        all_batch_results = []
        combined_energies = []
        best_omega = 0
        
        try:
            # Create batch-specific oracle for each call
            for batch_idx in range(num_batches):
                print(f"  Batch {batch_idx + 1}/{num_batches}...")
                
                # Create oracle with current batch size
                batch_oracle = DiracOracle(
                    num_samples=batch_size,
                    relax_schedule=self.config['relax_schedule'],
                    solution_precision=self.config['solution_precision'],
                    sum_constraint=self.config['sum_constraint'],
                    mean_photon_number=self.config['mean_photon_number'],
                    quantum_fluctuation_coefficient=self.config['quantum_fluctuation_coefficient'],
                    save_raw_data=True,  # Always save for aggregation
                    raw_data_path=self.config.get('raw_data_path', 'data')
                )
                
                # Get omega for this batch
                batch_omega = batch_oracle.get_omega(graph)
                best_omega = max(best_omega, batch_omega)
                
                # Extract energies from the saved raw data
                batch_energies = self._extract_energies_from_last_response(
                    self.config.get('raw_data_path', 'data')
                )
                
                if batch_energies:
                    combined_energies.extend(batch_energies)
                    all_batch_results.append({
                        'batch_index': batch_idx,
                        'omega': batch_omega,
                        'num_samples': len(batch_energies),
                        'energies': batch_energies
                    })
                
                # Add delay between API calls (except for last batch)
                if batch_idx < num_batches - 1:
                    time.sleep(batch_delay)
            
            # Save combined results
            if self.config.get('save_raw_data', False) and combined_energies:
                self._save_combined_batch_results(
                    all_batch_results, combined_energies, 
                    self.config.get('raw_data_path', 'data')
                )
            
            runtime = time.time() - start_time
            total_samples = len(combined_energies)
            
            print(f"✅ Batch collection complete: {total_samples} total samples, best ω = {best_omega}")
            
            return best_omega, runtime, True, f"Collected {total_samples} samples from {num_batches} batches"
            
        except Exception as e:
            runtime = time.time() - start_time
            return 0, runtime, False, f"Batch collection failed: {str(e)}"
    
    def _extract_energies_from_last_response(self, data_dir: str) -> List[float]:
        """Extract energies from the most recent Dirac response file."""
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                return []
            
            # Find the most recent Dirac response file
            json_files = list(data_path.glob("dirac_response_*.json"))
            if not json_files:
                return []
            
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                response = json.load(f)
            
            if "results" in response and "energies" in response["results"]:
                return response["results"]["energies"]
            else:
                return []
                
        except Exception as e:
            print(f"Warning: Could not extract energies from response: {e}")
            return []
    
    def _save_combined_batch_results(self, all_batch_results: List[Dict], 
                                   combined_energies: List[float], data_dir: str):
        """Save combined batch results in a format compatible with histogram plotting."""
        try:
            data_path = Path(data_dir)
            data_path.mkdir(exist_ok=True)
            
            # Create combined response structure
            combined_response = {
                "results": {
                    "energies": combined_energies,
                    "batch_info": {
                        "num_batches": len(all_batch_results),
                        "total_samples": len(combined_energies),
                        "batches": all_batch_results
                    }
                },
                "metadata": {
                    "is_batch_collection": True,
                    "batch_timestamp": time.time()
                }
            }
            
            # Save with timestamp
            timestamp = int(time.time())
            combined_file = data_path / f"dirac_response_batch_{timestamp}.json"
            
            with open(combined_file, 'w') as f:
                json.dump(combined_response, f, indent=2)
            
            print(f"💾 Saved combined batch results to: {combined_file}")
            
        except Exception as e:
            print(f"Warning: Could not save combined batch results: {e}")
    
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
                    dirac_sum_constraint=config['dirac_sum_constraint'],
                    dirac_mean_photon_number=config['dirac_mean_photon_number'],
                    dirac_quantum_fluctuation_coefficient=config['dirac_quantum_fluctuation_coefficient'],
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


def create_omega_solvers(omega_solvers: List[str], omega_config: Dict) -> List[OmegaSolver]:
    """
    Create solver instances based on configuration.
    
    Args:
        omega_solvers: List of solver names to create
        omega_config: Configuration dictionary with solver parameters
        
    Returns:
        List of OmegaSolver instances
    """
    solvers = []
    
    if "jax_pgd" in omega_solvers:
        solvers.append(JAXPGDOmegaSolver(omega_config['jax_config']))
    
    if "jax_pgd_regularized" in omega_solvers:
        solvers.append(RegularizedJAXPGDOmegaSolver(omega_config['jax_config']))
    
    if "jax_mirror" in omega_solvers:
        solvers.append(JAXMirrorOmegaSolver(omega_config['jax_config']))
    
    if "gurobi_oracle" in omega_solvers:
        solvers.append(GurobiOracleOmegaSolver(omega_config['gurobi_config']))
    
    if "gurobi_milp" in omega_solvers:
        solvers.append(GurobiMILPOmegaSolver(omega_config['gurobi_config']))
    
    if "scipy_milp" in omega_solvers:
        solvers.append(ScipyMILPOmegaSolver({}))
    
    if "networkx_exact" in omega_solvers:
        solvers.append(NetworkXOmegaSolver(omega_config['networkx_config']))
    
    if "dirac_oracle" in omega_solvers:
        solvers.append(DiracOracleOmegaSolver(omega_config['dirac_config']))
    
    if "dirac_pgd_hybrid" in omega_solvers:
        solvers.append(DiracPGDHybridOmegaSolver(omega_config['dirac_pgd_hybrid_config']))
    
    return solvers