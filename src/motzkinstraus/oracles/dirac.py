"""
Dirac-3 continuous cloud solver for the Motzkin-Straus quadratic program.

Dirac-3 API Parameters:

 - `num_samples`: The optional number of samples to run for the stochastic solver. The value must be between 1 and 100, with default 1.
- `relaxation_schedule`: A configuration selector that must be in the set {1,2,3,4}, representing four different schedules. The relaxation schedule controls multiple parameters of the quantum machine including the amount of loss, number of feedback loops, the amount of quantum fluctuation, and mean photon number measured. While the first two parameters are fixed, the last two can be further adjusted by users (see below). Lower relaxation schedules are set to larger amount of dissipation in a open quantum system setup, leading to more iterations needed to reach stable states. As a result, the probability of finding an optimal solution can be higher in higher schedules, especially on a competitive energy landscape with the trade-off of longer evolution time. This parameter is optional with a default of 1.
    
    {1,2,3,4}
    
- `sum_constraint`: A constraint applied to the problem space such that the solution variables must sum to the provided value. Optional value that must be between 1 and 10000, with default 1.
- `solution_precision`: An optional number that specifies the level of precision to apply to the solutions. Omit this when the highest precision continuous solutions are deisired. If specified, then a distillation method is applied to the continuous solutions to reduce them to the submitted `solution_precision`. Note that `sum_constraint` must be divisible by `solution_precision` when the latter is specified.
- `mean_photon_number`: An optional advanced configuration parameter that is normally set automatically through the choice of `relaxation_schedule`. A value specified here overrides the default value and should be a real-number value from from 0.0000667 to 0.0066666. This parameter is the average number of photons detected over a specific time period which is a time-bin representing a possible value of a variable in Dirac-3. This is a common metric used in photon statistics and quantum optics to approximate the probability of being in the single-photon regime of coherent light. Low mean photon number maintains the quantum superposition effect in high-dimensional time-bin modes of the wavefunction. Notice that extremely low mean photon number to the same order of thermal or electronic noise in single photon detector might affect the solution negatively. [Fox, M. (2006). *Quantum Optics*. Wiley, 75-104., [Pearson, D., Elliott, C. (2004). On the Optimal Mean Photon Number for Quantum Cryptography.](https://arxiv.org/abs/quant-ph/0403065v2), [Mower, J., Zhang, Z., Desjardins, P., Lee, C., Shapiro, J. H., Englund, D. (2013). High-dimensional quantum key distribution using dispersive optics. Phys. Rev. A, 87(6), 062322.](https://link.aps.org/doi/10.1103/PhysRevA.87.062322), [Nguyen, L., Rehain, P., Sua, Y. M., Huang, Y. (2018). Programmable quantum random number generator without postprocessing. Opt. Lett., 43(4), 631-634.](https://opg.optica.org/ol/abstract.cfm?URI=ol-43-4-631)]
- `quantum_fluctuation_coefficient`: An optional advanced configuration parameter that is normally set automatically through the choice of `relaxation_schedule`. A value specified here overrides the default value and should be an integer value *n*∈{1,2,…,100}, which is used to compute the coefficient *n*1 in the real-valued interval [1001,1]. The inherent randomness of photon arrival time comes from the quantum nature of light giving rise to a fundamental limitation in single photon counting, known as Poisson noise. Dirac-3 takes advantage of this noise arising from quantum fluctuation to gain opportunity to large search space and jump out of local minima. This parameter can be adjusted to allow high or low amount of quantum fluctuation into the open system. A low fluctuation tends to provide a worse solution than a high fluctuation. Notice that, to maintain a good returned solution, this parameter should not reach too high in the same order as the signal photon. [[Bédard, G. (1967). Analysis of Light Fluctuations from Photon Counting Statistics, J. Opt. Soc. Am., 57, 1201-1206.](https://doi.org/10.1364/JOSA.57.001201)]
    
    n∈{1,2,…,100}
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional
from .base import Oracle
from ..exceptions import OracleError, SolverUnavailableError

try:
    import qci_client as qc
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False
    qc = None
    Dirac3ContinuousCloudSolver = None
    QuadraticModel = None


class DiracOracle(Oracle):
    """
    Oracle implementation using QCi's Dirac-3 continuous cloud solver.
    
    This oracle uses the Dirac-3 quantum annealing service to solve the 
    Motzkin-Straus quadratic program over the continuous simplex.
    
    Args:
        num_samples: Number of solution samples to request (default: 100).
        relax_schedule: Solver relaxation schedule parameter (default: 2).
        solution_precision: Solution precision parameter (default: 0.001).
        sum_constraint: Constraint for solution variables sum (default: 1).
        mean_photon_number: Optional mean photon number override (default: None).
        quantum_fluctuation_coefficient: Optional quantum fluctuation coefficient override (default: None).
        save_raw_data: Whether to save the raw response from Dirac solver (default: False).
        raw_data_path: Directory path to save raw data files (default: 'data/' in project root).
    """
    
    def __init__(self, num_samples: int = 100, relax_schedule: int = 2, solution_precision: float = 0.001,
                 sum_constraint: int = 1, mean_photon_number: Optional[float] = None, 
                 quantum_fluctuation_coefficient: Optional[int] = None, save_raw_data: bool = False,
                 raw_data_path: Optional[str] = None):
        super().__init__()
        if not self.is_available:
            raise SolverUnavailableError(
                "Dirac solver is not available. Please install qci-client and "
                "eqc-models packages and ensure proper authentication."
            )
        
        self.num_samples = num_samples
        self.relax_schedule = relax_schedule
        self.solution_precision = solution_precision
        self.sum_constraint = sum_constraint
        self.mean_photon_number = mean_photon_number
        self.quantum_fluctuation_coefficient = quantum_fluctuation_coefficient
        self.save_raw_data = save_raw_data
        self.raw_data_path = raw_data_path or "data"
        
        # Test connection
        try:
            client = qc.QciClient()
            allocations = client.get_allocations()
            if "dirac" not in allocations.get("allocations", {}):
                raise SolverUnavailableError(
                    "Dirac solver allocation not available. Check your QCI account."
                )
        except Exception as e:
            raise SolverUnavailableError(f"Failed to connect to QCI services: {e}")
    
    @property
    def name(self) -> str:
        return "Dirac-3"
    
    @property
    def is_available(self) -> bool:
        return DIRAC_AVAILABLE
    
    def _solve_and_get_vector(self, adjacency_matrix: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Solve the Motzkin-Straus quadratic program and return both objective value and solution vector.
        
        The quadratic program is:
        max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
        
        We convert this to the Dirac model format where the energy function is:
        E(x) = x.T * C + x.T * J * x
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            Tuple of (optimal_value, solution_vector).
            
        Raises:
            OracleError: If the Dirac solver fails.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return 0.0, np.array([])
        
        try:
            # For the Dirac model, we need to convert our objective:
            # max(0.5 * x.T * A * x) -> min(-0.5 * x.T * A * x)
            # 
            # The Dirac model format is: E(x) = x.T * C + x.T * J * x
            # where C is linear coefficients and J is quadratic coefficients
            
            # Linear coefficients (no linear terms in our objective)
            # Note: C should be 1D array for Dirac solver
            C = np.zeros(n, dtype=np.float64)
            
            # Quadratic coefficients: -0.5 * A (negative for minimization)
            J = -0.5 * adjacency_matrix.astype(np.float64)
            
            # Create the QuadraticModel
            model = QuadraticModel(C, J)
            
            # Set upper bounds for continuous variables (required by Dirac solver)
            # For the simplex constraint, variables are in [0, 1]
            model.upper_bound = np.ones(n, dtype=np.float64)
            
            # Initialize the continuous solver
            solver = Dirac3ContinuousCloudSolver()
            
            print(f"Submitting {n}-variable quadratic program to Dirac-3 continuous solver...")
            params = [f"num_samples={self.num_samples}", f"relaxation_schedule={self.relax_schedule}"]
            if self.solution_precision is not None:
                params.append(f"solution_precision={self.solution_precision}")
            params.append(f"sum_constraint={self.sum_constraint}")
            if self.mean_photon_number is not None:
                params.append(f"mean_photon_number={self.mean_photon_number}")
            if self.quantum_fluctuation_coefficient is not None:
                params.append(f"quantum_fluctuation_coefficient={self.quantum_fluctuation_coefficient}")
            print(f"Parameters: {', '.join(params)}")
            
            # Solve with configured parameters
            solve_params = {
                'sum_constraint': self.sum_constraint,
                'num_samples': self.num_samples,
                'relaxation_schedule': self.relax_schedule
            }
            
            # Add optional parameters if specified
            if self.solution_precision is not None:
                solve_params['solution_precision'] = self.solution_precision
            if self.mean_photon_number is not None:
                solve_params['mean_photon_number'] = self.mean_photon_number
            if self.quantum_fluctuation_coefficient is not None:
                solve_params['quantum_fluctuation_coefficient'] = self.quantum_fluctuation_coefficient
            
            response = solver.solve(model, **solve_params)
            
            # Save raw response data if requested
            if self.save_raw_data:
                self._save_raw_response(response, n)
            
            # Process the response
            if response and "results" in response and "solutions" in response["results"]:
                solutions = response["results"]["solutions"]
                if solutions:
                    # Get the best solution (first one is typically best)
                    best_solution = np.array(solutions[0], dtype=np.float64)
                    
                    # Calculate the objective value: 0.5 * x.T * A * x
                    objective_value = 0.5 * (best_solution.T @ adjacency_matrix @ best_solution)
                    
                    print(f"Dirac solver returned solution with objective value: {objective_value}")
                    return float(objective_value), best_solution
                else:
                    raise OracleError("Dirac solver returned empty solutions list")
            else:
                raise OracleError("Dirac solver returned invalid response format")
                
        except Exception as e:
            if isinstance(e, OracleError):
                raise
            else:
                raise OracleError(f"Error in Dirac solver: {str(e)}")

    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Dirac-3 continuous solver.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If the Dirac solver fails.
        """
        objective_value, _ = self._solve_and_get_vector(adjacency_matrix)
        return objective_value

    def _save_raw_response(self, response: dict, n_vars: int) -> None:
        """
        Save the raw response from Dirac solver to a JSON file.
        
        Args:
            response: Raw response dictionary from Dirac solver.
            n_vars: Number of variables in the problem.
        """
        try:
            # Create save directory
            save_dir = Path(self.raw_data_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp and problem size
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"dirac_response_{timestamp}_{n_vars}vars.json"
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return str(obj)  # Fallback for other non-serializable types
            
            # Save the response
            with open(filename, 'w') as f:
                json.dump(response, f, indent=2, default=convert_for_json)
            
            print(f"Saved raw Dirac response to {filename}")
            
        except Exception as e:
            print(f"Warning: Failed to save raw Dirac response: {e}")