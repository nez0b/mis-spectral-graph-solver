"""
Dirac-3 continuous cloud solver for the Motzkin-Straus quadratic program.
"""

import numpy as np
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
    """
    
    def __init__(self, num_samples: int = 100, relax_schedule: int = 2):
        super().__init__()
        if not self.is_available:
            raise SolverUnavailableError(
                "Dirac solver is not available. Please install qci-client and "
                "eqc-models packages and ensure proper authentication."
            )
        
        self.num_samples = num_samples
        self.relax_schedule = relax_schedule
        
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
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Dirac-3 continuous solver.
        
        The quadratic program is:
        max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
        
        We convert this to the Dirac model format where the energy function is:
        E(x) = x.T * C + x.T * J * x
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If the Dirac solver fails.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return 0.0
        
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
            
            # Solve with simplex constraint (sum_constraint=1)
            response = solver.solve(
                model,
                sum_constraint=1,  # Enforce simplex constraint: sum(x_i) = 1
                num_samples=self.num_samples,
                relaxation_schedule=self.relax_schedule
            )
            
            # Process the response
            if response and "results" in response and "solutions" in response["results"]:
                solutions = response["results"]["solutions"]
                if solutions:
                    # Get the best solution (first one is typically best)
                    best_solution = np.array(solutions[0], dtype=np.float64)
                    
                    # Calculate the objective value: 0.5 * x.T * A * x
                    objective_value = 0.5 * (best_solution.T @ adjacency_matrix @ best_solution)
                    
                    print(f"Dirac solver returned solution with objective value: {objective_value}")
                    return float(objective_value)
                else:
                    raise OracleError("Dirac solver returned empty solutions list")
            else:
                raise OracleError("Dirac solver returned invalid response format")
                
        except Exception as e:
            if isinstance(e, OracleError):
                raise
            else:
                raise OracleError(f"Error in Dirac solver: {str(e)}")