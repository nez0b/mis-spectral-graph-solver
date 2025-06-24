"""
Gurobi-based oracle for solving the Motzkin-Straus quadratic program.
"""

import numpy as np
from .base import Oracle
from ..exceptions import OracleError, SolverUnavailableError

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None


class GurobiOracle(Oracle):
    """
    Oracle implementation using Gurobi's non-convex quadratic program solver.
    
    This oracle uses Gurobi to solve the Motzkin-Straus quadratic program:
    max(0.5 * x.T * A * x) subject to sum(x_i) = 1, x_i >= 0
    
    Args:
        suppress_output: Whether to suppress Gurobi's output (default: True).
    """
    
    def __init__(self, suppress_output: bool = True):
        super().__init__()
        if not self.is_available:
            raise SolverUnavailableError(
                "Gurobi is not available. Please install gurobipy and ensure "
                "you have a valid Gurobi license."
            )
        self.suppress_output = suppress_output
    
    @property
    def name(self) -> str:
        return "Gurobi"
    
    @property
    def is_available(self) -> bool:
        return GUROBI_AVAILABLE
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Gurobi.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If Gurobi fails to solve the problem.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return 0.0
        
        try:
            # Create Gurobi environment and model
            env = gp.Env(empty=True)
            if self.suppress_output:
                env.setParam('OutputFlag', 0)
            env.start()
            
            model = gp.Model("motzkin_straus", env=env)
            model.setParam('NonConvex', 2)  # Enable non-convex QP solver
            
            # Define variables: x_i >= 0 for i = 1, ..., n
            x = model.addMVar(shape=n, name="x", lb=0.0)
            
            # Add simplex constraint: sum(x_i) = 1
            model.addConstr(x.sum() == 1, "simplex_sum_constraint")
            
            # Define objective: maximize 0.5 * x.T * A * x
            objective = 0.5 * (x @ adjacency_matrix @ x)
            model.setObjective(objective, GRB.MAXIMIZE)
            
            # Solve the model
            model.optimize()
            
            # Check solution status
            if model.status != GRB.OPTIMAL:
                raise OracleError(f"Gurobi solver failed with status: {model.status}")
            
            return model.ObjVal
            
        except gp.GurobiError as e:
            raise OracleError(f"Gurobi error: {e.errno}, {e}")
        except Exception as e:
            raise OracleError(f"Unexpected error in Gurobi solver: {e}")
        finally:
            # Clean up
            try:
                env.dispose()
            except:
                pass