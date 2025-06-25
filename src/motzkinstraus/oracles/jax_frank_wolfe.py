"""
Frank-Wolfe algorithm oracle for the Motzkin-Straus quadratic program.

This oracle uses the Frank-Wolfe (Conditional Gradient) algorithm to solve:
    maximize f(x) = 0.5 * x^T * A * x
    subject to: x ∈ Δ_n = {x | sum(x_i) = 1, x_i ≥ 0}

The Frank-Wolfe algorithm is particularly well-suited for optimization over the probability simplex
because it avoids expensive simplex projections and naturally produces sparse iterates.

Key advantages:
- Projection-free: Only requires solving linear optimization over simplex (closed-form argmax)
- Sparse iterates: Solutions naturally become sparse
- Closed-form line search: Optimal step size for quadratic objectives
- GPU/TPU efficient: Fully vectorized with JAX
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from functools import partial

from .base import Oracle
from ..exceptions import OracleError

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class FrankWolfeOracle(Oracle):
    """
    Frank-Wolfe oracle for solving the Motzkin-Straus quadratic program.
    
    Uses the Frank-Wolfe (Conditional Gradient) algorithm with multi-restart strategy
    to find stationary points of the non-concave quadratic optimization problem.
    
    Args:
        num_restarts: Number of random restarts for global optimization (default: 16).
        max_iterations: Maximum iterations per restart (default: 1000).
        tolerance: Termination tolerance for Frank-Wolfe gap (default: 1e-6).
        verbose: Whether to print convergence information (default: False).
    """
    
    def __init__(self, num_restarts: int = 16, max_iterations: int = 1000, 
                 tolerance: float = 1e-6, verbose: bool = False):
        super().__init__()
        if not self.is_available:
            raise OracleError("JAX is not available. Please install JAX to use FrankWolfeOracle.")
        
        self.num_restarts = num_restarts
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # For tracking optimization details
        self._last_convergence_info = {}
    
    @property
    def name(self) -> str:
        return "Frank-Wolfe"
    
    @property
    def is_available(self) -> bool:
        return JAX_AVAILABLE
    
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """
        Solve the Motzkin-Straus quadratic program using Frank-Wolfe algorithm.
        
        Args:
            adjacency_matrix: The adjacency matrix of the graph.
            
        Returns:
            The optimal value of the quadratic program.
            
        Raises:
            OracleError: If the solver fails.
        """
        n = adjacency_matrix.shape[0]
        
        if n == 0:
            return 0.0
        
        if n == 1:
            return 0.0  # Single node has f(x) = 0.5 * x^T * 0 * x = 0
        
        try:
            # Convert to JAX array with proper dtype
            A = jnp.array(adjacency_matrix, dtype=jnp.float32)
            
            # Generate random key for initialization  
            seed = int(time.time() * 1000000) % (2**32)
            key = jr.PRNGKey(seed)
            
            # Run Frank-Wolfe with multi-restart
            best_x, best_val = self._frank_wolfe_multi_restart(A, key)
            
            if self.verbose:
                print(f"Frank-Wolfe Oracle: Best energy = {best_val:.6f}")
                print(f"Frank-Wolfe Oracle: Final solution sum = {jnp.sum(best_x):.6f}")
                print(f"Frank-Wolfe Oracle: Solution sparsity = {jnp.sum(best_x > 1e-6):.0f}/{n}")
            
            return float(best_val)
            
        except Exception as e:
            raise OracleError(f"Frank-Wolfe solver failed: {str(e)}")
    
    def _frank_wolfe_multi_restart(self, A: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
        """
        Run Frank-Wolfe algorithm with multiple random restarts.
        """
        n = A.shape[0]
        
        # Generate multiple random starting points on the simplex using Dirichlet distribution
        keys = jr.split(key, self.num_restarts)
        # Use uniform Dirichlet (all alphas = 1) for uniform distribution on simplex
        alphas = jnp.ones(n)
        initial_xs = jax.vmap(lambda k: jr.dirichlet(k, alphas))(keys)
        
        # Run Frank-Wolfe for all initial points in parallel
        results = []
        for i in range(self.num_restarts):
            x_init = initial_xs[i]
            final_x, final_obj, iterations, final_gap = self._single_frank_wolfe(A, x_init)
            results.append((final_x, final_obj, iterations, final_gap))
        
        # Find the best result across all restarts
        best_val = -jnp.inf
        best_x = None
        best_info = None
        
        for i, (x, obj, iters, gap) in enumerate(results):
            if obj > best_val:
                best_val = obj
                best_x = x
                best_info = {'restart': i, 'iterations': iters, 'final_gap': gap}
        
        # Store convergence information
        all_vals = [result[1] for result in results]
        all_iters = [result[2] for result in results]
        all_gaps = [result[3] for result in results]
        
        self._last_convergence_info = {
            'best_restart': best_info['restart'],
            'all_final_values': all_vals,
            'convergence_iterations': all_iters,
            'convergence_gaps': all_gaps
        }
        
        return best_x, best_val
    
    def _single_frank_wolfe(self, A: jnp.ndarray, x_init: jnp.ndarray) -> Tuple[jnp.ndarray, float, int, float]:
        """
        Single Frank-Wolfe optimization run.
        """
        x = x_init
        
        for k in range(self.max_iterations):
            # Frank-Wolfe iteration
            x_new, gap = self._frank_wolfe_step(A, x)
            x = x_new
            
            # Check convergence
            if gap <= self.tolerance:
                break
        
        # Compute final objective value
        final_objective = 0.5 * jnp.dot(x, A @ x)
        
        return x, final_objective, k + 1, gap
    
    @partial(jax.jit, static_argnums=(0,))
    def _frank_wolfe_step(self, A: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """
        Single Frank-Wolfe iteration step.
        """
        n = A.shape[0]
        
        # Step 1: Compute gradient
        grad = A @ x
        
        # Step 2: Solve Linear Minimization Oracle (LMO)
        # For maximizing grad^T @ s over simplex, put all weight on max component
        max_idx = jnp.argmax(grad)
        s = jax.nn.one_hot(max_idx, n, dtype=x.dtype)
        
        # Step 3: Compute Frank-Wolfe gap (termination criterion)
        gap = jnp.dot(grad, s - x)
        
        # Step 4: Compute step direction
        d = s - x
        
        # Step 5: Optimal step size via closed-form line search
        alpha = self._optimal_step_size(A, x, d, grad)
        
        # Step 6: Update
        x_new = x + alpha * d
        
        return x_new, gap
    
    @partial(jax.jit, static_argnums=(0,))
    def _optimal_step_size(self, A: jnp.ndarray, x: jnp.ndarray, d: jnp.ndarray, grad: jnp.ndarray) -> float:
        """
        Compute optimal step size for quadratic objective using closed-form line search.
        """
        # Linear coefficient: d^T * grad
        linear_coeff = jnp.dot(d, grad)
        
        # Quadratic coefficient: d^T * A * d
        quadratic_coeff = jnp.dot(d, A @ d)
        
        # For numerical stability, use a small epsilon
        eps = 1e-8
        
        alpha = jax.lax.cond(
            quadratic_coeff < -eps,
            # Case: concave quadratic, use interior optimum (clipped to [0,1])
            lambda: jnp.clip(-linear_coeff / quadratic_coeff, 0.0, 1.0),
            # Case: convex or linear, maximum is at boundary
            # Since d is an ascent direction (gap > 0 → linear_coeff > 0), use α = 1
            lambda: 1.0
        )
        
        return alpha
    
    def get_optimization_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the last optimization run.
        """
        return self._last_convergence_info.copy()
    
    def get_convergence_histories(self) -> Optional[Dict[str, Any]]:
        """
        Get convergence histories (for compatibility with other oracles).
        """
        if not self._last_convergence_info:
            return None
            
        return {
            'final_values': self._last_convergence_info.get('all_final_values', []),
            'iterations': self._last_convergence_info.get('convergence_iterations', []),
            'gaps': self._last_convergence_info.get('convergence_gaps', [])
        }