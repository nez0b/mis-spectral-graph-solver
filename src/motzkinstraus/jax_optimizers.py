"""
JAX-based optimization algorithms for the Motzkin-Straus quadratic program.

This module provides JAX implementations of gradient-based optimization methods
for solving quadratic programs over the probability simplex.
"""

import numpy as np
from typing import Tuple, List, Optional
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


class JAXOptimizerConfig:
    """Configuration for JAX-based optimizers."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 2000,
        tolerance: float = 1e-6,
        min_iterations: int = 50,
        num_restarts: int = 10,
        dirichlet_alpha: float = 1.0,
        verbose: bool = False
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.num_restarts = num_restarts
        self.dirichlet_alpha = dirichlet_alpha
        self.verbose = verbose


@partial(jax.jit, static_argnames=['poly_indices'])
def evaluate_polynomial(x, poly_indices, poly_coefficients):
    """
    Evaluates a polynomial defined by indices and coefficients using JAX.

    Args:
        x: Vector of variables (0-based indexing).
        poly_indices: Tuple of tuples containing 1-based indices for each term.
        poly_coefficients: JAX array of coefficients.

    Returns:
        The value of the polynomial.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
        
    total_value = 0.0
    
    for i, indices in enumerate(poly_indices):
        coeff = poly_coefficients[i]
        
        if not indices:  # Constant term
            term_value = coeff
        else:
            # Convert 1-based indices to 0-based
            zero_based_indices = jnp.array(indices, dtype=jnp.int32) - 1
            # Calculate product: x[i1]*x[i2]*...
            term_product = jnp.prod(x[zero_based_indices])
            term_value = coeff * term_product
            
        total_value += term_value
    
    return total_value


# Create gradient function
grad_eval_poly = jax.grad(evaluate_polynomial, argnums=0)


@jax.jit
def project_to_simplex(y):
    """
    Projects a vector y onto the standard simplex using Duchi et al. algorithm.
    
    Args:
        y: Vector to project.
        
    Returns:
        Projected vector on the simplex.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
        
    d = y.shape[0]
    y_sorted = jnp.sort(y)[::-1]  # Sort in descending order
    y_cumsum = jnp.cumsum(y_sorted)
    
    # Calculate condition for finding rho
    j_vals = jnp.arange(1, d + 1)
    condition_check = y_sorted + (1.0 - y_cumsum) / j_vals > 0
    
    # Find rho: largest index where condition holds
    rho = jnp.max(j_vals * condition_check).astype(jnp.int32)
    
    # Calculate theta
    y_cumsum_padded = jnp.pad(y_cumsum, (1, 0))
    sum_upto_rho = y_cumsum_padded[rho]
    theta = jnp.where(rho > 0, (sum_upto_rho - 1.0) / rho, jnp.inf)
    
    # Project: x_i = max(y_i - theta, 0)
    x_projected = jnp.maximum(y - theta, 0.0)
    
    return x_projected


def sample_dirichlet(key, alpha, sample_shape=()):
    """
    Samples from the Dirichlet distribution using the Gamma distribution method.

    Args:
        key: JAX random key.
        alpha: Concentration parameters vector.
        sample_shape: Shape of independent samples to draw.

    Returns:
        Dirichlet samples with shape sample_shape + alpha.shape.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
        
    alpha = jnp.asarray(alpha)
    if not jnp.all(alpha > 0):
        raise ValueError("Concentration parameters alpha must be positive.")
    
    output_shape = sample_shape + alpha.shape
    gamma_samples = jax.random.gamma(key, a=alpha, shape=output_shape)
    
    # Normalize along the last axis
    sum_gamma = jnp.sum(gamma_samples, axis=-1, keepdims=True)
    dirichlet_samples = gamma_samples / jnp.maximum(sum_gamma, 1e-38)
    
    return dirichlet_samples


def adjacency_to_polynomial(adjacency_matrix: np.ndarray) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """
    Convert adjacency matrix to polynomial format for JAX optimization.
    
    For the Motzkin-Straus theorem, we want to maximize 0.5 * x^T * A * x.
    This expands to: 0.5 * sum_{i,j} A[i,j] * x_i * x_j
    
    Args:
        adjacency_matrix: Graph adjacency matrix.
        
    Returns:
        Tuple of (poly_indices, poly_coefficients) for maximizing 0.5 * x^T * A * x.
    """
    n = adjacency_matrix.shape[0]
    poly_indices = []
    poly_coefficients = []
    
    # Add quadratic terms: 0.5 * x^T * A * x = 0.5 * sum_{i,j} A[i,j] * x_i * x_j
    # Include all matrix entries to match Gurobi's formulation exactly
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                if i == j:
                    # Diagonal term: 0.5 * A[i,i] * x_i^2
                    poly_indices.append((i + 1, i + 1))  # 1-based indexing
                    poly_coefficients.append(0.5 * adjacency_matrix[i, j])
                else:
                    # Off-diagonal term: 0.5 * A[i,j] * x_i * x_j
                    poly_indices.append((i + 1, j + 1))  # 1-based indexing  
                    poly_coefficients.append(0.5 * adjacency_matrix[i, j])
    
    # Convert to tuple format for JAX static args
    poly_indices_tuple = tuple(tuple(indices) for indices in poly_indices)
    poly_coefficients_array = jnp.array(poly_coefficients)
    
    return poly_indices_tuple, poly_coefficients_array


def run_projected_gradient_descent(
    poly_indices: Tuple[Tuple[int, ...], ...],
    poly_coefficients: jnp.ndarray,
    num_vars: int,
    config: JAXOptimizerConfig,
    x_init: Optional[jnp.ndarray] = None,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Projected Gradient Descent optimization.
    
    Args:
        poly_indices: Polynomial term indices (1-based).
        poly_coefficients: Polynomial coefficients.
        num_vars: Number of variables.
        config: Optimization configuration.
        x_init: Initial point (if None, uses uniform initialization).
        seed: Random seed.
        
    Returns:
        Tuple of (final_x, energy_history).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
    
    key = jax.random.PRNGKey(seed)
    
    # JIT compile the optimization step
    @partial(jax.jit, static_argnames=['poly_indices_static'])
    def pgd_step(x, lr, poly_indices_static, poly_coefficients_static):
        grad = grad_eval_poly(x, poly_indices_static, poly_coefficients_static)
        x_temp = x + lr * grad  # Gradient ascent for maximization
        x_new = project_to_simplex(x_temp)
        return x_new
    
    # Initialize
    if x_init is not None:
        x = x_init
    else:
        x = jnp.ones(num_vars) / num_vars
    
    # Use JAX arrays for energy history to be vmap-compatible
    energy_history = jnp.zeros(config.max_iterations + 1)
    
    if config.verbose:
        initial_energy = evaluate_polynomial(x, poly_indices, poly_coefficients)
        print(f"PGD Initial Energy: {initial_energy:.6f}")
        # For non-vmap case, we can set the first element
        energy_history = energy_history.at[0].set(initial_energy)
    else:
        # For vmap case, always compute initial energy
        initial_energy = evaluate_polynomial(x, poly_indices, poly_coefficients)
        energy_history = energy_history.at[0].set(initial_energy)
    
    previous_energy = initial_energy
    final_iteration = config.max_iterations
    
    # Optimization loop using jax.lax.fori_loop for vmap compatibility
    def body_fun(i, state):
        x, energy_history, previous_energy, converged = state
        
        # Skip if already converged
        x_new = jax.lax.cond(
            converged,
            lambda x: x,
            lambda x: pgd_step(x, config.learning_rate, poly_indices, poly_coefficients),
            x
        )
        
        current_energy = evaluate_polynomial(x_new, poly_indices, poly_coefficients)
        energy_history = energy_history.at[i + 1].set(current_energy)
        
        # Check convergence
        energy_change = jnp.abs(current_energy - previous_energy)
        new_converged = converged | ((i >= config.min_iterations) & (energy_change < config.tolerance))
        
        return x_new, energy_history, current_energy, new_converged
    
    # Initial state: (x, energy_history, previous_energy, converged)
    initial_state = (x, energy_history, previous_energy, False)
    
    # Run optimization loop
    final_x, final_energy_history, _, _ = jax.lax.fori_loop(
        0, config.max_iterations, body_fun, initial_state
    )
    
    if config.verbose:
        final_energy = final_energy_history[-1]
        print(f"PGD Final Energy: {final_energy:.6f}")
    
    return final_x, final_energy_history


def run_mirror_descent(
    poly_indices: Tuple[Tuple[int, ...], ...],
    poly_coefficients: jnp.ndarray,
    num_vars: int,
    config: JAXOptimizerConfig,
    x_init: Optional[jnp.ndarray] = None,
    seed: int = 42,
    epsilon: float = 1e-9
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Mirror Descent optimization.
    
    Args:
        poly_indices: Polynomial term indices (1-based).
        poly_coefficients: Polynomial coefficients.
        num_vars: Number of variables.
        config: Optimization configuration.
        x_init: Initial point (if None, uses uniform initialization).
        seed: Random seed.
        epsilon: Numerical stability parameter.
        
    Returns:
        Tuple of (final_x, energy_history).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
    
    key = jax.random.PRNGKey(seed)
    
    # JIT compile the optimization step
    @partial(jax.jit, static_argnames=['poly_indices_static'])
    def md_step(x, lr, poly_indices_static, poly_coefficients_static, eps):
        grad = grad_eval_poly(x, poly_indices_static, poly_coefficients_static)
        # Stabilized exponentiated gradient update for maximization
        grad_shifted = grad - jnp.min(grad)
        x_new_unnormalized = x * jnp.exp(lr * grad_shifted)  # Positive for maximization
        x_new = x_new_unnormalized / (jnp.sum(x_new_unnormalized) + eps)
        return x_new
    
    # Initialize (ensure positivity for MD)
    if x_init is not None:
        x = jnp.maximum(x_init, epsilon)
        x = x / jnp.sum(x)  # Re-normalize
    else:
        x = jnp.ones(num_vars) / num_vars
    
    # Use JAX arrays for energy history to be vmap-compatible
    energy_history = jnp.zeros(config.max_iterations + 1)
    
    if config.verbose:
        initial_energy = evaluate_polynomial(x, poly_indices, poly_coefficients)
        print(f"MD Initial Energy: {initial_energy:.6f}")
        energy_history = energy_history.at[0].set(initial_energy)
    else:
        # For vmap case, always compute initial energy
        initial_energy = evaluate_polynomial(x, poly_indices, poly_coefficients)
        energy_history = energy_history.at[0].set(initial_energy)
    
    previous_energy = initial_energy
    
    # Optimization loop using jax.lax.fori_loop for vmap compatibility
    def body_fun(i, state):
        x, energy_history, previous_energy, converged = state
        
        # Skip if already converged
        x_new = jax.lax.cond(
            converged,
            lambda x: x,
            lambda x: md_step(x, config.learning_rate, poly_indices, poly_coefficients, epsilon),
            x
        )
        
        current_energy = evaluate_polynomial(x_new, poly_indices, poly_coefficients)
        energy_history = energy_history.at[i + 1].set(current_energy)
        
        # Check convergence
        energy_change = jnp.abs(current_energy - previous_energy)
        new_converged = converged | ((i >= config.min_iterations) & (energy_change < config.tolerance))
        
        return x_new, energy_history, current_energy, new_converged
    
    # Initial state: (x, energy_history, previous_energy, converged)
    initial_state = (x, energy_history, previous_energy, False)
    
    # Run optimization loop
    final_x, final_energy_history, _, _ = jax.lax.fori_loop(
        0, config.max_iterations, body_fun, initial_state
    )
    
    if config.verbose:
        final_energy = final_energy_history[-1]
        print(f"MD Final Energy: {final_energy:.6f}")
    
    return final_x, final_energy_history


def run_multi_restart_optimization(
    poly_indices: Tuple[Tuple[int, ...], ...],
    poly_coefficients: jnp.ndarray,
    num_vars: int,
    config: JAXOptimizerConfig,
    algorithm: str = "pgd",
    seed: int = 42,
    key: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, float, List[jnp.ndarray], List[float]]:
    """
    Run optimization with multiple Dirichlet initializations using jax.vmap for parallel execution.
    
    Args:
        poly_indices: Polynomial term indices.
        poly_coefficients: Polynomial coefficients.
        num_vars: Number of variables.
        config: Optimization configuration.
        algorithm: "pgd" or "md".
        seed: Random seed (used only if key is None).
        key: JAX random key for reproducible randomness. If None, uses seed.
        
    Returns:
        Tuple of (best_x, best_energy, all_histories, all_final_energies).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
    
    # Use provided key or create one from seed
    if key is None:
        key = jax.random.PRNGKey(seed)
    
    # Generate Dirichlet initializations
    alpha = jnp.ones(num_vars) * config.dirichlet_alpha
    key, subkey = jax.random.split(key)
    initial_states = sample_dirichlet(subkey, alpha, sample_shape=(config.num_restarts,))
    
    # Generate integer seeds for each restart (for compatibility with existing functions)
    restart_seeds = jnp.arange(config.num_restarts) + seed
    
    if config.verbose:
        print(f"Running {algorithm.upper()} with {config.num_restarts} restarts using jax.vmap...")
    
    # Choose algorithm and create vectorized version
    if algorithm == "pgd":
        # Create a non-verbose config for batched execution
        batch_config = JAXOptimizerConfig(
            learning_rate=config.learning_rate,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            min_iterations=config.min_iterations,
            num_restarts=config.num_restarts,
            dirichlet_alpha=config.dirichlet_alpha,
            verbose=False  # Disable verbose for vmap (JAX can't trace print statements)
        )
        
        # Create vectorized PGD function
        # in_axes: (None, None, None, None, 0, 0) -> vmap over initial_states, restart_seeds
        vmapped_optimizer = jax.vmap(
            run_projected_gradient_descent,
            in_axes=(None, None, None, None, 0, 0)  # poly_indices, poly_coefficients, num_vars, config, x_init, seed
        )
        
        # Run all restarts in parallel
        all_final_x, all_histories_jax = vmapped_optimizer(
            poly_indices, poly_coefficients, num_vars, batch_config, initial_states, restart_seeds
        )
        
    elif algorithm == "md":
        # Create a non-verbose config for batched execution
        batch_config = JAXOptimizerConfig(
            learning_rate=config.learning_rate,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            min_iterations=config.min_iterations,
            num_restarts=config.num_restarts,
            dirichlet_alpha=config.dirichlet_alpha,
            verbose=False  # Disable verbose for vmap (JAX can't trace print statements)
        )
        
        # Create vectorized MD function  
        vmapped_optimizer = jax.vmap(
            run_mirror_descent,
            in_axes=(None, None, None, None, 0, 0, None)  # epsilon is broadcasted
        )
        
        # Run all restarts in parallel
        all_final_x, all_histories_jax = vmapped_optimizer(
            poly_indices, poly_coefficients, num_vars, batch_config, initial_states, restart_seeds, 1e-9
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Convert JAX arrays to Python lists for compatibility
    all_histories = [jnp.array(hist) for hist in all_histories_jax]
    all_final_energies = [float(hist[-1]) for hist in all_histories]
    
    # Find best solution
    best_idx = int(jnp.argmax(jnp.array(all_final_energies)))
    best_x = all_final_x[best_idx]
    best_energy = all_final_energies[best_idx]
    
    if config.verbose:
        for i, energy in enumerate(all_final_energies):
            print(f"  Restart {i+1}/{config.num_restarts}: Energy = {energy:.6f}")
        print(f"Best energy: {best_energy:.6f} (restart {best_idx+1})")
        print(f"Energy range: [{min(all_final_energies):.6f}, {max(all_final_energies):.6f}]")
    
    return best_x, best_energy, all_histories, all_final_energies