"""
Noise generation utilities for Perturb & MAP experiments.

This module provides functions for generating different types of noise
to perturb optimization problems, particularly for the Motzkin-Straus
quadratic program.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


def generate_gumbel_noise(
    key: jax.random.PRNGKey, 
    shape: Tuple[int, ...], 
    scale: float = 1.0
) -> jnp.ndarray:
    """
    Generate Gumbel noise using the direct formula.
    
    Gumbel(0, β) ~ -β * log(-log(U)) where U ~ Uniform(0,1)
    
    Args:
        key: JAX random key
        shape: Shape of the noise array to generate
        scale: Scale parameter β for Gumbel distribution
        
    Returns:
        Array of Gumbel-distributed noise
    """
    # Generate uniform random numbers
    uniform_noise = jax.random.uniform(key, shape, minval=1e-8, maxval=1.0)
    
    # Apply Gumbel transformation: -log(-log(U))
    gumbel_noise = -jnp.log(-jnp.log(uniform_noise))
    
    # Scale by β parameter
    return scale * gumbel_noise


def generate_gaussian_noise(
    key: jax.random.PRNGKey,
    shape: Tuple[int, ...],
    scale: float = 1.0
) -> jnp.ndarray:
    """
    Generate Gaussian noise for baseline comparison.
    
    Args:
        key: JAX random key
        shape: Shape of the noise array to generate
        scale: Standard deviation for Gaussian distribution
        
    Returns:
        Array of Gaussian-distributed noise
    """
    return scale * jax.random.normal(key, shape)


def perturb_adjacency_matrix(
    adjacency_matrix: jnp.ndarray,
    noise: jnp.ndarray,
    preserve_symmetry: bool = True
) -> jnp.ndarray:
    """
    Add noise to adjacency matrix while preserving graph properties.
    
    Args:
        adjacency_matrix: Original adjacency matrix A
        noise: Noise to add to the matrix
        preserve_symmetry: Whether to ensure the result is symmetric
        
    Returns:
        Perturbed adjacency matrix
    """
    perturbed = adjacency_matrix + noise
    
    if preserve_symmetry:
        # Ensure symmetry: A_new = (A_new + A_new.T) / 2
        perturbed = (perturbed + perturbed.T) / 2.0
    
    return perturbed


def compute_noise_scale(
    adjacency_matrix: jnp.ndarray,
    scale_factor: float = 0.1
) -> float:
    """
    Compute appropriate noise scale relative to the adjacency matrix.
    
    Args:
        adjacency_matrix: The graph's adjacency matrix
        scale_factor: Fraction of the mean absolute value to use as scale
        
    Returns:
        Computed noise scale
    """
    # Use mean of non-zero elements (edges) as reference
    non_zero_mask = adjacency_matrix != 0
    if jnp.sum(non_zero_mask) > 0:
        mean_edge_value = jnp.mean(jnp.abs(adjacency_matrix[non_zero_mask]))
    else:
        # If no edges, use 1.0 as default
        mean_edge_value = 1.0
    
    return scale_factor * mean_edge_value


def validate_noise_parameters(
    adjacency_matrix: jnp.ndarray,
    noise_scale: float,
    verbose: bool = True
) -> None:
    """
    Validate that noise parameters are reasonable for the given matrix.
    
    Args:
        adjacency_matrix: The graph's adjacency matrix
        noise_scale: Proposed noise scale
        verbose: Whether to print validation information
    """
    matrix_range = jnp.max(adjacency_matrix) - jnp.min(adjacency_matrix)
    noise_to_signal_ratio = noise_scale / (matrix_range + 1e-8)
    
    if verbose:
        print(f"Matrix range: [{jnp.min(adjacency_matrix):.3f}, {jnp.max(adjacency_matrix):.3f}]")
        print(f"Noise scale: {noise_scale:.3f}")
        print(f"Noise-to-signal ratio: {noise_to_signal_ratio:.3f}")
        
        if noise_to_signal_ratio > 1.0:
            print("⚠️  Warning: Noise scale is larger than matrix range")
        elif noise_to_signal_ratio < 0.01:
            print("⚠️  Warning: Noise scale might be too small to have effect")
        else:
            print("✅ Noise scale appears reasonable")


# Test functions for validation
if __name__ == "__main__":
    # Quick test of noise generation
    key = jax.random.PRNGKey(42)
    
    # Test Gumbel noise
    gumbel = generate_gumbel_noise(key, (5, 5), scale=0.1)
    print("Gumbel noise sample:")
    print(gumbel)
    
    # Test Gaussian noise
    key, subkey = jax.random.split(key)
    gaussian = generate_gaussian_noise(subkey, (5, 5), scale=0.1)
    print("\nGaussian noise sample:")
    print(gaussian)
    
    # Test with adjacency matrix
    A = jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)  # Triangle
    key, subkey = jax.random.split(key)
    noise = generate_gumbel_noise(subkey, A.shape, scale=0.1)
    A_perturbed = perturb_adjacency_matrix(A, noise)
    
    print("\nOriginal matrix:")
    print(A)
    print("\nPerturbed matrix:")
    print(A_perturbed)
    print("\nSymmetry check:", jnp.allclose(A_perturbed, A_perturbed.T))