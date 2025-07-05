"""
Benchmark JAX Metal GPU vs CPU performance for vmap batching.

Tests JAX performance on different devices and matrix operations.
"""

import pytest
import time
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMetalGPUBenchmark:
    """Benchmark JAX Metal GPU performance vs CPU."""
    
    @pytest.fixture(scope="class", autouse=True)
    def jax_info(self):
        """Print JAX device information."""
        if JAX_AVAILABLE:
            print(f"\nJAX version: {jax.__version__}")
            print(f"Available devices: {jax.devices()}")
            print(f"Default backend: {jax.default_backend()}")
    
    @pytest.fixture
    def test_matrices(self):
        """Create test matrices for benchmarking."""
        key = jax.random.PRNGKey(42)
        matrices = {}
        
        for n in [10, 50, 100]:
            matrices[n] = {
                'A': jax.random.normal(key, (n, n)),
                'x': jax.random.normal(key, (n,))
            }
        return matrices
    
    def test_basic_matrix_operations(self, test_matrices):
        """Test basic JAX matrix operations performance."""
        @jax.jit
        def matrix_op(A, x):
            return A @ x
        
        results = {}
        
        for n, data in test_matrices.items():
            A, x = data['A'], data['x']
            
            # Warm up compilation
            _ = matrix_op(A, x)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):  # Reduced iterations for pytest
                result = matrix_op(A, x)
            result.block_until_ready()  # Wait for computation to complete
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
            results[n] = avg_time
            
            # Basic validation
            assert result.shape == (n,), f"Wrong result shape for {n}x{n} matrix"
            assert avg_time > 0, f"Invalid timing for {n}x{n} matrix"
        
        # Performance should scale reasonably
        assert results[100] > results[10], "Performance should scale with matrix size"
    
    def test_vmap_batching_performance(self):
        """Test vmap batching performance."""
        n = 50  # Matrix size
        batch_sizes = [1, 5, 10]  # Reduced for pytest
        
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (n, n))
        
        @jax.jit
        def single_operation(x):
            # Simulate PGD-like operation: matrix multiply + normalization
            grad = A @ x
            x_new = x + 0.01 * grad
            return x_new / jnp.sum(x_new)
        
        # Create vectorized version
        vectorized_op = jax.vmap(single_operation)
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            # Create batch of inputs
            x_batch = jax.random.normal(key, (batch_size, n))
            x_batch = x_batch / jnp.sum(x_batch, axis=1, keepdims=True)  # Normalize
            
            # Warm up
            _ = vectorized_op(x_batch)
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):  # Reduced for pytest
                result = vectorized_op(x_batch)
            result.block_until_ready()
            end_time = time.time()
            
            total_time = end_time - start_time
            batch_results[batch_size] = total_time
            
            # Validate result
            assert result.shape == (batch_size, n), f"Wrong batch result shape"
            assert jnp.allclose(jnp.sum(result, axis=1), 1.0, atol=1e-6), "Results should be normalized"
        
        # Larger batches should be more efficient per item
        per_item_1 = batch_results[1] / 1
        per_item_10 = batch_results[10] / 10
        assert per_item_10 <= per_item_1 * 1.5, "Batching should provide some efficiency gain"
    
    def test_metal_gpu_availability(self):
        """Test if Metal GPU is available and functional."""
        devices = jax.devices()
        device_types = [str(device).lower() for device in devices]
        
        # Check if we have GPU devices
        has_gpu = any('gpu' in device_type or 'metal' in device_type for device_type in device_types)
        
        if has_gpu:
            # Test basic operation on GPU
            key = jax.random.PRNGKey(42)
            x = jax.random.normal(key, (100, 100))
            
            # Simple operation
            result = jnp.matmul(x, x.T)
            
            assert result.shape == (100, 100), "GPU operation should work correctly"
        else:
            # Just note that GPU is not available
            print("GPU/Metal not available, testing CPU only")
    
    @pytest.mark.slow
    def test_device_comparison(self):
        """Compare performance across available devices."""
        if len(jax.devices()) <= 1:
            pytest.skip("Multiple devices not available for comparison")
        
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (200, 200))
        x = jax.random.normal(key, (200,))
        
        @jax.jit
        def compute_operation(A, x):
            # More complex operation for meaningful comparison
            for _ in range(10):
                x = A @ x
                x = x / jnp.linalg.norm(x)
            return x
        
        device_results = {}
        
        for device in jax.devices():
            # Move data to specific device
            A_device = jax.device_put(A, device)
            x_device = jax.device_put(x, device)
            
            # Warm up
            _ = compute_operation(A_device, x_device)
            
            # Benchmark
            start_time = time.time()
            result = compute_operation(A_device, x_device)
            result.block_until_ready()
            end_time = time.time()
            
            device_results[str(device)] = end_time - start_time
            
            # Validate result
            assert result.shape == (200,), f"Wrong result shape on {device}"
            assert jnp.allclose(jnp.linalg.norm(result), 1.0, atol=1e-6), f"Result should be normalized on {device}"
        
        # All devices should produce similar results
        results_array = list(device_results.values())
        if len(results_array) > 1:
            max_time = max(results_array)
            min_time = min(results_array)
            # Allow for reasonable variation in performance
            assert max_time <= min_time * 10, f"Device performance too variable: {device_results}"
    
    def test_jax_compilation_caching(self):
        """Test JAX compilation caching behavior."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (50, 50))
        
        @jax.jit
        def test_function(x):
            return jnp.sum(x**2)
        
        # First call (compilation)
        start_time = time.time()
        result1 = test_function(x)
        first_call_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        result2 = test_function(x)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert jnp.allclose(result1, result2), "Compiled function should give consistent results"
        
        # Second call should be much faster (cached compilation)
        assert second_call_time < first_call_time, "Cached compilation should be faster"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of JAX operations."""
        key = jax.random.PRNGKey(42)
        
        # Test that we can handle reasonably large arrays
        try:
            large_array = jax.random.normal(key, (1000, 1000))
            result = jnp.sum(large_array)
            
            assert jnp.isfinite(result), "Large array operation should produce finite result"
            
        except Exception as e:
            # If we run out of memory, that's okay for this test
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                pytest.skip(f"Insufficient memory for large array test: {e}")
            else:
                raise
    
    @pytest.mark.performance
    def test_gradient_computation_performance(self):
        """Test performance of gradient computations (relevant for PGD)."""
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (100, 100))
        x = jax.random.normal(key, (100,))
        
        def objective(x):
            return 0.5 * x.T @ A @ x
        
        # Compile gradient function
        grad_fn = jax.jit(jax.grad(objective))
        
        # Warm up
        _ = grad_fn(x)
        
        # Benchmark gradient computation
        start_time = time.time()
        for _ in range(10):
            grad = grad_fn(x)
        grad.block_until_ready()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Validate gradient
        assert grad.shape == x.shape, "Gradient should have same shape as input"
        assert jnp.allclose(grad, A @ x), "Gradient should match analytical result"
        
        # Performance should be reasonable
        assert avg_time < 1.0, f"Gradient computation too slow: {avg_time:.3f}s"