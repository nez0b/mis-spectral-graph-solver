# JAX-Based Oracles

The JAX oracle family provides high-performance gradient-based optimization for the Motzkin-Straus quadratic program. These oracles leverage JAX's just-in-time (JIT) compilation and automatic differentiation to achieve excellent performance on modern hardware.

## JAX Framework Overview

### Key Advantages

- **JIT Compilation**: Automatic compilation to optimized XLA code
- **Automatic Differentiation**: Exact gradients without manual derivation
- **Vectorization**: Efficient parallel operations on modern hardware
- **GPU/TPU Support**: Seamless acceleration on specialized hardware

### Architectural Design

```python
# Common JAX oracle pattern
@jit
def optimization_step(x, adjacency_matrix, learning_rate):
    """JIT-compiled optimization step."""
    energy = 0.5 * x.T @ adjacency_matrix @ x
    gradient = adjacency_matrix @ x
    return update_rule(x, gradient, learning_rate)
```

## ProjectedGradientDescentOracle

### Mathematical Foundation

Projected Gradient Descent (PGD) solves the constrained optimization problem:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x$$

through the iterative updates:

1. **Gradient step**: $y^{(k+1)} = x^{(k)} + \alpha \nabla f(x^{(k)})$
2. **Simplex projection**: $x^{(k+1)} = \Pi_{\Delta_n}(y^{(k+1)})$

where $\Pi_{\Delta_n}$ is the projection onto the probability simplex.

### Simplex Projection Algorithm

The projection $\Pi_{\Delta_n}(y)$ finds the closest point in $\Delta_n$ to $y$:

```python
def project_simplex(y):
    """Project vector y onto probability simplex."""
    n = len(y)
    sorted_y = jnp.sort(y)[::-1]  # Descending order
    
    # Find the threshold for projection
    cumsum = jnp.cumsum(sorted_y)
    k = jnp.arange(1, n + 1)
    threshold_conditions = sorted_y - (cumsum - 1) / k > 0
    k_max = jnp.sum(threshold_conditions)
    theta = (jnp.sum(sorted_y[:k_max]) - 1) / k_max
    
    return jnp.maximum(y - theta, 0)
```

### API Reference

```python
ProjectedGradientDescentOracle(
    learning_rate: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    min_iterations: int = 50,
    num_restarts: int = 10,
    dirichlet_alpha: float = 1.0,
    verbose: bool = False
)
```

#### Parameters

<div class="parameter">
<span class="parameter-name">learning_rate</span>: <span class="parameter-type">float = 0.01</span><br>
Step size for gradient ascent. Higher values converge faster but may overshoot.
<span class="parameter-range">Typical range: 0.001-0.1</span>
</div>

<div class="parameter">
<span class="parameter-name">max_iterations</span>: <span class="parameter-type">int = 2000</span><br>
Maximum number of optimization iterations per restart.
</div>

<div class="parameter">
<span class="parameter-name">tolerance</span>: <span class="parameter-type">float = 1e-6</span><br>
Convergence tolerance for early stopping based on energy change.
</div>

<div class="parameter">
<span class="parameter-name">min_iterations</span>: <span class="parameter-type">int = 50</span><br>
Minimum iterations before early stopping can occur.
</div>

<div class="parameter">
<span class="parameter-name">num_restarts</span>: <span class="parameter-type">int = 10</span><br>
Number of random initializations. More restarts improve solution quality.
</div>

<div class="parameter">
<span class="parameter-name">dirichlet_alpha</span>: <span class="parameter-type">float = 1.0</span><br>
Concentration parameter for Dirichlet initialization. Lower values create more concentrated starting points.
</div>

### Usage Examples

#### Basic Usage

```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
import networkx as nx

G = nx.karate_club_graph()
oracle = ProjectedGradientDescentOracle()
omega = oracle.get_omega(G)
```

#### High-Quality Configuration

```python
# Configuration for best solution quality
oracle = ProjectedGradientDescentOracle(
    learning_rate=0.02,      # Moderate step size
    max_iterations=5000,     # More iterations
    num_restarts=20,         # Many restarts
    tolerance=1e-8,          # Tight convergence
    dirichlet_alpha=0.5      # Concentrated initialization
)
```

#### Fast Configuration

```python
# Configuration for speed over quality
oracle = ProjectedGradientDescentOracle(
    learning_rate=0.05,      # Larger steps
    max_iterations=500,      # Fewer iterations
    num_restarts=3,          # Fewer restarts
    tolerance=1e-5           # Looser convergence
)
```

## MirrorDescentOracle

### Mathematical Foundation

Mirror Descent uses the **exponentiated gradient** method, which is naturally suited for simplex constraints. The update rule works in the "dual space":

1. **Dual update**: $\theta^{(k+1)} = \theta^{(k)} + \alpha \nabla f(x^{(k)})$
2. **Primal mapping**: $x^{(k+1)} = \frac{\exp(\theta^{(k+1)})}{\sum_i \exp(\theta^{(k+1)}_i)}$

This naturally maintains the simplex constraint $\sum_i x_i = 1, x_i \geq 0$ without explicit projection.

### Entropic Regularization

The method can be viewed as solving the regularized problem:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x + \frac{1}{\beta} H(x)$$

where $H(x) = -\sum_i x_i \log x_i$ is the entropy regularizer and $\beta$ is the inverse temperature.

### API Reference

```python
MirrorDescentOracle(
    learning_rate: float = 0.005,
    max_iterations: int = 2000,
    tolerance: float = 1e-6,
    min_iterations: int = 50,
    num_restarts: int = 10,
    dirichlet_alpha: float = 1.0,
    verbose: bool = False
)
```

### Key Differences from PGD

- **Learning rate**: Typically needs smaller values (default 0.005 vs 0.01)
- **No projection**: Updates naturally stay on simplex
- **Entropy bias**: Tends toward uniform distributions
- **Numerical stability**: Better handling of boundary conditions

### Usage Examples

#### Comparison with PGD

```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle

# Same graph, different methods
G = nx.erdos_renyi_graph(50, 0.3)

pgd_oracle = ProjectedGradientDescentOracle(num_restarts=5)
mirror_oracle = MirrorDescentOracle(num_restarts=5)

omega_pgd = pgd_oracle.get_omega(G)
omega_mirror = mirror_oracle.get_omega(G)

print(f"PGD result: {omega_pgd}")
print(f"Mirror Descent result: {omega_mirror}")
```

#### Dense Graph Optimization

```python
# Mirror Descent often works better on dense graphs
dense_G = nx.erdos_renyi_graph(30, 0.8)  # 80% edge probability

oracle = MirrorDescentOracle(
    learning_rate=0.008,     # Slightly higher for dense graphs
    num_restarts=15,         # More restarts for difficult problems
    max_iterations=3000      # More iterations for convergence
)

omega = oracle.get_omega(dense_G)
```

## FrankWolfeOracle

### Mathematical Foundation

The Frank-Wolfe algorithm (also called conditional gradient) avoids explicit projection by solving linear optimization subproblems:

1. **Linear oracle**: $s^{(k)} = \arg\max_{s \in \Delta_n} \langle \nabla f(x^{(k)}), s \rangle$
2. **Line search**: $\gamma^{(k)} = \arg\max_{\gamma \in [0,1]} f((1-\gamma)x^{(k)} + \gamma s^{(k)})$
3. **Update**: $x^{(k+1)} = (1-\gamma^{(k)})x^{(k)} + \gamma^{(k)} s^{(k)}$

### Linear Subproblem Solution

For the simplex constraint, the linear subproblem has a simple solution:

$$s^{(k)} = e_j \text{ where } j = \arg\max_i (\nabla f(x^{(k)}))_i$$

This makes each iteration very efficient.

### Key Properties

- **Projection-free**: No explicit simplex projection needed
- **Sparse iterates**: Solutions tend to be sparse
- **Memory efficient**: Constant memory requirements
- **Convergence rate**: O(1/k) for smooth objectives

### Usage Examples

#### Large-Scale Problems

```python
from motzkinstraus.oracles.jax_frank_wolfe import FrankWolfeOracle

# Frank-Wolfe excels on large, sparse problems
large_sparse_G = nx.barabasi_albert_graph(500, 3)

oracle = FrankWolfeOracle(
    max_iterations=1000,     # Fewer iterations due to efficiency
    line_search_steps=20,    # Accurate line search
    verbose=True             # Monitor progress
)

omega = oracle.get_omega(large_sparse_G)
```

## Performance Comparison

### Computational Complexity

| Oracle | Per-Iteration Cost | Memory Usage | Convergence Rate |
|--------|-------------------|--------------|------------------|
| **PGD** | O(n² + projection) | O(n²) | O(1/√k) |
| **Mirror Descent** | O(n²) | O(n²) | O(log k/k) |
| **Frank-Wolfe** | O(n²) | O(n) | O(1/k) |

### Problem-Specific Recommendations

#### Graph Density

```python
def select_jax_oracle(graph):
    """Select JAX oracle based on graph properties."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    
    if density > 0.7:
        return MirrorDescentOracle()  # Better for dense graphs
    elif n > 200:
        return FrankWolfeOracle()     # Memory efficient for large graphs
    else:
        return ProjectedGradientDescentOracle()  # General purpose
```

#### Quality vs Speed

```python
# Quality-focused configuration
quality_config = {
    'num_restarts': 20,
    'max_iterations': 5000,
    'tolerance': 1e-8,
    'learning_rate': 0.01
}

# Speed-focused configuration  
speed_config = {
    'num_restarts': 3,
    'max_iterations': 500,
    'tolerance': 1e-5,
    'learning_rate': 0.05
}
```

## Advanced Features

### Multi-restart Strategy

All JAX oracles implement sophisticated multi-restart strategies:

```python
def multi_restart_optimization(adjacency_matrix, num_restarts, oracle_config):
    """Multi-restart optimization with Dirichlet initialization."""
    best_energy = -float('inf')
    best_solution = None
    
    for restart in range(num_restarts):
        # Dirichlet initialization
        alpha = oracle_config.dirichlet_alpha
        x_init = np.random.dirichlet([alpha] * n)
        
        # Run optimization
        x_final, energy = single_restart_optimize(adjacency_matrix, x_init)
        
        if energy > best_energy:
            best_energy = energy
            best_solution = x_final
    
    return best_solution, best_energy
```

### Convergence Monitoring

```python
# Enable detailed monitoring
oracle = ProjectedGradientDescentOracle(verbose=True)
omega = oracle.get_omega(G)

# Access convergence information
print(f"Converged in {oracle.last_iterations} iterations")
print(f"Final energy: {oracle.last_energy:.8f}")
print(f"Energy history: {oracle.convergence_history}")

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(oracle.convergence_history)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Convergence History')
```

### Hardware Acceleration

```python
# Verify GPU availability
import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# JAX oracles automatically use available accelerators
oracle = ProjectedGradientDescentOracle()  # Will use GPU if available
```

### Custom Initialization Strategies

```python
class CustomInitPGDOracle(ProjectedGradientDescentOracle):
    def __init__(self, init_strategy='dirichlet', **kwargs):
        super().__init__(**kwargs)
        self.init_strategy = init_strategy
    
    def get_initialization(self, n):
        if self.init_strategy == 'uniform':
            return np.ones(n) / n
        elif self.init_strategy == 'random_vertex':
            x = np.zeros(n)
            x[np.random.randint(n)] = 1.0
            return x
        elif self.init_strategy == 'degree_weighted':
            # Initialize based on node degrees (requires graph)
            degrees = np.array([self.current_graph.degree(i) for i in range(n)])
            return degrees / np.sum(degrees)
        else:  # dirichlet
            return np.random.dirichlet([self.dirichlet_alpha] * n)
```

## Troubleshooting

### Common Issues

#### Convergence Problems

```python
# If optimization fails to converge
oracle = ProjectedGradientDescentOracle(
    learning_rate=0.005,     # Reduce learning rate
    max_iterations=10000,    # Increase iterations
    num_restarts=30,         # More restarts
    tolerance=1e-7           # Tighter tolerance
)
```

#### Numerical Instability

```python
# For numerically challenging problems
oracle = MirrorDescentOracle(
    learning_rate=0.001,     # Very small steps
    min_iterations=100,      # Ensure minimum progress
    verbose=True             # Monitor for issues
)
```

#### Memory Issues

```python
# For large problems with memory constraints
oracle = FrankWolfeOracle(
    max_iterations=500,      # Fewer iterations
    line_search_steps=5      # Simpler line search
)
```

---

**Next Steps**:
- [Gurobi Oracle](gurobi.md) - Commercial solver integration
- [Hybrid Oracles](hybrid.md) - Combining JAX with other methods  
- [Performance Tuning](../../guides/performance-tuning.md) - Optimization strategies