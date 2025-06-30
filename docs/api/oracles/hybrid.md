# Hybrid Oracles

Hybrid oracles combine multiple solving approaches to achieve optimal performance across different problem types and sizes. They intelligently select or blend classical and quantum methods based on problem characteristics, providing the best of both worlds.

## Design Philosophy

### Adaptive Strategy Selection

Hybrid oracles implement **adaptive algorithms** that:

1. **Analyze problem characteristics**: Size, density, structure
2. **Select optimal method**: Based on empirical performance data
3. **Fall back gracefully**: When primary method fails
4. **Learn from experience**: Improve selection over time

### Performance Optimization Goals

- **Accuracy**: Maintain high solution quality across problem types
- **Speed**: Minimize total computation time
- **Robustness**: Handle edge cases and solver failures
- **Scalability**: Efficient performance from small to large problems

## DiracNetworkXHybridOracle

### Overview

The `DiracNetworkXHybridOracle` combines exact NetworkX algorithms for small graphs with Dirac-3 quantum annealing for larger problems.

### Decision Strategy

```python
def select_method(graph):
    """Select solving method based on graph size."""
    n = graph.number_of_nodes()
    
    if n <= threshold:
        return "networkx_exact"    # Use exact algorithms
    else:
        return "dirac_quantum"     # Use quantum annealing
```

### API Reference

```python
DiracNetworkXHybridOracle(
    networkx_size_threshold: int = 15,
    num_samples: int = 100,
    relax_schedule: int = 2,
    solution_precision: float = 0.001,
    sum_constraint: int = 1,
    mean_photon_number: Optional[float] = None,
    quantum_fluctuation_coefficient: Optional[int] = None
)
```

#### Parameters

<div class="parameter">
<span class="parameter-name">networkx_size_threshold</span>: <span class="parameter-type">int = 15</span><br>
Maximum graph size for using exact NetworkX algorithms.
<span class="parameter-range">Typical range: 10-25</span><br>
<strong>Rationale</strong>: Exact algorithms become prohibitively slow beyond this threshold.
</div>

<div class="parameter">
<span class="parameter-name">num_samples</span>: <span class="parameter-type">int = 100</span><br>
Number of samples for Dirac-3 quantum solver (when used).
</div>

<div class="parameter">
<span class="parameter-name">relax_schedule</span>: <span class="parameter-type">int = 2</span><br>
Relaxation schedule for quantum annealing (when used).
</div>

*Additional parameters follow Dirac-3 Oracle specification.*

### Usage Examples

#### Basic Hybrid Usage

```python
from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle

# Hybrid oracle with default threshold
hybrid_oracle = DiracNetworkXHybridOracle()

# Small graph - uses exact NetworkX
small_G = nx.cycle_graph(12)
omega_small = hybrid_oracle.get_omega(small_G)
print(f"Method used: {hybrid_oracle.last_method_used}")  # "networkx"

# Large graph - uses Dirac-3
large_G = nx.barabasi_albert_graph(50, 4)
omega_large = hybrid_oracle.get_omega(large_G)
print(f"Method used: {hybrid_oracle.last_method_used}")  # "dirac"
```

#### Custom Threshold Configuration

```python
# Conservative threshold for guaranteed exact solutions
conservative_oracle = DiracNetworkXHybridOracle(
    networkx_size_threshold=20,    # Larger exact region
    num_samples=50,                # Fewer quantum samples (faster)
    relax_schedule=1               # Fastest quantum schedule
)

# Aggressive threshold for speed
aggressive_oracle = DiracNetworkXHybridOracle(
    networkx_size_threshold=8,     # Smaller exact region
    num_samples=150,               # More quantum samples (quality)
    relax_schedule=4               # Best quantum schedule
)
```

### Performance Characteristics

```python
# Performance analysis by graph size
def analyze_hybrid_performance():
    oracle = DiracNetworkXHybridOracle(networkx_size_threshold=15)
    
    sizes = [5, 10, 15, 20, 30, 50, 100]
    results = []
    
    for n in sizes:
        G = nx.erdos_renyi_graph(n, 0.3)
        start_time = time.time()
        omega = oracle.get_omega(G)
        elapsed = time.time() - start_time
        
        results.append({
            'size': n,
            'method': oracle.last_method_used,
            'time': elapsed,
            'omega': omega
        })
    
    return results
```

## DiracPGDHybridOracle

### Overview

The `DiracPGDHybridOracle` combines JAX Projected Gradient Descent with Dirac-3 quantum annealing in a sequential or parallel fashion.

### Strategy Options

1. **PGD First**: Try fast classical optimization, fall back to quantum
2. **Quantum First**: Use quantum as primary, PGD for refinement
3. **Parallel**: Run both methods, select best result
4. **Sequential**: Use PGD result to initialize quantum solver

### API Reference

```python
DiracPGDHybridOracle(
    use_pgd_first: bool = True,
    pgd_time_limit: float = 10.0,
    parallel_execution: bool = False,
    # PGD parameters (prefixed with pgd_)
    pgd_learning_rate: float = 0.02,
    pgd_max_iterations: int = 1000,
    pgd_num_restarts: int = 5,
    pgd_tolerance: float = 1e-6,
    # Dirac parameters (prefixed with dirac_)
    dirac_num_samples: int = 50,
    dirac_relax_schedule: int = 2,
    dirac_sum_constraint: int = 1,
    dirac_mean_photon_number: Optional[float] = None,
    dirac_quantum_fluctuation_coefficient: Optional[int] = None
)
```

#### Key Parameters

<div class="parameter">
<span class="parameter-name">use_pgd_first</span>: <span class="parameter-type">bool = True</span><br>
Whether to try PGD optimization before quantum annealing.
</div>

<div class="parameter">
<span class="parameter-name">pgd_time_limit</span>: <span class="parameter-type">float = 10.0</span><br>
Maximum time (seconds) to spend on PGD optimization.
</div>

<div class="parameter">
<span class="parameter-name">parallel_execution</span>: <span class="parameter-type">bool = False</span><br>
Whether to run PGD and Dirac-3 in parallel and select best result.
</div>

### Usage Examples

#### Sequential Strategy (Default)

```python
from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle

# PGD first, quantum fallback
hybrid_oracle = DiracPGDHybridOracle(
    use_pgd_first=True,
    pgd_time_limit=15.0,           # Give PGD 15 seconds
    pgd_num_restarts=10,           # Thorough PGD search
    dirac_num_samples=30           # Moderate quantum effort
)

G = nx.karate_club_graph()
omega = hybrid_oracle.get_omega(G)
print(f"Methods used: {hybrid_oracle.methods_attempted}")
print(f"Best method: {hybrid_oracle.best_method}")
```

#### Parallel Strategy

```python
# Run both methods simultaneously
parallel_oracle = DiracPGDHybridOracle(
    parallel_execution=True,
    pgd_num_restarts=5,            # Balanced PGD effort
    dirac_num_samples=40,          # Balanced quantum effort
    dirac_relax_schedule=3         # High-quality quantum
)

# Both methods run concurrently
omega = parallel_oracle.get_omega(G)
print(f"PGD result: {parallel_oracle.pgd_result}")
print(f"Dirac result: {parallel_oracle.dirac_result}")
print(f"Selected: {parallel_oracle.selected_result}")
```

#### Quantum-First Strategy

```python
# Quantum primary, PGD refinement
quantum_first_oracle = DiracPGDHybridOracle(
    use_pgd_first=False,           # Quantum first
    dirac_num_samples=100,         # High quantum effort
    dirac_relax_schedule=4,        # Best quantum quality
    pgd_learning_rate=0.001,       # Fine-tuned refinement
    pgd_max_iterations=500         # Limited refinement
)
```

### Advanced Configurations

#### Problem-Adaptive Hybrid

```python
class AdaptiveHybridOracle(DiracPGDHybridOracle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = {}
    
    def solve_quadratic_program(self, adjacency_matrix):
        # Analyze problem characteristics
        n = adjacency_matrix.shape[0]
        density = np.sum(adjacency_matrix) / (n * (n - 1))
        problem_key = (n // 10, int(density * 10))  # Discretize characteristics
        
        # Adapt strategy based on historical performance
        if problem_key in self.performance_history:
            history = self.performance_history[problem_key]
            if history['pgd_success_rate'] > 0.8:
                self.pgd_time_limit *= 1.5  # Give PGD more time
            else:
                self.dirac_num_samples = min(150, self.dirac_num_samples * 1.2)
        
        # Execute hybrid strategy
        result = super().solve_quadratic_program(adjacency_matrix)
        
        # Update performance history
        self._update_history(problem_key, result)
        
        return result
```

#### Multi-Stage Hybrid

```python
class MultiStageHybridOracle(Oracle):
    def __init__(self):
        super().__init__()
        self.fast_oracle = ProjectedGradientDescentOracle(
            num_restarts=3, max_iterations=200
        )
        self.quality_oracle = DiracOracle(
            num_samples=50, relax_schedule=3
        )
        self.exact_oracle = GurobiOracle()
    
    def solve_quadratic_program(self, adjacency_matrix):
        n = adjacency_matrix.shape[0]
        
        # Stage 1: Fast approximation
        fast_result = self.fast_oracle.solve_quadratic_program(adjacency_matrix)
        
        # Stage 2: Quality check
        if n < 30:  # Small enough for exact solution
            exact_result = self.exact_oracle.solve_quadratic_program(adjacency_matrix)
            if abs(fast_result - exact_result) < 1e-6:
                return exact_result  # Fast result was optimal
        
        # Stage 3: High-quality approximation
        return self.quality_oracle.solve_quadratic_program(adjacency_matrix)
```

## Custom Hybrid Patterns

### Ensemble Hybrid Oracle

```python
class EnsembleHybridOracle(Oracle):
    """Weighted ensemble of multiple oracles."""
    
    def __init__(self, oracles, weights=None):
        super().__init__()
        self.oracles = oracles
        self.weights = weights or [1.0] * len(oracles)
        self.normalize_weights()
    
    def normalize_weights(self):
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def solve_quadratic_program(self, adjacency_matrix):
        results = []
        for oracle in self.oracles:
            try:
                result = oracle.solve_quadratic_program(adjacency_matrix)
                results.append(result)
            except OracleError:
                results.append(0.0)  # Fallback value
        
        # Weighted average
        return sum(w * r for w, r in zip(self.weights, results))

# Usage
ensemble = EnsembleHybridOracle(
    oracles=[
        ProjectedGradientDescentOracle(num_restarts=10),
        MirrorDescentOracle(num_restarts=10),
        DiracOracle(num_samples=30)
    ],
    weights=[0.4, 0.3, 0.3]  # Favor classical methods slightly
)
```

### Confidence-Based Hybrid

```python
class ConfidenceHybridOracle(Oracle):
    """Select oracle based on confidence in solution."""
    
    def __init__(self):
        super().__init__()
        self.oracles = [
            ProjectedGradientDescentOracle(num_restarts=5),
            DiracOracle(num_samples=30)
        ]
    
    def solve_quadratic_program(self, adjacency_matrix):
        results = []
        confidences = []
        
        for oracle in self.oracles:
            result = oracle.solve_quadratic_program(adjacency_matrix)
            confidence = self.estimate_confidence(oracle, adjacency_matrix, result)
            results.append(result)
            confidences.append(confidence)
        
        # Select result with highest confidence
        best_idx = np.argmax(confidences)
        return results[best_idx]
    
    def estimate_confidence(self, oracle, adjacency_matrix, result):
        """Estimate confidence in oracle result."""
        if hasattr(oracle, 'convergence_history'):
            # Check convergence stability
            history = oracle.convergence_history
            if len(history) > 10:
                recent_variance = np.var(history[-10:])
                return 1.0 / (1.0 + recent_variance)
        
        # Default confidence based on oracle type
        if isinstance(oracle, GurobiOracle):
            return 1.0  # Exact solver
        elif isinstance(oracle, DiracOracle):
            return 0.8  # High-quality quantum
        else:
            return 0.6  # Classical approximation
```

## Hybrid Performance Analysis

### Benchmarking Framework

```python
def benchmark_hybrid_oracles():
    """Compare different hybrid strategies."""
    
    oracles = {
        'NetworkX-Dirac': DiracNetworkXHybridOracle(networkx_size_threshold=15),
        'PGD-Dirac Sequential': DiracPGDHybridOracle(use_pgd_first=True),
        'PGD-Dirac Parallel': DiracPGDHybridOracle(parallel_execution=True),
        'Pure PGD': ProjectedGradientDescentOracle(num_restarts=10),
        'Pure Dirac': DiracOracle(num_samples=50)
    }
    
    # Test graphs of various sizes and types
    test_graphs = [
        (nx.cycle_graph(10), "cycle_10"),
        (nx.erdos_renyi_graph(25, 0.3), "er_25_sparse"),
        (nx.erdos_renyi_graph(25, 0.7), "er_25_dense"),
        (nx.barabasi_albert_graph(50, 3), "ba_50_3"),
        (nx.karate_club_graph(), "karate")
    ]
    
    results = {}
    for oracle_name, oracle in oracles.items():
        results[oracle_name] = {}
        for graph, graph_name in test_graphs:
            start_time = time.time()
            omega = oracle.get_omega(graph)
            elapsed = time.time() - start_time
            
            results[oracle_name][graph_name] = {
                'omega': omega,
                'time': elapsed,
                'calls': oracle.call_count
            }
    
    return results
```

### Performance Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hybrid_performance(results):
    """Visualize hybrid oracle performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time performance
    times = {oracle: [data['time'] for data in results[oracle].values()] 
             for oracle in results}
    axes[0,0].boxplot(times.values(), labels=times.keys())
    axes[0,0].set_title('Execution Time Distribution')
    axes[0,0].set_ylabel('Time (seconds)')
    
    # Solution quality (assuming we know optimal values)
    # ... additional plotting code
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

### Hybrid Oracle Selection

```python
def recommend_hybrid_oracle(graph_characteristics):
    """Recommend optimal hybrid oracle based on problem."""
    
    n = graph_characteristics['nodes']
    density = graph_characteristics['density']
    structure = graph_characteristics['structure']  # 'regular', 'random', 'scale-free'
    
    if n <= 20:
        return DiracNetworkXHybridOracle(networkx_size_threshold=25)
    elif density > 0.7:  # Dense graphs
        return DiracPGDHybridOracle(
            use_pgd_first=False,  # Quantum handles density well
            dirac_relax_schedule=4
        )
    elif structure == 'scale-free':  # Complex structure
        return DiracPGDHybridOracle(
            parallel_execution=True,  # Try both approaches
            pgd_num_restarts=15
        )
    else:  # General case
        return DiracPGDHybridOracle(use_pgd_first=True)
```

### Configuration Guidelines

1. **Small graphs (< 20 nodes)**: Use NetworkX-Dirac hybrid with threshold 20-25
2. **Medium graphs (20-100 nodes)**: Use PGD-Dirac sequential hybrid
3. **Large graphs (> 100 nodes)**: Consider parallel execution for time-critical applications
4. **Dense graphs**: Favor quantum-first strategies
5. **Sparse graphs**: Favor classical-first strategies

### Error Handling in Hybrids

```python
class RobustHybridOracle(DiracPGDHybridOracle):
    def solve_quadratic_program(self, adjacency_matrix):
        try:
            return super().solve_quadratic_program(adjacency_matrix)
        except Exception as e:
            # Fallback to most reliable method
            fallback_oracle = ProjectedGradientDescentOracle(num_restarts=20)
            return fallback_oracle.solve_quadratic_program(adjacency_matrix)
```

---

**Related Documentation**:
- [Oracle Overview](overview.md) - Base oracle concepts
- [JAX Solvers](jax-solvers.md) - Classical optimization methods
- [Dirac Oracle](dirac.md) - Quantum computing approach
- [Performance Tuning](../../guides/performance-tuning.md) - Optimization strategies