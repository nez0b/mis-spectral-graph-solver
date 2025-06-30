# Dirac-3 Configuration Guide

This comprehensive guide covers configuring the Dirac-3 quantum oracle for optimal performance across different problem types and requirements.

## Understanding Dirac-3 Parameters

### Core Parameters Overview

The Dirac-3 oracle provides fine-grained control over quantum computing parameters:

```python
DiracOracle(
    num_samples=100,                            # Quantum measurement samples
    relax_schedule=2,                          # Annealing schedule {1,2,3,4}
    sum_constraint=1,                          # Simplex constraint value
    solution_precision=0.001,                  # Continuous solution precision
    mean_photon_number=None,                   # Quantum coherence control
    quantum_fluctuation_coefficient=None       # Quantum noise level
)
```

### Quantum Physics Background

Understanding the physics helps optimize performance:

<div class="quantum-note">
**Photonic Quantum Computing**: Dirac-3 encodes optimization variables in photon arrival times using temporal multiplexing. This approach naturally handles continuous optimization problems like the Motzkin-Straus QP.
</div>

## Parameter Tuning Strategies

### 1. Relaxation Schedule Selection

The relaxation schedule controls the quantum annealing process:

| Schedule | Dissipation | Quality | Speed | Best For |
|----------|-------------|---------|-------|----------|
| **1** | High | Good | Fast | Quick approximations |
| **2** | Medium-High | Better | Medium | Balanced performance |
| **3** | Medium-Low | High | Slow | Quality solutions |
| **4** | Low | Highest | Slowest | Critical accuracy |

#### Configuration Examples

```python
# Fast approximation (research exploration)
fast_oracle = DiracOracle(
    relax_schedule=1,
    num_samples=20
)

# Production quality (balanced)
production_oracle = DiracOracle(
    relax_schedule=2,
    num_samples=50
)

# Research quality (best solutions)
research_oracle = DiracOracle(
    relax_schedule=4,
    num_samples=100
)
```

### 2. Sample Count Optimization

More samples improve solution quality but increase computation time:

```python
def optimize_sample_count(graph, time_budget_seconds=60):
    """Find optimal sample count within time budget."""
    
    base_samples = 10
    max_samples = 200
    
    # Estimate time per sample with small test
    test_oracle = DiracOracle(num_samples=base_samples, relax_schedule=1)
    start_time = time.time()
    test_oracle.get_omega(graph)
    time_per_sample = (time.time() - start_time) / base_samples
    
    # Calculate optimal sample count
    optimal_samples = min(max_samples, int(time_budget_seconds / time_per_sample))
    
    return DiracOracle(
        num_samples=optimal_samples,
        relax_schedule=2  # Balanced quality
    )
```

### 3. Advanced Quantum Parameters

#### Mean Photon Number Tuning

Controls quantum coherence strength:

```python
def select_photon_number(problem_type, graph_density):
    """Select optimal mean photon number."""
    
    if problem_type == "exploration":
        return 0.0001  # Maximum quantum coherence
    elif problem_type == "exploitation":
        return 0.005   # More classical behavior
    
    # Density-based selection
    if graph_density > 0.7:  # Dense graphs
        return 0.0003  # Strong coherence for complex landscapes
    elif graph_density < 0.3:  # Sparse graphs  
        return 0.002   # Moderate coherence
    else:
        return 0.001   # Default for medium density
```

#### Quantum Fluctuation Coefficient

Controls exploration vs exploitation balance:

```python
def select_fluctuation_coefficient(optimization_landscape):
    """Select quantum fluctuation based on problem complexity."""
    
    if optimization_landscape == "many_local_minima":
        return 90  # High exploration
    elif optimization_landscape == "few_local_minima":
        return 20  # Low exploration  
    elif optimization_landscape == "smooth":
        return 10  # Minimal exploration
    else:
        return 50  # Default moderate exploration
```

## Problem-Specific Configurations

### Graph Type Optimization

#### Dense Graphs (> 70% edge density)

```python
def dense_graph_config():
    """Optimal configuration for dense graphs."""
    return DiracOracle(
        num_samples=80,
        relax_schedule=4,                      # High quality needed
        mean_photon_number=0.0002,             # Strong coherence
        quantum_fluctuation_coefficient=80     # High exploration
    )

# Usage
dense_G = nx.erdos_renyi_graph(50, 0.8)
oracle = dense_graph_config()
omega = oracle.get_omega(dense_G)
```

#### Sparse Graphs (< 30% edge density)

```python
def sparse_graph_config():
    """Optimal configuration for sparse graphs."""
    return DiracOracle(
        num_samples=40,
        relax_schedule=2,                      # Balanced approach
        mean_photon_number=0.003,              # Faster convergence
        quantum_fluctuation_coefficient=30     # Moderate exploration
    )

# Usage  
sparse_G = nx.erdos_renyi_graph(50, 0.2)
oracle = sparse_graph_config()
omega = oracle.get_omega(sparse_G)
```

#### Scale-Free Networks

```python
def scale_free_config():
    """Configuration for scale-free networks (power-law degree distribution)."""
    return DiracOracle(
        num_samples=60,
        relax_schedule=3,                      # Good quality
        mean_photon_number=0.001,              # Balanced coherence
        quantum_fluctuation_coefficient=70     # High exploration for hubs
    )

# Usage
scale_free_G = nx.barabasi_albert_graph(100, 5)
oracle = scale_free_config()
omega = oracle.get_omega(scale_free_G)
```

### Problem Size Scaling

#### Small Problems (< 30 nodes)

```python
def small_problem_config():
    """High-precision configuration for small problems."""
    return DiracOracle(
        num_samples=150,                       # Many samples for precision
        relax_schedule=4,                      # Highest quality
        solution_precision=None,               # Continuous precision
        mean_photon_number=0.0001,             # Maximum coherence
        quantum_fluctuation_coefficient=90     # Full exploration
    )
```

#### Medium Problems (30-100 nodes)

```python
def medium_problem_config():
    """Balanced configuration for medium problems."""
    return DiracOracle(
        num_samples=75,
        relax_schedule=3,
        mean_photon_number=0.0005,
        quantum_fluctuation_coefficient=60
    )
```

#### Large Problems (> 100 nodes)

```python
def large_problem_config():
    """Efficient configuration for large problems."""
    return DiracOracle(
        num_samples=40,                        # Fewer samples for speed
        relax_schedule=2,                      # Faster schedule
        mean_photon_number=0.001,              # Moderate coherence
        quantum_fluctuation_coefficient=40     # Focused exploration
    )
```

## Adaptive Configuration Strategies

### Dynamic Parameter Adjustment

```python
class AdaptiveDiracOracle(DiracOracle):
    """Dirac oracle that adapts parameters based on performance."""
    
    def __init__(self, **initial_params):
        super().__init__(**initial_params)
        self.performance_history = []
        self.adaptation_enabled = True
    
    def solve_quadratic_program(self, adjacency_matrix):
        if self.adaptation_enabled and len(self.performance_history) > 3:
            self._adapt_parameters(adjacency_matrix)
        
        start_time = time.time()
        result = super().solve_quadratic_program(adjacency_matrix)
        elapsed = time.time() - start_time
        
        # Record performance
        self.performance_history.append({
            'graph_size': adjacency_matrix.shape[0],
            'graph_density': np.sum(adjacency_matrix) / (adjacency_matrix.shape[0]**2),
            'result': result,
            'time': elapsed,
            'params': self._get_current_params()
        })
        
        return result
    
    def _adapt_parameters(self, adjacency_matrix):
        """Adapt parameters based on recent performance."""
        recent = self.performance_history[-3:]
        avg_time = np.mean([h['time'] for h in recent])
        
        # If taking too long, reduce sample count
        if avg_time > 30:  # 30 second threshold
            self.num_samples = max(20, int(self.num_samples * 0.8))
        
        # If very fast, we can afford more samples
        elif avg_time < 5:
            self.num_samples = min(150, int(self.num_samples * 1.2))
```

### Multi-Phase Configuration

```python
class MultiPhaseDiracOracle(DiracOracle):
    """Multi-phase optimization with parameter progression."""
    
    def __init__(self):
        # Start with exploration phase
        super().__init__(
            num_samples=30,
            relax_schedule=1,  # Fast exploration
            mean_photon_number=0.0001,  # High coherence
            quantum_fluctuation_coefficient=90  # High exploration
        )
        self.phase = "exploration"
        self.exploration_results = []
    
    def solve_quadratic_program(self, adjacency_matrix):
        if self.phase == "exploration":
            # Exploration phase: fast, high exploration
            result = super().solve_quadratic_program(adjacency_matrix)
            self.exploration_results.append(result)
            
            # Switch to exploitation if we have enough data
            if len(self.exploration_results) >= 3:
                self._switch_to_exploitation()
                result = super().solve_quadratic_program(adjacency_matrix)
            
            return result
        else:
            # Exploitation phase: quality refinement
            return super().solve_quadratic_program(adjacency_matrix)
    
    def _switch_to_exploitation(self):
        """Switch to exploitation configuration."""
        self.phase = "exploitation"
        self.num_samples = 100
        self.relax_schedule = 4  # High quality
        self.quantum_fluctuation_coefficient = 30  # Lower exploration
```

## Performance Monitoring and Debugging

### Configuration Performance Analysis

```python
def analyze_configuration_performance(configs, test_graphs):
    """Compare different Dirac configurations."""
    
    results = {}
    
    for config_name, config_params in configs.items():
        results[config_name] = {}
        oracle = DiracOracle(**config_params)
        
        for graph_name, graph in test_graphs.items():
            start_time = time.time()
            omega = oracle.get_omega(graph)
            elapsed = time.time() - start_time
            
            results[config_name][graph_name] = {
                'omega': omega,
                'time': elapsed,
                'samples': config_params['num_samples'],
                'schedule': config_params['relax_schedule']
            }
    
    return results

# Example usage
configs = {
    'fast': {'num_samples': 20, 'relax_schedule': 1},
    'balanced': {'num_samples': 50, 'relax_schedule': 2},
    'quality': {'num_samples': 100, 'relax_schedule': 4}
}

test_graphs = {
    'cycle_20': nx.cycle_graph(20),
    'complete_15': nx.complete_graph(15),
    'erdos_renyi_30': nx.erdos_renyi_graph(30, 0.5)
}

analysis = analyze_configuration_performance(configs, test_graphs)
```

### Quantum Parameter Validation

```python
def validate_quantum_parameters(oracle):
    """Validate quantum parameters are in physical ranges."""
    
    issues = []
    
    # Check mean photon number
    if oracle.mean_photon_number is not None:
        if not (6.67e-5 <= oracle.mean_photon_number <= 6.67e-3):
            issues.append(f"mean_photon_number {oracle.mean_photon_number} outside valid range")
    
    # Check quantum fluctuation coefficient
    if oracle.quantum_fluctuation_coefficient is not None:
        if not (1 <= oracle.quantum_fluctuation_coefficient <= 100):
            issues.append(f"quantum_fluctuation_coefficient {oracle.quantum_fluctuation_coefficient} outside valid range")
    
    # Check for parameter conflicts
    if (oracle.mean_photon_number is not None and 
        oracle.quantum_fluctuation_coefficient is not None):
        
        # High coherence + high fluctuation may be suboptimal
        if (oracle.mean_photon_number < 0.0005 and 
            oracle.quantum_fluctuation_coefficient > 80):
            issues.append("High coherence + high fluctuation may cause instability")
    
    return issues
```

### Performance Optimization Tips

#### Memory Optimization

```python
# For memory-constrained environments
memory_efficient_oracle = DiracOracle(
    num_samples=20,          # Fewer samples
    relax_schedule=1,        # Faster processing
    solution_precision=0.01  # Lower precision
)
```

#### Time-Critical Applications

```python
# For real-time or time-critical applications
fast_oracle = DiracOracle(
    num_samples=15,
    relax_schedule=1,
    mean_photon_number=0.005,              # Faster convergence
    quantum_fluctuation_coefficient=20     # Focused search
)
```

#### Research Applications

```python
# For research requiring highest quality
research_oracle = DiracOracle(
    num_samples=200,         # Maximum samples
    relax_schedule=4,        # Highest quality
    solution_precision=None, # Continuous precision
    mean_photon_number=0.0001,             # Maximum coherence
    quantum_fluctuation_coefficient=95     # Full exploration
)
```

## Best Practices Summary

### Configuration Guidelines

1. **Start with defaults** for initial experiments
2. **Use relaxation schedule 2-3** for most applications  
3. **Increase samples** for critical accuracy requirements
4. **Tune quantum parameters** only after understanding base performance
5. **Monitor performance** and adapt based on problem characteristics

### Common Pitfalls to Avoid

- **Over-parameterization**: Don't tune all parameters simultaneously
- **Ignoring physics**: Quantum parameters have physical meaning
- **Single configuration**: Different problems need different settings
- **No validation**: Always validate parameter ranges

### Recommended Workflow

```python
def recommended_dirac_workflow(graph):
    """Recommended configuration workflow."""
    
    # 1. Analyze problem characteristics
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    
    # 2. Select base configuration
    if n < 30:
        base_config = small_problem_config()
    elif n < 100:
        base_config = medium_problem_config()
    else:
        base_config = large_problem_config()
    
    # 3. Adjust for graph density
    if density > 0.7:
        base_config.quantum_fluctuation_coefficient *= 1.5
        base_config.mean_photon_number /= 2
    elif density < 0.3:
        base_config.quantum_fluctuation_coefficient *= 0.7
        base_config.mean_photon_number *= 1.5
    
    # 4. Validate parameters
    issues = validate_quantum_parameters(base_config)
    if issues:
        print(f"Configuration issues: {issues}")
    
    return base_config
```

---

**Related Documentation**:
- [Dirac Oracle API](../api/oracles/dirac.md) - Complete API reference
- [Performance Tuning](performance-tuning.md) - General optimization strategies
- [Troubleshooting](troubleshooting.md) - Common issues and solutions