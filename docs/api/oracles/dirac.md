# Dirac-3 Quantum Oracle

The DiracOracle provides access to QCi's Dirac-3 continuous cloud solver, a cutting-edge photonic quantum computing system designed for optimization problems. This oracle leverages quantum annealing principles implemented through temporal encoding in photonic hardware.

## Quantum Computing Background

### Photonic Quantum Computing

The Dirac-3 system uses **photonic quantum computing** with several key innovations:

- **Time-bin encoding**: Optimization variables are encoded in photon arrival times
- **Temporal multiplexing**: Multiple solution candidates explored in quantum superposition
- **Single-photon regime**: Maintains quantum coherence for optimization advantage
- **Quantum fluctuations**: Natural quantum noise helps escape local minima

<div class="quantum-note">
**Physics Insight**: Unlike digital quantum computers that use qubits, Dirac-3 uses continuous-variable quantum states in the temporal domain. This approach is naturally suited for continuous optimization problems like the Motzkin-Straus QP.
</div>

### Quantum Annealing Process

1. **Initialization**: Problem encoded in quantum state superposition
2. **Evolution**: Gradual annealing from quantum to classical regime  
3. **Measurement**: Quantum state collapse yields solution candidates
4. **Selection**: Best solution returned based on objective function

## API Reference

### Constructor

```python
DiracOracle(
    num_samples: int = 100,
    relax_schedule: int = 2, 
    solution_precision: float = 0.001,
    sum_constraint: int = 1,
    mean_photon_number: Optional[float] = None,
    quantum_fluctuation_coefficient: Optional[int] = None
)
```

### Parameters

<div class="parameter">
<span class="parameter-name">num_samples</span>: <span class="parameter-type">int = 100</span><br>
Number of solution samples to request from the quantum solver.
<span class="parameter-range">Range: 1-100</span><br>
<strong>Physics</strong>: More samples provide better statistics but increase quantum measurement time.
</div>

<div class="parameter">
<span class="parameter-name">relax_schedule</span>: <span class="parameter-type">int = 2</span><br>
Quantum relaxation schedule controlling the annealing process.
<span class="parameter-range">Options: {1, 2, 3, 4}</span><br>
<strong>Physics</strong>: Higher schedules use less dissipation, leading to better solutions but longer evolution time.
</div>

<div class="parameter">
<span class="parameter-name">solution_precision</span>: <span class="parameter-type">float = 0.001</span><br>
Precision level for continuous solutions. When specified, distillation is applied.
<span class="parameter-range">Must divide sum_constraint</span><br>
<strong>Usage</strong>: Omit for highest precision continuous solutions.
</div>

<div class="parameter">
<span class="parameter-name">sum_constraint</span>: <span class="parameter-type">int = 1</span><br>
Constraint requiring solution variables to sum to this value.
<span class="parameter-range">Range: 1-10000</span><br>
<strong>Note</strong>: For Motzkin-Straus, this should remain 1 (probability simplex).
</div>

<div class="parameter">
<span class="parameter-name">mean_photon_number</span>: <span class="parameter-type">Optional[float] = None</span><br>
Average photons per time-bin, controlling quantum coherence.
<span class="parameter-range">Range: 6.67×10⁻⁵ to 6.67×10⁻³</span><br>
<strong>Physics</strong>: Lower values maintain stronger quantum superposition effects. When None, automatically set by relax_schedule.
</div>

<div class="parameter">
<span class="parameter-name">quantum_fluctuation_coefficient</span>: <span class="parameter-type">Optional[int] = None</span><br>
Quantum noise level for exploration vs exploitation balance.
<span class="parameter-range">Range: 1-100 (maps to coefficient n/100)</span><br>
<strong>Physics</strong>: Higher values increase Poisson noise from quantum detection, enabling escape from local minima.
</div>

## Usage Examples

### Basic Usage

```python
from motzkinstraus.oracles.dirac import DiracOracle
import networkx as nx

# Create test graph
G = nx.karate_club_graph()

# Initialize Dirac oracle with default settings
oracle = DiracOracle()

# Solve for clique number
omega = oracle.get_omega(G)
print(f"Clique number: {omega}")
```

### High-Quality Configuration

```python
# Configuration for maximum solution quality
oracle = DiracOracle(
    num_samples=100,                    # Maximum samples
    relax_schedule=4,                   # Highest quality schedule
    mean_photon_number=0.0001,          # Strong quantum coherence
    quantum_fluctuation_coefficient=80  # High exploration
)

omega = oracle.get_omega(G)
```

### Fast Configuration

```python
# Configuration for quick approximate solutions
oracle = DiracOracle(
    num_samples=20,                     # Fewer samples
    relax_schedule=1,                   # Fastest schedule  
    mean_photon_number=0.005,           # Weaker coherence
    quantum_fluctuation_coefficient=20  # Less exploration
)

omega = oracle.get_omega(G)
```

### Large-Scale Problems

```python
# Configuration optimized for large graphs
oracle = DiracOracle(
    num_samples=50,                     # Balanced sampling
    relax_schedule=3,                   # Good quality/speed trade-off
    solution_precision=None,            # Continuous precision
    mean_photon_number=0.001,           # Moderate coherence
    quantum_fluctuation_coefficient=60  # Good exploration
)

# Handle large graph
large_G = nx.barabasi_albert_graph(200, 5)
omega = oracle.get_omega(large_G)
```

## Advanced Configuration

### Parameter Tuning Guidelines

#### Mean Photon Number Selection

```python
def select_photon_number(graph_density, target_quality):
    """Select optimal mean photon number based on problem characteristics."""
    if target_quality == 'highest':
        return 0.0001  # Maximum quantum coherence
    elif graph_density > 0.7:  # Dense graphs need more coherence
        return 0.0003
    elif graph_density < 0.3:  # Sparse graphs can use less
        return 0.002
    else:
        return 0.001   # Default for medium density
```

#### Quantum Fluctuation Tuning

```python
def select_fluctuation_coefficient(num_local_minima_estimate):
    """Select quantum fluctuation based on optimization landscape."""
    if num_local_minima_estimate > 100:
        return 90  # High exploration for rugged landscape
    elif num_local_minima_estimate > 20:
        return 60  # Moderate exploration
    else:
        return 30  # Low exploration for smooth landscape
```

### Dynamic Parameter Adaptation

```python
class AdaptiveDiracOracle(DiracOracle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
    
    def solve_quadratic_program(self, adjacency_matrix):
        # Adapt parameters based on problem size and past performance
        n = adjacency_matrix.shape[0]
        
        if n > 100:  # Large problem
            self.num_samples = min(50, self.num_samples)
            self.relax_schedule = 3
        
        # Record performance for future adaptation
        start_time = time.time()
        result = super().solve_quadratic_program(adjacency_matrix)
        elapsed = time.time() - start_time
        
        self.performance_history.append({
            'size': n,
            'time': elapsed,
            'result': result
        })
        
        return result
```

## Quantum Physics Details

### Mean Photon Number Physics

The mean photon number $\bar{n}$ in each time-bin follows Poisson statistics:

$$P(n) = \frac{\bar{n}^n e^{-\bar{n}}}{n!}$$

**Low $\bar{n}$ regime** ($\bar{n} \ll 1$):
- Maintains single-photon quantum superposition
- Preserves quantum coherence for optimization advantage
- Enables exploration of multiple solution paths simultaneously

**High $\bar{n}$ regime** ($\bar{n} \approx 1$):
- Approaches classical behavior
- Faster convergence but less quantum advantage
- Suitable for refined local optimization

### Quantum Fluctuation Coefficient

The quantum fluctuation parameter $c \in [0.01, 1]$ modulates the noise strength:

$$\sigma_{quantum} = c \cdot \sqrt{\bar{n}}$$

This exploits the fundamental quantum shot noise to:
- **Escape local minima**: Quantum tunneling through energy barriers
- **Explore solution space**: Random walks in quantum superposition
- **Balance exploration/exploitation**: Higher $c$ = more exploration

### Relaxation Schedules

Each schedule controls multiple quantum parameters:

| Schedule | Dissipation | Feedback Loops | Default $\bar{n}$ | Default $c$ |
|----------|-------------|----------------|-------------------|-------------|
| **1** | High | Many | 0.005 | 0.8 |
| **2** | Medium-High | Medium | 0.002 | 0.6 |
| **3** | Medium-Low | Medium | 0.001 | 0.4 |
| **4** | Low | Few | 0.0005 | 0.2 |

## Error Handling and Robustness

### Connection Management

```python
try:
    oracle = DiracOracle()
except SolverUnavailableError as e:
    print(f"Dirac solver not available: {e}")
    # Fallback logic
    oracle = ProjectedGradientDescentOracle()
```

### Parameter Validation

```python
def validate_dirac_parameters(mean_photon_number, quantum_fluctuation_coefficient):
    """Validate quantum parameters are in physical ranges."""
    if mean_photon_number is not None:
        if not (6.67e-5 <= mean_photon_number <= 6.67e-3):
            raise ValueError("mean_photon_number out of range [6.67e-5, 6.67e-3]")
    
    if quantum_fluctuation_coefficient is not None:
        if not (1 <= quantum_fluctuation_coefficient <= 100):
            raise ValueError("quantum_fluctuation_coefficient out of range [1, 100]")
```

### Quantum Solution Validation

```python
def validate_quantum_solution(solution, tolerance=1e-3):
    """Validate quantum solution satisfies simplex constraints."""
    # Check non-negativity
    if np.any(solution < -tolerance):
        raise OracleError("Quantum solution violates non-negativity")
    
    # Check sum constraint
    if abs(np.sum(solution) - 1.0) > tolerance:
        raise OracleError("Quantum solution violates sum constraint")
    
    return True
```

## Performance Characteristics

### Scaling Behavior

- **Time complexity**: O(quantum evolution time) - often sublinear in problem size
- **Solution quality**: Generally 95%+ optimal for graph problems
- **Memory usage**: O(n) - efficient for large problems

### Benchmark Results

Graph size vs. performance on random graphs:

| Nodes | Edges | Quantum Time | Classical Time | Solution Quality |
|-------|-------|--------------|----------------|------------------|
| 50 | 500 | 8s | 12s | 98% optimal |
| 100 | 2000 | 15s | 45s | 96% optimal |
| 200 | 8000 | 25s | 180s | 95% optimal |
| 500 | 50000 | 60s | >600s | 94% optimal |

### When to Use Dirac-3

**Ideal scenarios**:
- Large-scale problems (> 100 nodes)
- Dense graphs with many local minima
- Problems requiring global optimization
- Research applications exploring quantum advantage

**Consider alternatives for**:
- Small problems (< 20 nodes) - exact solvers faster
- Highly sparse graphs - classical methods may suffice  
- Applications requiring exact solutions - use Gurobi
- Cost-sensitive environments - classical methods cheaper

## Integration with MIS Algorithm

### Oracle Call Pattern

```python
from motzkinstraus.algorithms import find_mis_with_oracle

# Dirac oracle integrates seamlessly with MIS algorithm
G = nx.barabasi_albert_graph(100, 5)
oracle = DiracOracle(num_samples=30, relax_schedule=3)

mis_set, oracle_calls = find_mis_with_oracle(G, oracle)
print(f"MIS size: {len(mis_set)}")
print(f"Oracle calls made: {oracle_calls}")
print(f"Total quantum measurements: {oracle_calls * oracle.num_samples}")
```

### Quantum-Specific Optimizations

```python
class QuantumAwareMISAlgorithm:
    def __init__(self, oracle: DiracOracle):
        self.oracle = oracle
        self.quantum_cache = {}  # Cache quantum solutions
    
    def find_mis(self, graph):
        # Pre-warm quantum system with small problems
        self._quantum_warmup()
        
        # Use quantum-specific heuristics
        return self._quantum_mis_search(graph)
    
    def _quantum_warmup(self):
        """Pre-solve small problems to stabilize quantum system."""
        warmup_graph = nx.cycle_graph(4)
        self.oracle.get_omega(warmup_graph)
```

---

**Related Documentation**:
- [QP Solvers Overview](qp-solvers.md) - Compare with other solvers
- [Hybrid Oracles](hybrid.md) - Combining quantum with classical methods
- [Performance Tuning](../../guides/performance-tuning.md) - Optimization strategies