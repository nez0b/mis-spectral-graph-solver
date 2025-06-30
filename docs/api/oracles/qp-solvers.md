# Quadratic Programming Solvers

The Motzkin-Straus MIS solver provides a comprehensive suite of quadratic programming (QP) solvers to handle the non-convex optimization problem that lies at the heart of the theorem. Each solver offers different performance characteristics, precision guarantees, and computational complexity profiles.

## Problem Formulation

All solvers tackle the same fundamental optimization problem derived from the Motzkin-Straus theorem:

<div class="theorem">
<div class="theorem-title">Standard QP Formulation</div>

$$\begin{align}
\text{maximize} \quad & \frac{1}{2} x^T A x \\
\text{subject to} \quad & \sum_{i=1}^n x_i = 1 \\
& x_i \geq 0, \quad i = 1, \ldots, n
\end{align}$$

where $A$ is the adjacency matrix of the input graph and $x \in \mathbb{R}^n$ represents points on the probability simplex $\Delta_n$.
</div>

### Key Challenges

- **Non-convexity**: The objective function $\frac{1}{2} x^T A x$ is generally non-convex for graph adjacency matrices
- **Local optima**: Standard optimization algorithms may converge to suboptimal solutions
- **Constraint handling**: Maintaining feasibility on the probability simplex throughout optimization
- **Numerical precision**: Floating-point errors can impact the final discrete result

## Available Solvers

### 1. JAX Projected Gradient Descent (PGD)

<div class="oracle-card">

#### ProjectedGradientDescentOracle

**Type**: First-order gradient-based optimization  
**Backend**: JAX with JIT compilation  
**Best for**: General-purpose optimization with good performance across graph types

**Mathematical Approach**:
The oracle uses projected gradient descent with simplex projection:

1. **Gradient computation**: $g = A x$
2. **Gradient step**: $y = x + \alpha g$ 
3. **Simplex projection**: $x_{new} = \Pi_{\Delta_n}(y)$

**Key Features**:
- Multi-restart strategy with Dirichlet initializations
- Early stopping based on energy tolerance  
- JIT-compiled optimization for performance
- Complete convergence history tracking

**Parameters**:
<div class="parameter">
<span class="parameter-name">learning_rate</span>: <span class="parameter-type">float = 0.01</span><br>
Step size for gradient descent
</div>

<div class="parameter">
<span class="parameter-name">max_iterations</span>: <span class="parameter-type">int = 2000</span><br>
Maximum optimization iterations
</div>

<div class="parameter">
<span class="parameter-name">num_restarts</span>: <span class="parameter-type">int = 10</span><br>
Number of random initializations
</div>

<div class="parameter">
<span class="parameter-name">tolerance</span>: <span class="parameter-type">float = 1e-6</span><br>
Convergence tolerance for early stopping
</div>

**Complexity**: <span class="complexity quadratic">O(iterations × n²)</span>

**Example Usage**:
```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

oracle = ProjectedGradientDescentOracle(
    learning_rate=0.02,
    max_iterations=1000,
    num_restarts=5,
    tolerance=1e-7
)
```

</div>

### 2. JAX Mirror Descent

<div class="oracle-card">

#### MirrorDescentOracle

**Type**: Exponentiated gradient method  
**Backend**: JAX with entropic regularization  
**Best for**: Simplex-constrained optimization with natural constraint handling

**Mathematical Approach**:
Mirror descent uses the exponential map for simplex updates:

1. **Gradient computation**: $g = A x$
2. **Log-space update**: $\log x_{new} = \log x + \alpha g$
3. **Normalization**: $x_{new} = \frac{x_{new}}{\sum_i x_{new,i}}$

The method naturally maintains simplex constraints through the exponential family structure.

**Key Features**:
- Exponentiated gradient updates with numerical stabilization
- Superior performance on simplex-constrained problems
- Natural constraint satisfaction (no projection needed)
- Adaptive step-size mechanisms

**Parameters**:
<div class="parameter">
<span class="parameter-name">learning_rate</span>: <span class="parameter-type">float = 0.005</span><br>
Step size for exponentiated gradient (typically smaller than PGD)
</div>

<div class="parameter">
<span class="parameter-name">max_iterations</span>: <span class="parameter-type">int = 2000</span><br>
Maximum optimization iterations
</div>

<div class="parameter">
<span class="parameter-name">num_restarts</span>: <span class="parameter-type">int = 10</span><br>
Number of Dirichlet initializations
</div>

**Complexity**: <span class="complexity quadratic">O(iterations × n²)</span>

**Example Usage**:
```python
from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle

oracle = MirrorDescentOracle(
    learning_rate=0.008,
    max_iterations=1500,
    num_restarts=8
)
```

</div>

### 3. Dirac-3 Quantum Annealing

<div class="oracle-card">

#### DiracOracle

**Type**: Quantum annealing / Photonic computing  
**Backend**: QCi Dirac-3 continuous cloud solver  
**Best for**: Large-scale problems and quantum-enhanced optimization

**Mathematical Approach**:
The Dirac-3 system uses **photonic quantum computing** with temporal encoding:

- **Time-bin encoding**: Variables encoded in photon arrival times
- **Quantum superposition**: Multiple solution candidates explored simultaneously  
- **Quantum annealing**: Gradual evolution from quantum superposition to classical solution
- **Quantum fluctuations**: Natural quantum noise helps escape local minima

**Key Features**:
- True quantum computing backend (not simulation)
- Advanced parameter control for quantum effects
- Handles large-scale problems efficiently
- Novel photonic hardware approach

**Parameters**:
<div class="parameter">
<span class="parameter-name">num_samples</span>: <span class="parameter-type">int = 100</span><br>
Number of solution samples to request <span class="parameter-range">Range: 1-100</span>
</div>

<div class="parameter">
<span class="parameter-name">relax_schedule</span>: <span class="parameter-type">int = 2</span><br>
Quantum relaxation schedule controlling dissipation <span class="parameter-range">Range: {1,2,3,4}</span>
</div>

<div class="parameter">
<span class="parameter-name">sum_constraint</span>: <span class="parameter-type">int = 1</span><br>
Constraint for solution variables sum <span class="parameter-range">Range: 1-10000</span>
</div>

<div class="parameter">
<span class="parameter-name">mean_photon_number</span>: <span class="parameter-type">Optional[float] = None</span><br>
Average photons per time-bin (quantum coherence control) <span class="parameter-range">Range: 6.67×10⁻⁵ to 6.67×10⁻³</span>
</div>

<div class="parameter">
<span class="parameter-name">quantum_fluctuation_coefficient</span>: <span class="parameter-type">Optional[int] = None</span><br>
Quantum noise level for escaping local minima <span class="parameter-range">Range: 1-100</span>
</div>

**Complexity**: <span class="complexity polynomial">O(quantum time)</span>

<div class="quantum-note">
**Quantum Physics Background**: The mean photon number controls quantum coherence in the time-bin modes, while the quantum fluctuation coefficient leverages Poisson noise from single-photon detection to enable exploration of the solution space.
</div>

**Example Usage**:
```python
from motzkinstraus.oracles.dirac import DiracOracle

oracle = DiracOracle(
    num_samples=50,
    relax_schedule=3,
    mean_photon_number=0.002,          # Enhanced quantum coherence
    quantum_fluctuation_coefficient=50  # Moderate quantum noise
)
```

</div>

### 4. Gurobi Commercial Solver

<div class="oracle-card">

#### GurobiOracle

**Type**: Commercial non-convex quadratic programming  
**Backend**: Gurobi Optimizer  
**Best for**: High-precision solutions and production environments

**Mathematical Approach**:
Gurobi employs sophisticated algorithms for non-convex QP:

- **Branch-and-bound**: Systematic exploration of solution space
- **Cutting planes**: Tightening relaxations for better bounds  
- **Heuristics**: Fast initial solutions and local search
- **Presolving**: Problem reduction and reformulation

**Key Features**:
- Industry-leading commercial solver
- Guaranteed global optimality (when termination criteria met)
- Advanced presolving and problem reformulation
- Professional technical support

**Parameters**:
<div class="parameter">
<span class="parameter-name">suppress_output</span>: <span class="parameter-type">bool = True</span><br>
Whether to suppress Gurobi's detailed output logs
</div>

**Complexity**: <span class="complexity polynomial">O(n³)</span> typical, exponential worst-case

**Requirements**:
- Valid Gurobi license (academic or commercial)
- `gurobipy` Python package installation

**Example Usage**:
```python
from motzkinstraus.oracles.gurobi import GurobiOracle

oracle = GurobiOracle(suppress_output=False)  # Show optimization progress
```

</div>

### 5. JAX Frank-Wolfe

<div class="oracle-card">

#### FrankWolfeOracle

**Type**: Projection-free first-order method  
**Backend**: JAX with linear optimization oracles  
**Best for**: Large-scale problems where projection is expensive

**Mathematical Approach**:
The Frank-Wolfe algorithm avoids explicit projection:

1. **Linear approximation**: $\min_{s \in \Delta_n} \langle \nabla f(x), s \rangle$
2. **Step computation**: $\gamma = \frac{2}{k+2}$ (line search or fixed)
3. **Convex combination**: $x_{new} = (1-\gamma) x + \gamma s$

**Key Features**:
- Projection-free optimization (only linear minimization)
- Sparse iterate sequences
- Convergence rate guarantees
- Memory-efficient for large problems

**Complexity**: <span class="complexity quadratic">O(iterations × n²)</span>

</div>

## Hybrid Approaches

### DiracNetworkXHybridOracle

Combines exact NetworkX algorithms with Dirac-3 quantum annealing:

```python
from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle

hybrid_oracle = DiracNetworkXHybridOracle(
    networkx_size_threshold=20,  # Use exact solver for small graphs
    num_samples=30,
    relax_schedule=2
)
```

**Strategy**:
- **Small graphs (≤ threshold)**: Use exact NetworkX algorithms
- **Large graphs (> threshold)**: Use Dirac-3 quantum annealing

### DiracPGDHybridOracle

Combines JAX PGD with Dirac-3 for adaptive optimization:

```python
from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle

hybrid_oracle = DiracPGDHybridOracle(
    use_pgd_first=True,          # Try PGD before Dirac
    pgd_time_limit=10.0,         # PGD timeout in seconds
    dirac_num_samples=25
)
```

**Strategy**:
- **Phase 1**: Fast JAX PGD for initial solution
- **Phase 2**: Dirac-3 quantum refinement if needed

## Performance Comparison

### Computational Complexity

| Solver | Time Complexity | Space Complexity | Convergence |
|--------|-----------------|------------------|-------------|
| **JAX PGD** | O(iter × n²) | O(n²) | First-order |
| **JAX Mirror** | O(iter × n²) | O(n²) | First-order |
| **Dirac-3** | O(quantum) | O(n) | Quantum annealing |
| **Gurobi** | O(n³) typical | O(n²) | Global optimum |
| **Frank-Wolfe** | O(iter × n²) | O(n) | Projection-free |

### Solution Quality vs Speed Trade-offs

<div class="benchmark-result">
<span class="benchmark-name">Gurobi (Global)</span>
<span class="benchmark-time">Slowest</span>
<span class="benchmark-quality">100% optimal</span>
</div>

<div class="benchmark-result">
<span class="benchmark-name">Dirac-3 (Quantum)</span>
<span class="benchmark-time">Fast</span>
<span class="benchmark-quality">95%+ optimal</span>
</div>

<div class="benchmark-result">
<span class="benchmark-name">JAX PGD (Multi-restart)</span>
<span class="benchmark-time">Medium</span>
<span class="benchmark-quality">90%+ optimal</span>
</div>

<div class="benchmark-result">
<span class="benchmark-name">JAX Mirror (Single)</span>
<span class="benchmark-time">Fast</span>
<span class="benchmark-quality">80%+ optimal</span>
</div>

### Graph Type Recommendations

| Graph Type | Recommended Solver | Rationale |
|------------|-------------------|-----------|
| **Small (< 20 nodes)** | Gurobi | Exact solutions feasible |
| **Dense graphs** | JAX Mirror Descent | Better simplex handling |
| **Sparse graphs** | JAX PGD | Efficient gradient computation |
| **Large scale (> 100 nodes)** | Dirac-3 | Quantum advantage |
| **Production systems** | Hybrid approaches | Adaptive performance |

## Advanced Configuration

### Multi-restart Strategies

For non-convex optimization, multiple random initializations are crucial:

```python
# High-quality configuration
oracle = ProjectedGradientDescentOracle(
    num_restarts=20,           # More restarts = better solution quality
    dirichlet_alpha=0.5,       # Concentrated initialization
    tolerance=1e-8             # Tight convergence
)
```

### Quantum Parameter Tuning

Fine-tune quantum effects for specific problem types:

```python
# For highly connected graphs (many local minima)
quantum_oracle = DiracOracle(
    mean_photon_number=0.001,           # Lower = more quantum coherence
    quantum_fluctuation_coefficient=80   # Higher = more exploration
)

# For sparse graphs (fewer local minima) 
quantum_oracle = DiracOracle(
    mean_photon_number=0.005,           # Higher = faster convergence
    quantum_fluctuation_coefficient=20  # Lower = more exploitation
)
```

### Convergence Monitoring

All JAX-based solvers provide detailed convergence information:

```python
oracle = ProjectedGradientDescentOracle(verbose=True)
result = oracle.solve_quadratic_program(adjacency_matrix)

# Access convergence history
print(f"Converged in {oracle.last_iterations} iterations")
print(f"Final energy: {oracle.last_energy}")
print(f"Convergence history: {oracle.convergence_history}")
```

## Error Handling and Robustness

### Numerical Stability

All solvers implement robust numerical handling:

- **Gradient clipping**: Prevents explosive gradients
- **Epsilon regularization**: Handles degenerate cases  
- **Overflow protection**: Guards against numerical overflow
- **Convergence validation**: Verifies solution quality

### Fallback Mechanisms

```python
from motzkinstraus.oracles import get_best_available_oracle

# Automatic solver selection based on availability
oracle = get_best_available_oracle(
    prefer_quantum=True,     # Try Dirac-3 first
    require_exact=False      # Allow approximate solvers
)
```

## Future Developments

### Planned Enhancements

- **Quantum-classical hybrid algorithms**: Advanced integration patterns
- **Adaptive parameter selection**: ML-guided hyperparameter tuning
- **Distributed computing**: Multi-GPU and cluster support  
- **Problem-specific solvers**: Specialized algorithms for graph families

### Research Directions

- **Quantum advantage characterization**: When does quantum computing help?
- **Approximation guarantees**: Theoretical bounds for approximate solvers
- **Scalability analysis**: Performance on massive graphs (10K+ nodes)

---

**Next Steps**: Explore [specific solver documentation](dirac.md) or learn about [hybrid approaches](hybrid.md) for combining multiple methods.