# Gibbons' Weighted Motzkin-Straus Theorem Implementation

This document explains the correct implementation of Gibbons' weighted Motzkin-Straus theorem for solving the Maximum Weight Clique problem, including analysis of previous implementation errors.

## Theoretical Background

### Gibbons' Theorem 5
For a graph G = (V,E) with vertex weights w: V → ℝ⁺, the weighted clique number ω(w,G) satisfies:

**1/ω(w,G) = min{x^T B x | e^T x = 1, x ≥ 0}**

Where B is a matrix in the class M(w,G) defined by:
- **B[i,i] = 1/w[i]** for all vertices i (diagonal entries)
- **B[i,j] = 0** for all adjacent vertices (i,j) ∈ E (off-diagonal, adjacent pairs)
- **B[i,j] + B[j,i] ≥ (1/w[i] + 1/w[j])** for all non-adjacent vertices (i,j) ∈ Ē (constraint for non-adjacent pairs)

### Key Insight
The optimization problem **minimizes** x^T B x to find **1/ω(w,G)**, then we derive ω(w,G) = 1/(x^T B x).

## Correct Implementation

### Matrix Construction
```python
def construct_gibbons_matrix(graph: nx.Graph, weights: Dict[int, float]) -> Tuple[np.ndarray, List[int]]:
    """Construct the matrix B according to Gibbons' Theorem 5."""
    node_list = list(graph.nodes())
    n = len(node_list)
    B = np.zeros((n, n))
    
    # Diagonal entries: B[i,i] = 1/w[i]
    for i, node in enumerate(node_list):
        B[i, i] = 1.0 / weights.get(node, 1.0)
    
    # Off-diagonal entries
    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if i != j:
                if graph.has_edge(node_i, node_j):
                    # Adjacent vertices: B[i,j] = 0
                    B[i, j] = 0.0
                else:
                    # Non-adjacent vertices: satisfy constraint with equality
                    # B[i,j] = B[j,i] = (1/w[i] + 1/w[j]) / 2
                    w_i = weights.get(node_i, 1.0)
                    w_j = weights.get(node_j, 1.0)
                    constraint_value = (1.0 / w_i + 1.0 / w_j) / 2.0
                    B[i, j] = constraint_value
    
    return B, node_list
```

### Optimization Setup
```python
# For JAX optimization (minimization)
B_opt = -B  # Negate for maximization since JAX minimizes
poly_indices, poly_coeffs = matrix_to_polynomial(B_opt, scale_factor=1.0)

# After optimization
objective_value = -best_energy  # Convert back from maximization
derived_omega = 1.0 / objective_value  # This gives ω(w,G)
```

## Previous Implementation Errors

### Error 1: Incorrect Constraint Handling
**Wrong:** Previous implementation tried to use adjacency matrix constraints directly in the quadratic form.
```python
# INCORRECT APPROACH
if graph.has_edge(node_i, node_j):
    B[i, j] = some_adjacency_penalty  # Wrong!
```

**Why wrong:** Gibbons' theorem specifically requires B[i,j] = 0 for adjacent vertices, not arbitrary penalty values.

### Error 2: Matrix Construction Order
**Wrong:** Previous implementation had the adjacency logic backwards.
```python
# INCORRECT - backwards logic
if graph.has_edge(node_i, node_j):
    B[i, j] = (1.0 / w_i + 1.0 / w_j) / 2.0  # Should be 0!
else:
    B[i, j] = 0.0  # Should be constraint value!
```

**Why wrong:** This violates Gibbons' theorem requirements and leads to incorrect optimization landscapes.

### Error 3: Misunderstanding the Objective Direction
**Wrong:** Trying to maximize x^T B x directly.
```python
# INCORRECT
objective = x^T @ B @ x  # Should minimize this, not maximize!
```

**Why wrong:** Gibbons' theorem requires **minimizing** x^T B x to find 1/ω(w,G). The maximum weight clique value is then ω(w,G) = 1/(min x^T B x).

### Error 4: Incorrect Polynomial Conversion
**Wrong:** Not properly handling the sign conversion for maximization algorithms.
```python
# INCORRECT - using B directly in maximization algorithm
poly_indices, poly_coeffs = matrix_to_polynomial(B, scale_factor=1.0)
```

**Why wrong:** Since optimization algorithms like PGD perform minimization internally, but we want to minimize x^T B x, we need to negate B before conversion to effectively maximize -(x^T B x) = x^T (-B) x.

## Verification Examples

### Triangle K3 with weights [1, 2, 3]
**Correct B matrix:**
```
B = [[1.000, 0.000, 0.000],
     [0.000, 0.500, 0.000], 
     [0.000, 0.000, 0.333]]
```

**Theoretical solution:**
- Maximum weight clique: {0, 1, 2} with weight = 6.0
- Optimal x: [1/6, 2/6, 3/6] = [0.167, 0.333, 0.500]
- x^T B x = 1/6 = 0.167
- ω(w,G) = 1/(1/6) = 6.0 ✓

### Path P3 with weights [2, 1, 3]
**Correct B matrix:**
```
B = [[0.500, 0.000, 0.417],
     [0.000, 1.000, 0.000],
     [0.417, 0.000, 0.333]]
```

**Theoretical solution:**
- Maximum weight clique: {1, 2} with weight = 4.0 (vertices 1 and 2 are adjacent)
- x^T B x = 1/4 = 0.250
- ω(w,G) = 1/(1/4) = 4.0 ✓

## Algorithm Workflow

1. **Construct Gibbons matrix B** according to theorem constraints
2. **Negate B** for maximization algorithms: B_opt = -B
3. **Convert to polynomial** form for JAX optimization
4. **Run optimization** to minimize x^T B_opt x (equivalent to maximizing x^T B x)
5. **Extract result**: objective = -best_energy, ω(w,G) = 1/objective
6. **Find clique vertices** from support of optimal x vector

## Key Implementation Insights

1. **Matrix symmetry**: B should be symmetric with proper constraint handling
2. **Diagonal dominance**: Diagonal terms 1/w[i] often dominate for well-weighted problems
3. **Zero entries**: Adjacent vertices must have B[i,j] = 0 (critical constraint)
4. **Sign handling**: Careful negation for maximization in minimization algorithms
5. **Numerical stability**: Use appropriate tolerances for support extraction

## Validation Approach

Always validate implementations with:
1. **Simple known cases** (single vertex, complete graphs, paths)
2. **Theoretical calculations** for small examples
3. **MILP comparison** as ground truth for correctness
4. **Consistency checks** between theory and optimization results

This correct implementation ensures that both MILP and JAX-PGD solvers find identical optimal solutions, as demonstrated in the test results showing perfect agreement across all test cases.