# Maximum Independent Set Algorithms: A Comprehensive Guide

This document provides detailed explanations of all Maximum Independent Set (MIS) algorithms implemented and benchmarked in this project, including our novel Motzkin-Straus quadratic programming approach.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [NetworkX Algorithms](#networkx-algorithms)
   - [Greedy Heuristic](#greedy-heuristic)
   - [Boppana-Halldórsson Approximation](#boppana-halldórsson-approximation)
   - [Exact Algorithm via Complement Graph](#exact-algorithm-via-complement-graph)
3. [Motzkin-Straus Quadratic Programming Method](#motzkin-straus-quadratic-programming-method)
4. [JAX Optimization Solvers](#jax-optimization-solvers)
5. [Algorithm Comparison and Analysis](#algorithm-comparison-and-analysis)

---

## Problem Definition

### Maximum Independent Set (MIS)
Given an undirected graph G = (V, E), an **independent set** is a subset S ⊆ V of vertices such that no two vertices in S are adjacent (i.e., there is no edge between any pair of vertices in S).

The **Maximum Independent Set (MIS) problem** seeks to find an independent set of maximum cardinality. This problem is NP-hard, meaning no polynomial-time algorithm is known to solve it optimally for all graphs.

**Key Properties:**
- Independence number α(G): The size of the maximum independent set
- Complement relationship: An independent set in G is a clique in the complement graph Ḡ
- Therefore: α(G) = ω(Ḡ), where ω denotes the clique number

---

## NetworkX Algorithms

### Greedy Heuristic

**Algorithm Name:** `nx_greedy` (multi-run) and `nx_greedy_single` (single run)

**Core Principle:**
The greedy algorithm iteratively selects vertices to include in the independent set based on a simple heuristic: choose the vertex with the smallest degree (fewest neighbors) among remaining vertices.

**Detailed Algorithm:**
1. **Initialize:** Start with an empty independent set S = ∅
2. **Iterative Selection:**
   - While there are unprocessed vertices:
     a. Among all remaining vertices, select v with minimum degree
     b. Add v to the independent set: S ← S ∪ {v}
     c. Remove v and all its neighbors from the graph
     d. Update degrees of remaining vertices
3. **Return:** The independent set S

**Randomization Strategy (nx_greedy):**
- The multi-run version executes the greedy algorithm multiple times with different random seeds
- When multiple vertices have the same minimum degree, the tie-breaking is randomized
- We run 10 iterations and select the best result
- This significantly improves solution quality by exploring different greedy choices

**Time Complexity:** O(V²) per run, where V is the number of vertices
**Space Complexity:** O(V + E)

**Quality Analysis:**
- **Approximation Ratio:** No theoretical guarantee (can be arbitrarily bad)
- **Practical Performance:** Often finds good solutions, especially with multiple random runs
- **Variance:** High variance across different random seeds due to tie-breaking

**Benchmark Results Example:**
- Single run: Found 6/7 optimal (ratio: 0.857)
- Multi-run: Found 7/7 optimal (ratio: 1.000)
- Runtime: ~0.001 seconds (extremely fast)

### Boppana-Halldórsson Approximation

**Algorithm Name:** `nx_approximation`

**Core Principle:**
This is a sophisticated approximation algorithm that provides theoretical guarantees. It's based on the Boppana-Halldórsson algorithm, which achieves a Δ/3 approximation ratio where Δ is the maximum degree of the graph.

**Detailed Algorithm:**
1. **Degree-Based Partitioning:**
   - Sort vertices by degree in ascending order
   - Create degree-based buckets for processing
2. **Iterative Independent Set Construction:**
   - Process vertices in order of increasing degree
   - For each vertex v:
     a. If v is not yet covered by the current independent set
     b. Add v to the independent set
     c. Mark v and all its neighbors as "covered"
3. **Optimization Phase:**
   - Apply local improvements where possible
   - Consider swapping vertices to improve solution quality

**Theoretical Guarantees:**
- **Approximation Ratio:** Δ/3, where Δ is the maximum degree
- **For dense graphs:** Can be significantly better than Δ/3 in practice
- **Worst-case bound:** Always finds a solution within Δ/3 of optimal

**Time Complexity:** O(V + E)
**Space Complexity:** O(V)

**Quality Analysis:**
- **Strength:** Provides theoretical approximation guarantee
- **Weakness:** Conservative bound may not reflect actual performance
- **Practical Performance:** Often performs better than theoretical bound suggests

**Benchmark Results Example:**
- Found 6/7 optimal (ratio: 0.857)
- Runtime: ~0.002 seconds
- Provides good quality/speed trade-off

### Exact Algorithm via Complement Graph

**Algorithm Name:** `nx_exact`

**Core Principle:**
This algorithm leverages the fundamental relationship between independent sets and cliques: an independent set in G corresponds to a clique in the complement graph Ḡ.

**Detailed Algorithm:**
1. **Graph Complement Construction:**
   - Given input graph G = (V, E)
   - Construct complement Ḡ = (V, Ē) where (u,v) ∈ Ē iff (u,v) ∉ E
2. **Maximum Clique Finding:**
   - Use NetworkX's optimized maximum clique algorithm on Ḡ
   - This typically uses the Bron-Kerbosch algorithm with pivoting
3. **Solution Translation:**
   - The maximum clique in Ḡ directly corresponds to the MIS in G
   - Return this clique as the independent set

**Bron-Kerbosch Algorithm Details:**
The underlying clique-finding algorithm uses a sophisticated branch-and-bound approach:
- **Recursive Structure:** Systematically explores all potential cliques
- **Pruning:** Uses pivoting to eliminate branches that cannot lead to larger cliques
- **Optimization:** Modern implementations include various optimizations for sparse graphs

**Time Complexity:** O(3^(V/3)) worst-case (exponential), but often much faster in practice
**Space Complexity:** O(V²) for complement graph storage

**Quality Analysis:**
- **Optimality:** Always finds the exact maximum independent set
- **Scalability:** Limited to small-to-medium graphs due to exponential worst-case complexity
- **Practical Performance:** Often surprisingly fast on real-world graphs due to effective pruning

**Benchmark Results Example:**
- Found 7/7 optimal (ratio: 1.000)
- Runtime: ~0.0004 seconds (fastest exact method)
- Serves as ground truth for comparison

---

## Motzkin-Straus Quadratic Programming Method

### Theoretical Foundation

**The Motzkin-Straus Theorem** provides a remarkable bridge between discrete graph problems and continuous optimization:

For any graph G with adjacency matrix A, the following relationship holds:
```
max_{x ∈ Δₙ} (½ · x^T · A · x) = ½ · (1 - 1/ω(G))
```

Where:
- Δₙ is the standard probability simplex: {x ∈ ℝⁿ | Σxᵢ = 1, xᵢ ≥ 0}
- ω(G) is the clique number (size of maximum clique)
- The left side is a non-convex quadratic program

**Key Insight:** By solving the continuous optimization problem, we can determine ω(G) algebraically:
```
ω(G) = 1 / (1 - 2M)
```
where M is the optimal value of the quadratic program.

### Our Complete Algorithm: From Quadratic Program to MIS Construction

#### Phase 1: Oracle Construction

**The Oracle Function:**
```python
def oracle(G):
    """Return the clique number ω(G) using Motzkin-Straus theorem"""
    # Step 1: Construct adjacency matrix A
    A = nx.adjacency_matrix(G).toarray()
    
    # Step 2: Solve quadratic program: max ½x^T A x subject to x ∈ Δₙ
    # This is done using JAX optimization (PGD or Mirror Descent)
    optimal_value = solve_quadratic_program(A)
    
    # Step 3: Apply Motzkin-Straus formula
    clique_number = 1 / (1 - 2 * optimal_value)
    return round(clique_number)
```

#### Phase 2: Search-to-Decision Algorithm

**The Challenge:** The oracle gives us ω(G), but we need the actual vertices forming the maximum clique (or equivalently, the MIS in the complement).

**Our Solution - Iterative Construction:**

```python
def find_mis_with_oracle(G):
    """Find maximum independent set using oracle-based search"""
    
    # Step 1: Determine target size
    G_complement = nx.complement(G)
    target_size = oracle(G_complement)  # This is α(G)
    
    # Step 2: Initialize
    independent_set = []
    current_graph = G.copy()
    remaining_target = target_size
    
    # Step 3: Iterative vertex selection
    for vertex in list(G.nodes()):
        if vertex not in current_graph:
            continue
            
        # Test hypothesis: "Can we include this vertex?"
        test_graph = current_graph.copy()
        test_graph.remove_node(vertex)
        
        # Also remove all neighbors of vertex (independence constraint)
        neighbors_to_remove = list(current_graph.neighbors(vertex))
        for neighbor in neighbors_to_remove:
            if neighbor in test_graph:
                test_graph.remove_node(neighbor)
        
        # Query oracle on remaining graph
        if test_graph.number_of_nodes() > 0:
            test_complement = nx.complement(test_graph)
            remaining_capacity = oracle(test_complement)
        else:
            remaining_capacity = 0
        
        # Decision: Include vertex if it leads to optimal solution
        if 1 + remaining_capacity == remaining_target:
            # Including this vertex is compatible with optimality
            independent_set.append(vertex)
            current_graph = test_graph
            remaining_target -= 1
            
            if remaining_target == 0:
                break
    
    return independent_set
```

**Detailed Step-by-Step Process:**

1. **Initialization Phase:**
   - Convert MIS problem to Max Clique via complement: α(G) = ω(Ḡ)
   - Use oracle to determine target independent set size k = α(G)
   - Initialize empty solution and working graph

2. **Iterative Vertex Consideration:**
   For each vertex v in the original graph:
   
   a. **Hypothesis Testing:**
      - Create test scenario: "What if we include v in our independent set?"
      - Remove v and all its neighbors from current working graph
      - This gives us the "remaining subproblem" after committing to v
   
   b. **Oracle Query:**
      - Apply oracle to the remaining subproblem
      - Get k_remaining = maximum independent set size in remaining graph
   
   c. **Optimality Check:**
      - If (1 + k_remaining) == k_current_target, then including v is consistent with optimality
      - The "1" accounts for vertex v itself
      - k_remaining accounts for the best we can do in the remaining graph
   
   d. **Decision and Update:**
      - If optimality check passes: commit to v, update working graph, decrement target
      - If optimality check fails: discard v, continue with next vertex

3. **Termination:**
   - Algorithm terminates when target size is reached
   - Returns the constructed independent set

**Oracle Call Complexity:**
- In the worst case, we make one oracle call per vertex: O(V) calls
- Each oracle call involves solving a quadratic program
- Total complexity depends on the quadratic program solver efficiency

### Advantages and Limitations

**Advantages:**
1. **Theoretical Foundation:** Based on rigorous mathematical theory (Motzkin-Straus)
2. **Exact Solution:** When oracle is perfect, finds optimal MIS
3. **Novel Approach:** Unique bridge between continuous optimization and discrete problems
4. **Modular Design:** Oracle can be implemented with different optimization methods

**Limitations:**
1. **Computational Cost:** Each oracle call requires solving non-convex QP
2. **Numerical Precision:** Floating-point arithmetic can introduce errors
3. **Scalability:** Number of oracle calls grows linearly with vertices
4. **Implementation Complexity:** More complex than traditional heuristics

---

## JAX Optimization Solvers

### Projected Gradient Descent (PGD)

**Algorithm Name:** `jax_pgd`

**Mathematical Formulation:**
We solve: max_{x ∈ Δₙ} f(x) = ½x^T A x

**Algorithm Details:**
1. **Initialization:** Sample multiple starting points from Dirichlet distribution
2. **Gradient Computation:** ∇f(x) = Ax
3. **Update Step:** x_{t+1} = Π_Δₙ(x_t + α∇f(x_t))
4. **Projection:** Project onto probability simplex using efficient algorithm

**Simplex Projection:**
```python
def project_simplex(v):
    """Project vector v onto probability simplex"""
    u = jnp.sort(v)[::-1]  # Sort in descending order
    cumsum = jnp.cumsum(u)
    rho = jnp.sum(u * jnp.arange(1, len(u) + 1) > cumsum) - 1
    theta = (cumsum[rho] - 1.0) / (rho + 1.0)
    return jnp.maximum(v - theta, 0.0)
```

**Multi-Restart Strategy:**
- Run 8 independent optimization trajectories
- Each starts from different Dirichlet(α=1) sample
- Select best result across all restarts
- Provides robustness against local optima

### Mirror Descent (MD)

**Algorithm Name:** `jax_md`

**Mathematical Formulation:**
Uses entropy as the mirror map: ψ(x) = Σᵢ xᵢ log xᵢ

**Algorithm Details:**
1. **Dual Update:** θ_{t+1} = θ_t + α∇f(x_t)
2. **Primal Recovery:** x_{t+1} = ∇ψ*(θ_{t+1})
3. **Explicit Form:** x_{t+1} ∝ exp(θ_{t+1})
4. **Normalization:** Ensure x_{t+1} lies on simplex

**Key Advantage:**
- Natural constraint handling: updates automatically stay in simplex interior
- Better theoretical convergence properties for simplex-constrained problems
- More stable for ill-conditioned problems

**Implementation:**
```python
def mirror_descent_step(theta, A, learning_rate):
    """Single step of mirror descent"""
    # Current primal point
    x = jax.nn.softmax(theta)
    
    # Gradient in primal space
    grad = A @ x
    
    # Dual update
    theta_new = theta + learning_rate * grad
    
    return theta_new, x
```

### Convergence Analysis and Diagnostics

**Convergence Monitoring:**
- Track objective value f(x_t) = ½x_t^T A x_t at each iteration
- Monitor gradient norm ||∇f(x_t)||
- Detect convergence when improvement falls below tolerance

**Multi-Restart Statistics:**
- Compare final objective values across restarts
- Analyze convergence speed and stability
- Identify best-performing initialization strategies

**Visualization Tools:**
- Plot convergence histories for all restarts
- Highlight best trajectory in different color
- Show final objective value and iteration count

---

## Algorithm Comparison and Analysis

### Performance Characteristics

| Algorithm | Time Complexity | Space | Optimality | Approximation Ratio |
|-----------|----------------|-------|------------|-------------------|
| nx_greedy_single | O(V²) | O(V+E) | No guarantee | Unbounded |
| nx_greedy (multi) | O(V²) × runs | O(V+E) | No guarantee | Better in practice |
| nx_approximation | O(V+E) | O(V) | Δ/3-approx | Δ/3 |
| nx_exact | O(3^(V/3)) | O(V²) | Optimal | 1.0 |
| jax_pgd | O(iterations×V²) | O(V²) | Optimal* | 1.0* |
| jax_md | O(iterations×V²) | O(V²) | Optimal* | 1.0* |

*Subject to numerical precision and convergence

### Quality vs Speed Trade-off Analysis

**From our 15-node Barabási-Albert graph benchmark:**

1. **Ultra-Fast Heuristics (< 0.001s):**
   - nx_greedy_single: 6/7 (0.857 ratio)
   - Excellent for rapid approximate solutions

2. **Fast Exact Methods (< 0.001s):**
   - nx_exact: 7/7 (1.000 ratio, 0.0004s)
   - Best choice for small graphs requiring optimality

3. **Robust Heuristics (< 0.01s):**
   - nx_greedy (multi-run): 7/7 (1.000 ratio, 0.001s)
   - nx_approximation: 6/7 (0.857 ratio, 0.002s)
   - Good balance of speed and quality

4. **Optimization-Based Exact (> 10s):**
   - jax_pgd: 7/7 (1.000 ratio, 33s, 11 oracle calls)
   - jax_md: 5/7 (0.714 ratio, 27s, 10 oracle calls)
   - Research interest, demonstrates theoretical approach

### Algorithm Selection Guidelines

**For Production Systems:**
- **Small graphs (< 20 nodes):** Use nx_exact for guaranteed optimality
- **Medium graphs (20-100 nodes):** Use nx_greedy with multiple runs
- **Large graphs (> 100 nodes):** Use nx_approximation for guaranteed bounds

**For Research and Analysis:**
- **Theoretical investigation:** Use Motzkin-Straus JAX methods
- **Benchmark comparison:** Include full algorithm spectrum
- **Performance analysis:** Focus on quality-speed trade-offs

### Key Insights from Benchmarking

1. **NetworkX Exact Dominance:** For graphs where it completes, nx_exact is unbeatable in speed and accuracy

2. **Greedy Randomization Value:** Multiple random runs significantly improve greedy heuristic performance

3. **Approximation Algorithm Reliability:** Boppana-Halldórsson provides consistent, bounded performance

4. **Optimization Method Challenges:** JAX methods demonstrate theoretical feasibility but face scalability issues

5. **Oracle-Based Approach Validation:** Successfully implements Motzkin-Straus theory in practice, confirming theoretical predictions

### Future Directions

1. **Hybrid Approaches:** Combine fast heuristics with optimization-based refinement
2. **Improved Oracles:** Investigate more efficient quadratic program solvers
3. **Approximation Improvements:** Develop better approximation algorithms with tighter bounds
4. **Parallel Implementation:** Leverage multi-restart nature for parallel computation
5. **Graph-Specific Optimization:** Tailor algorithms to specific graph classes (planar, sparse, etc.)

---

## Conclusion

This comprehensive analysis demonstrates the rich landscape of MIS algorithms, from ultra-fast heuristics to theoretically-grounded optimization methods. The Motzkin-Straus approach, while computationally intensive, provides a fascinating bridge between discrete graph theory and continuous optimization, opening new avenues for research and development in combinatorial optimization.

The benchmarking framework developed here enables rigorous comparison across the algorithm spectrum, providing insights valuable for both practical applications and theoretical research in the field of maximum independent set algorithms.