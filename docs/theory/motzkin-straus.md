# The Motzkin-Straus Theorem

The Motzkin-Straus theorem provides an elegant bridge between discrete graph theory and continuous optimization, enabling us to solve the NP-hard Maximum Independent Set problem through quadratic programming.

## Problem Definition

### Maximum Independent Set (MIS)

!!! abstract "Definition: Independent Set"
    
    Given an undirected graph $G = (V, E)$, an **independent set** is a subset $S \subseteq V$ of vertices such that no two vertices in $S$ are adjacent:
    
    $$\forall u, v \in S: (u, v) \notin E$$

The **Maximum Independent Set (MIS) problem** seeks to find an independent set of maximum cardinality:

$$\alpha(G) = \max\{|S| : S \text{ is an independent set in } G\}$$

where $\alpha(G)$ is called the **independence number** of $G$.

### Maximum Clique Problem

!!! abstract "Definition: Clique"
    
    A **clique** in graph $G = (V, E)$ is a subset $C \subseteq V$ where every pair of distinct vertices is adjacent:
    
    $$\forall u, v \in C, u \neq v: (u, v) \in E$$

The **clique number** $\omega(G)$ is the size of the maximum clique:

$$\omega(G) = \max\{|C| : C \text{ is a clique in } G\}$$

### Fundamental Relationship

The key insight connecting these problems lies in the **complement graph**:

!!! info "Graph Complement"
    
    For graph $G = (V, E)$, the complement $\overline{G} = (V, \overline{E})$ has the same vertices but complementary edges:
    
    $$\overline{E} = \{(u, v) : u, v \in V, u \neq v, (u, v) \notin E\}$$

This leads to the fundamental equivalence:

<div class="theorem">
<div class="theorem-title">MIS-Clique Equivalence</div>

For any graph $G$:

$$\alpha(G) = \omega(\overline{G})$$

An independent set in $G$ corresponds exactly to a clique in $\overline{G}$.
</div>

## The Motzkin-Straus Theorem

### Statement

<div class="theorem">
<div class="theorem-title">Motzkin-Straus Theorem (1965)</div>

Let $G = (V, E)$ be a graph with $n = |V|$ vertices and adjacency matrix $A$. Define the **standard simplex**:

$$\Delta_n = \left\{x \in \mathbb{R}^n : \sum_{i=1}^n x_i = 1, x_i \geq 0 \text{ for all } i\right\}$$

Then the following equality holds:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

where $\omega(G)$ is the clique number of $G$.
</div>

### Intuition

The theorem establishes that:

1. **Left side**: A continuous quadratic optimization problem over the probability simplex
2. **Right side**: A simple function of the discrete clique number

This remarkable connection allows us to solve discrete graph problems using continuous optimization techniques.

### Algebraic Rearrangement

From the Motzkin-Straus equality, we can solve for the clique number:

$$\frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

Let $M = \max_{x \in \Delta_n} \frac{1}{2} x^T A x$. Then:

$$M = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

Solving for $\omega(G)$:

$$2M = 1 - \frac{1}{\omega(G)}$$

$$\frac{1}{\omega(G)} = 1 - 2M$$

$$\omega(G) = \frac{1}{1 - 2M}$$

This formula is the **core of our oracle**: solve the quadratic program to get $M$, then compute $\omega(G)$ algebraically.

## Proof Sketch

The proof of the Motzkin-Straus theorem relies on several key insights:

### Step 1: Upper Bound via Clique

Consider a maximum clique $C$ of size $\omega(G)$. Define the characteristic vector:

$$x^* = \frac{1}{\omega(G)} \sum_{i \in C} e_i$$

where $e_i$ is the $i$-th standard basis vector. Then $x^* \in \Delta_n$ and:

$$\frac{1}{2} (x^*)^T A x^* = \frac{1}{2} \cdot \frac{1}{\omega(G)^2} \sum_{i,j \in C} A_{ij} = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

This shows the right-hand side is achievable.

### Step 2: Upper Bound via Optimization

For any $x \in \Delta_n$, we can show that:

$$\frac{1}{2} x^T A x \leq \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

This requires careful analysis of the quadratic form's structure relative to the clique structure.

### Step 3: Optimality Conditions

The maximum is achieved when the support of $x^*$ corresponds exactly to a maximum clique, with uniform weights $\frac{1}{\omega(G)}$ on clique vertices.

## Computational Implications

### Oracle Construction

The Motzkin-Straus theorem enables us to construct an **oracle** for the clique number:

<div class="algorithm">
<div class="algorithm-title">Oracle(G) → ω(G)</div>

**Input**: Graph $G$ with adjacency matrix $A$  
**Output**: Clique number $\omega(G)$

1. **Formulate QP**: $\max_{x \in \Delta_n} \frac{1}{2} x^T A x$
2. **Solve**: Use numerical optimization to find optimal value $M$
3. **Convert**: Return $\omega(G) = \frac{1}{1 - 2M}$
</div>

### MIS Algorithm

Combined with the MIS-clique equivalence, we get our complete algorithm:

<div class="algorithm">
<div class="algorithm-title">MIS Algorithm</div>

**Input**: Graph $G$  
**Output**: Maximum independent set of $G$

1. **Target Size**: Compute $k = \alpha(G) = \omega(\overline{G})$ using oracle
2. **Iterative Construction**: Build MIS vertex by vertex using oracle queries
3. **Optimality Check**: For each vertex $v$, test if including $v$ maintains optimality
4. **Decision**: Include $v$ if $1 + \text{remaining\_capacity} = \text{current\_target}$
</div>

## Quadratic Programming Formulation

### Standard Form

The optimization problem can be written in standard quadratic programming form:

$$\begin{align}
\text{maximize} \quad & \frac{1}{2} x^T A x \\
\text{subject to} \quad & \sum_{i=1}^n x_i = 1 \\
& x_i \geq 0, \quad i = 1, \ldots, n
\end{align}$$

### Properties

!!! note "Problem Characteristics"
    
    - **Objective**: Quadratic (generally non-convex for graph adjacency matrices)
    - **Constraints**: Linear (defines the probability simplex)
    - **Variables**: $n$ continuous variables
    - **Feasible Region**: Compact and convex
    - **Global Optimum**: Guaranteed to exist (Weierstrass theorem)

### Challenges

1. **Non-convexity**: The objective function $\frac{1}{2} x^T A x$ is generally non-convex
2. **Local Optima**: Standard optimization algorithms may get trapped
3. **Numerical Precision**: Floating-point errors can affect the final integer result

## Extensions and Variations

### Weighted Graphs

The theorem extends naturally to weighted graphs with adjacency matrix $A$ having non-negative weights:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega_w(G)}\right)$$

where $\omega_w(G)$ is the weight of the maximum weighted clique.

### Fractional Clique Number

The continuous relaxation provides insights into the **fractional clique number**:

$$\omega_f(G) = \frac{1}{1 - 2M}$$

where $M$ is the optimal value of the quadratic program. This quantity satisfies:

$$\omega(G) \leq \omega_f(G) \leq n$$

### Stability Number Connection

For the **stability number** (independence number), we have:

$$\alpha(G) = \omega(\overline{G}) = \frac{1}{1 - 2 \max_{x \in \Delta_n} \frac{1}{2} x^T \overline{A} x}$$

where $\overline{A}$ is the adjacency matrix of the complement graph.

## Theoretical Complexity

### Classical Complexity

- **MIS Problem**: NP-hard in general graphs
- **Approximation**: Hard to approximate within factor $n^{1-\epsilon}$ for any $\epsilon > 0$
- **Special Cases**: Polynomial-time solvable in bipartite graphs, trees, etc.

### Continuous Relaxation

The quadratic program provides:

- **Continuous Relaxation**: Polynomial-time solvable (in theory)
- **Rounding**: The continuous solution must be carefully rounded to discrete
- **Approximation Quality**: Depends on the optimization algorithm used

### Our Approach Complexity

The oracle-based MIS algorithm has complexity:

- **Oracle Calls**: $O(n)$ calls in the worst case
- **Per-Oracle Cost**: Depends on the quadratic programming solver
- **Total Complexity**: $O(n \cdot T_{QP}(n))$ where $T_{QP}(n)$ is QP solver time

## Historical Context

### Original Work

The theorem was first published by:

!!! quote "Citation"
    
    T. S. Motzkin and E. G. Straus, "Maxima for graphs and a new proof of a theorem of Turán," *Canadian Journal of Mathematics*, vol. 17, pp. 533-540, 1965.

### Subsequent Developments

- **1970s**: Extensions to hypergraphs and weighted cases
- **1980s**: Computational studies and approximation algorithms  
- **1990s**: Semidefinite programming relaxations
- **2000s**: Modern optimization approaches (interior point, first-order methods)
- **2020s**: Quantum computing applications (our work with Dirac-3)

## Summary

The Motzkin-Straus theorem provides a beautiful mathematical framework that:

1. **Bridges Domains**: Connects discrete graph theory with continuous optimization
2. **Enables Algorithms**: Provides a computational pathway for NP-hard problems
3. **Offers Flexibility**: Supports various optimization backends and solver choices
4. **Maintains Rigor**: Preserves mathematical exactness through the equivalence

This theoretical foundation underlies our entire computational approach, enabling us to leverage powerful optimization tools—from classical solvers to quantum annealers—for solving maximum independent set problems.

---

**Next**: Explore the [specific algorithms](algorithms.md) that implement this theory, or dive into the [quadratic programming solvers](../api/oracles/qp-solvers.md) that make it computationally feasible.