#!/usr/bin/env python3
"""
Quick test to validate the jax.vmap batching implementation.
"""

import os
import sys
import time
import networkx as nx

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle

def test_vmap_batching():
    """Test the vmap batching implementation on a small graph."""
    print("Testing JAX vmap batching implementation...")
    
    # Create a small test graph
    G = nx.erdos_renyi_graph(10, 0.3, seed=42)
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test PGD with verbose output
    print("\n=== Testing PGD with vmap batching ===")
    pgd_oracle = ProjectedGradientDescentOracle(
        num_restarts=5,
        max_iterations=100,
        verbose=True  # This will enable both solver and oracle call verbosity
    )
    
    start_time = time.time()
    omega_pgd = pgd_oracle.get_omega(G)
    pgd_time = time.time() - start_time
    
    print(f"PGD Result: ω = {omega_pgd}")
    print(f"PGD Time: {pgd_time:.4f} seconds")
    print(f"Oracle call count: {pgd_oracle.call_count}")
    
    # Test MD with verbose output
    print("\n=== Testing MD with vmap batching ===")
    md_oracle = MirrorDescentOracle(
        num_restarts=5,
        max_iterations=100,
        verbose=True  # This will enable both solver and oracle call verbosity
    )
    
    start_time = time.time()
    omega_md = md_oracle.get_omega(G)
    md_time = time.time() - start_time
    
    print(f"MD Result: ω = {omega_md}")
    print(f"MD Time: {md_time:.4f} seconds")
    print(f"Oracle call count: {md_oracle.call_count}")
    
    # Validate results are reasonable
    print(f"\n=== Results Summary ===")
    print(f"PGD: ω = {omega_pgd}, time = {pgd_time:.4f}s")
    print(f"MD:  ω = {omega_md}, time = {md_time:.4f}s")
    
    if omega_pgd == omega_md:
        print("✓ Both methods found same clique number")
    else:
        print("⚠ Methods found different clique numbers")
    
    print("\n✓ vmap batching test completed!")

if __name__ == "__main__":
    test_vmap_batching()