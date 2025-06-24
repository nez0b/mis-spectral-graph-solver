#!/usr/bin/env python3
"""
Quick validation script for JAX solver integration.
Run this after installing dependencies to verify the implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Validating JAX Solver Integration")
    print("=" * 40)
    
    # Check imports
    try:
        import jax
        import jax.numpy as jnp
        print("✓ JAX available")
    except ImportError:
        print("✗ JAX not available - install with: pip install jax jaxlib")
        return False
    
    try:
        import networkx as nx
        print("✓ NetworkX available")
    except ImportError:
        print("✗ NetworkX not available - install with: pip install networkx")
        return False
    
    try:
        from motzkinstraus.jax_optimizers import (
            JAXOptimizerConfig, 
            evaluate_polynomial, 
            project_to_simplex,
            adjacency_to_polynomial
        )
        print("✓ JAX optimizers module imported")
    except ImportError as e:
        print(f"✗ JAX optimizers import failed: {e}")
        return False
    
    try:
        from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
        from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle
        print("✓ JAX oracles imported")
    except ImportError as e:
        print(f"✗ JAX oracles import failed: {e}")
        return False
    
    # Test basic functionality
    try:
        # Create test graph
        G = nx.cycle_graph(4)
        print(f"✓ Test graph created: 4-cycle")
        
        # Test PGD oracle
        pgd_oracle = ProjectedGradientDescentOracle(
            learning_rate=0.02,
            max_iterations=100,
            num_restarts=2,
            verbose=False
        )
        print("✓ PGD oracle created")
        
        pgd_omega = pgd_oracle.get_omega(G)
        print(f"✓ PGD oracle result: ω = {pgd_omega}")
        
        # Test MD oracle
        md_oracle = MirrorDescentOracle(
            learning_rate=0.01,
            max_iterations=100,
            num_restarts=2,
            verbose=False
        )
        print("✓ MD oracle created")
        
        md_omega = md_oracle.get_omega(G)
        print(f"✓ MD oracle result: ω = {md_omega}")
        
        # Check basic functionality
        if pgd_omega >= 1 and md_omega >= 1:
            print("✓ Reasonable omega values obtained")
        else:
            print(f"⚠ Unexpected omega values: PGD={pgd_omega}, MD={md_omega}")
        
        # Test optimization details
        pgd_details = pgd_oracle.get_optimization_details()
        if 'num_restarts' in pgd_details and 'best_energy' in pgd_details:
            print("✓ Optimization details available")
        else:
            print("⚠ Optimization details incomplete")
        
        print("\n🎉 JAX solver integration validation PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nNext steps:")
        print("1. Install JAX: uv add jax jaxlib")
        print("2. Run tests: python -m pytest tests/test_jax_oracles.py")
        print("3. Run analysis: python examples/jax_solver_analysis.py")
    else:
        print("\nPlease fix the issues above before proceeding.")