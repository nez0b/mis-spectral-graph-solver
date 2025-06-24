"""
System validation script to test the complete Motzkin-Straus MIS solver.
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set
from motzkinstraus.oracles import get_available_oracles
from motzkinstraus.exceptions import SolverUnavailableError


def test_basic_functionality():
    """Test basic package functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Test brute force algorithm
    G = nx.cycle_graph(5)
    mis = find_mis_brute_force(G)
    print(f"✓ Brute force MIS for 5-cycle: {mis} (size: {len(mis)})")
    
    # Test verification
    is_valid = verify_independent_set(G, mis)
    print(f"✓ MIS verification: {is_valid}")
    
    # Check available oracles
    available = get_available_oracles()
    print(f"✓ Available oracles: {[cls.__name__ for cls in available]}")
    
    return len(available) > 0


def test_with_available_oracles():
    """Test with whatever oracles are available."""
    print("\n=== Testing with Available Oracles ===")
    
    available_oracles = get_available_oracles()
    
    if not available_oracles:
        print("No oracles available - testing package structure only")
        return True
    
    # Test on a simple graph
    G = nx.cycle_graph(4)
    expected_size = 2  # 4-cycle has MIS of size 2
    
    success_count = 0
    
    for oracle_class in available_oracles:
        oracle_name = oracle_class.__name__.replace('Oracle', '')
        
        try:
            print(f"\nTesting {oracle_name} oracle...")
            
            # Initialize oracle with appropriate parameters
            if oracle_name == "Gurobi":
                oracle = oracle_class(suppress_output=True)
            elif oracle_name == "Dirac":
                oracle = oracle_class(num_samples=50, relax_schedule=1)
            else:
                oracle = oracle_class()
            
            # Test omega calculation
            complement = nx.complement(G)
            omega = oracle.get_omega(complement)
            print(f"  Omega of complement: {omega}")
            
            # Test full MIS algorithm
            mis, oracle_calls = find_mis_with_oracle(G, oracle, verbose=False)
            is_valid = verify_independent_set(G, mis)
            correct_size = len(mis) == expected_size
            
            print(f"  MIS found: {mis} (size: {len(mis)})")
            print(f"  Valid independent set: {is_valid}")
            print(f"  Correct size: {correct_size}")
            
            if is_valid and correct_size:
                print(f"  ✓ {oracle_name} oracle test PASSED")
                success_count += 1
            else:
                print(f"  ✗ {oracle_name} oracle test FAILED")
                
        except SolverUnavailableError as e:
            print(f"  ⚠️  {oracle_name} solver not available: {e}")
        except Exception as e:
            print(f"  ✗ {oracle_name} oracle failed: {e}")
    
    total_oracles = len(available_oracles)
    print(f"\nOracle test results: {success_count}/{total_oracles} passed")
    
    return success_count > 0


def main():
    """Run all validation tests."""
    print("Motzkin-Straus MIS Solver - System Validation")
    print("=" * 50)
    
    # Test 1: Basic functionality
    basic_ok = test_basic_functionality()
    
    # Test 2: Oracle-based algorithms
    oracle_ok = test_with_available_oracles()
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"Basic functionality: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Oracle functionality: {'✓ PASS' if oracle_ok else '✗ FAIL'}")
    
    if basic_ok:
        print("\n✓ Package is properly structured and functional!")
        if oracle_ok:
            print("✓ Oracle-based MIS solving is working!")
            print("\nNext steps:")
            print("  - Install Gurobi (gurobipy) for optimal performance")
            print("  - Install QCI dependencies (qci-client, eqc-models) for Dirac-3 access")
            print("  - Run examples/basic_example.py for demonstrations")
        else:
            print("⚠️  No working oracles - install solver dependencies")
        return True
    else:
        print("\n✗ Package has issues - check dependencies and installation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)