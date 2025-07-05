"""
Test oracle counting on small graphs with different oracle implementations.

Validates oracle call counting and creates comparison visualizations.
"""

import pytest
import time
import numpy as np
import networkx as nx
import os

from motzkinstraus.algorithms import find_mis_with_oracle, find_mis_brute_force, verify_independent_set

try:
    from motzkinstraus.oracles.gurobi import GurobiOracle
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestOracleCallCounting:
    """Test oracle call counting and comparison."""
    
    @pytest.fixture
    def counting_test_graphs(self):
        """Test graphs for oracle counting validation."""
        return [
            (nx.cycle_graph(5), "5-cycle", 2, 3),  # Expected MIS size=2, oracle calls=3
            (nx.path_graph(5), "5-path", 3, 3),    # Expected MIS size=3, oracle calls=3 
            (nx.complete_graph(4), "K4", 1, 1),    # Expected MIS size=1, oracle calls=1
            (nx.empty_graph(4), "Empty-4", 4, 1),  # Expected MIS size=4, oracle calls=1
        ]
    
    @pytest.fixture
    def gurobi_oracle(self):
        """Create Gurobi oracle for testing."""
        if not GUROBI_AVAILABLE:
            pytest.skip("Gurobi oracle not available")
        return GurobiOracle(suppress_output=True)
    
    @pytest.fixture
    def dirac_oracle(self):
        """Create Dirac oracle for testing."""
        if not DIRAC_AVAILABLE:
            pytest.skip("Dirac oracle not available")
        return DiracOracle(num_samples=50, relax_schedule=2)
    
    def test_ground_truth_computation(self, counting_test_graphs):
        """Test ground truth computation using brute force."""
        for graph, description, expected_mis_size, _ in counting_test_graphs:
            ground_truth = find_mis_brute_force(graph)
            
            # Verify the ground truth is valid
            assert verify_independent_set(graph, ground_truth), \
                f"Invalid ground truth MIS for {description}"
            
            # Check expected size
            assert len(ground_truth) == expected_mis_size, \
                f"Ground truth MIS size wrong for {description}: got {len(ground_truth)}, expected {expected_mis_size}"
    
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi oracle not available")
    def test_gurobi_oracle_counting(self, counting_test_graphs, gurobi_oracle):
        """Test oracle call counting with Gurobi oracle."""
        for graph, description, expected_mis_size, expected_calls in counting_test_graphs:
            # Reset call count and solve
            gurobi_oracle.call_count = 0
            mis_found, oracle_calls = find_mis_with_oracle(graph, gurobi_oracle, verbose=False)
            
            # Validate solution
            assert verify_independent_set(graph, mis_found), \
                f"Invalid MIS from Gurobi for {description}"
            assert len(mis_found) == expected_mis_size, \
                f"Gurobi MIS size wrong for {description}: got {len(mis_found)}, expected {expected_mis_size}"
            
            # Check oracle call count
            assert oracle_calls == expected_calls, \
                f"Gurobi oracle call count wrong for {description}: got {oracle_calls}, expected {expected_calls}"
            
            # Verify call count tracking
            assert gurobi_oracle.call_count == oracle_calls, \
                f"Oracle call count tracking inconsistent for {description}"
    
    @pytest.mark.skipif(not DIRAC_AVAILABLE, reason="Dirac oracle not available")
    @pytest.mark.slow
    def test_dirac_oracle_counting(self, counting_test_graphs, dirac_oracle):
        """Test oracle call counting with Dirac oracle."""
        # Only test on smaller graphs due to Dirac's computational cost
        small_graphs = [g for g in counting_test_graphs if g[0].number_of_nodes() <= 5]
        
        for graph, description, expected_mis_size, expected_calls in small_graphs:
            # Reset call count and solve
            dirac_oracle.call_count = 0
            mis_found, oracle_calls = find_mis_with_oracle(graph, dirac_oracle, verbose=False)
            
            # Validate solution
            assert verify_independent_set(graph, mis_found), \
                f"Invalid MIS from Dirac for {description}"
            assert len(mis_found) == expected_mis_size, \
                f"Dirac MIS size wrong for {description}: got {len(mis_found)}, expected {expected_mis_size}"
            
            # Check oracle call count (should match expected)
            assert oracle_calls == expected_calls, \
                f"Dirac oracle call count wrong for {description}: got {oracle_calls}, expected {expected_calls}"
    
    @pytest.mark.skipif(not (GUROBI_AVAILABLE and DIRAC_AVAILABLE), 
                       reason="Both Gurobi and Dirac oracles required")
    @pytest.mark.slow
    def test_oracle_call_count_agreement(self, gurobi_oracle, dirac_oracle):
        """Test that both oracles make the same number of calls."""
        test_graph = nx.cycle_graph(5)
        
        # Test Gurobi
        gurobi_oracle.call_count = 0
        gurobi_mis, gurobi_calls = find_mis_with_oracle(test_graph, gurobi_oracle, verbose=False)
        
        # Test Dirac
        dirac_oracle.call_count = 0
        dirac_mis, dirac_calls = find_mis_with_oracle(test_graph, dirac_oracle, verbose=False)
        
        # Both should find optimal solution
        assert len(gurobi_mis) == len(dirac_mis) == 2, "Both should find MIS of size 2"
        
        # Both should make same number of oracle calls
        assert gurobi_calls == dirac_calls, \
            f"Oracle call count mismatch: Gurobi={gurobi_calls}, Dirac={dirac_calls}"
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available for visualization")
    def test_visualization_creation(self, counting_test_graphs):
        """Test creation of oracle comparison visualizations."""
        if not (GUROBI_AVAILABLE and DIRAC_AVAILABLE):
            pytest.skip("Both oracles required for visualization test")
        
        # Create figures directory
        figures_dir = "figures/tests"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Test visualization on 5-cycle
        graph = nx.cycle_graph(5)
        
        # Mock results for visualization test
        gurobi_result = ({0, 2}, 3)  # MIS nodes and call count
        dirac_result = ({1, 3}, 3)   # Different MIS nodes, same call count
        
        # Create visualization
        self._create_dual_panel_visualization(
            graph, gurobi_result, dirac_result, 
            os.path.join(figures_dir, "test_oracle_counting_comparison.png")
        )
        
        # Verify file was created
        assert os.path.exists(os.path.join(figures_dir, "test_oracle_counting_comparison.png")), \
            "Visualization file not created"
    
    def _create_dual_panel_visualization(self, graph, gurobi_result, dirac_result, save_path):
        """Create dual-panel comparison visualization."""
        gurobi_mis, gurobi_calls = gurobi_result
        dirac_mis, dirac_calls = dirac_result
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Use consistent layout
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # Panel 1: Gurobi results
        node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                             for node in graph.nodes()]
        
        nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
                node_size=1200, edge_color='gray', width=3, font_size=16, 
                font_weight='bold', font_color='white')
        
        ax1.set_title(f'Gurobi Oracle\nMIS Size: {len(gurobi_mis)}\nOracle Calls: {gurobi_calls}', 
                      fontsize=14, fontweight='bold')
        
        # Panel 2: Dirac results
        node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                            for node in graph.nodes()]
        
        nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
                node_size=1200, edge_color='gray', width=3, font_size=16, 
                font_weight='bold', font_color='white')
        
        ax2.set_title(f'Dirac Oracle\nMIS Size: {len(dirac_mis)}\nOracle Calls: {dirac_calls}', 
                      fontsize=14, fontweight='bold')
        
        # Add overall title
        fig.suptitle(f'Oracle Call Counting Comparison\n{graph.number_of_nodes()}-Node Graph', 
                     fontsize=16, fontweight='bold')
        
        # Add legend
        red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set Nodes')
        blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
        fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8, bottom=0.15)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_oracle_call_efficiency(self, counting_test_graphs):
        """Test that oracle call count grows as expected with problem size."""
        if not GUROBI_AVAILABLE:
            pytest.skip("Gurobi oracle not available")
        
        oracle = GurobiOracle(suppress_output=True)
        call_counts = []
        
        for graph, description, expected_mis_size, expected_calls in counting_test_graphs:
            oracle.call_count = 0
            mis_found, oracle_calls = find_mis_with_oracle(graph, oracle, verbose=False)
            call_counts.append((graph.number_of_nodes(), oracle_calls))
        
        # Oracle calls should be reasonable (not exponential)
        for nodes, calls in call_counts:
            assert calls <= nodes, f"Too many oracle calls: {calls} for {nodes} nodes"
            assert calls >= 1, f"Should make at least 1 oracle call: {calls}"
    
    @pytest.mark.integration
    def test_full_oracle_counting_workflow(self):
        """Test complete oracle counting workflow."""
        if not (GUROBI_AVAILABLE and DIRAC_AVAILABLE):
            pytest.skip("Both oracles required for full workflow test")
        
        # Test on 5-cycle graph
        graph = nx.cycle_graph(5)
        
        # Ground truth
        ground_truth = find_mis_brute_force(graph)
        assert len(ground_truth) == 2, "5-cycle should have MIS size 2"
        
        # Test both oracles
        gurobi_oracle = GurobiOracle(suppress_output=True)
        dirac_oracle = DiracOracle(num_samples=50, relax_schedule=2)
        
        # Gurobi test
        gurobi_oracle.call_count = 0
        start_time = time.time()
        gurobi_mis, gurobi_calls = find_mis_with_oracle(graph, gurobi_oracle, verbose=False)
        gurobi_time = time.time() - start_time
        
        # Dirac test
        dirac_oracle.call_count = 0
        start_time = time.time()
        dirac_mis, dirac_calls = find_mis_with_oracle(graph, dirac_oracle, verbose=False)
        dirac_time = time.time() - start_time
        
        # Validate results
        assert verify_independent_set(graph, gurobi_mis), "Invalid Gurobi MIS"
        assert verify_independent_set(graph, dirac_mis), "Invalid Dirac MIS"
        assert len(gurobi_mis) == len(dirac_mis) == 2, "Both should find optimal MIS"
        assert gurobi_calls == dirac_calls == 3, "Both should make 3 oracle calls"
        
        # Performance check (Gurobi should be faster)
        assert gurobi_time < dirac_time or gurobi_time < 1.0, "Gurobi should be reasonably fast"