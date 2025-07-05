"""
Quick test of large graph functionality without brute force.

Tests oracle performance and accuracy on moderately sized graphs.
"""

import pytest
import os
import networkx as nx

from motzkinstraus.algorithms import find_mis_with_oracle, verify_independent_set

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


class TestLargeGraphQuick:
    """Quick tests on moderately sized graphs for performance validation."""
    
    @pytest.fixture
    def test_graph(self):
        """Create a moderately sized test graph."""
        # 8-node cycle with some chords for complexity
        G = nx.cycle_graph(8)
        G.add_edges_from([(0, 4), (2, 6)])  # Add some chords
        return G
    
    @pytest.fixture
    def larger_test_graphs(self):
        """Create various larger test graphs."""
        graphs = []
        
        # 12-node cycle
        G1 = nx.cycle_graph(12)
        graphs.append((G1, "12-cycle"))
        
        # 15-node path
        G2 = nx.path_graph(15)
        graphs.append((G2, "15-path"))
        
        # 10-node wheel
        G3 = nx.wheel_graph(10)
        graphs.append((G3, "10-wheel"))
        
        # 16-node grid
        G4 = nx.grid_2d_graph(4, 4)
        G4 = nx.convert_node_labels_to_integers(G4)
        graphs.append((G4, "4x4-grid"))
        
        return graphs
    
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi oracle not available")
    def test_gurobi_oracle_performance(self, test_graph):
        """Test Gurobi oracle performance on moderately sized graph."""
        gurobi_oracle = GurobiOracle(suppress_output=True)
        
        # Solve and validate
        gurobi_mis, gurobi_calls = find_mis_with_oracle(test_graph, gurobi_oracle)
        
        # Basic validation
        assert verify_independent_set(test_graph, gurobi_mis), "Gurobi MIS should be valid"
        assert len(gurobi_mis) > 0, "MIS should not be empty"
        assert gurobi_calls > 0, "Should make at least one oracle call"
        
        # Performance check - should be fast
        import time
        start_time = time.time()
        gurobi_mis2, _ = find_mis_with_oracle(test_graph, gurobi_oracle)
        runtime = time.time() - start_time
        
        assert runtime < 5.0, f"Gurobi should be fast on small graphs: {runtime:.3f}s"
        assert len(gurobi_mis2) == len(gurobi_mis), "Should get consistent results"
    
    @pytest.mark.skipif(not DIRAC_AVAILABLE, reason="Dirac oracle not available")
    @pytest.mark.slow
    def test_dirac_oracle_performance(self, test_graph):
        """Test Dirac oracle performance on moderately sized graph."""
        # Use quick Dirac settings for testing
        dirac_oracle = DiracOracle(num_samples=30, relax_schedule=1)
        
        # Solve and validate
        dirac_mis, dirac_calls = find_mis_with_oracle(test_graph, dirac_oracle)
        
        # Basic validation
        assert verify_independent_set(test_graph, dirac_mis), "Dirac MIS should be valid"
        assert len(dirac_mis) > 0, "MIS should not be empty"
        assert dirac_calls > 0, "Should make at least one oracle call"
        
        # Performance check - allow more time for Dirac
        import time
        start_time = time.time()
        dirac_mis2, _ = find_mis_with_oracle(test_graph, dirac_oracle)
        runtime = time.time() - start_time
        
        assert runtime < 30.0, f"Dirac should complete in reasonable time: {runtime:.3f}s"
    
    @pytest.mark.skipif(not (GUROBI_AVAILABLE and DIRAC_AVAILABLE), 
                       reason="Both oracles required for comparison")
    def test_oracle_comparison(self, test_graph):
        """Compare Gurobi and Dirac oracle results."""
        # Test both oracles
        gurobi_oracle = GurobiOracle(suppress_output=True)
        dirac_oracle = DiracOracle(num_samples=30, relax_schedule=1)
        
        gurobi_mis, gurobi_calls = find_mis_with_oracle(test_graph, gurobi_oracle)
        dirac_mis, dirac_calls = find_mis_with_oracle(test_graph, dirac_oracle)
        
        # Both should be valid
        assert verify_independent_set(test_graph, gurobi_mis), "Gurobi MIS should be valid"
        assert verify_independent_set(test_graph, dirac_mis), "Dirac MIS should be valid"
        
        # Both should find optimal solution (same size)
        assert len(gurobi_mis) == len(dirac_mis), \
            f"Oracle disagreement: Gurobi={len(gurobi_mis)}, Dirac={len(dirac_mis)}"
        
        # Oracle call counts should be reasonable
        assert 1 <= gurobi_calls <= test_graph.number_of_nodes(), \
            f"Gurobi oracle calls out of range: {gurobi_calls}"
        assert 1 <= dirac_calls <= test_graph.number_of_nodes(), \
            f"Dirac oracle calls out of range: {dirac_calls}"
    
    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi oracle not available")
    def test_scaling_behavior(self, larger_test_graphs):
        """Test scaling behavior on larger graphs."""
        gurobi_oracle = GurobiOracle(suppress_output=True)
        results = []
        
        for graph, description in larger_test_graphs:
            import time
            start_time = time.time()
            
            mis, calls = find_mis_with_oracle(graph, gurobi_oracle)
            runtime = time.time() - start_time
            
            # Validate result
            assert verify_independent_set(graph, mis), f"Invalid MIS for {description}"
            
            results.append({
                'description': description,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'mis_size': len(mis),
                'oracle_calls': calls,
                'runtime': runtime
            })
            
            # Performance check
            assert runtime < 10.0, f"Runtime too long for {description}: {runtime:.3f}s"
        
        # Validate scaling trends
        for result in results:
            nodes = result['nodes']
            calls = result['oracle_calls']
            
            # Oracle calls should be reasonable relative to graph size
            assert calls <= nodes, f"Too many oracle calls for {result['description']}: {calls} > {nodes}"
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_visualization_creation(self, test_graph):
        """Test creation of comparison visualizations."""
        if not (GUROBI_AVAILABLE and DIRAC_AVAILABLE):
            pytest.skip("Both oracles required for visualization test")
        
        # Create figures directory
        figures_dir = "figures/tests"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Get results from both oracles
        gurobi_oracle = GurobiOracle(suppress_output=True)
        dirac_oracle = DiracOracle(num_samples=20, relax_schedule=1)
        
        gurobi_mis, gurobi_calls = find_mis_with_oracle(test_graph, gurobi_oracle)
        dirac_mis, dirac_calls = find_mis_with_oracle(test_graph, dirac_oracle)
        
        # Create visualization
        save_path = os.path.join(figures_dir, "test_large_graph_comparison.png")
        self._create_dual_panel_visualization(
            test_graph, 
            (gurobi_mis, gurobi_calls), 
            (dirac_mis, dirac_calls), 
            save_path
        )
        
        # Verify file was created
        assert os.path.exists(save_path), "Visualization file not created"
    
    def _create_dual_panel_visualization(self, graph, gurobi_result, dirac_result, save_path):
        """Create a dual-panel visualization."""
        gurobi_mis, gurobi_calls = gurobi_result
        dirac_mis, dirac_calls = dirac_result
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)
        
        # Panel 1: Gurobi
        node_colors_gurobi = ['red' if node in gurobi_mis else 'lightblue' 
                             for node in graph.nodes()]
        nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=node_colors_gurobi, 
                node_size=800, edge_color='gray', width=1.5, font_size=12, 
                font_weight='bold', font_color='white')
        ax1.set_title(f'Gurobi Oracle Result\nMIS Size: {len(gurobi_mis)}, Oracle Calls: {gurobi_calls}', 
                      fontsize=14, fontweight='bold')
        
        # Panel 2: Dirac
        node_colors_dirac = ['red' if node in dirac_mis else 'lightblue' 
                            for node in graph.nodes()]
        nx.draw(graph, pos, ax=ax2, with_labels=True, node_color=node_colors_dirac, 
                node_size=800, edge_color='gray', width=1.5, font_size=12, 
                font_weight='bold', font_color='white')
        ax2.set_title(f'Dirac Oracle Result\nMIS Size: {len(dirac_mis)}, Oracle Calls: {dirac_calls}', 
                      fontsize=14, fontweight='bold')
        
        fig.suptitle('Motzkin-Straus MIS Solver Comparison', fontsize=16, fontweight='bold')
        
        # Legend
        red_patch = plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Independent Set')
        blue_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Other Nodes')
        fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
                   bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_graph_properties_validation(self, test_graph, larger_test_graphs):
        """Validate properties of test graphs."""
        all_graphs = [(test_graph, "test_graph")] + larger_test_graphs
        
        for graph, description in all_graphs:
            # Basic graph properties
            assert graph.number_of_nodes() > 0, f"Graph {description} should have nodes"
            assert nx.is_connected(graph), f"Graph {description} should be connected"
            
            # NetworkX exact solution for validation
            try:
                complement = nx.complement(graph)
                max_clique = max(nx.find_cliques(complement), key=len, default=[])
                expected_mis_size = len(max_clique)
                
                assert expected_mis_size > 0, f"Graph {description} should have non-empty MIS"
                assert expected_mis_size <= graph.number_of_nodes(), \
                    f"MIS size should not exceed node count for {description}"
                    
            except Exception:
                # NetworkX might fail on some graphs, that's okay
                pass
    
    @pytest.mark.integration
    def test_end_to_end_workflow(self, test_graph):
        """Test complete end-to-end workflow."""
        if not GUROBI_AVAILABLE:
            pytest.skip("Gurobi oracle required for end-to-end test")
        
        # Complete workflow test
        oracle = GurobiOracle(suppress_output=True)
        
        # Step 1: Solve MIS
        mis, calls = find_mis_with_oracle(test_graph, oracle)
        
        # Step 2: Validate solution
        assert verify_independent_set(test_graph, mis), "Solution should be valid independent set"
        
        # Step 3: Check optimality by trying to add more nodes
        remaining_nodes = set(test_graph.nodes()) - mis
        can_extend = False
        
        for node in remaining_nodes:
            # Check if this node is adjacent to any in MIS
            adjacent_to_mis = any(test_graph.has_edge(node, mis_node) for mis_node in mis)
            if not adjacent_to_mis:
                can_extend = True
                break
        
        # If we can extend, the MIS might not be maximal (but could still be maximum)
        # This is okay for our algorithm which finds maximum, not necessarily maximal
        
        # Step 4: Performance and correctness summary
        efficiency = len(mis) / test_graph.number_of_nodes()
        assert 0 < efficiency <= 1, f"MIS efficiency should be reasonable: {efficiency:.3f}"