#!/usr/bin/env python3
"""
Output to Histogram: Generate histograms from existing Dirac response JSON files

This script takes a Dirac response JSON file and generates an energy histogram
with theoretical omega lines, similar to the functionality in graph_to_omega.py.

Usage:
    python scripts/output_to_histogram.py dirac_response_file.json [OPTIONS]

Examples:
    python scripts/output_to_histogram.py dirac_response_20250706_131958_125vars.json
    python scripts/output_to_histogram.py response.json --show-theory --save-histogram
    python scripts/output_to_histogram.py data.json --no-theory --output-prefix analysis
    python scripts/output_to_histogram.py regularized_response.json --show-theory --regularized 0.1
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# OMEGA CALCULATION FUNCTIONS (copied from graph_to_omega.py)
# =============================================================================

def energy_to_omega(energy: float) -> float:
    """
    Convert energy to equivalent clique number using inverse Motzkin-Straus formula.
    
    Formula: energy = -1/2 * (1 - 1/ω), so ω = 1/(1 + 2*energy)
    
    Args:
        energy: Energy value from Dirac solver (should be negative)
        
    Returns:
        Equivalent clique number (omega), or NaN if energy is non-negative
    """
    if energy >= 0:
        return float('nan')
    denominator = 1.0 + 2.0 * energy
    if abs(denominator) < 1e-12:  # Avoid division by zero
        return float('inf')
    return 1.0 / denominator


def omega_to_energy(omega: int) -> float:
    """
    Convert clique number to theoretical energy using Motzkin-Straus formula.
    
    Formula: energy = -1/2 * (1 - 1/ω) = -0.5 * (1 - 1/ω)
    
    Args:
        omega: Clique number (should be >= 2)
        
    Returns:
        Theoretical energy value, or -inf if omega <= 1
    """
    if omega <= 1:
        return float('-inf')
    return -0.5 * (1.0 - 1.0 / omega)


def regularized_energy_to_omega(energy: float, c: float) -> float:
    """
    Convert energy to equivalent clique number using inverse regularized Motzkin-Straus formula.
    
    For polynomial representation with coefficient 1.0, the relationship is:
    energy = -(ω-1)/(2ω) - c/ω, solving for ω:
    ω = (1-2c)/(1 + 2*energy)
    
    Args:
        energy: Energy value from Dirac solver (should be negative)
        c: Regularization parameter in [0, 1]
        
    Returns:
        Equivalent clique number (omega), or NaN if energy is non-negative
    """
    if energy >= 0:
        return float('nan')
    
    # Special case: c = 0 (no regularization)
    if c == 0.0:
        return energy_to_omega(energy)
    
    # Special case: c = 0.5 (degenerate case for polynomial representation)
    if abs(c - 0.5) < 1e-12:
        return float('nan')  # Not well-defined (numerator becomes 0)
    
    # General case: c ∈ [0, 1], c ≠ 0.5
    denominator = 1.0 + 2.0 * energy
    if abs(denominator) < 1e-12:  # Avoid division by zero
        return float('inf')
    
    return (1.0 - 2.0 * c) / denominator


def regularized_omega_to_energy(omega: int, c: float) -> float:
    """
    Convert clique number to theoretical energy using regularized Motzkin-Straus formula.
    
    Formula for polynomial representation: energy = -(ω-1)/(2ω) - c/ω
    
    Args:
        omega: Clique number (should be >= 2)
        c: Regularization parameter in [0, 1]
        
    Returns:
        Theoretical energy value, or -inf if omega <= 1
    """
    if omega <= 1:
        return float('-inf')
    
    # Special case: c = 0 (no regularization)
    if c == 0.0:
        return omega_to_energy(omega)
    
    # General case: regularized formula
    return -(omega - 1.0) / (2.0 * omega) - c / omega


# =============================================================================
# DIRAC RESPONSE PARSING FUNCTIONS
# =============================================================================

def load_dirac_response(file_path: str) -> Dict[str, Any]:
    """
    Load Dirac response JSON data from file.
    
    Args:
        file_path: Path to Dirac response JSON file
        
    Returns:
        Dictionary containing Dirac response data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dirac response file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            response_data = json.load(f)
        
        # Validate required structure
        if 'results' not in response_data:
            raise ValueError("Missing 'results' section in Dirac response file")
        
        results = response_data['results']
        if 'energies' not in results:
            raise ValueError("Missing 'energies' in results section")
        
        energies = results['energies']
        if not energies:
            raise ValueError("Empty energies array in Dirac response")
        
        print(f"✅ Loaded Dirac response: {len(energies)} energy samples")
        return response_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Dirac response file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading Dirac response file: {e}")


def extract_energies_and_metadata(response_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extract energies and metadata from Dirac response.
    
    Args:
        response_data: Dirac response dictionary
        
    Returns:
        Tuple of (energies_list, metadata_dict)
        
    Raises:
        ValueError: If data extraction fails
    """
    try:
        results = response_data['results']
        energies = results['energies']
        
        # Extract metadata
        metadata = {
            'job_id': response_data.get('job_info', {}).get('job_id', 'unknown'),
            'job_name': response_data.get('job_info', {}).get('job_submission', {}).get('job_name', 'unknown'),
            'status': response_data.get('status', 'unknown'),
            'num_samples': len(energies),
            'best_energy': min(energies),
            'energy_range': [min(energies), max(energies)],
            'energy_stats': {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(min(energies)),
                'max': float(max(energies))
            }
        }
        
        # Extract device info if available
        job_submission = response_data.get('job_info', {}).get('job_submission', {})
        device_config = job_submission.get('device_config', {})
        if device_config:
            device_name = list(device_config.keys())[0] if device_config else 'unknown'
            device_params = device_config.get(device_name, {})
            metadata['device'] = device_name
            metadata['device_params'] = device_params
        
        print(f"📊 Energy statistics:")
        print(f"   Best energy: {metadata['best_energy']:.6f}")
        print(f"   Energy range: [{metadata['energy_range'][0]:.6f}, {metadata['energy_range'][1]:.6f}]")
        print(f"   Mean: {metadata['energy_stats']['mean']:.6f} ± {metadata['energy_stats']['std']:.6f}")
        print(f"   Total samples: {metadata['num_samples']}")
        
        return energies, metadata
        
    except Exception as e:
        raise ValueError(f"Error extracting energies and metadata: {e}")


# =============================================================================
# ENHANCED THEORY LINES FUNCTIONALITY (copied from graph_to_omega.py)
# =============================================================================

def get_omega_range(energies: List[float], buffer: int = 1, regularization_c: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine relevant omega range from observed energies with adaptive buffering.
    
    Args:
        energies: List of energy values from Dirac solver
        buffer: Number of extra omega values to show on each side
        regularization_c: Regularization parameter for regularized formulas (None for standard)
        
    Returns:
        Tuple of (start_omega, end_omega) for plotting, or (None, None) if invalid
    """
    if not energies:
        return None, None
    
    min_energy = min(energies)
    max_energy = max(energies)
    
    if min_energy >= 0:
        return None, None  # Invalid energy range for Motzkin-Straus
    
    # Use appropriate formula based on regularization
    if regularization_c is not None:
        omega_for_min = regularized_energy_to_omega(min_energy, regularization_c)
        omega_for_max = regularized_energy_to_omega(max_energy, regularization_c)
    else:
        omega_for_min = energy_to_omega(min_energy)
        omega_for_max = energy_to_omega(max_energy)
    
    if not (omega_for_min > 1 and omega_for_max > 1):
        return None, None
    
    # Add buffer and ensure valid range (start from omega=2 minimum)
    start_omega = max(2, int(omega_for_max) - buffer)
    end_omega = int(omega_for_min) + buffer
    
    return start_omega, end_omega


def add_theory_lines(ax, energies: List[float], show_theory_lines: bool = False, 
                    max_lines: int = 8, regularization_c: Optional[float] = None) -> bool:
    """
    Add theoretical omega lines to the histogram based on Motzkin-Straus formula.
    
    Args:
        ax: Matplotlib axes object
        energies: List of energy values from Dirac solver
        show_theory_lines: Whether to add the theoretical lines
        max_lines: Maximum number of lines to prevent visual clutter
        regularization_c: Regularization parameter for regularized formulas (None for standard)
        
    Returns:
        True if lines were added successfully, False otherwise
    """
    if not show_theory_lines:
        return False
    
    if not energies or min(energies) >= 0:
        print("Warning: Cannot display theoretical lines for non-negative energies")
        return False
    
    start_omega, end_omega = get_omega_range(energies, regularization_c=regularization_c)
    if start_omega is None or end_omega is None:
        print("Warning: Could not determine valid omega range for theoretical lines")
        return False
    
    # Limit number of lines to prevent visual clutter
    if end_omega - start_omega > max_lines:
        end_omega = start_omega + max_lines
        print(f"Limiting theoretical lines to {max_lines} (ω = {start_omega} to {end_omega})")
    
    # Get plot boundaries and actual energy data bounds
    y_max = ax.get_ylim()[1]
    energy_min, energy_max = min(energies), max(energies)
    
    # Add small buffer to energy range for theoretical line visibility
    energy_range = energy_max - energy_min
    buffer = max(0.1 * energy_range, 0.01)  # 10% buffer or minimum 0.01
    data_x_min = energy_min - buffer
    data_x_max = energy_max + buffer
    
    # Debug logging
    print(f"Debug: Energy range: {energy_min:.6f} to {energy_max:.6f}")
    print(f"Debug: Plot range: ω={start_omega} to ω={end_omega}")
    
    lines_added = 0
    for omega in range(start_omega, end_omega + 1):
        # Use appropriate formula based on regularization
        if regularization_c is not None:
            energy_val = regularized_omega_to_energy(omega, regularization_c)
        else:
            energy_val = omega_to_energy(omega)
        
        # Check if theoretical energy is within reasonable range of actual data
        in_data_range = data_x_min <= energy_val <= data_x_max
        print(f"Debug: ω={omega}: energy={energy_val:.6f}, in_data_range={in_data_range}")
        
        # Always draw lines for omega values derived from the energy data
        # This ensures relevant theoretical lines are always visible
        if in_data_range:
            ax.axvline(x=energy_val, color='#7f7f7f', linestyle='--', 
                      alpha=0.8, zorder=2, linewidth=1.5)
            ax.text(energy_val, y_max * 0.9, f' ω={omega}',
                   rotation=90, verticalalignment='bottom', 
                   color='#7f7f7f', fontsize=9, fontweight='bold')
            lines_added += 1
            
            # Extend plot x-axis to ensure theoretical line is visible
            current_xlim = ax.get_xlim()
            new_x_min = min(current_xlim[0], energy_val * 1.001)
            new_x_max = max(current_xlim[1], energy_val * 0.999)
            ax.set_xlim(new_x_min, new_x_max)
    
    if lines_added > 0:
        # Add formula annotation in upper-left corner
        if regularization_c is not None:
            formula_text = r'$E = -\frac{\omega-1}{2\omega} - \frac{c}{\omega}$' + f' (c={regularization_c})'
            label_text = f'Theoretical Energy (ω, c={regularization_c})'
        else:
            formula_text = r'$E = -\frac{1}{2}(1-\frac{1}{\omega})$'
            label_text = 'Theoretical Energy (ω)'
            
        ax.text(0.05, 0.95, formula_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        
        # Add single legend entry for all theory lines
        ax.plot([], [], color='#7f7f7f', linestyle='--', alpha=0.8, 
               label=label_text)
        
        print(f"Added {lines_added} theoretical omega lines (ω = {start_omega} to {start_omega + lines_added - 1})")
        return True
    
    return False


# =============================================================================
# HISTOGRAM PLOTTING FUNCTION
# =============================================================================

def plot_energy_histogram(
    energies: List[float], 
    metadata: Dict[str, Any],
    show_theory_lines: bool = False,
    save_path: Optional[str] = None,
    output_prefix: str = "output",
    regularization_c: Optional[float] = None
) -> bool:
    """
    Plot histogram of energies from Dirac response with enhanced theoretical omega lines.
    
    Args:
        energies: List of energy values
        metadata: Metadata dictionary with job info
        show_theory_lines: Whether to show theoretical omega lines
        save_path: Optional path to save plot
        output_prefix: Prefix for output filename
        
    Returns:
        True if plot was created successfully
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, cannot plot histogram")
        return False
    
    try:
        best_energy = metadata['best_energy']
        job_name = metadata.get('job_name', 'Unknown Job')
        num_samples = metadata['num_samples']
        
        # Calculate omega from best energy using appropriate formula
        if regularization_c is not None:
            omega = regularized_energy_to_omega(best_energy, regularization_c)
        else:
            omega = energy_to_omega(best_energy)
        
        plt.figure(figsize=(12, 7))
        plt.hist(energies, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Mark best energy
        plt.axvline(x=best_energy, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Energy: {best_energy:.6f}')
        
        # Add enhanced theoretical omega lines if requested
        theory_added = add_theory_lines(plt.gca(), energies, show_theory_lines, regularization_c=regularization_c)
        plt.xlim(min(energies) - 0.005, max(energies) + 0.005)
        
        plt.xlabel('Energy', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Enhanced title with more information
        title = f'{job_name}: Energy Distribution'
        subtitle = f'Samples: {num_samples} | Best Energy: {best_energy:.6f} | Estimated ω: {omega:.2f}'
        plt.title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Statistics:
Mean: {metadata['energy_stats']['mean']:.6f}
Std: {metadata['energy_stats']['std']:.6f}
Range: [{metadata['energy_stats']['min']:.6f}, {metadata['energy_stats']['max']:.6f}]"""
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Auto-generate save path if not provided but saving is requested
        if save_path is None and output_prefix != "output":
            save_path = f"{output_prefix}_histogram.png"
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved histogram to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        return True
        
    except Exception as e:
        print(f"❌ Error creating histogram: {e}")
        return False


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate histograms from existing Dirac response JSON files with theoretical omega lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/output_to_histogram.py dirac_response_20250706_131958_125vars.json
  python scripts/output_to_histogram.py response.json --show-theory --save-histogram
  python scripts/output_to_histogram.py data.json --no-theory --output-prefix analysis
  python scripts/output_to_histogram.py regularized_response.json --show-theory --regularized 0.1
        """
    )
    
    parser.add_argument("dirac_response_file", help="Path to Dirac response JSON file")
    parser.add_argument("--show-theory", action="store_true", 
                       help="Show theoretical omega lines in histogram")
    parser.add_argument("--no-theory", action="store_true",
                       help="Explicitly disable theoretical omega lines")
    parser.add_argument("--regularized", type=float, metavar="C", default=None,
                       help="Use regularized formulas with parameter c (e.g., --regularized 0.1)")
    parser.add_argument("--save-histogram", action="store_true",
                       help="Save histogram to PNG file")
    parser.add_argument("--output-prefix", type=str, default="output",
                       help="Prefix for output filename (default: output)")
    parser.add_argument("--bins", type=int, default=25,
                       help="Number of histogram bins (default: 25)")
    parser.add_argument("--format", choices=["show", "save", "both"], default="show",
                       help="Output format: show plot, save only, or both (default: show)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Please install: pip install matplotlib")
        return 1
    
    # Validate input file
    if not os.path.exists(args.dirac_response_file):
        print(f"❌ Dirac response file not found: {args.dirac_response_file}")
        return 1
    
    # Resolve theory lines setting
    show_theory = args.show_theory and not args.no_theory
    
    print("📊 Output to Histogram: Dirac Response Analysis")
    print("=" * 60)
    
    try:
        # Load and parse Dirac response
        response_data = load_dirac_response(args.dirac_response_file)
        energies, metadata = extract_energies_and_metadata(response_data)
        
        # Determine save path
        save_path = None
        if args.save_histogram or args.format in ["save", "both"]:
            file_stem = Path(args.dirac_response_file).stem
            save_path = f"{args.output_prefix}_{file_stem}_histogram.png"
        
        # Generate histogram
        success = plot_energy_histogram(
            energies=energies,
            metadata=metadata,
            show_theory_lines=show_theory,
            save_path=save_path,
            output_prefix=args.output_prefix,
            regularization_c=args.regularized
        )
        
        if success:
            # Print summary
            print(f"\n{'='*60}")
            print("ANALYSIS SUMMARY")
            print(f"{'='*60}")
            print(f"Input file: {args.dirac_response_file}")
            print(f"Job ID: {metadata['job_id']}")
            print(f"Job name: {metadata['job_name']}")
            print(f"Status: {metadata['status']}")
            print(f"Total samples: {metadata['num_samples']}")
            print(f"Best energy: {metadata['best_energy']:.6f}")
            if args.regularized is not None:
                estimated_omega = regularized_energy_to_omega(metadata['best_energy'], args.regularized)
                print(f"Estimated ω: {estimated_omega:.3f} (regularized c={args.regularized})")
            else:
                print(f"Estimated ω: {energy_to_omega(metadata['best_energy']):.3f}")
            print(f"Energy range: [{metadata['energy_range'][0]:.6f}, {metadata['energy_range'][1]:.6f}]")
            if show_theory:
                print(f"Theoretical lines: Enabled")
            print(f"{'='*60}")
            
            print("\n🎉 Histogram generation completed successfully!")
            return 0
        else:
            print("\n❌ Failed to generate histogram")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())