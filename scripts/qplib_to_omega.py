#!/usr/bin/env python3
"""
QPLIB to Omega Workflow using Direct QCI Client API

This script implements a complete workflow for computing maximum clique size (omega)
from QPLIB JSON files using direct submission to Dirac-3 via qci_client API.

Workflow: QPLIB JSON → QCI Polynomial File → Dirac-3 Job → Energy Results → Omega

Usage:
    python scripts/qplib_to_omega.py input.json [OPTIONS]

Examples:
    python scripts/qplib_to_omega.py qplib_18.json
    python scripts/qplib_to_omega.py triangle.qplib.json --num-samples 200
    python scripts/qplib_to_omega.py data.json --relax-schedule 4 --format json
"""

import sys
import os
import json
import time
import argparse
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# No additional imports needed for QPLIB-only version

# Import QCI client
try:
    import qci_client as qc
    print("✅ Successfully imported qci_client")
    QCI_AVAILABLE = True
except ImportError as e:
    print(f"❌ qci_client not available: {e}")
    print("Please install: pip install qci-client")
    QCI_AVAILABLE = False

# Import plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# CORE QPLIB TO QCI TRANSFORMATION FUNCTIONS
# =============================================================================

def load_qplib_file(file_path: str) -> Dict[str, Any]:
    """
    Load QPLIB JSON data from file.
    
    Args:
        file_path: Path to QPLIB JSON file
        
    Returns:
        Dictionary containing QPLIB data with keys: poly_indices, poly_coefficients, sum_constraint
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"QPLIB file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            qplib_data = json.load(f)
        
        # Validate required keys
        required_keys = ['poly_indices', 'poly_coefficients', 'sum_constraint']
        for key in required_keys:
            if key not in qplib_data:
                raise ValueError(f"Missing required key in QPLIB file: {key}")
        
        print(f"✅ Loaded QPLIB file: {len(qplib_data['poly_indices'])} polynomial terms")
        return qplib_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in QPLIB file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading QPLIB file: {e}")


def qplib_to_polynomial_file(qplib_data: Dict[str, Any], file_name: str = "qplib_optimization") -> Dict[str, Any]:
    """
    Transform QPLIB data to QCI polynomial file format.
    
    CRITICAL: This function converts a MAXIMIZATION problem (Motzkin-Straus) to 
    MINIMIZATION format (Dirac solver) by negating all coefficients.
    
    Mathematical Background:
    - Motzkin-Straus: MAXIMIZE f(x) = (1/2) * x^T * A * x
    - Dirac Solver: MINIMIZES the objective function
    - Conversion: minimize(-f(x)) ≡ maximize(f(x))
    - Therefore: All coefficients are negated in the transformation
    
    Args:
        qplib_data: QPLIB data dictionary with positive coefficients
        file_name: Name for the polynomial file
        
    Returns:
        QCI polynomial file configuration dictionary with negated coefficients
        
    Raises:
        ValueError: If QPLIB data is invalid
    """
    try:
        poly_indices = qplib_data['poly_indices']
        poly_coefficients = qplib_data['poly_coefficients']
        
        if len(poly_indices) != len(poly_coefficients):
            raise ValueError("poly_indices and poly_coefficients must have same length")
        
        # Calculate number of variables and degrees
        all_indices = np.array(poly_indices).flatten()
        if len(all_indices) == 0:
            raise ValueError("Empty polynomial data")
        
        ind_dict = Counter(all_indices.tolist())
        num_vars = int(max(all_indices)) if len(all_indices) > 0 else 0  # Convert to native int
        max_degree = len(poly_indices[0]) if len(poly_indices) > 0 else 2
        min_degree = max_degree  # Assume all terms have same degree for Motzkin-Straus
        
        print(f"Polynomial structure: {num_vars} variables, degree {min_degree}-{max_degree}")
        
        # Transform to QCI format: [{"idx": [i,j], "val": coefficient}]
        # Ensure all values are native Python types (not numpy types)
        data = []
        for idx, val in zip(poly_indices, poly_coefficients):
            # Convert indices to native Python ints and coefficients to native Python floats
            if isinstance(idx, (list, tuple)):
                idx_converted = [int(i) for i in idx]
            else:
                idx_converted = int(idx)
            
            val_converted = -float(val)  # Negate for maximization->minimization conversion
            data.append({"idx": idx_converted, "val": val_converted})
        
        # Create QCI polynomial file configuration
        file_config = {
            "file_name": file_name,
            "file_config": {
                "polynomial": {
                    "num_variables": int(num_vars),  # Ensure native int
                    "min_degree": int(min_degree),   # Ensure native int
                    "max_degree": int(max_degree),   # Ensure native int
                    "data": data
                }
            }
        }
        
        print(f"✅ Created QCI polynomial file config with {len(data)} terms")
        return file_config
        
    except Exception as e:
        raise ValueError(f"Error transforming QPLIB to QCI format: {e}")


# =============================================================================
# DIRAC-3 JOB SUBMISSION FUNCTIONS
# =============================================================================

def submit_to_dirac(
    polynomial_file: Dict[str, Any],
    job_name: str = "qplib_omega_computation",
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    solution_precision: Optional[float] = None,
    sum_constraint: int = 1,
    wait: bool = True,
    job_tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Submit polynomial optimization job to Dirac-3 via QCI client.
    
    Args:
        polynomial_file: QCI polynomial file configuration
        job_name: Name for the job
        num_samples: Number of samples to request
        relaxation_schedule: Relaxation schedule (1-4)
        solution_precision: Optional solution precision
        sum_constraint: Constraint for solution variables sum
        wait: Whether to wait for job completion
        job_tags: Optional list of job tags
        
    Returns:
        Job response dictionary containing results
        
    Raises:
        RuntimeError: If QCI client not available or job fails
        ValueError: If parameters are invalid
    """
    if not QCI_AVAILABLE:
        raise RuntimeError("qci_client not available")
    
    # Validate parameters
    if not 1 <= relaxation_schedule <= 4:
        raise ValueError("relaxation_schedule must be between 1 and 4")
    if not 1 <= num_samples <= 1000:
        raise ValueError("num_samples must be between 1 and 1000")
    if not 1 <= sum_constraint <= 10000:
        raise ValueError("sum_constraint must be between 1 and 10000")
    
    try:
        # Initialize QCI client
        client = qc.QciClient()
        
        # Verify access to Dirac
        allocations = client.get_allocations()
        if "dirac" not in allocations.get("allocations", {}):
            raise RuntimeError("Dirac solver allocation not available. Check your QCI account.")
        
        print(f"🔗 Connected to QCI. Available Dirac allocation: {allocations['allocations']['dirac']}")
        
        # Upload polynomial file
        print("📤 Uploading polynomial file...")
        file_response = client.upload_file(file=polynomial_file)
        file_id = file_response['file_id']
        print(f"✅ File uploaded successfully. File ID: {file_id}")
        
        # Build job parameters
        job_params = {
            'device_type': 'dirac-3',
            'relaxation_schedule': relaxation_schedule,
            'sum_constraint': sum_constraint,
            'num_samples': num_samples
        }
        
        # Add optional parameters
        if solution_precision is not None:
            job_params['solution_precision'] = solution_precision
        
        # Build job body
        job_body = client.build_job_body(
            job_type='sample-hamiltonian',
            job_name=job_name,
            job_tags=job_tags or ['qplib', 'omega_computation'],
            job_params=job_params,
            polynomial_file_id=file_id
        )
        
        # Submit job
        print(f"🚀 Submitting job to Dirac-3...")
        print(f"   Parameters: samples={num_samples}, relaxation_schedule={relaxation_schedule}")
        if solution_precision is not None:
            print(f"   Solution precision: {solution_precision}")
        print(f"   Sum constraint: {sum_constraint}")
        
        job_response = client.process_job(job_body=job_body, wait=wait)
        
        # Verify job completion
        if job_response["status"] != qc.JobStatus.COMPLETED.value:
            raise RuntimeError(f"Job failed with status: {job_response['status']}")
        
        print(f"✅ Job completed successfully!")
        return job_response
        
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError)):
            raise
        else:
            raise RuntimeError(f"Error submitting job to Dirac-3: {e}")


def extract_best_energy(job_response: Dict[str, Any]) -> Tuple[float, List[float], np.ndarray]:
    """
    Extract best energy and full energy array from Dirac job response.
    
    Args:
        job_response: Job response from Dirac-3
        
    Returns:
        Tuple of (best_energy, all_energies, best_solution)
        
    Raises:
        ValueError: If response format is invalid
    """
    try:
        results = job_response.get('results', {})
        energies = results.get('energies', [])
        solutions = results.get('solutions', [])
        
        if not energies:
            raise ValueError("No energies found in job response")
        if not solutions:
            raise ValueError("No solutions found in job response")
        
        # Find best (minimum) energy
        best_energy = min(energies)
        best_idx = energies.index(best_energy)
        best_solution = np.array(solutions[best_idx])
        
        print(f"📊 Energy statistics:")
        print(f"   Best energy: {best_energy:.6f}")
        print(f"   Energy range: [{min(energies):.6f}, {max(energies):.6f}]")
        print(f"   Total samples: {len(energies)}")
        
        return best_energy, energies, best_solution
        
    except Exception as e:
        raise ValueError(f"Error extracting energy from job response: {e}")


# =============================================================================
# OMEGA CALCULATION FUNCTIONS
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




# =============================================================================
# COMPLETE WORKFLOW FUNCTION
# =============================================================================

def compute_omega_from_qplib_file(
    qplib_file: str,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    solution_precision: Optional[float] = None,
    save_raw_data: bool = False,
    job_name: Optional[str] = None
) -> Tuple[float, float, List[float], np.ndarray]:
    """
    Complete workflow: Load QPLIB JSON → Submit to Dirac → Compute omega.
    
    Args:
        qplib_file: Path to QPLIB JSON file
        num_samples: Number of Dirac samples
        relaxation_schedule: Relaxation schedule parameter
        solution_precision: Optional solution precision
        save_raw_data: Whether to save raw job response
        job_name: Optional job name
        
    Returns:
        Tuple of (omega, best_energy, all_energies, best_solution)
        
    Raises:
        Various exceptions if workflow fails
    """
    input_path = Path(qplib_file)
    
    # Validate input file type
    if not (input_path.suffix.lower() == '.json' or '.qplib' in input_path.name):
        raise ValueError(f"Unsupported file type: {input_path.suffix}. This script accepts only QPLIB JSON files (.json, .qplib.json)")
    
    # Load QPLIB data
    print(f"📁 Loading QPLIB file: {qplib_file}")
    qplib_data = load_qplib_file(qplib_file)
    
    # Transform to QCI format
    file_name = f"{input_path.stem}_optimization"
    polynomial_file = qplib_to_polynomial_file(qplib_data, file_name)
    
    # Submit to Dirac-3
    job_name = job_name or f"{input_path.stem}_omega_job"
    job_response = submit_to_dirac(
        polynomial_file=polynomial_file,
        job_name=job_name,
        num_samples=num_samples,
        relaxation_schedule=relaxation_schedule,
        solution_precision=solution_precision
    )
    
    # Save raw data if requested
    if save_raw_data:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_file = f"dirac_response_{timestamp}_{polynomial_file['file_config']['polynomial']['num_variables']}vars.json"
        with open(raw_file, 'w') as f:
            json.dump(job_response, f, indent=2)
        print(f"💾 Saved raw response to: {raw_file}")
    
    # Extract energy and compute omega
    best_energy, all_energies, best_solution = extract_best_energy(job_response)
    omega = energy_to_omega(best_energy)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best energy: {best_energy:.6f}")
    print(f"   Omega (ω): {omega:.3f}")
    print(f"   Solution vector shape: {best_solution.shape}")
    
    return omega, best_energy, all_energies, best_solution


# =============================================================================
# ENHANCED ENERGY HISTOGRAM PLOTTING WITH THEORY LINES
# =============================================================================

def get_omega_range(energies: List[float], buffer: int = 1) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine relevant omega range from observed energies with adaptive buffering.
    
    Args:
        energies: List of energy values from Dirac solver
        buffer: Number of extra omega values to show on each side
        
    Returns:
        Tuple of (start_omega, end_omega) for plotting, or (None, None) if invalid
    """
    if not energies:
        return None, None
    
    min_energy = min(energies)
    max_energy = max(energies)
    
    if min_energy >= 0:
        return None, None  # Invalid energy range for Motzkin-Straus
    
    omega_for_min = energy_to_omega(min_energy)
    omega_for_max = energy_to_omega(max_energy)
    
    if not (omega_for_min > 1 and omega_for_max > 1):
        return None, None
    
    # Add buffer and ensure valid range (start from omega=2 minimum)
    start_omega = max(2, int(omega_for_max) - buffer)
    end_omega = int(omega_for_min) + buffer
    
    return start_omega, end_omega


def add_theory_lines(ax, energies: List[float], show_theory_lines: bool = False, 
                    max_lines: int = 8) -> bool:
    """
    Add theoretical omega lines to the histogram based on Motzkin-Straus formula.
    
    Args:
        ax: Matplotlib axes object
        energies: List of energy values from Dirac solver
        show_theory_lines: Whether to add the theoretical lines
        max_lines: Maximum number of lines to prevent visual clutter
        
    Returns:
        True if lines were added successfully, False otherwise
    """
    if not show_theory_lines:
        return False
    
    if not energies or min(energies) >= 0:
        print("Warning: Cannot display theoretical lines for non-negative energies")
        return False
    
    start_omega, end_omega = get_omega_range(energies)
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
            new_x_min = min(current_xlim[0], energy_val - 0.01)
            new_x_max = max(current_xlim[1], energy_val + 0.01)
            ax.set_xlim(new_x_min, new_x_max)
    
    if lines_added > 0:
        # Add formula annotation in upper-left corner
        formula_text = r'$E = -\frac{1}{2}(1-\frac{1}{\omega})$'
        ax.text(0.05, 0.95, formula_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        
        # Add single legend entry for all theory lines
        ax.plot([], [], color='#7f7f7f', linestyle='--', alpha=0.8, 
               label='Theoretical Energy (ω)')
        
        print(f"Added {lines_added} theoretical omega lines (ω = {start_omega} to {start_omega + lines_added - 1})")
        return True
    
    return False


def plot_energy_histogram(
    energies: List[float], 
    title: str, 
    best_energy: float,
    show_theory_lines: bool = False,
    save_path: Optional[str] = None
) -> bool:
    """
    Plot histogram of energies with enhanced theoretical omega lines.
    
    Args:
        energies: List of energy values
        title: Plot title
        best_energy: Best (minimum) energy value
        show_theory_lines: Whether to show theoretical omega lines
        save_path: Optional path to save plot
        
    Returns:
        True if plot was created successfully
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, cannot plot histogram")
        return False
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(energies, bins=20, edgecolor='black', alpha=0.7)
        
        # Mark best energy
        plt.axvline(x=best_energy, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Energy: {best_energy:.6f}')
        
        # Add enhanced theoretical omega lines if requested
        add_theory_lines(plt.gca(), energies, show_theory_lines)
        
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title(f'{title}\nSamples: {len(energies)}, Best Energy: {best_energy:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved histogram to: {save_path}")
        
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
        description="QPLIB to Omega computation using direct QCI client API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/qplib_to_omega.py qplib_18.json
  python scripts/qplib_to_omega.py triangle.qplib.json --num-samples 200
  python scripts/qplib_to_omega.py data.json --relax-schedule 4 --format json
  python scripts/qplib_to_omega.py input.qplib.json --solution-precision 0.001
        """
    )
    
    parser.add_argument("qplib_file", help="Path to QPLIB JSON file")
    parser.add_argument("--num-samples", type=int, default=100, 
                       help="Number of Dirac samples (default: 100)")
    parser.add_argument("--relax-schedule", type=int, default=2,
                       help="Dirac relaxation schedule 1-4 (default: 2)")
    parser.add_argument("--solution-precision", type=float, default=None,
                       help="Solution precision (optional)")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--save-raw", action="store_true",
                       help="Save raw Dirac response to JSON file")
    parser.add_argument("--show-theory", action="store_true", 
                       help="Show theoretical omega lines in histogram")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable energy histogram plotting")
    parser.add_argument("--job-name", type=str, default=None,
                       help="Custom job name for Dirac submission")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not QCI_AVAILABLE:
        print("❌ qci_client not available. Please install: pip install qci-client")
        return 1
    
    # Validate input file
    if not os.path.exists(args.qplib_file):
        print(f"❌ QPLIB file not found: {args.qplib_file}")
        return 1
    
    print("🎯 QPLIB to Omega Computation via Direct QCI Client")
    print("=" * 60)
    
    try:
        # Run complete workflow
        omega, best_energy, all_energies, best_solution = compute_omega_from_qplib_file(
            qplib_file=args.qplib_file,
            num_samples=args.num_samples,
            relaxation_schedule=args.relax_schedule,
            solution_precision=args.solution_precision,
            save_raw_data=args.save_raw,
            job_name=args.job_name
        )
        
        # Generate output
        if args.format == "json":
            result = {
                "qplib_file": args.qplib_file,
                "omega": float(omega),
                "best_energy": float(best_energy),
                "num_samples": len(all_energies),
                "energy_statistics": {
                    "min": float(min(all_energies)),
                    "max": float(max(all_energies)),
                    "mean": float(np.mean(all_energies)),
                    "std": float(np.std(all_energies))
                },
                "parameters": {
                    "num_samples": args.num_samples,
                    "relaxation_schedule": args.relax_schedule,
                    "solution_precision": args.solution_precision
                },
                "workflow": "QPLIB->QCI->Dirac-3->Omega"
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print(f"{'='*60}")
            print(f"QPLIB file: {args.qplib_file}")
            print(f"Omega (ω): {omega:.3f}")
            print(f"Best energy: {best_energy:.6f}")
            print(f"Samples processed: {len(all_energies)}")
            print(f"Energy range: [{min(all_energies):.6f}, {max(all_energies):.6f}]")
            print(f"Workflow: QPLIB → QCI Client → Dirac-3 → Omega")
            print(f"{'='*60}")
        
        # Create energy histogram
        if not args.no_plot and MATPLOTLIB_AVAILABLE:
            file_stem = Path(args.qplib_file).stem
            plot_title = f"{file_stem}: Energy Distribution (Dirac-3)"
            save_path = f"energy_histogram_{file_stem}.png" if args.save_raw else None
            
            plot_energy_histogram(
                energies=all_energies,
                title=plot_title,
                best_energy=best_energy,
                show_theory_lines=args.show_theory,
                save_path=save_path
            )
        
        print("\n🎉 Computation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())