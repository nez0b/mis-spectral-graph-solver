#!/usr/bin/env python3
"""
Regularized Graph to Omega Workflow using Direct QCI Client API

This script implements a complete workflow for computing maximum clique size (omega)
from both DIMACS and QPLIB JSON files using regularized Motzkin-Straus formulation
and direct submission to Dirac-3 via qci_client API.

The key difference from the standard graph_to_omega.py is the use of regularization:
- Standard formulation: x^T A x
- Regularized formulation: x^T (A + cI) x

Regularization benefits:
- Eliminates spurious solutions (non-clique local optima)
- Ensures one-to-one correspondence between optima and cliques
- Makes optimization landscape strictly concave
- More robust convergence to true maximum cliques

Workflow: DIMACS/QPLIB → Regularized QCI Polynomial File → Dirac-3 Job → Energy Results → Omega

Usage:
    python scripts/regularized_graph_to_omega.py input.dimacs [OPTIONS]
    python scripts/regularized_graph_to_omega.py input.json [OPTIONS]

Examples:
    python scripts/regularized_graph_to_omega.py qplib_18.json
    python scripts/regularized_graph_to_omega.py DIMACS/triangle.dimacs --regularization-c 0.5 --num-samples 200
    python scripts/regularized_graph_to_omega.py data.json --regularization-c 0.8 --relax-schedule 4 --format json
    python scripts/regularized_graph_to_omega.py input.dimacs --regularization-c 0.25 --solution-precision 0.001
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

# Import existing utilities
try:
    from motzkinstraus.io import read_dimacs_graph
    print("✅ Successfully imported motzkinstraus.io")
except ImportError as e:
    print(f"❌ Error importing motzkinstraus.io: {e}")
    sys.exit(1)

# Import dimacs_to_qplib with path handling
try:
    # Add scripts directory to path for import
    scripts_dir = os.path.join(os.path.dirname(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    # Import the dimacs_to_qplib module
    import dimacs_to_qplib
    dimacs_to_qplib_func = dimacs_to_qplib.dimacs_to_qplib
    print("✅ Successfully imported dimacs_to_qplib")
except ImportError as e:
    print(f"❌ Error importing dimacs_to_qplib: {e}")
    sys.exit(1)

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
# REGULARIZATION FUNCTIONS
# =============================================================================

def apply_identity_regularization(qplib_data: Dict[str, Any], c: float) -> Dict[str, Any]:
    """
    Apply identity regularization to QPLIB data: A → A + cI.
    
    This transforms the Motzkin-Straus objective from x^T A x to x^T (A + cI) x,
    which eliminates spurious solutions and makes the optimization landscape
    strictly concave.
    
    Args:
        qplib_data: Original QPLIB data dictionary
        c: Regularization parameter (typically 0.5 or 1.0)
        
    Returns:
        Regularized QPLIB data with identity matrix added to adjacency matrix
    """
    if c < 0 or c > 1:
        raise ValueError(f"Regularization parameter c must be in [0, 1], got {c}")
    
    # Special case: c = 0 means no regularization, return original data
    if c == 0.0:
        print(f"✅ No regularization applied: c = 0.0 (standard Motzkin-Straus)")
        return qplib_data
    
    # Extract original data
    poly_indices = qplib_data['poly_indices']
    poly_coefficients = qplib_data['poly_coefficients']
    
    # Find the number of variables
    all_indices = np.array(poly_indices).flatten()
    if len(all_indices) == 0:
        # No edges in graph, just return original data
        print(f"⚠️  No edges found, regularization has no effect")
        return qplib_data
    
    num_vars = int(max(all_indices))
    
    # Create regularized polynomial terms
    regularized_indices = list(poly_indices)  # Copy original terms
    regularized_coefficients = list(poly_coefficients)  # Copy original coefficients
    
    # Add diagonal terms for regularization: c * x_i^2 for each variable
    diagonal_terms_added = 0
    for i in range(1, num_vars + 1):  # Variables are 1-indexed in QPLIB
        # Check if diagonal term already exists
        diagonal_exists = any(
            len(idx) == 2 and idx[0] == i and idx[1] == i 
            for idx in poly_indices
        )
        
        if diagonal_exists:
            # Update existing diagonal term to c (replace, don't add)
            # For regularized formulation A + cI, diagonal should be c, not 1.0 + c
            for j, idx in enumerate(regularized_indices):
                if len(idx) == 2 and idx[0] == i and idx[1] == i:
                    regularized_coefficients[j] = c  # Set to c, not add c
                    diagonal_terms_added += 1
                    break
        else:
            # Add new diagonal term with coefficient c
            regularized_indices.append([i, i])
            regularized_coefficients.append(c)
            diagonal_terms_added += 1
    
    # Create regularized QPLIB data
    regularized_data = qplib_data.copy()
    regularized_data['poly_indices'] = regularized_indices
    regularized_data['poly_coefficients'] = regularized_coefficients
    
    print(f"✅ Applied identity regularization: c = {c}")
    print(f"   Added/updated {diagonal_terms_added} diagonal terms")
    print(f"   Total polynomial terms: {len(regularized_indices)} (was {len(poly_indices)})")
    
    return regularized_data


def validate_regularization_parameter(c: Union[float, int]) -> float:
    """
    Validate and convert regularization parameter.
    
    Args:
        c: Regularization parameter
        
    Returns:
        Validated float parameter
        
    Raises:
        ValueError: If parameter is invalid
    """
    try:
        c_float = float(c)
        if c_float < 0 or c_float > 1:
            raise ValueError(f"Regularization parameter must be in [0, 1], got {c_float}")
        if c_float > 10:
            # Warn about very large values that might hurt performance
            import warnings
            warnings.warn(f"Large regularization parameter c={c_float} may hurt optimization performance")
        return c_float
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid regularization parameter: {c}") from e


# =============================================================================
# CORE QPLIB TO QCI TRANSFORMATION FUNCTIONS (ADAPTED FOR REGULARIZATION)
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


def qplib_to_polynomial_file(qplib_data: Dict[str, Any], file_name: str = "regularized_qplib_optimization") -> Dict[str, Any]:
    """
    Transform regularized QPLIB data to QCI polynomial file format.
    
    CRITICAL: This function converts a MAXIMIZATION problem (regularized Motzkin-Straus) to 
    MINIMIZATION format (Dirac solver) by negating all coefficients.
    
    Mathematical Background:
    - Regularized Motzkin-Straus: MAXIMIZE f(x) = (1/2) * x^T * (A + cI) * x
    - Dirac Solver: MINIMIZES the objective function
    - Conversion: minimize(-f(x)) ≡ maximize(f(x))
    - Therefore: All coefficients are negated in the transformation
    
    Args:
        qplib_data: Regularized QPLIB data dictionary with positive coefficients
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
        
        print(f"✅ Created regularized QCI polynomial file config with {len(data)} terms")
        return file_config
        
    except Exception as e:
        raise ValueError(f"Error transforming regularized QPLIB to QCI format: {e}")


# =============================================================================
# DIMACS INTEGRATION FUNCTIONS (ADAPTED FOR REGULARIZATION)
# =============================================================================

def dimacs_to_regularized_qplib_file(dimacs_file: str, c: float) -> str:
    """
    Convert DIMACS file to regularized QPLIB JSON file.
    
    Args:
        dimacs_file: Path to DIMACS format file
        c: Regularization parameter
        
    Returns:
        Path to created regularized QPLIB JSON file
        
    Raises:
        FileNotFoundError: If DIMACS file doesn't exist
        ValueError: If conversion fails
    """
    if not os.path.exists(dimacs_file):
        raise FileNotFoundError(f"DIMACS file not found: {dimacs_file}")
    
    try:
        # Convert DIMACS to standard QPLIB
        qplib_data = dimacs_to_qplib_func(dimacs_file)
        
        # Apply regularization
        regularized_data = apply_identity_regularization(qplib_data, c)
        
        # Create output file path
        dimacs_path = Path(dimacs_file)
        qplib_file = dimacs_path.with_suffix(f'.regularized_c{c}.qplib.json')
        
        # Save regularized QPLIB data
        with open(qplib_file, 'w') as f:
            json.dump(regularized_data, f, indent=2)
        
        print(f"✅ Converted {dimacs_file} → {qplib_file} (regularized with c={c})")
        return str(qplib_file)
        
    except Exception as e:
        raise ValueError(f"Error converting DIMACS to regularized QPLIB: {e}")


# =============================================================================
# DIRAC-3 JOB SUBMISSION FUNCTIONS (REUSED FROM ORIGINAL)
# =============================================================================

def submit_to_dirac(
    polynomial_file: Dict[str, Any],
    job_name: str = "regularized_graph_omega_computation",
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    solution_precision: Optional[float] = None,
    sum_constraint: int = 1,
    wait: bool = True,
    job_tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Submit regularized polynomial optimization job to Dirac-3 via QCI client.
    
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
    if not 1 <= num_samples <= 100:
        raise ValueError("num_samples must be between 1 and 1000")
    if not 1 <= sum_constraint <= 100:
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
        print("📤 Uploading regularized polynomial file...")
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
            job_tags=job_tags or ['graph', 'omega_computation', 'regularized'],
            job_params=job_params,
            polynomial_file_id=file_id
        )
        
        # Submit job
        print(f"🚀 Submitting regularized job to Dirac-3...")
        print(f"   Parameters: samples={num_samples}, relaxation_schedule={relaxation_schedule}")
        if solution_precision is not None:
            print(f"   Solution precision: {solution_precision}")
        print(f"   Sum constraint: {sum_constraint}")
        
        job_response = client.process_job(job_body=job_body, wait=wait)
        
        # Verify job completion
        if job_response["status"] != qc.JobStatus.COMPLETED.value:
            raise RuntimeError(f"Job failed with status: {job_response['status']}")
        
        print(f"✅ Regularized job completed successfully!")
        return job_response
        
    except Exception as e:
        if isinstance(e, (RuntimeError, ValueError)):
            raise
        else:
            raise RuntimeError(f"Error submitting regularized job to Dirac-3: {e}")


# Reuse energy extraction and omega calculation functions from original script
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


def regularized_energy_to_omega(energy: float, c: float) -> float:
    """
    Convert energy to equivalent clique number using inverse regularized Motzkin-Straus formula.
    
    For polynomial representation with coefficient 1.0, the relationship is:
    energy = -(ω-1)/(2ω) - c/ω, solving for ω:
    ω = (1-2c)/(1 + 2*energy)
    
    Special cases:
    - c = 0: Reduces to standard formula ω = 1/(1 + 2*energy) 
    - c = 0.5: ω = 0/(1 + 2*energy) = 0 (degenerate case)
    
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
    
    # General case: regularized polynomial formula
    return -(omega - 1.0) / (2.0 * omega) - c / omega


# =============================================================================
# COMPLETE REGULARIZED WORKFLOW FUNCTION
# =============================================================================

def compute_omega_from_file_regularized(
    input_file: str,
    regularization_c: float = 0.5,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    solution_precision: Optional[float] = None,
    save_raw_data: bool = False,
    save_submission: bool = False,
    job_name: Optional[str] = None
) -> Tuple[float, float, List[float], np.ndarray]:
    """
    Complete regularized workflow: Load file → Apply regularization → Submit to Dirac → Compute omega.
    
    Args:
        input_file: Path to QPLIB JSON or DIMACS file
        regularization_c: Regularization parameter for identity regularization
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
    input_path = Path(input_file)
    
    # Validate regularization parameter
    regularization_c = validate_regularization_parameter(regularization_c)
    
    # Determine file type and load/create regularized QPLIB data
    if input_path.suffix.lower() in ['.dimacs', '.clq']:
        print(f"📁 Converting DIMACS file to regularized QPLIB: {input_file}")
        print(f"   Regularization parameter: c = {regularization_c}")
        qplib_file = dimacs_to_regularized_qplib_file(input_file, regularization_c)
        qplib_data = load_qplib_file(qplib_file)
    elif input_path.suffix.lower() == '.json' or '.qplib' in input_path.name:
        print(f"📁 Loading QPLIB file and applying regularization: {input_file}")
        print(f"   Regularization parameter: c = {regularization_c}")
        qplib_data = load_qplib_file(input_file)
        qplib_data = apply_identity_regularization(qplib_data, regularization_c)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .json, .qplib.json, .dimacs, or .clq")
    
    # Transform to QCI format
    file_name = f"{input_path.stem}_regularized_c{regularization_c}_optimization"
    polynomial_file = qplib_to_polynomial_file(qplib_data, file_name)
    
    # Save polynomial submission file if requested
    if save_submission:
        submission_filename = f"{input_path.stem}_c{regularization_c}_dirac_submission.json"
        with open(submission_filename, 'w') as f:
            json.dump(polynomial_file, f, indent=2)
        print(f"💾 Saved Dirac submission file: {submission_filename}")
    
    # Submit to Dirac-3
    job_name = job_name or f"{input_path.stem}_regularized_omega_job"
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
        raw_file = f"dirac_response_regularized_c{regularization_c}_{timestamp}_{polynomial_file['file_config']['polynomial']['num_variables']}vars.json"
        with open(raw_file, 'w') as f:
            json.dump(job_response, f, indent=2)
        print(f"💾 Saved raw response to: {raw_file}")
    
    # Extract energy and compute omega using regularized formula
    best_energy, all_energies, best_solution = extract_best_energy(job_response)
    omega = regularized_energy_to_omega(best_energy, regularization_c)
    
    print(f"\n🎯 FINAL REGULARIZED RESULTS:")
    print(f"   Regularization parameter: c = {regularization_c}")
    print(f"   Best energy: {best_energy:.6f}")
    print(f"   Omega (ω): {omega:.3f}")
    print(f"   Solution vector shape: {best_solution.shape}")
    
    return omega, best_energy, all_energies, best_solution


# =============================================================================
# ENHANCED ENERGY HISTOGRAM PLOTTING (REUSED FROM ORIGINAL)
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


def add_theory_lines(ax, energies: List[float], regularization_c: float = 0.0,
                    show_theory_lines: bool = False, max_lines: int = 8) -> bool:
    """
    Add theoretical omega lines to the histogram based on regularized Motzkin-Straus formula.
    
    Args:
        ax: Matplotlib axes object
        energies: List of energy values from Dirac solver
        regularization_c: Regularization parameter for theory line calculation
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
    
    lines_added = 0
    omega_lines_added = []
    for omega in range(start_omega, end_omega + 1):
        energy_val = regularized_omega_to_energy(omega, regularization_c)
        
        # Check if theoretical energy is within reasonable range of actual data
        in_data_range = data_x_min <= energy_val <= data_x_max
        
        # Always draw lines for omega values derived from the energy data
        if in_data_range:
            ax.axvline(x=energy_val, color='#7f7f7f', linestyle='--', 
                      alpha=0.8, zorder=2, linewidth=1.5)
            ax.text(energy_val, y_max * 0.9, f' ω={omega}',
                   rotation=90, verticalalignment='bottom', 
                   color='#7f7f7f', fontsize=9, fontweight='bold')
            lines_added += 1
            omega_lines_added.append(omega)
            
            # Extend plot x-axis to ensure theoretical line is visible
            current_xlim = ax.get_xlim()
            new_x_min = min(current_xlim[0], energy_val*1.01)
            new_x_max = max(current_xlim[1], energy_val*0.99)
            ax.set_xlim(new_x_min, new_x_max)
    
    if lines_added > 0:
        # Add formula annotation in upper-left corner
        if regularization_c > 0:
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
        
        if omega_lines_added:
            omega_range_str = f"ω = {min(omega_lines_added)}" + (f" to {max(omega_lines_added)}" if len(omega_lines_added) > 1 else "")
            print(f"Added {lines_added} theoretical omega lines ({omega_range_str})")
        return True
    
    return False


def plot_energy_histogram(
    energies: List[float], 
    title: str, 
    best_energy: float,
    regularization_c: float,
    show_theory_lines: bool = False,
    save_path: Optional[str] = None
) -> bool:
    """
    Plot histogram of energies with regularization info and enhanced theoretical omega lines.
    
    Args:
        energies: List of energy values
        title: Plot title
        best_energy: Best (minimum) energy value
        regularization_c: Regularization parameter used
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
        add_theory_lines(plt.gca(), energies, regularization_c, show_theory_lines)
        
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title(f'{title} (Regularized c={regularization_c})\nSamples: {len(energies)}, Best Energy: {best_energy:.6f}')
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
    """Main function with CLI interface for regularized Motzkin-Straus optimization."""
    parser = argparse.ArgumentParser(
        description="Regularized Graph to Omega computation using direct QCI client API - supports DIMACS/CLQ and QPLIB formats with regularization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/regularized_graph_to_omega.py qplib_18.json
  python scripts/regularized_graph_to_omega.py DIMACS/triangle.dimacs --regularization-c 0.5 --num-samples 200
  python scripts/regularized_graph_to_omega.py graph.clq --regularization-c 0.3 --num-samples 100
  python scripts/regularized_graph_to_omega.py data.json --regularization-c 0.8 --relax-schedule 4 --format json
  python scripts/regularized_graph_to_omega.py input.dimacs --regularization-c 0.25 --solution-precision 0.001

Regularization Benefits:
  - Eliminates spurious solutions (non-clique local optima)
  - Ensures one-to-one correspondence between optima and cliques  
  - Makes optimization landscape strictly concave
  - More robust convergence to true maximum cliques
        """
    )
    
    parser.add_argument("input_file", help="Path to DIMACS/CLQ or QPLIB JSON file")
    parser.add_argument("--regularization-c", type=float, default=0.5,
                       help="Regularization parameter for identity regularization, must be in [0, 1] (default: 0.5)")
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
    parser.add_argument("--save-submission", action="store_true",
                       help="Save the polynomial file submitted to Dirac for inspection")
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
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    # Validate regularization parameter
    if args.regularization_c < 0 or args.regularization_c > 1:
        print(f"❌ Error: --regularization-c must be in [0, 1], got {args.regularization_c}")
        return 1
    
    print("🎯 Regularized Graph to Omega Computation via Direct QCI Client")
    print("=" * 70)
    print(f"Regularization: Identity matrix with c = {args.regularization_c}")
    print("Benefits: Eliminates spurious solutions, ensures clean clique correspondence")
    print("=" * 70)
    
    try:
        # Run complete regularized workflow
        omega, best_energy, all_energies, best_solution = compute_omega_from_file_regularized(
            input_file=args.input_file,
            regularization_c=args.regularization_c,
            num_samples=args.num_samples,
            relaxation_schedule=args.relax_schedule,
            solution_precision=args.solution_precision,
            save_raw_data=args.save_raw,
            save_submission=args.save_submission,
            job_name=args.job_name
        )
        
        # Generate output
        if args.format == "json":
            result = {
                "input_file": args.input_file,
                "regularization": {
                    "type": "identity",
                    "parameter_c": float(args.regularization_c)
                },
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
                "workflow": "Graph->Regularized_QCI->Dirac-3->Omega"
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*70}")
            print("FINAL REGULARIZED RESULTS")
            print(f"{'='*70}")
            print(f"Input file: {args.input_file}")
            print(f"Regularization: Identity matrix with c = {args.regularization_c}")
            print(f"Omega (ω): {omega:.3f}")
            print(f"Best energy: {best_energy:.6f}")
            print(f"Samples processed: {len(all_energies)}")
            print(f"Energy range: [{min(all_energies):.6f}, {max(all_energies):.6f}]")
            print(f"Workflow: Graph → Regularized QCI → Dirac-3 → Omega")
            print(f"{'='*70}")
        
        # Create energy histogram
        if not args.no_plot and MATPLOTLIB_AVAILABLE:
            file_stem = Path(args.input_file).stem
            plot_title = f"{file_stem}: Regularized Energy Distribution (Dirac-3)"
            save_path = f"energy_histogram_regularized_c{args.regularization_c}_{file_stem}.png" if args.save_raw else None
            
            plot_energy_histogram(
                energies=all_energies,
                title=plot_title,
                best_energy=best_energy,
                regularization_c=args.regularization_c,
                show_theory_lines=args.show_theory,
                save_path=save_path
            )
        
        print("\n🎉 Regularized computation completed successfully!")
        print("Note: Regularization eliminates spurious solutions for more reliable results.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())