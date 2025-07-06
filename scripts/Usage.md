# Scripts Usage Guide

This directory contains scripts for computing maximum clique size (ω) using quantum annealing with Dirac-3 via the QCI client API.

## 🔧 Script Overview

### `dimacs_to_qplib.py`
Converts DIMACS format graph files to QPLIB JSON format for optimization.

### `qplib_to_omega.py` 
Computes ω from **QPLIB JSON files only**. Optimized for direct QPLIB input workflow.

### `graph_to_omega.py`
Universal script that accepts **both DIMACS and QPLIB JSON files**. Auto-detects format and handles conversion.

## 📁 Recommended Workflow

### 1. Convert DIMACS graphs to QPLIB (one-time setup)
```bash
# Create directory for converted files
mkdir -p QPLIBS

# Convert individual graphs
python scripts/dimacs_to_qplib.py DIMACS/triangle.dimacs QPLIBS/triangle.qplib.json --verbose

# Batch convert multiple graphs
for file in DIMACS/*.dimacs; do
    basename=$(basename "$file" .dimacs)
    python scripts/dimacs_to_qplib.py "$file" "QPLIBS/${basename}.qplib.json"
done
```

### 2. Compute ω from QPLIB files
```bash
# Using the QPLIB-only script (recommended for QPLIB files)
python scripts/qplib_to_omega.py QPLIBS/triangle.qplib.json --num-samples 100 --show-theory

# Using the universal script
python scripts/graph_to_omega.py QPLIBS/triangle.qplib.json --num-samples 100 --show-theory
```

### 3. Direct computation from DIMACS files
```bash
# Only possible with the universal script
python scripts/graph_to_omega.py DIMACS/triangle.dimacs --num-samples 100 --show-theory
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Quick test with small graph
python scripts/qplib_to_omega.py QPLIBS/triangle.qplib.json --num-samples 50

# Larger graph with enhanced parameters
python scripts/graph_to_omega.py DIMACS/erdos_renyi_15_p03_seed123.dimacs \
    --num-samples 200 \
    --relax-schedule 3 \
    --show-theory \
    --save-raw
```

### Advanced Configuration
```bash
# High precision computation
python scripts/qplib_to_omega.py QPLIBS/keller4.qplib.json \
    --num-samples 500 \
    --relax-schedule 4 \
    --solution-precision 0.001 \
    --show-theory \
    --save-raw \
    --format json

# Custom job naming and no plotting
python scripts/graph_to_omega.py DIMACS/brock200_1.clq \
    --num-samples 100 \
    --job-name "brock200_test" \
    --no-plot \
    --format table
```

### Theory Lines Visualization
```bash
# Enable theoretical omega lines on histograms
python scripts/graph_to_omega.py QPLIBS/erdos_renyi_10_p07_seed42.qplib.json \
    --num-samples 100 \
    --show-theory \
    --save-raw

# The --show-theory flag adds:
# • Theoretical energy lines for different ω values
# • Mathematical formula: E = -½(1-1/ω)
# • Automatic range detection and line limiting
```

## ⚡ **IMPORTANT: Energy Values**

**All energy values returned by the scripts are NEGATIVE.** This is the correct behavior:

- **Mathematical Background**: Motzkin-Straus maximizes f(x), but Dirac minimizes -f(x)
- **Expected Results**: 
  - Triangle graph: ω = 3, best energy ≈ -0.333
  - Larger graphs: ω varies, energies closer to -0.5
- **Zero Energies**: If you see all zero energies, there's a configuration issue

## 📊 Output Formats

### Table Format (default)
```
============================================================
FINAL RESULTS
============================================================
Input file: QPLIBS/triangle.qplib.json
Omega (ω): 3.000
Best energy: -0.333333
Samples processed: 100
Energy range: [-0.340000, -0.320000]
Workflow: QPLIB → QCI Client → Dirac-3 → Omega
============================================================
```

### JSON Format
```bash
python scripts/qplib_to_omega.py QPLIBS/triangle.qplib.json --format json
```
```json
{
  "qplib_file": "QPLIBS/triangle.qplib.json",
  "omega": 3.0,
  "best_energy": -0.333333,
  "num_samples": 100,
  "energy_statistics": {
    "min": -0.340000,
    "max": -0.320000,
    "mean": -0.331500,
    "std": 0.005234
  },
  "parameters": {
    "num_samples": 100,
    "relaxation_schedule": 2,
    "solution_precision": null
  },
  "workflow": "QPLIB->QCI->Dirac-3->Omega"
}
```

## ⚙️ Command Line Options

### Common Options (both scripts)
- `--num-samples N`: Number of Dirac samples (default: 100, max: 1000)
- `--relax-schedule N`: Relaxation schedule 1-4 (default: 2)
- `--solution-precision F`: Solution precision (optional)
- `--format {table,json}`: Output format (default: table)
- `--save-raw`: Save raw Dirac response to JSON file
- `--show-theory`: Show theoretical omega lines in histogram
- `--no-plot`: Disable energy histogram plotting
- `--job-name NAME`: Custom job name for Dirac submission

### Input Differences
- `qplib_to_omega.py`: Accepts only `*.json` or `*.qplib.json` files
- `graph_to_omega.py`: Accepts both `*.dimacs` and `*.json` files

## 🔬 Theory Lines Feature

The enhanced theory lines functionality shows theoretical energy values for different clique sizes:

- **Formula**: `E = -½(1-1/ω)` where ω is the clique size
- **Automatic Range**: Calculates relevant ω range from observed energies
- **Smart Limiting**: Maximum 8 lines to prevent visual clutter
- **Debug Output**: Shows energy range and theory line placement

Example theory line output:
```
Debug: Energy range: -0.340000 to -0.320000
Debug: Plot range: ω=2 to ω=4
Debug: ω=2: energy=-0.250000, in_data_range=False
Debug: ω=3: energy=-0.333333, in_data_range=True
Debug: ω=4: energy=-0.375000, in_data_range=True
Added 2 theoretical omega lines (ω = 3 to 4)
```

## 🚨 Prerequisites

1. **QCI Client**: Install with `pip install qci-client`
2. **Virtual Environment**: Activate with `source .venv/bin/activate`
3. **Dirac Access**: Valid QCI account with Dirac-3 allocation
4. **Dependencies**: matplotlib (optional, for plotting)

## 📁 File Organization

```
scripts/
├── dimacs_to_qplib.py      # DIMACS → QPLIB converter
├── qplib_to_omega.py       # QPLIB-only ω computation
├── graph_to_omega.py       # Universal ω computation
└── Usage.md               # This file

DIMACS/                     # Input DIMACS files
├── triangle.dimacs
├── erdos_renyi_*.dimacs
└── *.clq

QPLIBS/                     # Converted QPLIB files
├── triangle.qplib.json
├── erdos_renyi_*.qplib.json
└── *.qplib.json
```

## 🔧 Troubleshooting

### Common Issues

1. **Zero Energies**: Small graphs may converge to zero energy (ω = NaN)
   - Try larger graphs or different relaxation schedules
   - Use graphs with at least 10+ nodes for meaningful results

2. **Theory Lines Not Showing**: Requires negative energies
   - Zero or positive energies don't support theory line visualization
   - Use `--show-theory` flag and ensure graph produces negative energies

3. **QCI Client Errors**: Check Dirac allocation and credentials
   - Verify QCI account access and available compute time
   - Check network connectivity for API calls

4. **Plot Display Issues**: Matplotlib backend problems
   - Use `--no-plot` flag to disable plotting
   - Check display environment for plot rendering

### Performance Tips

- Start with small graphs (10-20 nodes) for testing
- Use `--save-raw` to preserve results for analysis
- Increase `--num-samples` for better statistics (up to 1000)
- Higher `--relax-schedule` values may improve solution quality