# Installation Guide

This guide covers installing the Motzkin-Straus MIS Solver and its dependencies across different environments and use cases.

## Quick Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the fastest Python package manager and our recommended installation method:

```bash
# Clone the repository
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus

# Install with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Dependency Groups

The project uses dependency groups for different use cases:

### Core Dependencies (Default)

```bash
uv sync  # Installs core dependencies only
```

**Includes**:
- NetworkX for graph operations
- NumPy for numerical computing
- JAX for gradient-based optimization
- SciPy for scientific computing

### Documentation

```bash
uv sync --group docs
```

**Includes**:
- MkDocs with Material theme
- Mathematical extensions (KaTeX)
- Jupyter notebook support

### Development

```bash
uv sync --group dev
```

**Includes**:
- Testing frameworks (pytest)
- Code formatting (black, isort)
- Type checking (mypy)
- Linting (ruff)

### All Dependencies

```bash
uv sync --all-groups
```

## Optional Solver Dependencies

### Dirac-3 Quantum Computing

For quantum annealing capabilities:

```bash
pip install qci-client eqc-models
```

**Requirements**:
- QCi account with Dirac-3 allocation
- Valid authentication credentials

**Setup**:
```bash
# Configure QCi credentials
export QCI_TOKEN="your_token_here"
# Or create ~/.qci/config.json with authentication
```

### Gurobi Commercial Solver

For exact optimization:

```bash
pip install gurobipy
```

**Requirements**:
- Valid Gurobi license (academic or commercial)
- License file or network license server

**Setup**:
```bash
# Academic license
grbgetkey your_license_key

# Or set environment variable
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

## Platform-Specific Instructions

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev python3-venv git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus
uv sync
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and git
brew install python git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus
uv sync
```

### Windows

Using PowerShell:

```powershell
# Install Python from Microsoft Store or python.org
# Install Git from git-scm.com

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and install
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus
uv sync
```

## Hardware Acceleration

### GPU Support (NVIDIA)

For accelerated JAX operations:

```bash
# Install CUDA-enabled JAX
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU availability
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

### Apple Silicon (M1/M2)

JAX automatically uses Metal acceleration:

```bash
# Standard installation works
uv sync

# Verify Metal backend
python -c "import jax; print(f'Backend: {jax.default_backend()}')"
```

## Docker Installation

### Pre-built Image

```bash
# Pull and run the container
docker pull ghcr.io/your-org/motzkinstraus:latest
docker run -it --rm ghcr.io/your-org/motzkinstraus:latest
```

### Build from Source

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync --no-dev

ENTRYPOINT ["python", "-m", "motzkinstraus"]
```

```bash
# Build and run
docker build -t motzkinstraus .
docker run -it --rm motzkinstraus
```

## Conda/Mamba Installation

```bash
# Create environment
conda create -n motzkinstraus python=3.11
conda activate motzkinstraus

# Install from conda-forge
conda install -c conda-forge networkx numpy scipy jax

# Install package
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus
pip install -e .
```

## Development Installation

For contributing to the project:

```bash
# Clone your fork
git clone https://github.com/your-username/MotzkinStraus.git
cd MotzkinStraus

# Install all development dependencies
uv sync --all-groups

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

## Verification

### Basic Functionality Test

```python
import networkx as nx
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.algorithms import find_mis_with_oracle

# Create test graph
G = nx.cycle_graph(6)

# Test basic oracle
oracle = ProjectedGradientDescentOracle()
omega = oracle.get_omega(G)
print(f"Clique number: {omega}")  # Should be 2

# Test MIS algorithm
mis_set, calls = find_mis_with_oracle(G, oracle)
print(f"MIS size: {len(mis_set)}")  # Should be 3
print(f"Oracle calls: {calls}")
```

### Solver Availability Check

```python
from motzkinstraus.oracles import check_solver_availability

# Check which solvers are available
availability = check_solver_availability()
for solver, available in availability.items():
    status = "✓" if available else "✗"
    print(f"{status} {solver}")
```

### Performance Benchmark

```python
from motzkinstraus.benchmarks import quick_benchmark

# Run quick performance test
results = quick_benchmark(
    graph_sizes=[10, 20, 30],
    num_trials=3
)
print(f"Benchmark completed: {results}")
```

## Troubleshooting

### Common Installation Issues

#### JAX Installation Problems

```bash
# Clear JAX cache
rm -rf ~/.cache/jax_cache/

# Reinstall with specific version
pip install --upgrade "jax==0.4.20" "jaxlib==0.4.20"
```

#### Import Errors

```python
# Check Python path
import sys
print(sys.path)

# Verify installation
import pkg_resources
print(pkg_resources.get_distribution('motzkinstraus'))
```

#### Permission Issues (Linux/macOS)

```bash
# Fix ownership
sudo chown -R $USER:$USER ~/.local/

# Use user installation
pip install --user -e .
```

### Dependency Conflicts

```bash
# Check for conflicts
uv pip check

# Create clean environment
uv venv --clean
uv sync
```

### Performance Issues

```python
# Check JAX compilation
import jax
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Verify NumPy BLAS
import numpy as np
np.show_config()
```

## Next Steps

After successful installation:

1. **[Quick Start](quickstart.md)** - Basic usage examples
2. **[Examples](examples.md)** - Comprehensive examples
3. **[API Documentation](../api/oracles/overview.md)** - Detailed reference
4. **[Performance Tuning](../guides/performance-tuning.md)** - Optimization tips

## Getting Help

If you encounter installation issues:

- **Documentation**: Check the [troubleshooting guide](../guides/troubleshooting.md)
- **GitHub Issues**: [Report installation problems](https://github.com/your-org/MotzkinStraus/issues)
- **Discussions**: [Community support](https://github.com/your-org/MotzkinStraus/discussions)

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for installation, additional space for data
- **OS**: Linux, macOS, Windows (64-bit)

### Recommended Requirements

- **Python**: 3.11 or higher
- **RAM**: 16GB for large-scale problems
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: SSD for better I/O performance

### Tested Platforms

| Platform | Python Versions | Status |
|----------|-----------------|--------|
| Ubuntu 22.04 | 3.9, 3.10, 3.11, 3.12 | ✅ Fully supported |
| macOS 13+ | 3.10, 3.11, 3.12 | ✅ Fully supported |
| Windows 11 | 3.9, 3.10, 3.11 | ✅ Fully supported |
| CentOS 8 | 3.9, 3.10 | ⚠️ Limited testing |

---

**Ready to start?** Continue with the [Quick Start Guide](quickstart.md) to learn basic usage patterns.