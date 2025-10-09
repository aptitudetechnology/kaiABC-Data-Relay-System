# KaiABC Hypothesis Tests - Quick Start

## Installation

### 1. Install Python Dependencies

```bash
cd /home/chris/kaiABC-Data-Relay-System/hypothesis/tests
pip install -r requirements.txt
```

Or if you prefer conda:

```bash
conda install numpy scipy matplotlib
```

### 2. Verify Installation

```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__} installed ✓')"
```

## Running the MVP Test

### Quick Test (5 minutes)

```bash
cd /home/chris/kaiABC-Data-Relay-System/hypothesis/tests
python3 test_basin_volume.py
```

### Interactive Mode

```python
python3
>>> from test_basin_volume import *
>>> config = SimulationConfig(N=10, Q10=1.1, sigma_T=5.0)
>>> result = test_basin_volume(config, trials=50)
>>> print(f"Predicted: {result['V_predicted']:.2%}")
>>> print(f"Empirical: {result['V_empirical']:.2%}")
```

## Expected Runtime

- **MVP Test**: ~5 minutes (100 trials, N=10, 30 days simulation)
- **Full Parameter Sweep**: ~30 minutes (multiple Q10 and N values)
- **Complete Hypothesis Suite**: ~2-3 hours (all 6 tests)

## Minimal Requirements

**The MVP (`test_basin_volume.py`) only needs:**
- Python 3.7+
- NumPy

**Optional (for full suite):**
- Matplotlib (plotting)
- SciPy (statistical tests)
- tqdm (progress bars)

## Quick Dependency Check

```bash
python3 << 'EOF'
try:
    import numpy as np
    print("✓ NumPy installed")
except ImportError:
    print("✗ NumPy missing - install with: pip install numpy")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib installed")
except ImportError:
    print("⚠ Matplotlib missing (optional)")

try:
    import scipy
    print("✓ SciPy installed")
except ImportError:
    print("⚠ SciPy missing (optional)")
EOF
```

## Troubleshooting

### NumPy not found
```bash
pip3 install --user numpy
# or
sudo apt install python3-numpy
```

### Permission denied
```bash
chmod +x test_basin_volume.py
```

### Python version too old
```bash
python3 --version  # Should be 3.7+
```

## Next Steps After MVP

1. If MVP passes (error < 15%) → Run parameter sweep
2. If parameter sweep passes → Run full 6-hypothesis suite
3. If full suite passes (5/6) → Order hardware
4. If hardware syncs → Write paper! 📝
