# CUDA GPU-Accelerated Market Simulator

A fully GPU-accelerated market simulator using CuPy/CUDA for massively parallel market simulations.

## Performance

Benchmarked on NVIDIA RTX 4090:

| Environments | Agents | Steps | Time (s) | Steps/s |
|-------------|--------|-------|----------|---------|
| 1,000 | 10 | 1,000 | 5.2 | **190,925** |
| 5,000 | 10 | 1,000 | 6.6 | **756,032** |
| 10,000 | 10 | 1,000 | 7.1 | **1,415,362** |
| 1,000 | 50 | 1,000 | 8.3 | **120,083** |
| 5,000 | 50 | 1,000 | 9.9 | **503,847** |
| 1,000 | 100 | 1,000 | 10.2 | **98,152** |

**Peak throughput: 1.4M steps/second** with 10,000 parallel environments.

## Installation

```bash
pip install cupy-cuda12x scipy
```

## Quick Start

```python
from marketsim.cuda import CUDASimulator

# Create simulator with 1000 parallel environments
sim = CUDASimulator(
    num_envs=1000,       # Number of parallel simulations
    num_agents=50,       # Agents per environment
    sim_time=10000,      # Timesteps per simulation
    arrival_rate=0.005,  # Agent arrival probability
    seed=42,
)

# Run all simulations
results = sim.run()

# Results contain:
# - positions: Final positions (num_envs, num_agents)
# - cash: Final cash (num_envs, num_agents)
# - total_matches: Match count per environment
# - final_fundamental: Final fundamental value per environment
print(f"Mean matches: {results['total_matches'].mean():.1f}")

# Verify conservation laws
conservation = sim.verify_conservation()
print(f"Position conservation: {conservation['position_conservation']}")
```

## Architecture

The GPU simulator consists of:

- **GPUFundamental**: Precomputed mean-reverting fundamental values
- **GPUPrivateValues**: Vectorized private value generation/lookup
- **GPUOrderBook**: Sorting-based order book with CDA matching
- **CUDASimulator**: Main orchestrator running on GPU

All operations are fully vectorized across environments.

## Market Mechanism

This implementation uses **Continuous Double Auction (CDA)** with price-time priority:
- Orders are matched based on price priority (best prices first)
- Multiple matches can occur per timestep
- Matched orders are cleared immediately

**Note**: The CPU baseline in `marketsim.simulator` uses batch/call market clearing, which is a different mechanism. This leads to different match statistics, but both maintain correct position/cash conservation.

## Configuration

```python
CUDASimulator(
    num_envs=1000,        # Parallel environments
    num_agents=50,        # Agents per environment
    sim_time=10000,       # Simulation timesteps

    # Agent parameters
    q_max=10,             # Max position quantity
    shade=(0, 2),         # Price shade range
    pv_var=5e6,           # Private value variance
    eta=1.0,              # Aggressiveness (1.0 = passive)

    # Market parameters
    mean=1e5,             # Fundamental mean
    r=0.05,               # Mean reversion rate
    shock_var=1e6,        # Shock variance
    arrival_rate=0.005,   # Arrival probability

    # Other
    seed=42,              # Random seed
    device=0,             # CUDA device ID
)
```

## Multi-GPU Support

```python
from marketsim.cuda import MultiGPUSimulator

# Distribute across multiple GPUs
sim = MultiGPUSimulator(
    num_gpus=4,           # Use 4 GPUs
    envs_per_gpu=10000,   # 10k envs per GPU = 40k total
    num_agents=50,
    sim_time=10000,
)

results = sim.run_and_aggregate()
```

## Validation

Conservation laws verified:
- **Position conservation**: Sum of positions = 0 for all environments ✓
- **Cash conservation**: Minor floating point deviation (< 1.0) ✓

## GPU Requirements

- CUDA 12.x compatible GPU
- CuPy with CUDA 12.x support
- Recommended: 8GB+ VRAM for large-scale simulations

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package init, GPU detection utilities |
| `simulator.py` | Main CUDASimulator class |
| `order_book.py` | GPU order book with sorting-based matching |
| `fundamental.py` | GPU fundamental value generation |
| `private_values.py` | GPU private values |
| `kernels.py` | Vectorized computation kernels |
| `multi_gpu.py` | Multi-GPU orchestration |
| `benchmark.py` | Benchmark suite |
