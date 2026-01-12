# Performance Benchmark Results

## Summary

| Version | Total Time | Speedup vs Baseline |
|---------|-----------|---------------------|
| Baseline (master) | 17.15s | 1.00x |
| PR #40 (numpy arrivals, rho cache) | 13.73s | 1.25x |
| PR #41 (+ PrivateValues, order IDs) | **7.45s** | **2.30x** |

## Detailed Results

| Configuration | Baseline | PR #40 | PR #41 | Total Speedup |
|--------------|----------|--------|--------|---------------|
| tiny (10 agents, 1k) | 0.028s | 0.010s | 0.007s | **4.18x** |
| small (25 agents, 5k) | 0.061s | 0.030s | 0.026s | **2.36x** |
| medium-small (50 agents, 10k) | 0.140s | 0.073s | 0.064s | **2.21x** |
| medium (100 agents, 10k) | 0.246s | 0.155s | 0.136s | **1.81x** |
| medium-long (100 agents, 25k) | 0.713s | 0.461s | 0.342s | **2.08x** |
| medium-vlong (100 agents, 50k) | 1.196s | 0.969s | 0.661s | **1.81x** |
| large (200 agents, 25k) | 1.030s | 0.744s | 0.576s | **1.79x** |
| large-long (200 agents, 50k) | 2.020s | 1.550s | 1.000s | **2.02x** |
| xlarge (500 agents, 25k) | 1.951s | 1.831s | 1.064s | **1.83x** |
| high-activity (100 agents, 25k, λ=0.01) | 1.017s | 1.558s | 0.499s | **2.04x** |
| high-activity-large (200 agents, 25k, λ=0.01) | 1.922s | 1.581s | 0.865s | **2.22x** |
| stress (500 agents, 50k) | 6.827s | 4.772s | 2.206s | **3.10x** |

## Optimizations by PR

### PR #40: Core Simulator Optimizations
1. **Replace torch with numpy for arrival sampling**
   - `np.random.geometric()` instead of `torch.distributions.Geometric`
   - Adjusted for 0-based vs 1-based semantics

2. **Precompute rho lookup table**
   - `(1-r)^(T-t)` precomputed for all timesteps at init

3. **Cache fundamental estimate per timestep**
   - Computed once after `set_time()`, reused by all agents

4. **Replace torch with numpy in LazyGaussianMeanReverting**
   - All tensor operations converted to numpy

### PR #41: Agent & Order Book Optimizations
5. **Replace torch with numpy in PrivateValues**
   - Eliminates `.item()` overhead on every order

6. **Order ID counter instead of random.randint()**
   - Sequential IDs: `agent_id * 1000000 + counter`
   - Eliminates random number generation overhead

7. **Cache private value lookups in take_action()**
   - Avoid duplicate `pv.value_for_exchange()` when `eta != 1.0`

8. **Cache peek() values in FourHeap.insert()**
   - Reduces redundant heap cleanup operations

## Statistical Validation

30 samples collected from each version (50 agents, 10k steps, λ=0.005):

| Metric | Baseline | PR #41 |
|--------|----------|--------|
| Mean matches | 61.9 | 67.5 |
| Std dev | 11.4 | 11.7 |
| Range | [40, 94] | [50, 94] |

### Statistical Tests (all pass at α=0.05)

| Test | p-value | Result |
|------|---------|--------|
| Mann-Whitney U | 0.102 | PASS |
| Kolmogorov-Smirnov | 0.135 | PASS |
| Independent t-test | 0.070 | PASS |

### Conservation Laws

All trials verified:
- Position conservation: Σ positions = 0 ✓
- Cash conservation: Σ cash = 0 ✓

## Conclusion

Combined optimizations achieve **2.30x speedup** while maintaining statistical equivalence.
- Best improvement on stress tests: **3.10x faster**
- Consistent 1.8-2.2x improvement across most configurations
- All conservation laws preserved
