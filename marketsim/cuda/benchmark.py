"""
Benchmark suite for CUDA GPU simulator.

Compares GPU performance against CPU baseline and provides comprehensive
benchmarking across various configurations.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
import warnings

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available. GPU benchmarks will be skipped.")


def run_cpu_baseline(
    num_agents: int,
    sim_time: int,
    arrival_rate: float = 0.005,
    **kwargs
) -> float:
    """
    Run CPU baseline simulation and return execution time.

    Args:
        num_agents: Number of agents
        sim_time: Simulation timesteps
        arrival_rate: Agent arrival rate
        **kwargs: Additional market parameters

    Returns:
        Execution time in seconds
    """
    from marketsim.market.market import Market
    from marketsim.agent.zero_intelligence_agent import ZIAgent
    from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

    # Default parameters
    mean = kwargs.get('mean', 1e5)
    r = kwargs.get('r', 0.05)
    shock_var = kwargs.get('shock_var', 1e6)
    q_max = kwargs.get('q_max', 10)
    shade = kwargs.get('shade', [0, 2])
    pv_var = kwargs.get('pv_var', 5e6)

    # Create market
    fundamental = LazyGaussianMeanReverting(
        final_time=sim_time,
        mean=mean,
        r=r,
        shock_var=shock_var
    )

    market = Market(
        fundamental=fundamental,
        time_steps=sim_time,
        n_agents=num_agents,
        lmbda=arrival_rate
    )

    # Create agents
    agents = []
    for i in range(num_agents):
        agent = ZIAgent(
            agent_id=i,
            market=market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var
        )
        agents.append(agent)
        market.register_agent(agent)

    # Run simulation
    start = time.perf_counter()
    market.run()
    elapsed = time.perf_counter() - start

    return elapsed


def run_gpu_benchmark(
    num_envs: int,
    num_agents: int,
    sim_time: int,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Run GPU benchmark and return timing statistics.

    Args:
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        sim_time: Simulation timesteps
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        **kwargs: Additional simulator parameters

    Returns:
        Dictionary with timing statistics
    """
    if not CUPY_AVAILABLE:
        return {'error': 'CuPy not available'}

    from .simulator import CUDASimulator

    sim = CUDASimulator(
        num_envs=num_envs,
        num_agents=num_agents,
        sim_time=sim_time,
        **kwargs
    )

    # Warmup
    for _ in range(warmup_runs):
        sim.run()
        sim.reset()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        sim.run()
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        sim.reset()

    times = np.array(times)
    total_steps = num_envs * sim_time

    return {
        'mean_time': float(times.mean()),
        'std_time': float(times.std()),
        'min_time': float(times.min()),
        'max_time': float(times.max()),
        'steps_per_second': float(total_steps / times.mean()),
        'env_steps_per_second': float(sim_time / times.mean()),
        'memory_mb': sim.memory_usage_mb,
    }


def benchmark_configurations() -> List[Dict[str, Any]]:
    """
    Get standard benchmark configurations.

    Returns:
        List of configuration dictionaries
    """
    return [
        # Small scale tests
        {'name': 'tiny', 'num_agents': 10, 'sim_time': 1000, 'arrival_rate': 0.005},
        {'name': 'small', 'num_agents': 25, 'sim_time': 5000, 'arrival_rate': 0.005},

        # Medium scale
        {'name': 'medium-small', 'num_agents': 50, 'sim_time': 10000, 'arrival_rate': 0.005},
        {'name': 'medium', 'num_agents': 100, 'sim_time': 10000, 'arrival_rate': 0.005},
        {'name': 'medium-long', 'num_agents': 100, 'sim_time': 25000, 'arrival_rate': 0.005},
        {'name': 'medium-vlong', 'num_agents': 100, 'sim_time': 50000, 'arrival_rate': 0.005},

        # Large scale
        {'name': 'large', 'num_agents': 200, 'sim_time': 25000, 'arrival_rate': 0.005},
        {'name': 'large-long', 'num_agents': 200, 'sim_time': 50000, 'arrival_rate': 0.005},
        {'name': 'xlarge', 'num_agents': 500, 'sim_time': 25000, 'arrival_rate': 0.005},

        # High activity
        {'name': 'high-activity', 'num_agents': 100, 'sim_time': 25000, 'arrival_rate': 0.01},
        {'name': 'high-activity-large', 'num_agents': 200, 'sim_time': 25000, 'arrival_rate': 0.01},

        # Stress test
        {'name': 'stress', 'num_agents': 500, 'sim_time': 50000, 'arrival_rate': 0.005},
    ]


def jaxmarl_configurations() -> List[Dict[str, Any]]:
    """
    Configurations matching JaxMARL-HFT benchmark table.

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for data_msgs in [1, 100]:
        for num_agents in [1, 5, 10]:
            configs.append({
                'name': f'{data_msgs}msg_{num_agents}agent',
                'num_agents': num_agents,
                'sim_time': 10000,
                'arrival_rate': data_msgs / 1000,  # Approximate mapping
            })

    return configs


def run_full_benchmark(
    num_envs: int = 1000,
    include_cpu: bool = True,
    configurations: List[Dict] = None,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> Dict[str, Any]:
    """
    Run full benchmark suite.

    Args:
        num_envs: Number of parallel GPU environments
        include_cpu: Whether to include CPU baseline
        configurations: List of configurations (default: standard configs)
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs

    Returns:
        Dictionary with all benchmark results
    """
    if configurations is None:
        configurations = benchmark_configurations()

    results = {
        'num_envs': num_envs,
        'configurations': {},
    }

    # Get GPU info
    if CUPY_AVAILABLE:
        from . import get_device_info
        results['gpu_info'] = get_device_info()

    for config in configurations:
        name = config['name']
        print(f"\nBenchmarking: {name}")
        print(f"  Agents: {config['num_agents']}, Steps: {config['sim_time']}")

        config_results = {}

        # CPU baseline (single environment)
        if include_cpu:
            print("  Running CPU baseline...", end=' ', flush=True)
            try:
                cpu_time = run_cpu_baseline(
                    num_agents=config['num_agents'],
                    sim_time=config['sim_time'],
                    arrival_rate=config['arrival_rate'],
                )
                config_results['cpu'] = {
                    'time': cpu_time,
                    'steps_per_second': config['sim_time'] / cpu_time,
                }
                print(f"{cpu_time:.3f}s ({config_results['cpu']['steps_per_second']:.0f} steps/s)")
            except Exception as e:
                config_results['cpu'] = {'error': str(e)}
                print(f"Error: {e}")

        # GPU benchmark
        if CUPY_AVAILABLE:
            print(f"  Running GPU ({num_envs} envs)...", end=' ', flush=True)
            try:
                gpu_results = run_gpu_benchmark(
                    num_envs=num_envs,
                    num_agents=config['num_agents'],
                    sim_time=config['sim_time'],
                    arrival_rate=config['arrival_rate'],
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                )
                config_results['gpu'] = gpu_results
                print(f"{gpu_results['mean_time']:.3f}s ({gpu_results['steps_per_second']:.0f} steps/s)")

                # Calculate speedup
                if 'cpu' in config_results and 'error' not in config_results['cpu']:
                    # Speedup per environment (GPU runs num_envs in parallel)
                    gpu_per_env = gpu_results['mean_time'] / num_envs
                    speedup = config_results['cpu']['time'] / gpu_per_env
                    config_results['speedup'] = speedup
                    print(f"  Speedup (per env): {speedup:.1f}x")

            except Exception as e:
                config_results['gpu'] = {'error': str(e)}
                print(f"Error: {e}")

        results['configurations'][name] = config_results

    return results


def print_results_table(results: Dict[str, Any]):
    """
    Print benchmark results in a formatted table.

    Args:
        results: Results from run_full_benchmark()
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    if 'gpu_info' in results:
        info = results['gpu_info']
        if info['available']:
            print(f"GPU: {info['devices'][0]['name']}")
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"Parallel Environments: {results['num_envs']}")
        print()

    # Table header
    print(f"{'Configuration':<25} {'CPU (s)':<12} {'GPU (s)':<12} {'GPU steps/s':<15} {'Speedup':<10}")
    print("-" * 80)

    for name, config_results in results['configurations'].items():
        cpu_time = '--'
        gpu_time = '--'
        gpu_steps = '--'
        speedup = '--'

        if 'cpu' in config_results and 'error' not in config_results['cpu']:
            cpu_time = f"{config_results['cpu']['time']:.3f}"

        if 'gpu' in config_results and 'error' not in config_results['gpu']:
            gpu_time = f"{config_results['gpu']['mean_time']:.3f}"
            gpu_steps = f"{config_results['gpu']['steps_per_second']:.0f}"

        if 'speedup' in config_results:
            speedup = f"{config_results['speedup']:.1f}x"

        print(f"{name:<25} {cpu_time:<12} {gpu_time:<12} {gpu_steps:<15} {speedup:<10}")

    print("=" * 80)


def run_statistical_validation(
    num_samples: int = 30,
    num_envs: int = 100,
    num_agents: int = 50,
    sim_time: int = 10000,
    arrival_rate: float = 0.005,
) -> Dict[str, Any]:
    """
    Run statistical validation comparing GPU vs CPU results.

    Args:
        num_samples: Number of samples to collect
        num_envs: Number of parallel GPU environments
        num_agents: Number of agents
        sim_time: Simulation steps
        arrival_rate: Arrival rate

    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats

    print(f"\nStatistical Validation")
    print(f"  Collecting {num_samples} samples...")

    # Collect CPU samples
    cpu_matches = []
    print("  CPU samples: ", end='', flush=True)
    for i in range(num_samples):
        from marketsim.market.market import Market
        from marketsim.agent.zero_intelligence_agent import ZIAgent
        from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

        fundamental = LazyGaussianMeanReverting(
            final_time=sim_time, mean=1e5, r=0.05, shock_var=1e6
        )
        market = Market(
            fundamental=fundamental,
            time_steps=sim_time,
            n_agents=num_agents,
            lmbda=arrival_rate
        )
        for j in range(num_agents):
            agent = ZIAgent(j, market, 10, [0, 2], 5e6)
            market.register_agent(agent)
        market.run()
        cpu_matches.append(market.matched_orders)
        print('.', end='', flush=True)
    print()

    # Collect GPU samples
    if not CUPY_AVAILABLE:
        return {'error': 'CuPy not available'}

    from .simulator import CUDASimulator

    print("  GPU samples: ", end='', flush=True)
    gpu_matches = []

    # Run multiple batches to get enough samples
    samples_per_run = num_envs
    runs_needed = (num_samples + samples_per_run - 1) // samples_per_run

    for run in range(runs_needed):
        sim = CUDASimulator(
            num_envs=samples_per_run,
            num_agents=num_agents,
            sim_time=sim_time,
            arrival_rate=arrival_rate,
            seed=run * 1000,
        )
        results = sim.run()
        gpu_matches.extend(results['total_matches'].tolist())
        print('.', end='', flush=True)
    print()

    gpu_matches = gpu_matches[:num_samples]

    # Statistical tests
    cpu_matches = np.array(cpu_matches)
    gpu_matches = np.array(gpu_matches)

    mann_whitney = stats.mannwhitneyu(cpu_matches, gpu_matches, alternative='two-sided')
    ks_test = stats.ks_2samp(cpu_matches, gpu_matches)
    t_test = stats.ttest_ind(cpu_matches, gpu_matches)

    results = {
        'cpu_stats': {
            'mean': float(cpu_matches.mean()),
            'std': float(cpu_matches.std()),
            'min': int(cpu_matches.min()),
            'max': int(cpu_matches.max()),
        },
        'gpu_stats': {
            'mean': float(gpu_matches.mean()),
            'std': float(gpu_matches.std()),
            'min': int(gpu_matches.min()),
            'max': int(gpu_matches.max()),
        },
        'tests': {
            'mann_whitney': {
                'statistic': float(mann_whitney.statistic),
                'p_value': float(mann_whitney.pvalue),
                'pass': mann_whitney.pvalue > 0.05,
            },
            'kolmogorov_smirnov': {
                'statistic': float(ks_test.statistic),
                'p_value': float(ks_test.pvalue),
                'pass': ks_test.pvalue > 0.05,
            },
            't_test': {
                'statistic': float(t_test.statistic),
                'p_value': float(t_test.pvalue),
                'pass': t_test.pvalue > 0.05,
            },
        },
    }

    # Print results
    print("\n  Results:")
    print(f"    CPU: mean={results['cpu_stats']['mean']:.1f}, std={results['cpu_stats']['std']:.1f}")
    print(f"    GPU: mean={results['gpu_stats']['mean']:.1f}, std={results['gpu_stats']['std']:.1f}")
    print(f"\n  Statistical Tests (α=0.05):")
    for test_name, test_results in results['tests'].items():
        status = "PASS" if test_results['pass'] else "FAIL"
        print(f"    {test_name}: p={test_results['p_value']:.3f} [{status}]")

    return results


if __name__ == '__main__':
    # Run benchmark when executed directly
    results = run_full_benchmark(num_envs=1000)
    print_results_table(results)

    # Run statistical validation
    validation = run_statistical_validation(num_samples=30)
