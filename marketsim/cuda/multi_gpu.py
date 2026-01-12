"""
Multi-GPU support for CUDA market simulator.

This module provides orchestration for running simulations across
multiple GPUs in parallel.
"""

import concurrent.futures
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class MultiGPUSimulator:
    """
    Multi-GPU orchestrator for parallel market simulation.

    Distributes simulation environments across multiple GPUs and runs
    them in parallel using threading.

    Attributes:
        num_gpus: Number of GPUs to use
        envs_per_gpu: Number of environments per GPU
        simulators: List of CUDASimulator instances (one per GPU)
    """

    def __init__(
        self,
        num_gpus: int = None,
        envs_per_gpu: int = 1000,
        num_agents: int = 100,
        sim_time: int = 10000,
        **kwargs
    ):
        """
        Initialize multi-GPU simulator.

        Args:
            num_gpus: Number of GPUs to use (default: all available)
            envs_per_gpu: Number of environments per GPU
            num_agents: Number of agents per environment
            sim_time: Simulation timesteps
            **kwargs: Additional arguments passed to CUDASimulator
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for MultiGPUSimulator")

        # Detect available GPUs
        available_gpus = cp.cuda.runtime.getDeviceCount()

        if num_gpus is None:
            num_gpus = available_gpus
        elif num_gpus > available_gpus:
            raise ValueError(
                f"Requested {num_gpus} GPUs but only {available_gpus} available"
            )

        self.num_gpus = num_gpus
        self.envs_per_gpu = envs_per_gpu
        self.num_agents = num_agents
        self.sim_time = sim_time
        self.kwargs = kwargs

        # Create simulators on each GPU
        from .simulator import CUDASimulator

        self.simulators = []
        for gpu_id in range(num_gpus):
            with cp.cuda.Device(gpu_id):
                sim = CUDASimulator(
                    num_envs=envs_per_gpu,
                    num_agents=num_agents,
                    sim_time=sim_time,
                    device=gpu_id,
                    seed=kwargs.get('seed', None),
                    **{k: v for k, v in kwargs.items() if k != 'seed'}
                )
                # Adjust seed for each GPU if seed was provided
                if kwargs.get('seed') is not None:
                    sim.reset(seed=kwargs['seed'] + gpu_id * 10000)
                self.simulators.append(sim)

    @property
    def total_envs(self) -> int:
        """Total number of environments across all GPUs."""
        return self.num_gpus * self.envs_per_gpu

    def _run_on_gpu(self, gpu_id: int) -> Dict[str, Any]:
        """
        Run simulation on a specific GPU.

        Args:
            gpu_id: GPU device ID

        Returns:
            Simulation results
        """
        with cp.cuda.Device(gpu_id):
            return self.simulators[gpu_id].run()

    def run(self, max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Run simulations on all GPUs in parallel.

        Args:
            max_workers: Maximum number of worker threads (default: num_gpus)

        Returns:
            List of result dictionaries from each GPU
        """
        if max_workers is None:
            max_workers = self.num_gpus

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._run_on_gpu, gpu_id)
                for gpu_id in range(self.num_gpus)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        return results

    def run_and_aggregate(self) -> Dict[str, np.ndarray]:
        """
        Run simulations and aggregate results.

        Returns:
            Dictionary with concatenated results from all GPUs
        """
        results = self.run()

        # Aggregate results
        aggregated = {
            'positions': np.concatenate([r['positions'] for r in results], axis=0),
            'cash': np.concatenate([r['cash'] for r in results], axis=0),
            'total_matches': np.concatenate([r['total_matches'] for r in results]),
            'final_fundamental': np.concatenate([r['final_fundamental'] for r in results]),
        }

        return aggregated

    def reset(self, seed: int = None):
        """
        Reset all simulators.

        Args:
            seed: Optional random seed
        """
        for gpu_id, sim in enumerate(self.simulators):
            with cp.cuda.Device(gpu_id):
                if seed is not None:
                    sim.reset(seed=seed + gpu_id * 10000)
                else:
                    sim.reset()

    def verify_conservation(self) -> Dict[str, Any]:
        """
        Verify conservation laws across all GPUs.

        Returns:
            Dictionary with conservation check results per GPU
        """
        results = {}
        for gpu_id, sim in enumerate(self.simulators):
            with cp.cuda.Device(gpu_id):
                results[f'gpu_{gpu_id}'] = sim.verify_conservation()

        # Aggregate check
        all_position_ok = all(
            r['position_conservation']
            for r in results.values()
        )
        all_cash_ok = all(
            r['cash_conservation']
            for r in results.values()
        )

        results['all_conservation_ok'] = all_position_ok and all_cash_ok

        return results

    @property
    def memory_usage_mb(self) -> Dict[str, float]:
        """Get memory usage per GPU."""
        usage = {}
        total = 0
        for gpu_id, sim in enumerate(self.simulators):
            with cp.cuda.Device(gpu_id):
                mem = sim.memory_usage_mb
                usage[f'gpu_{gpu_id}'] = mem
                total += mem
        usage['total'] = total
        return usage

    def __repr__(self) -> str:
        return (
            f"MultiGPUSimulator(num_gpus={self.num_gpus}, "
            f"envs_per_gpu={self.envs_per_gpu}, "
            f"total_envs={self.total_envs})"
        )


def benchmark_multi_gpu(
    num_gpus: int = None,
    envs_per_gpu: int = 1000,
    num_agents: int = 100,
    sim_time: int = 10000,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark multi-GPU performance.

    Args:
        num_gpus: Number of GPUs to use
        envs_per_gpu: Environments per GPU
        num_agents: Agents per environment
        sim_time: Simulation steps
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    import time

    sim = MultiGPUSimulator(
        num_gpus=num_gpus,
        envs_per_gpu=envs_per_gpu,
        num_agents=num_agents,
        sim_time=sim_time,
    )

    # Warmup
    for _ in range(warmup_runs):
        sim.run()
        sim.reset()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        # Sync all GPUs
        for gpu_id in range(sim.num_gpus):
            with cp.cuda.Device(gpu_id):
                cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        sim.run()

        # Sync all GPUs
        for gpu_id in range(sim.num_gpus):
            with cp.cuda.Device(gpu_id):
                cp.cuda.Stream.null.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        sim.reset()

    times = np.array(times)
    total_steps = sim.total_envs * sim_time

    return {
        'num_gpus': sim.num_gpus,
        'total_envs': sim.total_envs,
        'mean_time': float(times.mean()),
        'std_time': float(times.std()),
        'min_time': float(times.min()),
        'max_time': float(times.max()),
        'steps_per_second': float(total_steps / times.mean()),
        'memory_usage': sim.memory_usage_mb,
    }


def print_gpu_scaling_results(results: List[Dict[str, Any]]):
    """
    Print GPU scaling benchmark results.

    Args:
        results: List of benchmark results with different GPU counts
    """
    print("\n" + "=" * 70)
    print("MULTI-GPU SCALING RESULTS")
    print("=" * 70)

    print(f"{'GPUs':<8} {'Envs':<10} {'Time (s)':<12} {'Steps/s':<15} {'Scaling':<10}")
    print("-" * 70)

    base_steps_per_second = None
    for result in results:
        if base_steps_per_second is None:
            base_steps_per_second = result['steps_per_second']
            scaling = 1.0
        else:
            scaling = result['steps_per_second'] / base_steps_per_second

        print(
            f"{result['num_gpus']:<8} "
            f"{result['total_envs']:<10} "
            f"{result['mean_time']:<12.3f} "
            f"{result['steps_per_second']:<15.0f} "
            f"{scaling:<10.2f}x"
        )

    print("=" * 70)
