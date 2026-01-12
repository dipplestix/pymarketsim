"""
CUDA GPU-accelerated market simulator.

This module provides the main CUDASimulator class that runs fully
GPU-accelerated market simulations.
"""

import cupy as cp
import numpy as np
from typing import Optional, Dict, Any

from .fundamental import GPUFundamental
from .private_values import GPUPrivateValues
from .order_book import GPUOrderBook
from .kernels import compute_orders_vectorized, update_positions_fast


class CUDASimulator:
    """
    Fully GPU-accelerated market simulator.

    Runs multiple parallel market simulations entirely on the GPU.
    Supports ZI (Zero Intelligence) agents with configurable parameters.

    Target performance:
        - >25,000 steps/s (100 agents) on RTX 4090
        - >400,000 steps/s (1 agent) on RTX 4090

    Attributes:
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        sim_time: Total simulation timesteps
        positions: GPU array (num_envs, num_agents) current positions
        cash: GPU array (num_envs, num_agents) current cash
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        sim_time: int,
        # Agent parameters
        q_max: int = 10,
        shade: tuple = (0, 2),
        pv_var: float = 5e6,
        eta: float = 1.0,
        # Fundamental parameters
        mean: float = 1e5,
        r: float = 0.05,
        shock_var: float = 1e6,
        # Arrival rate
        arrival_rate: float = 0.005,
        # Order book
        max_orders: int = None,
        # Random seed
        seed: int = None,
        # Device
        device: int = 0,
    ):
        """
        Initialize CUDA simulator.

        Args:
            num_envs: Number of parallel environments
            num_agents: Number of agents per environment
            sim_time: Total simulation timesteps
            q_max: Maximum position quantity
            shade: Tuple of (min, max) shade values
            pv_var: Private value variance
            eta: Aggressiveness parameter (1.0 = passive)
            mean: Fundamental mean value
            r: Mean reversion rate
            shock_var: Fundamental shock variance
            arrival_rate: Agent arrival rate (probability per timestep)
            max_orders: Maximum orders per side (default: num_agents)
            seed: Random seed
            device: CUDA device ID
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.sim_time = sim_time
        self.q_max = q_max
        self.shade = shade
        self.pv_var = pv_var
        self.eta = eta
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.arrival_rate = arrival_rate
        self.device = device

        if max_orders is None:
            max_orders = num_agents  # Each agent can have at most one order

        self.max_orders = max_orders

        # Set device
        cp.cuda.Device(device).use()

        # Set seed
        if seed is not None:
            cp.random.seed(seed)

        # Initialize state arrays
        self.positions = cp.zeros((num_envs, num_agents), dtype=cp.int32)
        self.cash = cp.zeros((num_envs, num_agents), dtype=cp.float32)
        self.current_time = 0

        # Initialize components
        self.fundamental = GPUFundamental(
            num_envs=num_envs,
            sim_time=sim_time,
            mean=mean,
            r=r,
            shock_var=shock_var,
            seed=seed,
        )

        self.private_values = GPUPrivateValues(
            num_envs=num_envs,
            num_agents=num_agents,
            q_max=q_max,
            val_var=pv_var,
            seed=seed + 1 if seed is not None else None,
        )

        self.order_book = GPUOrderBook(
            num_envs=num_envs,
            max_orders=max_orders,
        )

        # Precompute arrival times using geometric distribution
        self._precompute_arrivals(seed + 2 if seed is not None else None)

        # Statistics tracking
        self.total_matches = cp.zeros(num_envs, dtype=cp.int32)
        self.midprices = []

    def _precompute_arrivals(self, seed: int = None):
        """
        Precompute agent arrival times using geometric distribution.

        This is more efficient than sampling per-step.
        """
        if seed is not None:
            cp.random.seed(seed)

        # For each agent in each env, precompute arrival times
        # Use geometric distribution with p = arrival_rate
        # We need enough arrivals to cover sim_time

        # Estimate max arrivals per agent
        expected_arrivals = int(self.sim_time * self.arrival_rate * 2)
        max_arrivals = max(expected_arrivals, 100)

        # Generate inter-arrival times
        self.arrival_times = []

        # For efficiency, generate arrival times as a mask per timestep
        # arrivals[t] contains which agents arrive at time t
        self.arrival_mask = cp.random.random(
            (self.sim_time, self.num_envs, self.num_agents),
            dtype=cp.float32
        ) < self.arrival_rate

    def step(self):
        """
        Execute one simulation step.

        1. Get arriving agents
        2. Compute fundamental estimates
        3. Compute orders for arriving agents
        4. Insert orders into book (clears previous orders)
        5. Match orders (single match for speed)
        6. Update positions/cash
        """
        t = self.current_time

        if t >= self.sim_time:
            return

        # Get which agents arrive this timestep
        arrivals = self.arrival_mask[t]  # (num_envs, num_agents)

        # Compute fundamental estimates
        estimates = self.fundamental.compute_estimates(t)  # (num_envs,)

        # Compute orders for all agents
        prices, sides = compute_orders_vectorized(
            estimates=estimates,
            positions=self.positions,
            pv_values=self.private_values,
            shade_min=self.shade[0],
            shade_max=self.shade[1],
            q_max=self.q_max,
            best_bids=None,  # Simplified: no eta adjustment for speed
            best_asks=None,
            eta=1.0,
        )

        # Insert orders (clears previous and inserts new in one operation)
        self.order_book.insert_orders_fast(prices, sides, arrivals)

        # Fast matching: single match per timestep for speed
        matched, trade_prices, buyer_ids, seller_ids = self.order_book.match_one()

        # Update positions and cash using vectorized scatter
        # Create one-hot encoding for buyer/seller and use broadcasting
        buyer_onehot = (cp.arange(self.num_agents)[None, :] == buyer_ids[:, None]).astype(cp.int32)
        seller_onehot = (cp.arange(self.num_agents)[None, :] == seller_ids[:, None]).astype(cp.int32)
        matched_exp = matched[:, None].astype(cp.int32)

        # Position updates: buyers +1, sellers -1
        self.positions += buyer_onehot * matched_exp
        self.positions -= seller_onehot * matched_exp

        # Cash updates: buyers pay, sellers receive
        price_exp = trade_prices[:, None] * matched_exp
        self.cash -= buyer_onehot * price_exp
        self.cash += seller_onehot * price_exp

        # Count matches
        self.total_matches += matched.astype(cp.int32)
        self.current_time += 1

    def run(self, progress: bool = False) -> Dict[str, Any]:
        """
        Run the full simulation.

        Args:
            progress: Whether to show progress bar

        Returns:
            Dictionary with simulation results
        """
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(self.sim_time), desc="Simulating")
            except ImportError:
                iterator = range(self.sim_time)
        else:
            iterator = range(self.sim_time)

        for _ in iterator:
            self.step()

        # Synchronize GPU
        cp.cuda.Stream.null.synchronize()

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results.

        Returns:
            Dictionary with:
                - positions: final positions (numpy array)
                - cash: final cash (numpy array)
                - total_matches: matches per environment (numpy array)
                - fundamental_values: final fundamental values (numpy array)
        """
        return {
            'positions': cp.asnumpy(self.positions),
            'cash': cp.asnumpy(self.cash),
            'total_matches': cp.asnumpy(self.total_matches),
            'final_fundamental': cp.asnumpy(self.fundamental.get_final_values()),
        }

    def reset(self, seed: int = None):
        """
        Reset the simulator for a new run.

        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            cp.random.seed(seed)

        self.positions.fill(0)
        self.cash.fill(0)
        self.current_time = 0
        self.total_matches.fill(0)
        self.midprices = []

        self.fundamental.reset(seed)
        self.private_values.reset(seed + 1 if seed is not None else None)
        self.order_book.clear()
        self._precompute_arrivals(seed + 2 if seed is not None else None)

    def verify_conservation(self) -> Dict[str, bool]:
        """
        Verify conservation laws are satisfied.

        Returns:
            Dictionary with conservation check results
        """
        positions_sum = cp.asnumpy(self.positions.sum(axis=1))
        cash_sum = cp.asnumpy(self.cash.sum(axis=1))

        return {
            'position_conservation': np.allclose(positions_sum, 0),
            'cash_conservation': np.allclose(cash_sum, 0),
            'max_position_deviation': float(np.abs(positions_sum).max()),
            'max_cash_deviation': float(np.abs(cash_sum).max()),
        }

    @property
    def memory_usage_mb(self) -> float:
        """Get total GPU memory usage in MB."""
        state_bytes = self.positions.nbytes + self.cash.nbytes
        fundamental_bytes = self.fundamental.memory_usage_mb * 1024 * 1024
        pv_bytes = self.private_values.memory_usage_mb * 1024 * 1024
        order_book_bytes = self.order_book.memory_usage_mb * 1024 * 1024
        arrival_bytes = self.arrival_mask.nbytes

        total_bytes = state_bytes + fundamental_bytes + pv_bytes + order_book_bytes + arrival_bytes
        return total_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"CUDASimulator(num_envs={self.num_envs}, "
            f"num_agents={self.num_agents}, "
            f"sim_time={self.sim_time}, "
            f"memory={self.memory_usage_mb:.1f}MB)"
        )


def benchmark_simulator(
    num_envs: int = 1000,
    num_agents: int = 100,
    sim_time: int = 1000,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark the CUDA simulator.

    Args:
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        sim_time: Simulation timesteps
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        **kwargs: Additional arguments for CUDASimulator

    Returns:
        Dictionary with benchmark results
    """
    import time

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
        'envs_per_second': float(num_envs / times.mean()),
    }
