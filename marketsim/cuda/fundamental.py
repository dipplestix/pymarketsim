"""
GPU-accelerated fundamental value generation.

This module provides GPU-based generation of mean-reverting fundamental values
for all parallel environments in a single operation.
"""

import cupy as cp


class GPUFundamental:
    """
    GPU-accelerated mean-reverting fundamental value generator.

    Precomputes all fundamental values for the entire simulation timeline
    across all parallel environments on the GPU.

    The fundamental follows a mean-reverting process:
        V_t = (1-r) * V_{t-1} + r * mean + shock_t

    Attributes:
        num_envs: Number of parallel environments
        sim_time: Total simulation timesteps
        mean: Long-term mean value
        r: Rate of mean reversion
        shock_var: Variance of Gaussian shocks
        values: GPU array of shape (num_envs, sim_time) with fundamental values
        rho_table: GPU array of shape (sim_time,) with precomputed (1-r)^(T-t) values
    """

    def __init__(
        self,
        num_envs: int,
        sim_time: int,
        mean: float,
        r: float,
        shock_var: float,
        seed: int = None,
    ):
        """
        Initialize GPU fundamental values.

        Args:
            num_envs: Number of parallel environments
            sim_time: Total simulation timesteps
            mean: Long-term mean value for mean reversion
            r: Rate of mean reversion (0 < r < 1)
            shock_var: Variance of Gaussian shocks
            seed: Optional random seed for reproducibility
        """
        self.num_envs = num_envs
        self.sim_time = sim_time
        self.mean = cp.float32(mean)
        self.r = cp.float32(r)
        self.shock_var = cp.float32(shock_var)
        self.shock_std = cp.sqrt(shock_var)
        self.one_minus_r = cp.float32(1.0 - r)

        # Set seed if provided
        if seed is not None:
            cp.random.seed(seed)

        # Precompute fundamental values for all environments and timesteps
        self.values = self._generate_all_values()

        # Precompute rho table: (1-r)^(T-t) for t in 0..sim_time-1
        self.rho_table = self._compute_rho_table()

    def _generate_all_values(self) -> cp.ndarray:
        """
        Generate all fundamental values using vectorized operations.

        Returns:
            GPU array of shape (num_envs, sim_time) with fundamental values
        """
        # Initialize with mean at t=0
        values = cp.empty((self.num_envs, self.sim_time), dtype=cp.float32)
        values[:, 0] = self.mean

        # Generate all shocks at once
        shocks = cp.random.randn(self.num_envs, self.sim_time - 1, dtype=cp.float32)
        shocks *= self.shock_std

        # Iterate through time (this loop is small, ~1000-50000 iterations)
        # Each iteration is fully vectorized across environments
        for t in range(1, self.sim_time):
            values[:, t] = (
                self.one_minus_r * values[:, t - 1] +
                self.r * self.mean +
                shocks[:, t - 1]
            )

        return values

    def _compute_rho_table(self) -> cp.ndarray:
        """
        Precompute rho values: (1-r)^(T-t) for all timesteps.

        This is used for fundamental estimate computation:
            estimate = (1-rho) * mean + rho * observed_value

        Returns:
            GPU array of shape (sim_time,) with rho values
        """
        t_values = cp.arange(self.sim_time, dtype=cp.float32)
        exponents = cp.float32(self.sim_time - 1) - t_values
        rho_table = cp.power(self.one_minus_r, exponents)
        return rho_table

    def get_values_at_time(self, t: int) -> cp.ndarray:
        """
        Get fundamental values at a specific timestep for all environments.

        Args:
            t: Timestep

        Returns:
            GPU array of shape (num_envs,) with fundamental values
        """
        return self.values[:, t]

    def get_final_values(self) -> cp.ndarray:
        """
        Get final fundamental values for all environments.

        Returns:
            GPU array of shape (num_envs,) with final fundamental values
        """
        return self.values[:, -1]

    def compute_estimates(self, t: int) -> cp.ndarray:
        """
        Compute fundamental estimates at timestep t for all environments.

        Uses precomputed rho values:
            estimate = (1-rho) * mean + rho * observed_value

        Args:
            t: Current timestep

        Returns:
            GPU array of shape (num_envs,) with estimates
        """
        rho = self.rho_table[t]
        return (1 - rho) * self.mean + rho * self.values[:, t]

    def reset(self, seed: int = None):
        """
        Reset/regenerate all fundamental values.

        Args:
            seed: Optional random seed
        """
        if seed is not None:
            cp.random.seed(seed)
        self.values = self._generate_all_values()

    def reset_envs(self, env_mask: cp.ndarray, seed: int = None):
        """
        Reset fundamental values for specific environments.

        Args:
            env_mask: Boolean array of shape (num_envs,) indicating which envs to reset
            seed: Optional random seed
        """
        if seed is not None:
            cp.random.seed(seed)

        num_reset = int(env_mask.sum())
        if num_reset == 0:
            return

        # Generate new values for reset environments
        new_values = cp.empty((num_reset, self.sim_time), dtype=cp.float32)
        new_values[:, 0] = self.mean

        shocks = cp.random.randn(num_reset, self.sim_time - 1, dtype=cp.float32)
        shocks *= self.shock_std

        for t in range(1, self.sim_time):
            new_values[:, t] = (
                self.one_minus_r * new_values[:, t - 1] +
                self.r * self.mean +
                shocks[:, t - 1]
            )

        self.values[env_mask] = new_values

    def get_info(self) -> tuple:
        """
        Get fundamental parameters.

        Returns:
            Tuple of (mean, r, sim_time)
        """
        return float(self.mean), float(self.r), self.sim_time

    @property
    def memory_usage_mb(self) -> float:
        """Get approximate GPU memory usage in MB."""
        values_bytes = self.values.nbytes
        rho_bytes = self.rho_table.nbytes
        return (values_bytes + rho_bytes) / (1024 * 1024)
