"""
GPU-accelerated private values generation and lookup.

This module provides GPU-based private value generation for all agents
across all parallel environments in a single operation.
"""

import cupy as cp


class GPUPrivateValues:
    """
    GPU-accelerated private values for vectorized market simulation.

    Generates private values for all agents across all environments on the GPU.
    Values are pre-sorted in descending order for proper value-for-exchange lookups.

    Attributes:
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        q_max: Maximum position quantity
        val_var: Variance for value generation
        values: GPU array of shape (num_envs, num_agents, 2*q_max) containing sorted private values
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        q_max: int,
        val_var: float = 5e6,
        seed: int = None,
    ):
        """
        Initialize GPU private values.

        Args:
            num_envs: Number of parallel environments
            num_agents: Number of agents per environment
            q_max: Maximum position quantity (values array size = 2*q_max)
            val_var: Variance for private value generation
            seed: Optional random seed for reproducibility
        """
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.q_max = q_max
        self.val_var = val_var
        self.pv_size = 2 * q_max

        # Set seed if provided
        if seed is not None:
            cp.random.seed(seed)

        # Generate random values: shape (num_envs, num_agents, 2*q_max)
        values = cp.random.randn(num_envs, num_agents, self.pv_size, dtype=cp.float32)
        values *= cp.sqrt(cp.float32(val_var))

        # Sort each agent's values in descending order along last axis
        # CuPy sort is ascending, so we negate, sort, and negate back
        values = -cp.sort(-values, axis=-1)

        self.values = values

        # Precompute extra values for boundary conditions
        # extra_buy = min(values[-1], 0) per agent
        # extra_sell = max(values[0], 0) per agent
        self.extra_buy = cp.minimum(values[:, :, -1], 0)  # (num_envs, num_agents)
        self.extra_sell = cp.maximum(values[:, :, 0], 0)  # (num_envs, num_agents)

    def value_for_exchange_vectorized(
        self,
        positions: cp.ndarray,
        sides: cp.ndarray,
    ) -> cp.ndarray:
        """
        Vectorized lookup of private values for all agents.

        Args:
            positions: GPU array of shape (num_envs, num_agents) with current positions
            sides: GPU array of shape (num_envs, num_agents) with order sides (1=BUY, -1=SELL)

        Returns:
            GPU array of shape (num_envs, num_agents) with private values
        """
        # Compute indices: position + q_max - (1 if SELL else 0)
        # sides: 1=BUY, -1=SELL
        # For SELL (side=-1): subtract 1
        # For BUY (side=1): subtract 0
        sell_offset = (sides == -1).astype(cp.int32)
        indices = positions + self.q_max - sell_offset

        # Handle boundary conditions
        # index >= pv_size: return extra_buy
        # index < 0: return extra_sell
        # otherwise: return values[index]

        # Clamp indices for safe indexing
        safe_indices = cp.clip(indices, 0, self.pv_size - 1)

        # Create environment and agent indices for advanced indexing
        env_idx = cp.arange(self.num_envs)[:, None]
        agent_idx = cp.arange(self.num_agents)[None, :]

        # Lookup values
        result = self.values[env_idx, agent_idx, safe_indices]

        # Apply boundary conditions
        result = cp.where(indices >= self.pv_size, self.extra_buy, result)
        result = cp.where(indices < 0, self.extra_sell, result)

        return result

    def reset(self, seed: int = None):
        """
        Reset/regenerate all private values.

        Args:
            seed: Optional random seed
        """
        if seed is not None:
            cp.random.seed(seed)

        values = cp.random.randn(self.num_envs, self.num_agents, self.pv_size, dtype=cp.float32)
        values *= cp.sqrt(cp.float32(self.val_var))
        values = -cp.sort(-values, axis=-1)

        self.values = values
        self.extra_buy = cp.minimum(values[:, :, -1], 0)
        self.extra_sell = cp.maximum(values[:, :, 0], 0)

    def reset_envs(self, env_mask: cp.ndarray, seed: int = None):
        """
        Reset private values for specific environments.

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
        new_values = cp.random.randn(num_reset, self.num_agents, self.pv_size, dtype=cp.float32)
        new_values *= cp.sqrt(cp.float32(self.val_var))
        new_values = -cp.sort(-new_values, axis=-1)

        # Update only the reset environments
        self.values[env_mask] = new_values
        self.extra_buy[env_mask] = cp.minimum(new_values[:, :, -1], 0)
        self.extra_sell[env_mask] = cp.maximum(new_values[:, :, 0], 0)

    def get_flat_values(self) -> cp.ndarray:
        """
        Get flattened values array for CUDA kernel.

        Returns:
            GPU array of shape (num_envs * num_agents * pv_size,)
        """
        return self.values.ravel()

    @property
    def memory_usage_mb(self) -> float:
        """Get approximate GPU memory usage in MB."""
        # values: num_envs * num_agents * pv_size * 4 bytes
        # extra_buy/sell: num_envs * num_agents * 4 bytes each
        values_bytes = self.values.nbytes
        extra_bytes = self.extra_buy.nbytes + self.extra_sell.nbytes
        return (values_bytes + extra_bytes) / (1024 * 1024)
