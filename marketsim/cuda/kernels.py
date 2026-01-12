"""
CUDA kernels for agent decision-making.

This module provides optimized kernels for computing agent orders.
Uses CuPy vectorized operations for portability, with optional raw CUDA
kernels for maximum performance when NVCC is available.
"""

import cupy as cp


def compute_orders_vectorized(
    estimates: cp.ndarray,
    positions: cp.ndarray,
    pv_values: cp.ndarray,
    shade_min: float,
    shade_max: float,
    q_max: int,
    best_bids: cp.ndarray = None,
    best_asks: cp.ndarray = None,
    eta: float = 1.0,
) -> tuple:
    """
    Compute orders for all agents using vectorized CuPy operations.

    Args:
        estimates: GPU array (num_envs,) with fundamental estimates
        positions: GPU array (num_envs, num_agents) with current positions
        pv_values: GPUPrivateValues object or GPU array (num_envs, num_agents, pv_size)
        shade_min: Minimum shade value
        shade_max: Maximum shade value
        q_max: Maximum position quantity
        best_bids: Optional GPU array (num_envs,) with best bid prices (for eta != 1)
        best_asks: Optional GPU array (num_envs,) with best ask prices (for eta != 1)
        eta: Aggressiveness parameter (1.0 = never take liquidity)

    Returns:
        Tuple of (prices, sides) GPU arrays of shape (num_envs, num_agents)
    """
    num_envs, num_agents = positions.shape
    pv_size = 2 * q_max

    # Random sides: 1=BUY, -1=SELL
    sides = cp.where(
        cp.random.random((num_envs, num_agents), dtype=cp.float32) < 0.5,
        cp.int32(1),
        cp.int32(-1)
    )

    # Random shade offsets
    spread = shade_max - shade_min
    offsets = cp.random.random((num_envs, num_agents), dtype=cp.float32) * spread + shade_min

    # Compute PV indices
    # For SELL (side=-1): position + q_max - 1
    # For BUY (side=1): position + q_max
    sell_offset = (sides == -1).astype(cp.int32)
    pv_indices = positions + q_max - sell_offset

    # Clamp indices
    pv_indices = cp.clip(pv_indices, 0, pv_size - 1)

    # Lookup private values
    # If pv_values is a GPUPrivateValues object, use its values array
    if hasattr(pv_values, 'values'):
        pv_array = pv_values.values
        extra_buy = pv_values.extra_buy
        extra_sell = pv_values.extra_sell
    else:
        pv_array = pv_values
        extra_buy = cp.minimum(pv_array[:, :, -1], 0)
        extra_sell = cp.maximum(pv_array[:, :, 0], 0)

    env_idx = cp.arange(num_envs)[:, None]
    agent_idx = cp.arange(num_agents)[None, :]

    pv = pv_array[env_idx, agent_idx, pv_indices]

    # Apply boundary conditions
    raw_indices = positions + q_max - sell_offset
    pv = cp.where(raw_indices >= pv_size, extra_buy, pv)
    pv = cp.where(raw_indices < 0, extra_sell, pv)

    # Compute prices
    # BUY: estimate + pv - offset
    # SELL: estimate + pv + offset
    estimates_expanded = estimates[:, None]  # (num_envs, 1)
    base_price = estimates_expanded + pv

    prices = cp.where(
        sides == 1,
        base_price - offsets,  # BUY
        base_price + offsets   # SELL
    )

    # Apply eta adjustment (aggressive taking)
    if eta != 1.0 and best_bids is not None and best_asks is not None:
        best_bids_exp = best_bids[:, None]
        best_asks_exp = best_asks[:, None]

        # For BUY: if (base_price - best_ask) > eta * offset and best_ask != inf, take ask
        buy_take_condition = (
            (sides == 1) &
            ((base_price - best_asks_exp) > eta * offsets) &
            (best_asks_exp != cp.inf)
        )
        prices = cp.where(buy_take_condition, best_asks_exp, prices)

        # For SELL: if (best_bid - base_price) > eta * offset and best_bid != inf, take bid
        sell_take_condition = (
            (sides == -1) &
            ((best_bids_exp - base_price) > eta * offsets) &
            (best_bids_exp != -cp.inf)
        )
        prices = cp.where(sell_take_condition, best_bids_exp, prices)

    return prices, sides


def update_positions_vectorized(
    positions: cp.ndarray,
    cash: cp.ndarray,
    matched: cp.ndarray,
    trade_prices: cp.ndarray,
    buyer_ids: cp.ndarray,
    seller_ids: cp.ndarray,
):
    """
    Update positions and cash after trades.

    Args:
        positions: GPU array (num_envs, num_agents) - modified in place
        cash: GPU array (num_envs, num_agents) - modified in place
        matched: GPU array (num_envs,) - whether match occurred
        trade_prices: GPU array (num_envs,) - trade prices
        buyer_ids: GPU array (num_envs,) - buyer agent IDs
        seller_ids: GPU array (num_envs,) - seller agent IDs
    """
    num_envs, num_agents = positions.shape

    # Create masks for buyers and sellers
    env_idx = cp.arange(num_envs)

    # Update buyers: position += 1, cash -= price
    for agent_id in range(num_agents):
        is_buyer = (buyer_ids == agent_id) & matched
        positions[is_buyer, agent_id] += 1
        cash[is_buyer, agent_id] -= trade_prices[is_buyer]

        is_seller = (seller_ids == agent_id) & matched
        positions[is_seller, agent_id] -= 1
        cash[is_seller, agent_id] += trade_prices[is_seller]


# Optimized position update using advanced indexing
def update_positions_fast(
    positions: cp.ndarray,
    cash: cp.ndarray,
    matched: cp.ndarray,
    trade_prices: cp.ndarray,
    buyer_ids: cp.ndarray,
    seller_ids: cp.ndarray,
):
    """
    Fast position/cash update using scatter operations.

    This version avoids the agent loop by using advanced indexing.
    """
    num_envs = positions.shape[0]

    # Get indices of matched environments
    matched_envs = cp.where(matched)[0]

    if len(matched_envs) == 0:
        return

    # Get buyer and seller IDs for matched environments
    matched_buyers = buyer_ids[matched_envs]
    matched_sellers = seller_ids[matched_envs]
    matched_prices = trade_prices[matched_envs]

    # Update positions using scatter_add equivalent
    # Buyers: +1 position, -price cash
    # Sellers: -1 position, +price cash

    # Use cupyx.scatter_add for atomic updates
    try:
        from cupyx import scatter_add

        # Create delta arrays
        pos_delta = cp.zeros_like(positions)
        cash_delta = cp.zeros_like(cash)

        # This approach creates index arrays for scatter operations
        buyer_indices = (matched_envs, matched_buyers)
        seller_indices = (matched_envs, matched_sellers)

        # Scatter add for positions
        scatter_add(pos_delta, buyer_indices, 1)
        scatter_add(pos_delta, seller_indices, -1)

        # Scatter add for cash
        scatter_add(cash_delta, buyer_indices, -matched_prices)
        scatter_add(cash_delta, seller_indices, matched_prices)

        positions += pos_delta
        cash += cash_delta

    except ImportError:
        # Fall back to loop version
        for i in range(len(matched_envs)):
            env = matched_envs[i]
            buyer = matched_buyers[i]
            seller = matched_sellers[i]
            price = matched_prices[i]

            positions[env, buyer] += 1
            cash[env, buyer] -= price
            positions[env, seller] -= 1
            cash[env, seller] += price
