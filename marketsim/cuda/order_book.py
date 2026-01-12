"""
GPU-accelerated order book with sorting-based matching.

This module provides a fully GPU-based order book implementation that uses
sorting for price-time priority matching. Optimized for ZI agents with
single-unit orders.
"""

import cupy as cp


class GPUOrderBook:
    """
    GPU-accelerated order book using sorting-based matching.

    This simplified implementation assumes:
    - Each agent submits at most one order per timestep
    - Orders are withdrawn at the start of each timestep (ZI behavior)
    - Single-unit orders only

    This allows for a much more efficient "clear and insert" approach.

    Attributes:
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
        bid_prices: GPU array (num_envs, num_agents) - bid prices (-inf for no order)
        ask_prices: GPU array (num_envs, num_agents) - ask prices (inf for no order)
    """

    def __init__(self, num_envs: int, max_orders: int):
        """
        Initialize GPU order book.

        Args:
            num_envs: Number of parallel environments
            max_orders: Maximum number of orders per side per environment (typically = num_agents)
        """
        self.num_envs = num_envs
        self.max_orders = max_orders

        # Simplified order storage: one slot per agent
        # For ZI agents, each agent has at most one active order
        self.bid_prices = cp.full((num_envs, max_orders), -cp.inf, dtype=cp.float32)
        self.bid_agents = cp.full((num_envs, max_orders), -1, dtype=cp.int32)

        self.ask_prices = cp.full((num_envs, max_orders), cp.inf, dtype=cp.float32)
        self.ask_agents = cp.full((num_envs, max_orders), -1, dtype=cp.int32)

        # Sorted views for matching
        self._bid_sorted_prices = cp.full((num_envs, max_orders), -cp.inf, dtype=cp.float32)
        self._bid_sorted_agents = cp.full((num_envs, max_orders), -1, dtype=cp.int32)
        self._ask_sorted_prices = cp.full((num_envs, max_orders), cp.inf, dtype=cp.float32)
        self._ask_sorted_agents = cp.full((num_envs, max_orders), -1, dtype=cp.int32)

    def clear(self):
        """Clear all orders from the book."""
        self.bid_prices.fill(-cp.inf)
        self.bid_agents.fill(-1)
        self.ask_prices.fill(cp.inf)
        self.ask_agents.fill(-1)

    def insert_orders_fast(
        self,
        prices: cp.ndarray,
        sides: cp.ndarray,
        arrivals: cp.ndarray,
    ):
        """
        Insert orders using fully vectorized operations.

        For arriving agents: withdraw existing order and place new one.
        For non-arriving agents: keep existing order.

        Args:
            prices: GPU array (num_envs, num_agents) with order prices
            sides: GPU array (num_envs, num_agents) with sides (1=BUY, -1=SELL)
            arrivals: GPU array (num_envs, num_agents) boolean mask of arriving agents
        """
        num_agents = prices.shape[1]
        agent_ids = cp.arange(num_agents)[None, :]  # (1, num_agents)

        # For arriving agents: clear their old orders and place new ones
        # For non-arriving agents: keep their existing orders

        # Clear old orders for arriving agents (they withdraw before placing new)
        self.bid_prices = cp.where(arrivals, -cp.inf, self.bid_prices)
        self.bid_agents = cp.where(arrivals, -1, self.bid_agents)
        self.ask_prices = cp.where(arrivals, cp.inf, self.ask_prices)
        self.ask_agents = cp.where(arrivals, -1, self.ask_agents)

        # Place new orders for arriving agents
        # Buy orders: where side == 1 and agent arrives
        buy_mask = (sides == 1) & arrivals
        self.bid_prices = cp.where(buy_mask, prices, self.bid_prices)
        self.bid_agents = cp.where(buy_mask, agent_ids, self.bid_agents)

        # Sell orders: where side == -1 and agent arrives
        sell_mask = (sides == -1) & arrivals
        self.ask_prices = cp.where(sell_mask, prices, self.ask_prices)
        self.ask_agents = cp.where(sell_mask, agent_ids, self.ask_agents)

        # Sort for matching
        self._sort_orders()

    def _sort_orders(self):
        """
        Sort orders by price priority.

        Bids: descending price (highest first)
        Asks: ascending price (lowest first)
        """
        # Sort bids by descending price
        # Negate to get descending order, then sort
        bid_sort_idx = cp.argsort(-self.bid_prices, axis=1)
        env_idx = cp.arange(self.num_envs)[:, None]
        self._bid_sorted_prices = self.bid_prices[env_idx, bid_sort_idx]
        self._bid_sorted_agents = self.bid_agents[env_idx, bid_sort_idx]

        # Sort asks by ascending price
        ask_sort_idx = cp.argsort(self.ask_prices, axis=1)
        self._ask_sorted_prices = self.ask_prices[env_idx, ask_sort_idx]
        self._ask_sorted_agents = self.ask_agents[env_idx, ask_sort_idx]

    def get_best_bid_ask(self) -> tuple:
        """
        Get best bid and ask prices for all environments.

        Returns:
            Tuple of (best_bid, best_ask) GPU arrays of shape (num_envs,)
        """
        return self._bid_sorted_prices[:, 0], self._ask_sorted_prices[:, 0]

    def match_single(self) -> tuple:
        """
        Match best crossing orders (single match per call).

        Returns:
            Tuple of:
                - matched: bool array (num_envs,) - whether match occurred
                - trade_prices: float array (num_envs,) - trade prices (0 if no match)
                - buyer_ids: int array (num_envs,) - buyer agent IDs (-1 if no match)
                - seller_ids: int array (num_envs,) - seller agent IDs (-1 if no match)
        """
        best_bid = self._bid_sorted_prices[:, 0]
        best_ask = self._ask_sorted_prices[:, 0]
        best_bid_agent = self._bid_sorted_agents[:, 0]
        best_ask_agent = self._ask_sorted_agents[:, 0]

        # Can only match if prices cross AND both sides have valid orders
        can_match = (best_bid >= best_ask) & (best_bid_agent >= 0) & (best_ask_agent >= 0)

        buyer_ids = cp.where(can_match, best_bid_agent, -1)
        seller_ids = cp.where(can_match, best_ask_agent, -1)
        trade_prices = cp.where(can_match, best_bid, cp.float32(0))

        # Clear matched orders from sorted arrays
        if can_match.any():
            # Shift remaining orders left
            self._bid_sorted_prices[can_match, :-1] = self._bid_sorted_prices[can_match, 1:]
            self._bid_sorted_prices[can_match, -1] = -cp.inf
            self._bid_sorted_agents[can_match, :-1] = self._bid_sorted_agents[can_match, 1:]
            self._bid_sorted_agents[can_match, -1] = -1

            self._ask_sorted_prices[can_match, :-1] = self._ask_sorted_prices[can_match, 1:]
            self._ask_sorted_prices[can_match, -1] = cp.inf
            self._ask_sorted_agents[can_match, :-1] = self._ask_sorted_agents[can_match, 1:]
            self._ask_sorted_agents[can_match, -1] = -1

        return can_match, trade_prices, buyer_ids, seller_ids

    def match_one(self) -> tuple:
        """
        Fast single-match operation for high-throughput simulation.

        Matches only the best crossing pair per environment, without
        iterating. This is faster for simulations where few orders
        cross per timestep.

        Returns:
            Tuple of:
                - matched: bool array (num_envs,) - whether match occurred
                - trade_prices: float array (num_envs,) - trade prices (0 if no match)
                - buyer_ids: int array (num_envs,) - buyer agent IDs (-1 if no match)
                - seller_ids: int array (num_envs,) - seller agent IDs (-1 if no match)
        """
        return self.match_single()

    def match_all(self, max_rounds: int = 3) -> tuple:
        """
        Match all crossing orders using vectorized batch matching.

        Args:
            max_rounds: Maximum number of matching rounds (limits iterations)

        Returns:
            Tuple of:
                - total_matches: int array (num_envs,) - total matches per env
                - all_buyer_ids: list of GPU arrays with buyer IDs per round
                - all_seller_ids: list of GPU arrays with seller IDs per round
                - all_prices: list of GPU arrays with trade prices per round
        """
        total_matches = cp.zeros(self.num_envs, dtype=cp.int32)
        all_buyers = []
        all_sellers = []
        all_prices = []

        # Match until no more crosses or max rounds reached
        # Limit rounds to avoid excessive Python loop overhead
        for _ in range(max_rounds):
            matched, prices, buyers, sellers = self.match_single()

            # Fast exit check using sum instead of any()
            num_matched = int(matched.sum())
            if num_matched == 0:
                break

            total_matches += matched.astype(cp.int32)
            all_buyers.append(buyers)
            all_sellers.append(sellers)
            all_prices.append(prices)

        return total_matches, all_buyers, all_sellers, all_prices

    @property
    def memory_usage_mb(self) -> float:
        """Get approximate GPU memory usage in MB."""
        total_bytes = (
            self.bid_prices.nbytes + self.bid_agents.nbytes +
            self.ask_prices.nbytes + self.ask_agents.nbytes +
            self._bid_sorted_prices.nbytes + self._bid_sorted_agents.nbytes +
            self._ask_sorted_prices.nbytes + self._ask_sorted_agents.nbytes
        )
        return total_bytes / (1024 * 1024)
