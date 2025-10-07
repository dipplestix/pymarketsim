# PyMarketSim

PyMarketSim is a research-oriented sandbox for building and evaluating agent-based limit order book markets. The package provides reusable components for modeling fundamentals, simulating heterogeneous trading agents, and instrumenting the resulting market dynamics so that you can prototype new strategies or reinforcement-learning environments with minimal boilerplate.

## Key capabilities

- **Limit order book microstructure.** Matching is handled by a four-heap order book that respects price-time priority and exposes utilities for monitoring mid-prices and execution statistics.
- **Customizable fundamentals.** Plug in stochastic processes (for example mean-reverting Gaussian fundamentals) to drive the latent asset value observed by your agents.
- **Agent library.** Combine zero-intelligence, market making, informed, and noise agents or author your own policies by extending the base agent interface.
- **Event-driven simulation loop.** A discrete event queue coordinates order arrivals and market clearing, letting you scale to multiple agents and assets while keeping control over the simulation clock.
- **Reinforcement learning wrappers.** Gym-style wrappers make it straightforward to expose the simulator as an RL environment for training custom policies and benchmarking existing ones.

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies and package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

This registers the `marketsim` package locally so you can import it from notebooks or scripts.

## Quick start

The snippet below runs a short background-agent simulation with a mean-reverting fundamental process. It demonstrates how to instantiate the core components and iterate the simulator.

```python
from marketsim.simulator.simulator import Simulator

sim = Simulator(
    num_background_agents=50,
    sim_time=1_000,
    num_assets=1,
    lam=0.1,           # arrival intensity for background agents
    mean=100.0,        # long-run fundamental value
    r=0.6,             # mean-reversion strength
    shock_var=10.0,    # volatility of fundamental shocks
    q_max=10,          # max order size for background agents
)

sim.run()

# Inspect market statistics once the run completes
market = sim.markets[0]
mid_prices = market.get_midprices()
matched_orders = market.matched_orders
```

You can replace or augment the background agents with your own implementations by subclassing `marketsim.agent.agent.Agent` and registering instances in `sim.agents`. Fundamentals are swappable as long as they implement the `marketsim.fundamental.fundamental_abc.Fundamental` interface.

## Working with agents and markets

- **Agents:** Agent policies live under `marketsim/agent`. They encapsulate order submission logic via a `take_action` method and maintain inventory through helper utilities like `update_position`. Use the provided zero-intelligence and market-making agents as blueprints for new behaviors.
- **Markets:** The `Market` class manages the event queue, order book, and matching process. At each step it ingests orders from agents, clears the book, and updates mid-prices so you can compute downstream metrics.
- **Fundamentals:** Mean-reverting fundamentals offer a simple default latent value process. Implement `get_value_at` and `get_info` to introduce new information structures.

## Reinforcement learning workflows

The `marketsim.wrappers` package contains ready-to-use wrappers that expose the simulator through stable, vectorized interfaces. They provide observation builders, reward functions, and benchmarking utilities so you can plug the environment into RL pipelines with minimal glue code. Explore the examples in `marketsim/wrappers/examples` for end-to-end demonstrations of training or evaluating custom agents.

## Testing and notebooks

Lightweight regression tests live under `marketsim/tests`, and exploratory notebooks (such as `test_sim.ipynb` and `marketsim/intro_notebook.ipynb`) showcase typical analysis workflows. Running these notebooks is a good way to familiarize yourself with the API before embedding the simulator into your own research projects.

## Contributing

1. Fork the repository and create a feature branch.
2. Install the development dependencies listed in `requirements.txt`.
3. Ensure unit tests pass before submitting a pull request.
4. Describe your changes clearly and include references to any new strategies or environments you add.

## License

PyMarketSim is distributed under the MIT License. See `LICENSE.txt` for details.
