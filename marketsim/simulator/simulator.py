from ..market import Market
from ..fundamental import Fundamental

class Simulator:
    def __init__(self, num_agents, num_assets=1):
        self.num_agents = num_agents
        self.num_assets = num_assets

        self.markets = []
        for _ in range(num_assets):
            self.markets.append(Market())

        self.agents = []
