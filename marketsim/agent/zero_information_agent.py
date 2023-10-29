from agent import Agent


class ZIAgent(Agent):
    def __init__(self, id: int, market, q_max):
        self.id = id
        self.market = market
        self.pv = PrivateValues(q_max)

    def get_id(self) -> int:
        return self.id

    def estimate_fundamental(self, t):
        T = self.market.fundamental.get_final_time()

    def __str__(self):
        return f'ZI{self.id}'
