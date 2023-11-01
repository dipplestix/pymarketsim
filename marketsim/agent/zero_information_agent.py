from agent import Agent

BUY = 1
SELL = -1

class ZIAgent(Agent):
    def __init__(self, id: int, market: Market, q_max: int, offset: float, eta: float):
        self.id = id
        self.market = market
        self.pv = PrivateValues(q_max)
        self.position = 0
        self.offset = offset
        self.eta = eta

    def get_id(self) -> int:
        return self.id

    def estimate_fundamental(self, t):
        T = self.market.fundamental.get_final_time()
        r = self.market.fundamental.get_r()
        mean = self.market.fundamental.get_mean()
        val = self.market.fundamental.get_value_at(t)
        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        return estimate

    def take_action(self, side, t):
        estimate = self.estimate_fundamental(t)
        private_value = self.pv.value_for_exchange(self.position, side)


    def __str__(self):
        return f'ZI{self.id}'
