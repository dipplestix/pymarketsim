import torch
from fundamental_abc import Fundamental


class LazyGaussianMeanReverting(Fundamental):
    def __init__(self, final_time: int, mean: float, r: float, shock_var: float, shock_mean: float = 0):
        self.final_time = final_time
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.shock_mean = shock_mean
        self.shock_std = torch.sqrt(torch.tensor(shock_var, dtype=torch.float32))
        self.fundamental_values = {0: mean}
        self.latest_t = 0

    def _generate_at(self, t: int):
        dt = t - self.latest_t

        shocks = torch.randn(dt) * self.shock_std + self.shock_mean
        weights = torch.pow(1 - self.r, torch.arange(dt, dtype=torch.float32))
        total_shock = torch.sum(weights * shocks)
        value_at_t = (
                torch.pow(1 - self.r, dt) * self.fundamental_values[self.latest_t] +
                self.r * self.mean * torch.sum(weights) +
                total_shock
        )

        self.fundamental_values[t] = value_at_t
        self.latest_t = t

    def get_value_at(self, time: int) -> float:
        if time not in self.fundamental_values:
            self._generate_at(time)
        return self.fundamental_values[time]

    def get_fundamental_values(self):
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        return self.get_value_at(self.final_time)

    def get_r(self) -> float:
        return self.r.item()

    def get_mean(self) -> float:
        return self.mean.item()

    def get_info(self):
        return self.get_mean(), self.get_r(), self.final_time
