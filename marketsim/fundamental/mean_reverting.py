import random
import torch
from .fundamental_abc import Fundamental


class GaussianMeanReverting(Fundamental):
    def __init__(self, final_time: int, mean: float, r: float, shock_var: float, shock_mean: float = 0, random_seed: int = None):
        
        random.seed(random_seed)
        torch.manual_seed(random.randint(1, 4096))

        self.final_time = final_time
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.shock_mean = shock_mean
        self.shock_std = torch.sqrt(torch.tensor(shock_var, dtype=torch.float32))
        self.fundamental_values = torch.zeros(final_time, dtype=torch.float32)
        self.fundamental_values[0] = mean
        self._generate()

    def _generate(self):
        shocks = torch.randn(self.final_time)*self.shock_std + self.shock_mean
        for t in range(1, self.final_time):
            self.fundamental_values[t] = (
                max(0, self.r*self.mean + (1 - self.r)*self.fundamental_values[t - 1] + shocks[t])
            )

    def get_value_at(self, time: int) -> float:
        return self.fundamental_values[time].item()

    def get_fundamental_values(self) -> torch.Tensor:
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        return self.fundamental_values[-1].item()

    def get_r(self) -> float:
        return self.r.item()

    def get_mean(self) -> float:
        return self.mean.item()

    def get_info(self):
        return self.get_mean(), self.get_r(), self.final_time
