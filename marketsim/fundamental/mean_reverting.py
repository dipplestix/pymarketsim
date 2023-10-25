from fundamental import Fundamental
import torch


class GaussianMeanReverting(Fundamental):
    def __init__(self, final_time: int, mean: float, r: float, shock_var: float):
        self.final_time = final_time
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.r = torch.tensor(r, dtype=torch.float32)
        self.shock_std = torch.sqrt(torch.tensor(shock_var, dtype=torch.float32))
        self.fundamental_values = torch.zeros(final_time, dtype=torch.float32)
        self.fundamental_values[0] = mean
        self._generate()

    def _generate(self):
        shocks = torch.randn(self.final_time)*self.shock_std
        for t in range(1, self.final_time):
            self.fundamental_values[t] = (
                max(0, self.r*self.mean + (1 - self.r)*self.fundamental_values[t - 1] + shocks[t])
            )

    def get_value_at(self, time: int) -> float:
        return self.fundamental_values[time].item()

    def get_fundamental_values(self) -> torch.Tensor:
        return self.fundamental_values
