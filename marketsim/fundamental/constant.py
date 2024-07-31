import random
import torch

from fundamental_abc import Fundamental


class Constant(Fundamental):
    def __init__(self, final_time: int, value: float, random_seed: int = None):
               
        random.seed(random_seed)
        torch.manual_seed(random.randint(1, 4096))

        self.fundamental_values = torch.ones(final_time, dtype=torch.float32)*value

    def get_value_at(self, time: int) -> float:
        return self.fundamental_values[time].item()

    def get_fundamental_values(self) -> torch.Tensor:
        return self.fundamental_values
