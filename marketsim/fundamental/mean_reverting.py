from fundamental import Fundamental
import numpy as np
from typing import List


class GaussianMeanReverting(Fundamental):
    def __init__(self, final_time, mean, r, shock_var):
        self.final_time = final_time
        self.mean = mean
        self.r = r
        self.shock_std = np.sqrt(shock_var)
        self.fundamental_values = []

        self._generate()

    def _generate(self):
        self.fundamental_values[0] = self.mean
        for t in range(1, self.final_time):
            shock = np.random.normal(0, self.shock_std)
            val = self.r*self.mean + (1-self.r)*self.fundamental_values[t - 1] + shock
            self.fundamental_values.append(val)

    def get_value_at(self, time: int) -> float:
        return self.fundamental_values[time]

    def get_fundamental_values(self) -> List[float]:
        return self.fundamental_values
