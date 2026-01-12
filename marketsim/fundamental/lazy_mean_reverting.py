import numpy as np
from marketsim.fundamental.fundamental_abc import Fundamental


class LazyGaussianMeanReverting(Fundamental):
    """
    A class representing a fundamental value that follows a mean-reverting process with Gaussian shocks.

    Args:
        final_time (int): The final time step of the process.
        mean (float): The long-term mean value that the process reverts to.
        r (float): The rate of mean reversion.
        shock_var (float): The variance of the Gaussian shocks.
        shock_mean (float, optional): The mean of the Gaussian shocks. Default is 0.
    """
    def __init__(self, final_time: int, mean: float, r: float, shock_var: float, shock_mean: float = 0):
        self.final_time = final_time
        self.mean = float(mean)
        self.r = float(r)
        self.shock_mean = shock_mean
        self.shock_std = np.sqrt(shock_var)
        self.shock_var = shock_var
        self.fundamental_values = {0: mean}
        self.latest_t = 0
        # Cache for (1-r)^k values to avoid repeated computation
        self._one_minus_r = 1.0 - self.r

    def _generate_at(self, t: int):
        """
        Generate the fundamental value at a specific time step.

        Args:
            t (int): The time step to generate the value for.
        """
        dt = t - self.latest_t

        shocks = np.random.randn(dt) * self.shock_std + self.shock_mean
        weights = np.power(self._one_minus_r, np.arange(dt, dtype=np.float64))
        total_shock = np.sum(weights * shocks)
        value_at_t = (
                np.power(self._one_minus_r, dt) * self.fundamental_values[self.latest_t] +
                self.r * self.mean * np.sum(weights) +
                total_shock
        )

        self.fundamental_values[t] = float(value_at_t)
        self.latest_t = t

    def get_value_at(self, time: int) -> float:
        """
        Get the fundamental value at a specific time step.

        Args:
            time (int): The time step to retrieve the value for.

        Returns:
            float: The fundamental value at the specified time step.
        """
        if time not in self.fundamental_values:
            self._generate_at(time)
        return self.fundamental_values[time]

    def get_fundamental_values(self):
        """
        Get the entire dictionary of fundamental values.

        Returns:
            Dict[int, float]: The dictionary of fundamental values.
        """
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        """
        Get the fundamental value at the final time step.

        Returns:
            float: The fundamental value at the final time step.
        """
        return self.get_value_at(self.final_time)

    def get_r(self) -> float:
        """
        Get the rate of mean reversion.

        Returns:
            float: The rate of mean reversion.
        """
        return self.r

    def get_mean(self) -> float:
        """
        Get the long-term mean value.

        Returns:
            float: The long-term mean value.
        """
        return self.mean

    def get_info(self):
        """
        Get the mean, rate of mean reversion, and final time step.

        Returns:
            Tuple[float, float, int]: A tuple containing the mean, rate of mean reversion, and final time step.
        """
        return self.get_mean(), self.get_r(), self.final_time
