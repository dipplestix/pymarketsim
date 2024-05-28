# %%
from lazy_mean_reverting import LazyGaussianMeanReverting
from mean_reverting import GaussianMeanReverting

def estimate_fundamental(fun:GaussianMeanReverting):
    mean, r, T = 1e5,0.05,10000
    t = 9990
    rho = (1 - r) ** (T - t)

    estimate = (1 - rho) * mean + rho * fun.get_value_at(t)
    return estimate
# %%
vals = []
estimates = []
for i in range(10000):
    fun = LazyGaussianMeanReverting(final_time=10000, mean=1e5, r=0.05, shock_var=1e6)
    print(i)
    vals.append(fun.get_value_at(9950))
    estimates.append(estimate_fundamental(fun))
mean = sum(vals)/len(vals)
mean_estimates = sum(estimates)/len(estimates)
print(mean)
print(mean_estimates)
print(abs(mean_estimates - 1e5)/mean_estimates)
print(abs(mean - 1e5)/mean)
input()

# %%
vals = []
for _ in range(100000):
    fun = GaussianMeanReverting(final_time=1000, mean=1e5, r=0.05, shock_var=1e6)
    fun.fundamental_values[10] = 2000
    fun.latest_t = 10
    vals.append(fun.get_value_at(30))
rho = (1-.05)**(30-10)

estimate = (1-rho)*1e5 + rho*2000

mean = sum(vals)/len(vals)
print(abs(mean - estimate)/mean)

# %%
vals = []
for _ in range(100000):
    fun = LazyGaussianMeanReverting(final_time=1000, mean=1e5, r=0.05, shock_var=1e6)
    fun.fundamental_values[10] = 2000
    fun.latest_t = 10
    vals.append(fun.get_final_fundamental())
rho = (1-.05)**(1000-10)

estimate = (1-rho)*1e5 + rho*2000
mean = sum(vals)/len(vals)
print(abs(mean - estimate)/mean)

# %%
import matplotlib.pyplot as plt
from scipy.stats import normaltest

plt.hist(vals)

# %%



