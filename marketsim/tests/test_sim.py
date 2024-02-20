from marketsim.simulator.simulator import Simulator
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival

sim = Simulator(num_agents=66, sim_time=60000, lam=1e-4, mean=1e7, r=.05, shock_var=1e6)
sim.run()



sim = SimulatorSampledArrival(num_agents=66, sim_time=60000, lam=1e-4, mean=1e7, r=.05, shock_var=1e6)
sim.run()



