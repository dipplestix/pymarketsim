from marketsim.MM.simMM import SimulatorSampledArrival_MM
import numpy as np


def run():
    values_over_sim = []

    for _ in range(10):
        sim = SimulatorSampledArrival_MM(num_background_agents=25,
                                          sim_time=1200,
                                          lam=1e-3,
                                          lamMM=5e-3,
                                          mean=1e5,
                                          r=0.05,
                                          shock_var=5e6,
                                          q_max=10,
                                          pv_var=5e6,
                                          shade=[250, 500],
                                          xi= 100,  # rung size
                                          omega= 64,  # spread,
                                          K = 100  # n_level - 1
                                          )

        sim.run()
        fundamental_val = sim.markets[0].get_final_fundamental()
        # print("fundamental:", fundamental_val)
        values = []
        for agent_id in sim.agents:
            agent = sim.agents[agent_id]
            if agent_id == len(sim.agents):
                value = agent.position * fundamental_val + agent.cash
            else:
                value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
            # print(agent.cash, agent.position, agent.get_pos_value(), value)
            values.append(value)
        values_over_sim.append(values)

    # Simulation Output

    average_values = np.mean(values_over_sim, axis=0)

    print("=============== END of SIM ================")
    print("MM Profit:", average_values[-1])
    print("SW:", sum(average_values))


if __name__ == "__main__":
    run()