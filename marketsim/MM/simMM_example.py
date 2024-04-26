from marketsim.MM.simMM import SimulatorSampledArrival_MM

surpluses = []

for _ in range(10):
    sim = SimulatorSampledArrival_MM(num_background_agents=25,
                                      sim_time=1200,
                                      lam=0.1,
                                      lamMM=0.2,
                                      mean=1e5,
                                      r=0.05,
                                      shock_var=5e6,
                                      q_max=10,
                                      pv_var=5e6,
                                      shade=[250,500],
                                      xi= 10,  # rung size
                                      omega= 10,  # spread,
                                      K = 4  # n_level - 1
                                      )

    sim.run()
    fundamental_val = sim.markets[0].get_final_fundamental()
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        if agent_id == len(sim.agents):
            value = agent.position * fundamental_val + agent.cash
        else:
            value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, agent.position, agent.get_pos_value(), value)
        values.append(value)
    print(f'At the end of the simulation we get {values}')
    surpluses.append(sum(values)/len(values))

print(sum(surpluses)/len(surpluses)*25)