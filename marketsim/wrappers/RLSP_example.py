import gymnasium
from tqdm import tqdm
from stable_baselines3 import SAC

from wrappers.SP_wrapper import SPEnv
import numpy as np
def run():
    normalizers = {"fundamental": 1e5, "order_price":1e5, "min_order_val": 1e5, "invt": 10, "cash": 1e7}

    env = SPEnv(num_background_agents=10,
            sim_time=6000,
            lam=5e-3,
            mean=1e5,
            r=0.05,
            shock_var=5e6,
            q_max=10,
            pv_var=5e6,
            shade=[250,500],
            normalizers=normalizers)

    model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0006)
    # Total timesteps: # of spoofer steps to learn on.
    model.learn(total_timesteps=100, log_interval=3)
    # model.save("sac_spoofer")

    # del model # remove to demonstrate saving and loading
    # model = SAC.load("sac_spoofer")

    obs, info = env.reset()
    sim_not_terminated = True
    value_agents = []
    for j in tqdm(range(10000)):
        for i in range(8000):
            while sim_not_terminated:
                # print("Current Iter:")
                # print("Internal steps:", env.time)
                action, _states = model.predict(obs, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                # print("Observations:", observation)
                # print("Reward:", reward)
                # print("terminated:", terminated)
                # print(env.markets[0].order_book.observe())
                # print("---------------------------")
                if terminated or truncated:
                    values = []
                    for agent_id in env.agents:
                        agent = env.agents[agent_id]
                        value = agent.get_pos_value() + agent.position * env.fundamental_value + agent.cash
                        # print(agent.cash, agent.position, agent.get_pos_value(), value)
                        values.append(value)
                    obs, info = env.reset()
                    sim_not_terminated = False
            value_agents.append(values)
    print(np.mean(value_agents, axis=0))
if __name__ == "__main__":
    run()