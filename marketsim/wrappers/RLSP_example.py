import gymnasium
from stable_baselines3 import SAC

from wrappers.SP_wrapper import SPEnv

def run():
    normalizers = {"fundamental": 5e5, "invt": 1e3, "cash": 1e5}

    env = SPEnv(num_background_agents=25,
            sim_time=1000,
            lam=0.1,
            mean=1e5,
            r=0.05,
            shock_var=5e6,
            q_max=10,
            pv_var=5e6,
            shade=[250,500],
            normalizers=normalizers)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)
    # model.save("sac_spoofer")

    # del model # remove to demonstrate saving and loading
    # model = SAC.load("sac_spoofer")

    obs, info = env.reset()
    while True:
        print("Current Iter:")
        print("Internal steps:", env.time)
        action, _states = model.predict(obs, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("=========================")
            observation, info = env.reset()
        print("Observations:", observation)
        print("Reward:", reward)
        print("terminated:", terminated)
        print(env.markets[0].order_book.observe())
        print("---------------------------")
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    run()