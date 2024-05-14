import gymnasium
import os
from tqdm import tqdm
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from wrappers.SP_wrapper import SPEnv
import numpy as np
from typing import Callable


def make_env(spEnv: SPEnv, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> SPEnv:
        env = spEnv
        env.reset(seed=seed + rank)
        return env

    seed = 0
    return _init

def run():

    # eval_log_dir = "./eval_logs/"
    # os.makedirs(eval_log_dir, exist_ok=True)

    normalizers = {"fundamental": 1e5, "reward":1e4, "min_order_val": 1e5, "invt": 10, "cash": 1e7}

    num_cpu = 4

    spEnv = SPEnv(num_background_agents=15,
            sim_time=10000,
            lam=8e-3,
            lamSP=5e-2,
            mean=1e5,
            r=0.05,
            shock_var=5e6,
            q_max=10,
            pv_var=5e6,
            shade=[250,500],
            normalizers=normalizers)

    eval_env = SPEnv(num_background_agents=15,
            sim_time=10000,
            lam=5e-3,
            lamSP=5e-2,
            mean=1e5,
            r=0.05,
            shock_var=5e6,
            q_max=10,
            pv_var=5e6,
            shade=[250,500],
            normalizers=normalizers)

    # env = SubprocVecEnv([make_env(spEnv, i) for i in range(num_cpu)])

    # eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
    #                           log_path=eval_log_dir, eval_freq=1e4,
    #                           n_eval_episodes=5, deterministic=True,
    #                           render=False)

# train_freq=1, gradient_steps=2,
    model = SAC("MlpPolicy", spEnv,  verbose=1)
    # model = PPO("MlpPolicy", spEnv, verbose=1)
    # Total timesteps: # of spoofer steps to learn on.
    # model.learn(total_timesteps=1e6, log_interval=3)
    model.learn(total_timesteps=1e6)
    # model.learn(total_timesteps=5e5, log_interval=3)
    # model.save("sac_spoofer")

    # del model # remove to demonstrate saving and loading
    # model = SAC.load("sac_spoofer")

    # obs, info = env.reset()
    obs, info = spEnv.reset()
    sim_not_terminated = True
    value_agents = []
    #eval_episodes = num simulations?
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
    # for j in tqdm(range(10000)):
    #     for i in range(8000):
    #         while sim_not_terminated:
    #             # print("Current Iter:")
    #             # print("Internal steps:", env.time)
    #             action, _states = model.predict(obs, deterministic=True)
    #             observation, reward, terminated, truncated, info = env.step(action)
    #             # print("Observations:", observation)
    #             # print("Reward:", reward)
    #             # print("terminated:", terminated)
    #             # print(env.markets[0].order_book.observe())
    #             # print("---------------------------")
    #             if terminated or truncated:
    #                 values = []
    #                 for agent_id in env.agents:
    #                     agent = env.agents[agent_id]
    #                     value = agent.get_pos_value() + agent.position * env.fundamental_value + agent.cash
    #                     # print(agent.cash, agent.position, agent.get_pos_value(), value)
    #                     values.append(value)
    #                 obs, info = env.reset()
    #                 sim_not_terminated = False
    #         value_agents.append(values)
    # print(np.mean(value_agents, axis=0))
if __name__ == "__main__":
    run()