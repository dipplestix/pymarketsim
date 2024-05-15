from marketsim.wrappers.SP_wrapper import SPEnv

def run():

    normalizers = {"fundamental": 5e5, "invt": 1e3, "cash": 1e5}
    env = SPEnv(num_background_agents=25,
                sim_time=100,
                lam=0.1,
                mean=1e5,
                r=0.05,
                shock_var=5e6,
                q_max=10,
                pv_var=5e6,
                shade=[250,500],
                normalizers=normalizers)

    obs, info = env.reset()

    for i in range(200):
        print("Current Iter:", i)
        print("Internal steps:", env.time)
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        print("Observations:", observation)
        print("Reward:", reward)
        print("terminated:", terminated)
        print(env.markets[0].order_book.observe())
        print("---------------------------")


        if terminated or truncated:
            print("=========================")
            observation, info = env.reset()


if __name__ == "__main__":
    run()