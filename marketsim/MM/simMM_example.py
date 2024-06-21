from marketsim.MM.simMM import SimulatorSampledArrival_MM
import numpy as np
from absl import app
from absl import flags
import datetime
from marketsim.MM.utils import write_to_csv
from utils import replace_inf_with_nearest_2d
import os
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string("game_name", "LadderMM", "Game name.")
flags.DEFINE_string("root_result_folder", './root_result_static', "root directory of saved results")
flags.DEFINE_integer("num_iteration", 1, "num_iteration")

flags.DEFINE_integer("num_background_agents", 100, "Number of background agents.")
flags.DEFINE_integer("sim_time", int(1e4), "Simulation time.")
flags.DEFINE_float("lam", 5e-3, "Lambda.")
flags.DEFINE_float("lamMM", 5e-2, "Lambda MM.")
flags.DEFINE_float("mean", 1e5, "Mean.")
flags.DEFINE_float("r", 0.05, "Interest rate.")
flags.DEFINE_float("shock_var", 5e6, "Shock variance.")
flags.DEFINE_integer("q_max", 10, "Maximum quantity.")
flags.DEFINE_float("pv_var", 5e6, "PV variance.")
flags.DEFINE_list("shade", [250, 500], "Shade.")
flags.DEFINE_integer("xi", 100, "Rung size.")
flags.DEFINE_integer("omega", 10, "Spread.")
flags.DEFINE_integer("K", 10, "Number of levels - 1.")
flags.DEFINE_integer("n_levels", 11, "n_levels.")
flags.DEFINE_integer("total_volume", 50, "total_volume.")
flags.DEFINE_string("policy", None, "Policy.")
flags.DEFINE_boolean("beta_MM", True, "Beta MM.")
flags.DEFINE_boolean("inv_driven", False, "Inventory driven.")
flags.DEFINE_integer("w0", 5, "Initial wealth.")
flags.DEFINE_integer("p", 2, "Parameter p.")
flags.DEFINE_integer("k_min", 5, "Minimum k.")
flags.DEFINE_integer("k_max", 20, "Maximum k.")
flags.DEFINE_integer("max_position", 20, "Maximum position.")
flags.DEFINE_boolean("agents_only", False, "agents_only.")

# Beta Policy
flags.DEFINE_float("a_sell", 1, "a_sell.")
flags.DEFINE_float("b_sell", 2, "b_sell.")
flags.DEFINE_float("a_buy", 1, "a_buy.")
flags.DEFINE_float("b_buy", 2, "b_buy.")

def run(argv):
    # Set up working directory.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    seed = np.random.randint(0, 10000)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir = checkpoint_dir + "agonly_" + str(FLAGS.agents_only) + "se_" + str(
        seed) + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the original standard output
    # sys.stdout = open(checkpoint_dir + '/stdout.txt', 'w+')

    print("========== Parameters ==========")
    print(f"game_name: {FLAGS.game_name}")
    print(f"root_result_folder: {FLAGS.root_result_folder}")
    print(f"num_iteration: {FLAGS.num_iteration}")
    print(f"num_background_agents: {FLAGS.num_background_agents}")
    print(f"sim_time: {FLAGS.sim_time}")
    print(f"lam: {FLAGS.lam}")
    print(f"lamMM: {FLAGS.lamMM}")
    print(f"mean: {FLAGS.mean}")
    print(f"r: {FLAGS.r}")
    print(f"shock_var: {FLAGS.shock_var}")
    print(f"q_max: {FLAGS.q_max}")
    print(f"pv_var: {FLAGS.pv_var}")
    print(f"shade: {FLAGS.shade}")
    print(f"xi: {FLAGS.xi}")
    print(f"omega: {FLAGS.omega}")
    print(f"n_levels: {FLAGS.n_levels}")
    print(f"K: {FLAGS.K}")
    print(f"total_volume: {FLAGS.total_volume}")
    print(f"a_sell: {FLAGS.a_sell}")
    print(f"b_sell: {FLAGS.b_sell}")
    print(f"a_buy: {FLAGS.a_buy}")
    print(f"b_buy: {FLAGS.b_buy}")
    print(f"policy: {FLAGS.policy}")
    print(f"beta_MM: {FLAGS.beta_MM}")
    print(f"inv_driven: {FLAGS.inv_driven}")
    print(f"w0: {FLAGS.w0}")
    print(f"p: {FLAGS.p}")
    print(f"k_min: {FLAGS.k_min}")
    print(f"k_max: {FLAGS.k_max}")
    print(f"max_position: {FLAGS.max_position}")

    all_spreads, all_midprices, all_inventory, all_tq, all_MM_q , MM_values = [], [], [], [], [], []
    beta_params = {}
    beta_params["a_sell"] = FLAGS.a_sell
    beta_params["b_sell"] = FLAGS.b_sell
    beta_params["a_buy"] = FLAGS.a_buy
    beta_params["b_buy"] = FLAGS.b_buy


    sim = SimulatorSampledArrival_MM(num_background_agents = FLAGS.num_background_agents,
                                    sim_time = FLAGS.sim_time,
                                    lam = FLAGS.lam,
                                    lamMM = FLAGS.lamMM,
                                    mean = FLAGS.mean,
                                    r = FLAGS.r,
                                    shock_var = FLAGS.shock_var,
                                    q_max = FLAGS.q_max,
                                    pv_var = FLAGS.pv_var,
                                    shade = FLAGS.shade,
                                    xi = FLAGS.xi,
                                    omega = FLAGS.omega,
                                    K = FLAGS.K,
                                    n_levels=FLAGS.n_levels,
                                    total_volume=FLAGS.total_volume,
                                    beta_params = beta_params,
                                    policy = FLAGS.policy,
                                    beta_MM = FLAGS.beta_MM,
                                    inv_driven = FLAGS.inv_driven,
                                    w0 = FLAGS.w0,
                                    p = FLAGS.p,
                                    k_min = FLAGS.k_min,
                                    k_max = FLAGS.k_max,
                                    max_position = FLAGS.max_position
                                    )

    sim.reset()
    print("=============== START of SIM ================")
    for iteration in range(FLAGS.num_iteration):
        print("Running Simulation Iteration {}".format(iteration))
        if FLAGS.agents_only:
            stats = sim.run_agents_only(all_time_steps=True)
        else:
            stats = sim.run()

        sim.reset()
        all_spreads.append(stats["spreads"])
        all_midprices.append(stats["midprices"])
        all_inventory.append(stats["inventory"])
        all_tq.append(stats["total_quantity"])
        all_MM_q.append(stats["MM_quantity"])
        MM_values.append(stats["MM_value"])


    # Remove inf
    all_spreads = replace_inf_with_nearest_2d(all_spreads)
    all_midprices = replace_inf_with_nearest_2d(all_midprices)

    # Simulation Output
    average_spreads = np.mean(all_spreads, axis=0)
    average_midprices = np.mean(all_midprices, axis=0)
    average_inventory = np.mean(all_inventory, axis=0)
    average_tq = np.mean(all_tq)
    average_MM_q = np.mean(all_MM_q)
    average_values = np.mean(MM_values)

    print("Average Spreads:", np.mean(average_spreads))
    print("Average Midprices:", np.mean(average_midprices))
    print("Average Inventory:", np.mean(average_inventory))

    print("Average Total Quantity:", average_tq)
    print("Average MM Quantity:", average_MM_q)
    print("Average Values:", average_values)

    print("=============== END of SIM ================")


    # Save everything
    write_to_csv(checkpoint_dir + "/average_spreads.csv", average_spreads)
    write_to_csv(checkpoint_dir + "/average_midprices.csv", average_midprices)
    write_to_csv(checkpoint_dir + "/average_inventory.csv", average_inventory)
    # write_to_csv(average_trade_market_share + "/average_spreads.csv", average_spreads)
    # write_to_csv(average_values + "/average_spreads.csv", average_spreads)



if __name__ == "__main__":
    app.run(run)