# Define the list of variables
game_name_list = ["LadderMM"]
root_result_folder_list = ["./root_result_static_beta"]
num_iteration_list = [10000]
num_background_agents_list = [100]
sim_time_list = [100000]
lam_list = [5e-3]
lamMM_list = [5e-2]
omega_list = [10, 30, 60]
K_list = [10]
n_levels_list = [11]
total_volume_list = [50]
beta_MM_list = ["True"]
inv_driven_list = ["False"]
beta_params = [
    "--a_sell=1 --b_sell=1, --a_buy=1, b_buy=1",
    "--a_sell=1 --b_sell=2, --a_buy=1, b_buy=2",
    "--a_sell=1 --b_sell=5, --a_buy=1, b_buy=5",
    "--a_sell=2 --b_sell=1, --a_buy=2, b_buy=1",
    "--a_sell=2 --b_sell=2, --a_buy=2, b_buy=2",
    "--a_sell=2 --b_sell=5, --a_buy=2, b_buy=5",
    "--a_sell=5 --b_sell=1, --a_buy=5, b_buy=1",
    "--a_sell=5 --b_sell=2, --a_buy=5, b_buy=2",
    "--a_sell=5 --b_sell=5, --a_buy=5, b_buy=5"
]
w0_list = [5]
p_list = [2]
k_min_list = [5]
k_max_list = [20]
max_position_list = [20]
agents_only_list = ["False"]

file_name = "../run_static_beta.sh"

# Generate the bash script content
bash_script_content = ""

for game_name in game_name_list:
    for root_result_folder in root_result_folder_list:
        for num_iteration in num_iteration_list:
            for num_background_agents in num_background_agents_list:
                for sim_time in sim_time_list:
                    for lam in lam_list:
                        for lamMM in lamMM_list:
                            for omega in omega_list:
                                for K in K_list:
                                    for n_levels in n_levels_list:
                                        for total_volume in total_volume_list:
                                            for beta_MM in beta_MM_list:
                                                for inv_driven in inv_driven_list:
                                                    for w0 in w0_list:
                                                        for p in p_list:
                                                            for k_min in k_min_list:
                                                                for k_max in k_max_list:
                                                                    for max_position in max_position_list:
                                                                        for agents_only in agents_only_list:
                                                                            for param in beta_params:
                                                                                bash_script_content += (
                                                                                    f"python simMM_example.py --game_name={game_name} "
                                                                                    f"--root_result_folder={root_result_folder} "
                                                                                    f"--num_iteration={num_iteration} "
                                                                                    f"--num_background_agents={num_background_agents} "
                                                                                    f"--sim_time={sim_time} "
                                                                                    f"--lam={lam} "
                                                                                    f"--lamMM={lamMM} "
                                                                                    f"--omega={omega} "
                                                                                    f"--K={K} "
                                                                                    f"--n_levels={n_levels} "
                                                                                    f"--total_volume={total_volume} "
                                                                                    f"--beta_MM={beta_MM} "
                                                                                    f"--inv_driven={inv_driven} "
                                                                                    f"--w0={w0} "
                                                                                    f"--p={p} "
                                                                                    f"--k_min={k_min} "
                                                                                    f"--k_max={k_max} "
                                                                                    f"--max_position={max_position} "
                                                                                    f"--agents_only={agents_only}"
                                                                                )
                                                                                bash_script_content += param + "&& \\\n"

# Remove the last "&& \\\n"
bash_script_content = bash_script_content.rstrip(" && \\\n")

# Write the bash script to a file
with open(file_name, 'w') as file:
    file.write(bash_script_content)

print(f"Bash script generated and saved as '{file_name}'")
