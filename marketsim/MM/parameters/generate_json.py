import itertools
import json

def generate_combinations(parameters):
    keys = parameters.keys()
    values = parameters.values()
    combinations = list(itertools.product(*values))
    result = [{k: v for k, v in zip(keys, combination)} for combination in combinations]
    return result

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Parameters
parameters = {
    "num_background_agents" : [100, 200],
    "shock_var": [1e6, 5e6],
    "lam": [0.001, 0.005],
    "lamMM": [0.005],
    "K": [50, 100], # number of rungs
    "xi": [100], # two-rung space
    "omega": [64, 128, 256, 512, 1024, 2048], #spread
    "num_simulations": [1000],
    "sim_time": [1000, 2000, 4000, 8000, 12000],
    "eta": [1, 0.8, 0.6, 0.4], # surplus required
    "buy_alpha": [1,2,5],
    "buy_beta": [1,2,5],
    "sell_alpha": [1,2,5],
    "sell_beta": [1,2,5],
    "n_level": [101]
}

# Generate combinations
combinations = generate_combinations(parameters)

# Save combinations to JSON file
save_to_json(combinations, './combinations.json')
