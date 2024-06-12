import os
import numpy as np
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.last_model_path = os.path.join(log_dir, "last model")
        self.best_mean_reward = -np.inf
        self.cumulative_window_rewards = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            # Mean training reward over the last 100 episodes
            mean_reward = np.mean(y[-min(len(y),10):])
            self.cumulative_window_rewards.append(mean_reward)
            if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward and self.n_calls > 20000:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

    def _on_training_end(self) -> None:
        print("End of training.")
        self.model.save(self.last_model_path)