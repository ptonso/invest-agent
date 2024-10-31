import os
import json
from torch import manual_seed

class Configurations:
    def __init__(self, seed=123, device="cpu",
                stock_csv = "data/01_clean/total_return_var.csv",
                ipca_csv = "data/01_clean/ipca.csv",
                action_csv = "data/model/agent_actions.csv",
                start_date = "2005-01-03",
                save_actions = True,
                n_stocks = -1,    # -1
                window_size = 30, # 30
                n_steps = 5,
                actor_lr=0.001,
                critic_lr=0.001,
                gamma = 0.99,
                std_noise = 1.,
                describe_flag = True,
                baseline_flag = True,
                savefig = False
                ):

        self.SEED = seed
        self.device = device
        manual_seed(self.SEED)

        self.stock_csv = stock_csv
        self.ipca_csv = ipca_csv
        self.action_csv = action_csv
        self.start_date = start_date

        self.save_actions = save_actions

        self.n_stocks = n_stocks
        self.window_size = window_size
        
        self.n_steps = n_steps
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.std_noise = std_noise

        self.describe_flag = describe_flag
        self.baseline_flag = baseline_flag
        self.savefig = savefig
    

    def get_experiment_results(self, experiment):
        self.cum_value = experiment.cum_value
        self.exp_name = experiment.exp_name

    def save_config(self, experiment, filepath):
        self.get_experiment_results(experiment)
        os.makedirs(os.basename(filepath), exists_ok=True)

        json_data = {attr:value for attr, value in self.__dict__.items()}

        with open(filepath, 'w') as f:
            json.dump(json_data, f)