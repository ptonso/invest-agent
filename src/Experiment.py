

import os
import pandas as pd

from src.Agent import Agent, NNActor, Critic
from src.Environment import PortfolioEnv
from src.Descriptor import Descriptor

class Experiment:
    def __init__(self, exp_name, config):
        self.exp_name = exp_name
        self.config = config

        self.env = PortfolioEnv(
            stock_data_csv=config.stock_csv,
            ipca_csv=config.ipca_csv,
            window_size=config.window_size,
            n_stocks=config.n_stocks,
            baseline_flag=config.baseline_flag
            )

        stock_size = self.env.stock_data.shape[1]
        state_size = stock_size * config.window_size
        action_size = stock_size

        actor = NNActor(state_size=state_size, action_size=action_size, device=config.device)
        critic = Critic(state_size=state_size, action_size=action_size, device=config.device)

        self.agent = Agent(
            actor=actor,
            critic=critic,
            gamma=config.gamma,
            n_steps=config.n_steps,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            std_noise=config.std_noise,
            device=config.device
        )

        if config.describe_flag:
            self.descriptor = Descriptor(n_stocks=stock_size, 
                                         start_date=config.start_date,
                                         baseline_flag=config.baseline_flag)
            self.current_step = 0

    def run(self):

        state = self.env.reset()
        self.agent.reset()
        done = False
        current_step = 0

        self.cum_value = []

        while not done:
            state_tensor = state.unsqueeze(0)
            action = self.agent.policy(state_tensor)
            next_state, reward, done = self.env.step(action.squeeze(0))
            next_state_tensor = next_state.unsqueeze(0)

            self.agent.store_transition(state_tensor, action, next_state_tensor, reward, done)
            self.agent.train()

            actual_cum_value = self.env.get_cumulative_value()
            self.cum_value.append(actual_cum_value)

            if self.config.describe_flag:
                baseline_value = self.env.get_baseline_cumulative_value()
                current_date = self.env.get_current_date()
                window_date = self.env.get_window_date_string()
            
                self.descriptor.update(action.squeeze(0), 
                                       actual_cum_value, 
                                       current_date=current_date,
                                       baseline_value=baseline_value)
            
            if self.config.save_actions:
                self.save_action_to_csv(action, window_date)
            
            current_step += 1
            state = next_state

            if done:
                break

        if self.config.savefig:
            self.descriptor.save_plot(self.config.savefig)

        print(self.env.get_cumulative_value())


    def save_action_to_csv(self, action, window_date):
        csv_filename = self.config.action_csv
        action_list = action.detach().cpu().numpy().tolist()
        data = {'Window Date': [window_date]}
        for i, alloc in enumerate(action_list):
            data[f'Stock_{i}'] = [alloc]
        df = pd.DataFrame(data)
        file_exists = os.path.isfile(csv_filename)
        
        if file_exists:
            df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filename, mode='w', header=True, index=False)
