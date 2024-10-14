import os
import pandas as pd
import torch.optim as optim
from torch import manual_seed

from src.Agent import Agent, NNActor, QCritic
from src.Environment import PortfolioEnv
from src.Descriptor import DescriptorControl


def save_action_to_csv(csv_filename, action, window_date):
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




n_stocks = -1
window_size = 30
save_actions = True
SEED = 0

add_noise = True
noise_scale = 0.01
entropy_beta = 0.01

manual_seed(SEED)


stock_csv = "data/01_clean/total-return-var.csv"
ipca_csv = "data/01_clean/ipca.csv"
action_csv = "data/model/agent_actions.csv"

env = PortfolioEnv(
    stock_data_csv=stock_csv,
    ipca_csv=ipca_csv,
    window_size=window_size,
    n_stocks=n_stocks
    )

stock_size = env.stock_data.shape[1]
state_size = stock_size * window_size
action_size = stock_size

actor = NNActor(state_size=state_size, action_size=action_size, 
                add_noise=add_noise, 
                noise_scale=noise_scale, 
                entropy_beta=entropy_beta)

critic = QCritic(state_size=state_size, action_size=action_size)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

agent = Agent(
    actor=actor,
    critic=critic,
    gamma=0.001
)

descriptor = DescriptorControl(n_stocks=stock_size)
state = env.reset()

while True:
    action = agent.select_action(state)
    
    next_state, reward, done = env.step(action)

    current_date = env.get_current_date()
    window_date = env.get_window_date_string()
    
    descriptor.update(action, env.get_cumulative_value(), current_date)
    
    if save_actions:
        save_action_to_csv(action_csv, action, window_date)

    agent.train(state, action, reward, next_state, done, actor_optimizer, critic_optimizer)
    state = next_state


    if done:
        break

print(env.get_cumulative_value())