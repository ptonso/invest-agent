
import random
import torch
import torch.nn as nn

from .ReplayBuffer import ReplayBuffer

class Agent:
    def __init__(self, actor, critic, gamma=0.99, buffer_size=10000, batch_size=64):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size

        self.starting_exploration=0.6
        self.episode_length=30
        self.n_exploratory_stocks=2
        self.current_step=0


    def select_action(self, state, current_step=0):
        self.current_step = current_step
        on_policy_action = self.policy(state)
        off_policy_action = self.apply_exploratory_behavior(on_policy_action)
        return off_policy_action
    

    def policy(self, state):
        action = self.actor.predict(state)
        assert torch.isclose(torch.sum(action), torch.tensor(1.0), atol=1e-4), f"Action must sum to 1, but got {torch.sum(action)}"
        return action

    
    def predict_value(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Shape: [1, state_size]
        # Ensure action is a 2D tensor
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Shape: [1, action_size]
        q_value = self.critic.forward(state, action)
        return q_value.item()

    def store_transition(self, state, action, reward, next_state, done):
        state = state.detach()
        action = action.detach()
        next_state = next_state.detach()
        self.replay_buffer.add(state, action, reward, next_state, done)



    def apply_exploratory_behavior(self, action):

        exploration_rate = self.starting_exploration * (1 - self.current_step / self.episode_length)
        exploration_rate = max(0.0, exploration_rate)

        n_stocks = action.size(0)
        exploratory_stocks = random.sample(range(n_stocks), self.n_exploratory_stocks)

        exploratory_action = torch.zeros_like(action)
        exploratory_allocation = exploration_rate / self.n_exploratory_stocks

        remaining_allocation = action.clone()
        for stock_idx in exploratory_stocks:
            exploratory_action[stock_idx] = exploratory_allocation
            remaining_allocation[stock_idx] = 0

        if remaining_allocation.sum() > 0:
            remaining_allocation = remaining_allocation * (1 - exploration_rate) / remaining_allocation.sum()

        final_action = exploratory_action + remaining_allocation

        final_action = final_action / final_action.sum()
        return final_action



    def train(self, actor_optimizer, critic_optimizer):
        if self.replay_buffer.size() < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            next_actions = self.actor.forward(next_states)
            next_q_values = self.critic.forward(next_states, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

        self.critic.update(states, actions, target_q_values, critic_optimizer)

        self.actor.update(states, self.critic, actor_optimizer)


    def save(self, actor_filename, critic_filename):
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic.state_dict(), critic_filename)

    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename))
        self.critic.load_state_dict(torch.load(critic_filename))


class NNActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 32], random_state=123,
                 add_noise=False, noise_scale=0.01, entropy_beta=0.01):
        torch.manual_seed(random_state)
        super(NNActor, self).__init__()
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ELU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.entropy_beta = entropy_beta
        self.action_size = action_size

    def forward(self, state):
        logits = self.model(state)
        if self.add_noise:
            noise = torch.randn_like(logits) * self.noise_scale
            logits += noise
        return torch.softmax(logits, dim=-1)

    def predict(self, state):
        action = self.forward(state.unsqueeze(0))
        return action.squeeze(0)
    
    def update(self, state, critic, optimizer):
        action = self.forward(state)
        with torch.no_grad():
            q_value = critic.forward(state, action).squeeze(1)
        entropy_loss = -torch.sum(action * torch.log(action + 1e-6))
        policy_loss = -q_value.mean() - self.entropy_beta * entropy_loss
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


class QCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 32], random_state=123):
        torch.manual_seed(random_state)
        super(QCritic, self).__init__()
        input_size = state_size + action_size
        layers = []
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ELU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        state_action = torch.cat([state, action], dim=1)
        return self.model(state_action)

    def predict(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        q_value = self.forward(state, action)
        return q_value.item()

    def update(self, state, action, target_q_value, optimizer):

        target_q_value = target_q_value.unsqueeze(1)
        
        predicted_q_value = self.forward(state, action)
        loss = nn.MSELoss()(predicted_q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
