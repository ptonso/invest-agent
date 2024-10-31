
import torch
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, actor, critic, gamma=0.99, n_steps=5, actor_lr=0.001, critic_lr=0.001, std_noise=0, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = gamma
        self.n_steps = n_steps
        self.std_noise = std_noise
        self.device = device

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.trajectory = Trajectory(gamma, self.n_steps, device)


    def policy(self, state):
        action = self.actor(state)
        action = action.clone()
        if self.std_noise > 0:
            noise = torch.normal(mean=0, std=self.std_noise, size=action.size(), device=self.device)
            action = action + noise
            action = torch.clamp(action, min=0)
        action = action / action.sum(dim=-1, keepdim=True)
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        state = state.to(self.device)
        action = action.detach().to(self.device)
        next_state = next_state.to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.trajectory.add(state, action, next_state, reward, done)

    def train(self):
        if self.trajectory.is_ready():
            G = self.trajectory.compute_n_step_return(self.actor, self.critic)
            state, action, next_state, done = self.trajectory.get()

            state = state.to(self.device)
            action = action.to(self.device)
            next_state = next_state.to(self.device)

            self.critic_optimizer.zero_grad()
            value = self.critic(state, action)
            loss_critic = nn.functional.mse_loss(value, G)
            loss_critic.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            action = self.actor(state)
            actor_loss = -self.critic(state, action).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

    def reset(self):
        self.trajectory.reset()


    def save(self, actor_filename, critic_filename):
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic.state_dict(), critic_filename)

    def load(self, actor_filename, critic_filename):
        self.actor.load_state_dict(torch.load(actor_filename, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_filename, map_location=self.device))


class NNActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 32], random_state=123, device="cpu"):
        torch.manual_seed(random_state)
        super(NNActor, self).__init__()
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)
        self.action_size = action_size
        self.device = device
        self.to(device)

    def forward(self, state):
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.model(state)
        return torch.softmax(logits, dim=-1)



class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64, 32], random_state=123, device="cpu"):
        torch.manual_seed(random_state)
        super(Critic, self).__init__()
        layers = []
        input_size = state_size + action_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=1)
        value = self.model(x)
        return value.squeeze(-1)

class Trajectory:
    def __init__(self, gamma, n_steps, device="cpu"):
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, next_state, reward, done):
        self.states.append(state.to(self.device))
        self.actions.append(action.to(self.device))
        self.next_states.append(next_state.to(self.device))
        self.rewards.append(reward.to(dtype=torch.float32, device=self.device))
        self.dones.append(done.to(dtype=torch.bool, device=self.device))

    def is_ready(self):
        return len(self.rewards) >= self.n_steps
    
    def compute_n_step_return(self, actor, critic):
        n = len(self.rewards)
        with torch.no_grad():
            if not self.dones[-1]:
                next_state = self.next_states[-1]
                if next_state.dim() == 1:
                    next_state = next_state.unsqueeze(0)
                next_action = actor(next_state)
                V_next = critic(next_state, next_action)
            else:
                V_next = torch.tensor([0.0], device=self.device)

        G = V_next * (self.gamma ** n)
        for i in reversed(range(n)):
            reward = self.rewards[i]
            G = reward + self.gamma * G
        return G
    
    def get(self):
        state = self.states.pop(0)
        action = self.actions.pop(0)
        next_state = self.next_states.pop(0)
        done = self.dones.pop(0)
        return state, action, next_state, done
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []