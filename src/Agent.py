import torch
import torch.nn as nn

class Agent:
    def __init__(self, actor, critic, gamma=0.99):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma


    def select_action(self, state):
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

    def train(self, state, action, reward, next_state, done, actor_optimizer, critic_optimizer):
        if done:
            target_q_value = torch.tensor(reward, dtype=torch.float32)
        else:
            next_action = self.select_action(next_state)
            next_q_value = self.predict_value(next_state, next_action)
            target_q_value = reward + self.gamma * next_q_value
            target_q_value.clone().detach()

        # Train the critic to minimize (Q(s, a) - target)^2
        self.critic.update(state, action, target_q_value, critic_optimizer)

        # Update the actor to maximize the expected Q-value
        self.actor.update(state, self.critic, actor_optimizer)



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
        action = self.forward(state.unsqueeze(0))
        q_value = critic.forward(state.unsqueeze(0), action)
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
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)

        target_q_value = target_q_value.unsqueeze(0).unsqueeze(1)
        
        predicted_q_value = self.forward(state, action)
        loss = nn.MSELoss()(predicted_q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
