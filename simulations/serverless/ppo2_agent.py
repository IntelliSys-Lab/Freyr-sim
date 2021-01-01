import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from mlp import MLP


class ActorCritic(nn.Module):
    def __init__(
        self, 
        actor,
        critic
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, observation):
        action_pred = self.actor(observation)
        value_pred = self.critic(observation)
        
        return action_pred, value_pred


class PPO2Agent():
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dims=[64,32],
        learning_rate=0.001,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_steps=5
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = discount_factor

        self.ppo_clip = ppo_clip
        self.ppo_steps = ppo_steps

        self.observation_history = []
        self.action_history = []
        self.value_history = []
        self.log_prob_history = []
        self.reward_history = []
        
        self.model = self.build_model()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr= self.learning_rate)

        self.eps = np.finfo(np.float32).eps.item()

    def build_model(self):
        actor = MLP(
            observation_dim=self.observation_dim,
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
            is_actor=True
        )

        critic = MLP(
            observation_dim=self.observation_dim,
            hidden_dims=self.hidden_dims,
            action_dim=1,
            is_actor=False
        )

        ac_model = ActorCritic(actor, critic)

        return ac_model

    def choose_action(self, observation):
        self.model.eval()
        
        action_pred, value_pred = self.model(torch.Tensor(observation[np.newaxis, :]))
        action_prob = F.softmax(action_pred, dim=-1)
        dist = Categorical(action_prob)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
            
        return action, value_pred, log_prob

    def record_trajectory(
        self, 
        observation, 
        action, 
        value, 
        reward, 
        log_prob
    ):
        self.observation_history.append(observation.unsqueeze(0))
        self.action_history.append(action.unsqueeze(0))
        self.value_history.append(value)
        self.log_prob_history.append(log_prob.unsqueeze(0))
        self.reward_history.append(reward)

    def propagate(self):
        self.model.train()
        
        # Discount rewards
        reward_history = self.discount_rewards()

        # Concatenate trajectory
        observation_history = torch.cat(self.observation_history, dim=0)
        action_history = torch.cat(self.action_history, dim=0)
        value_history = torch.cat(self.value_history).squeeze()
        log_prob_history = torch.cat(self.log_prob_history, dim=0)

        # Calculate advantage
        advantage = reward_history - value_history
        
        # Detach trajectory 
        observation_history = observation_history.detach()
        action_history = action_history.detach()
        log_prob_history = log_prob_history.detach()
        advantage = advantage.detach()
        reward_history = reward_history.detach()
        
        loss = 0
        for _ in range(self.ppo_steps):
            # Get new log probs of actions for all input states
            new_log_prob_history = []
            new_value_history = []
            for i in range (observation_history.size(0)):
                observation_i = observation_history[i, :]
                action_pred, value_pred = self.model(observation_i)
                action_prob = F.softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)

                # New log probs using old actions
                new_log_prob = dist.log_prob(action_history[i])

                new_log_prob_history.append(new_log_prob.unsqueeze(0))
                new_value_history.append(value_pred)

            new_log_prob_history = torch.cat(new_log_prob_history, dim=0)
            new_value_history = torch.cat(new_value_history).squeeze()

            policy_ratio = (new_log_prob_history - log_prob_history).exp()
            policy_loss_1 = policy_ratio * advantage
            policy_loss_2 = torch.clamp(
                policy_ratio, 
                min = 1.0 - self.ppo_clip, 
                max = 1.0 + self.ppo_clip
            ) * advantage

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(reward_history, new_value_history).mean()
            loss = loss + policy_loss.item() + value_loss.item()

            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer.step()

        loss = loss / self.ppo_steps
        return loss
    
    def norm(self, x):
        x = (x - x.mean()) / (x.std() + self.eps)
        return x

    def discount_rewards(self):
        discounted_rewards = []
        tmp = 0
        for reward in self.reward_history[::-1]:
            tmp = tmp * self.gamma + reward
            discounted_rewards.append(tmp)
        
        discounted_rewards = torch.Tensor(discounted_rewards[::-1])
        discounted_rewards = self.norm(discounted_rewards)

        return discounted_rewards
    
    def reset(self):
        self.observation_history = []
        self.action_history = []
        self.value_history = []
        self.log_prob_history = []
        self.reward_history = []

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
