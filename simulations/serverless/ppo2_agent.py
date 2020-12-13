import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np



class PGNet(nn.Module):
    def __init__(
        self, 
        observation_dim, 
        hidden_dims,
        action_dim
    ):
        super().__init__()
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        self.layers = []
        
        layer_input = nn.Sequential(
            nn.Linear(observation_dim, hidden_dims[0]),
            nn.Tanh()
        )
        self.layers.append(layer_input)
        
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims)-1:
                layer_output = nn.Linear(hidden_dims[i], action_dim)
                self.layers.append(layer_output)
            else:
                layer_hidden = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.Tanh()
                )
                self.layers.append(layer_hidden)
        
        self.layer_module = nn.ModuleList(self.layers)
        
    def forward(self, x):
        for layer in self.layer_module:
            x = layer(x)

        return x


class ActorCritic(nn.Module):
    def __init__(
        self, 
        actor_cpu,
        actor_memory,
        critic_cpu,
        critic_memory,
    ):
        super().__init__()

        self.actor_cpu = actor_cpu
        self.actor_memory = actor_memory
        self.critic_cpu = critic_cpu
        self.critic_memory = critic_memory

    def forward(self, observation_cpu, observation_memory):
        action_pred_cpu = self.actor_cpu(observation_cpu)
        action_pred_memory = self.actor_memory(observation_memory)
        value_pred_cpu = self.critic_cpu(observation_cpu)
        value_pred_memory = self.critic_memory(observation_memory)
        
        return action_pred_cpu, action_pred_memory, value_pred_cpu, value_pred_memory


class PPO2Agent():
    def __init__(
        self,
        observation_dim_cpu,
        observation_dim_memory,
        action_dim_cpu,
        action_dim_memory,
        hidden_dims_cpu=[64, 32],
        hidden_dims_memory=[64, 32],
        learning_rate=0.001,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_steps=5
    ):
        self.observation_dim_cpu = observation_dim_cpu
        self.observation_dim_memory = observation_dim_memory
        self.action_dim_cpu = action_dim_cpu
        self.action_dim_memory = action_dim_memory
        self.hidden_dims_cpu = hidden_dims_cpu
        self.hidden_dims_memory = hidden_dims_memory
        self.learning_rate = learning_rate
        self.gamma = discount_factor

        self.ppo_clip = ppo_clip
        self.ppo_steps = ppo_steps

        self.observation_list_cpu = []
        self.observation_list_memory = []
        self.action_list_cpu = []
        self.action_list_memory = []
        self.value_list_cpu = []
        self.value_list_memory = []
        self.log_prob_list_cpu = []
        self.log_prob_list_memory = []
        self.reward_list = []
        
        self.model = self.build_model()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def build_model(self):
        actor_cpu = PGNet(
            observation_dim=self.observation_dim_cpu,
            hidden_dims=self.hidden_dims_cpu,
            action_dim=self.action_dim_cpu,
        )
        actor_memory = PGNet(
            observation_dim=self.observation_dim_memory,
            hidden_dims=self.hidden_dims_memory,
            action_dim=self.action_dim_memory,
        )
        critic_cpu = PGNet(
            observation_dim=self.observation_dim_cpu,
            hidden_dims=self.hidden_dims_cpu,
            action_dim=1,
        )
        critic_memory = PGNet(
            observation_dim=self.observation_dim_memory,
            hidden_dims=self.hidden_dims_memory,
            action_dim=1,
        )

        ac_model = ActorCritic(
            actor_cpu=actor_cpu, 
            actor_memory=actor_memory, 
            critic_cpu=critic_cpu,
            critic_memory=critic_memory
        )

        return ac_model

    def choose_action(self, observation_cpu, observation_memory):
        self.model.eval()
        
        observation_cpu = torch.Tensor(observation_cpu[np.newaxis, :])
        observation_memory = torch.Tensor(observation_memory[np.newaxis, :])
        action_pred_cpu, action_pred_memory, value_pred_cpu, value_pred_memory = self.model(
            observation_cpu=observation_cpu, 
            observation_memory=observation_memory
        )

        action_prob_cpu = F.softmax(action_pred_cpu, dim=-1)
        action_prob_memory = F.softmax(action_pred_memory, dim=-1)
        dist_cpu = Categorical(action_prob_cpu)
        dist_memory = Categorical(action_prob_memory)
        action_cpu = dist_cpu.sample()
        action_memory = dist_memory.sample()
        log_prob_cpu = dist_cpu.log_prob(action_cpu)
        log_prob_memory = dist_memory.log_prob(action_memory)
            
        return action_cpu, action_memory, value_pred_cpu, value_pred_memory, log_prob_cpu, log_prob_memory

    def record_trajectory(
        self, 
        observation_cpu, 
        observation_memory, 
        action_cpu, 
        action_memory, 
        value_cpu, 
        value_memory, 
        log_prob_cpu,
        log_prob_memory,
        reward
    ):
        self.observation_list_cpu.append(torch.Tensor(observation_cpu[np.newaxis, :]))
        self.observation_list_memory.append(torch.Tensor(observation_memory[np.newaxis, :]))
        self.action_list_cpu.append(action_cpu)
        self.action_list_memory.append(action_memory)
        self.value_list_cpu.append(value_cpu)
        self.value_list_memory.append(value_memory)
        self.log_prob_list_cpu.append(log_prob_cpu)
        self.log_prob_list_memory.append(log_prob_memory)
        self.reward_list.append(reward)

    def learn(self):
        self.model.train()
        
        # Discount rewards
        reward_list = self.discount_rewards()

        # Concatenate trajectory
        observation_list_cpu = torch.cat(self.observation_list_cpu, dim=0)
        observation_list_memory = torch.cat(self.observation_list_memory, dim=0)
        action_list_cpu = torch.cat(self.action_list_cpu, dim=0)
        action_list_memory = torch.cat(self.action_list_memory, dim=0)
        value_list_cpu = torch.cat(self.value_list_cpu).squeeze(-1)
        value_list_memory = torch.cat(self.value_list_memory).squeeze(-1)
        log_prob_list_cpu = torch.cat(self.log_prob_list_cpu, dim=0)
        log_prob_list_memory = torch.cat(self.log_prob_list_memory, dim=0)

        # Calculate advantage
        advantage = reward_list - (value_list_cpu + value_list_memory)

        # Detach trajectory 
        observation_list_cpu = observation_list_cpu.detach()
        observation_list_memory = observation_list_memory.detach()
        action_list_cpu = action_list_cpu.detach()
        action_list_memory = action_list_memory.detach()
        log_prob_list_cpu = log_prob_list_cpu.detach()
        log_prob_list_memory = log_prob_list_memory.detach()
        advantage = advantage.detach()
        reward_list = reward_list.detach()
        
        loss = 0
        for _ in range(self.ppo_steps):
            # Get new log probs of actions for all input states
            action_pred_cpu, action_pred_memory, value_pred_cpu, value_pred_memory = self.model(
                observation_cpu=observation_list_cpu, 
                observation_memory=observation_list_memory
            )
            value_pred_cpu = value_pred_cpu.squeeze(-1)
            value_pred_memory = value_pred_memory.squeeze(-1)
            action_prob_cpu = F.softmax(action_pred_cpu, dim=-1)
            action_prob_memory = F.softmax(action_pred_memory, dim=-1)
            dist_cpu = Categorical(action_prob_cpu)
            dist_memory = Categorical(action_prob_memory)

            # New log probs using old actions
            new_log_prob_cpu = dist_cpu.log_prob(action_list_cpu)
            new_log_prob_memory = dist_memory.log_prob(action_list_memory)

            policy_ratio = (new_log_prob_cpu * new_log_prob_memory - log_prob_list_cpu * log_prob_list_memory).exp()
            policy_loss_1 = policy_ratio * advantage
            policy_loss_2 = torch.clamp(
                policy_ratio, 
                min = 1.0 - self.ppo_clip, 
                max = 1.0 + self.ppo_clip
            ) * advantage

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(reward_list, value_pred_cpu + value_pred_memory).mean()
            loss = loss + policy_loss.item() + value_loss.item()

            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer.step()

        loss = loss / self.ppo_steps
        return loss
    
    def norm(self, x):
        eps = np.finfo(np.float32).eps.item()
        x = (x - x.mean()) / (x.std() + eps)

        return x

    def discount_rewards(self):
        discounted_rewards = []
        tmp = 0
        for reward in self.reward_list[::-1]:
            tmp = tmp * self.gamma + reward
            discounted_rewards.append(tmp)
        
        discounted_rewards = torch.Tensor(discounted_rewards[::-1])
        discounted_rewards = self.norm(discounted_rewards)

        return discounted_rewards
    
    def reset(self):
        self.observation_list_cpu = []
        self.observation_list_memory = []
        self.action_list_cpu = []
        self.action_list_memory = []
        self.value_list_cpu = []
        self.value_list_memory = []
        self.log_prob_list_cpu = []
        self.log_prob_list_memory = []
        self.reward_list = []

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
