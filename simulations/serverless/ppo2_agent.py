import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from seq2seq import Encoder, Decoder,Seq2Seq


class ActorCritic(nn.Module):
    def __init__(
        self, 
        actor,
        critic,
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, observation):
        start_of_predict = torch.ones(observation.size(1)) * self.actor.decoder.output_size

        action_pred = self.actor(observation, start_of_predict)
        value_pred = self.critic(observation, start_of_predict)
        
        return action_pred, value_pred


class PPO2Agent():
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        learning_rate=0.001,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_epoch=5,
        value_loss_coef=0.5,
        entropy_coef=0.01
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor

        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.observation_history = []
        self.action_history = []
        self.value_history = []
        self.log_prob_history = []
        self.reward_history = []
        
        self.model = self.build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.eps = np.finfo(np.float32).eps.item()

    def build_model(self):
        actor = Seq2Seq(
            encoder=Encoder(
                input_size=self.observation_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
            ),
            decoder=Decoder(
                is_actor=True,
                input_size=self.action_dim + 1, # Add <predict>
                hidden_size=self.hidden_dim,
                output_size=self.action_dim,
                num_layers=1,
            )
        )

        critic = Seq2Seq(
            encoder=Encoder(
                input_size=self.observation_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
            ),
            decoder=Decoder(
                is_actor=False,
                input_size=self.action_dim + 1, # Add <predict>
                hidden_size=self.hidden_dim,
                output_size=1,
                num_layers=1,
            )
        )

        ac_model = ActorCritic(
            actor=actor, 
            critic=critic,
        )

        return ac_model

    def choose_action(self, observation):
        self.model.eval()
        
        action_prob, value_pred = self.model(observation)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action, value_pred, log_prob

    def record_trajectory(
        self, 
        observation, 
        action, 
        value, 
        log_prob,
        reward
    ):
        self.observation_history.append(observation)
        self.action_history.append(action)
        self.value_history.append(value)
        self.log_prob_history.append(log_prob)

        # Only return reward when a series of actions finish
        rewards = []
        for i in range(action.size(0)):
            if i < action.size(0) - 1:
                rewards.append(0)
            else:
                rewards.append(reward)
        self.reward_history = self.reward_history + rewards

    def update(self):
        self.model.train()
        
        # Discount rewards
        reward_history = self.discount_rewards()

        # Concatenate trajectory
        observation_history = torch.cat(self.observation_history, dim=1)
        action_history = torch.cat(self.action_history, dim=1)
        value_history = torch.cat(self.value_history).squeeze()
        log_prob_history = torch.cat(self.log_prob_history, dim=0)
        reward_history = torch.Tensor(self.reward_history)
        # print("history after cat: ")
        # print("observation_history shape: {}".format(observation_history.shape))
        # print("action_history shape: {}".format(action_history.shape))
        # print("value_history shape: {}".format(value_history.shape))
        # print("log_prob_history shape: {}".format(log_prob_history.shape))
        # print("reward_history shape: {}".format(reward_history.shape))

        # Calculate advantage
        advantage = reward_history - value_history

        # Detach trajectory 
        observation_history = observation_history.detach()
        action_history = action_history.detach()
        log_prob_history = log_prob_history.detach()
        advantage = advantage.detach()
        reward_history = reward_history.detach()
        
        loss_epoch = 0
        for _ in range(self.ppo_epoch):
            # Get new log probs of actions for all input states
            action_prob_all, value_pred_all = self.model(observation_history)
            dist = Categorical(action_prob_all)
            dist_entropy = dist.entropy().mean()

            # New log probs using old actions
            new_log_prob_all = dist.log_prob(action_history)
            new_log_prob_history = torch.reshape(new_log_prob_all, (-1, 1))
            new_value_history = torch.flatten(value_pred_all)

            policy_ratio = (new_log_prob_history - log_prob_history).exp()
            policy_loss_1 = policy_ratio * advantage
            policy_loss_2 = torch.clamp(
                policy_ratio, 
                min = 1.0 - self.ppo_clip, 
                max = 1.0 + self.ppo_clip
            ) * advantage

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(reward_history, new_value_history).mean()
            
            self.optimizer.zero_grad()
            loss = policy_loss + value_loss * self.value_loss_coef - dist_entropy * self.entropy_coef
            loss.backward()
            self.optimizer.step()

            loss_epoch = loss_epoch + loss.item()
        loss_epoch = loss_epoch / self.ppo_epoch
        
        return loss_epoch
    
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
