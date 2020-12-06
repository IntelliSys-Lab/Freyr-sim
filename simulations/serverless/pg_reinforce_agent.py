import torch
import torch.nn as nn
import numpy as np
import copy as cp


class PGNet(nn.Module):
    def __init__(self, 
                 observation_dim, 
                 hidden_dims,
                 action_dim
                 ):
        super(PGNet, self).__init__()
        
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
                layer_output = nn.Sequential(
                    nn.Linear(hidden_dims[i], action_dim),
                    nn.Softmax()
                    )
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


class ReinforceAgent():
    def __init__(
            self,
            observation_dim,
            action_dim,
            hidden_dims=64,
            learning_rate=0.01,
            discount_factor=1
            ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.lr = learning_rate
        self.gamma = discount_factor

        self.observations = []
        self.actions = []
        self.rewards = []
        
        self.values = []
        self.value_max_len = 0
        
        self.net = PGNet(
            observation_dim=self.observation_dim,
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
            )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def choose_action(self, observation):
        self.net.eval()
        
        actions = self.net(torch.Tensor(observation[np.newaxis, :]))
        action = np.random.choice(range(actions.shape[1]), p=actions.view(-1).detach().numpy())
            
        return action

    def record_trajectory(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def propagate(self):
        self.net.train()
        
        value = self.discount_rewards()
        baseline = self.compute_naive_baseline(value)
        advantage = value - baseline
        
        output = self.net(torch.Tensor(self.observations))
        one_hot = self.one_hot(self.actions, self.action_dim)
        neg_log_prob = torch.sum(-torch.log(output) * one_hot, 1)
        loss = (neg_log_prob * torch.Tensor(advantage)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset()

        return loss
    
    def one_hot(self, indices, depth):
        dim = 1
        index = torch.LongTensor(indices).view(-1,1)
        src = 1
        
        return torch.zeros(len(indices), depth).scatter_(dim, index, src)
    
    def zero_padding(self, values, max_len):
        values_padded = cp.deepcopy(values)
        
        if isinstance(values_padded[0], list):
            for value in values_padded:
                for i in range(max_len-len(value)):
                    value.append(0)
        else:
            for i in range(max_len-len(values_padded)):
                values_padded.append(0)
                
        return values_padded
    
    def compute_naive_baseline(self, value):
        if len(self.values) == 0:
            baseline = 0
        else:
            if len(value) > self.value_max_len:
                self.value_max_len = len(value)
                self.values = self.zero_padding(self.values, self.value_max_len)
            else: 
                value_padded = self.zero_padding(value, self.value_max_len)
                
            baseline = np.mean(self.values[:, :len(value)])
            self.values.append(value_padded)
        
        return baseline

    def norm(self, x):
        x = x - np.mean(x)
        x = x / np.std(x)

        return x

    def discount_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards)
        tmp = 0
        for i in reversed(range(len(self.rewards))):
            tmp = tmp * self.gamma + self.rewards[i]
            discounted_rewards[i] = tmp

        return discounted_rewards

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
    
    