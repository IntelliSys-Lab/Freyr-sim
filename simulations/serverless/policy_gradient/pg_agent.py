import torch
import torch.nn as nn
import numpy as np


class PGNet(nn.Module):
    def __init__(self, n_actions, n_features, n_hiddens):
        super(PGNet, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(n_features, n_hiddens),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(n_hiddens, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


class PGAgent():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            discount_factor=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = discount_factor

        self.obs = []
        self.acs = []
        self.rws = []

        self.net = PGNet(n_actions, n_features, 128)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def choose_action(self, observation):
        self.net.eval()
        actions = self.net(torch.Tensor(observation[np.newaxis, :]))
        action = np.random.choice(range(actions.shape[1]), p=actions.view(-1).detach().numpy())
        return action

    def store_transition(self, s, a, r):
        self.obs.append(s)
        self.acs.append(a)
        self.rws.append(r)

    def learn(self):
        self.net.train()
        discount = self.discount_and_norm_rewards()
        output = self.net(torch.Tensor(self.obs))
        one_hot = torch.zeros(len(self.acs), self.n_actions).\
            scatter_(1, torch.LongTensor(self.acs).view(-1,1), 1)
        neg = torch.sum(-torch.log(output) * one_hot, 1)
        loss = neg * torch.Tensor(discount)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.acs = []
        self.obs = []
        self.rws = []
        return discount

    def discount_and_norm_rewards(self):
        discount = np.zeros_like(self.rws)
        tmp = 0
        for i in reversed(range(len(self.rws))):
            tmp = tmp * self.gamma + self.rws[i]
            discount[i] = tmp

        discount -= np.mean(discount)
        discount /= np.std(discount)

        return discount