import torch.nn as nn
from attn import Attn
from params import *


class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        embed_dim,
        num_heads,
        hidden_dims,
        action_dim,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.action_dim = action_dim
        
        self.layers = []
        self.attn = Attn(
            state_dim=state_dim, 
            embed_dim=embed_dim, 
            num_heads=num_heads
        )
        
        layer_input = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            ACTIVATION
        )
        self.layers.append(layer_input)
        
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims)-1:
                layer_output = nn.Linear(hidden_dims[i], action_dim)
                self.layers.append(layer_output)
            else:
                layer_hidden = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    ACTIVATION
                )
                self.layers.append(layer_hidden)
        
        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        x, weights = self.attn(x)
        for layer in self.layer_module:
            x = layer(x)
        return x, weights
