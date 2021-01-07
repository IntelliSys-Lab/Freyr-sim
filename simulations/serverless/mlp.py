import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, 
        observation_dim, 
        hidden_dims,
        action_dim,
        is_actor
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_dims = hidden_dims
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.action_dim = action_dim
        self.is_actor = is_actor
        
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

    def get_encode_pos(self, index):
        observation_base = self.observation_dim - 8 * (index + 1)
        observation_offset = {}
        observation_offset["avg_interval"] = 0
        observation_offset["avg_completion_time"] = 1
        observation_offset["is_cold_start"] = 2
        observation_offset["cpu"] = 3
        observation_offset["memory"] = 4
        observation_offset["cpu_direction"] = 5
        observation_offset["memory_direction"] = 6
        observation_offset["invoke_num"] = 7

        observation_pos = {}
        observation_pos["avg_interval"] = observation_base + observation_offset["avg_interval"]
        observation_pos["avg_completion_time"] = observation_base + observation_offset["avg_completion_time"]
        observation_pos["is_cold_start"] = observation_base + observation_offset["is_cold_start"]
        observation_pos["cpu"] = observation_base + observation_offset["cpu"]
        observation_pos["memory"] = observation_base + observation_offset["memory"]
        observation_pos["cpu_direction"] = observation_base + observation_offset["cpu_direction"]
        observation_pos["memory_direction"] = observation_base + observation_offset["memory_direction"]
        observation_pos["invoke_num"] = observation_base + observation_offset["invoke_num"]

        action_base = (self.action_dim - 1) - 4 * (index + 1)
        action_offset = {}
        action_offset["decrease_cpu"] = 0
        action_offset["increase_cpu"] = 1
        action_offset["decrease_memory"] = 2
        action_offset["increase_memory"] = 3

        action_pos = {}
        action_pos["decrease_cpu"] = action_base + action_offset["decrease_cpu"]
        action_pos["increase_cpu"] = action_base + action_offset["increase_cpu"]
        action_pos["decrease_memory"] = action_base + action_offset["decrease_memory"]
        action_pos["increase_memory"] = action_base + action_offset["increase_memory"]

        return observation_pos, action_pos

    def get_mask(self, observation):
        # Hard-coded limits
        cpu_cap_per_function = 8
        cpu_least_hint = 1
        memory_cap_per_function = 8
        memory_least_hint = 1

        function_num = int((self.action_dim - 1) / 4)

        # Create mask(s)
        if observation.size(0) == 1:
            observation = observation.squeeze()

            # Init mask
            mask = torch.zeros(self.action_dim)

            for i in range(function_num):
                observation_pos, action_pos = self.get_encode_pos(i)
                # No invocation
                if observation[observation_pos["invoke_num"]] == 0: 
                    # print("function {} no invoke".format(i))
                    mask[action_pos["decrease_cpu"]] = -10e6
                    mask[action_pos["increase_cpu"]] = -10e6
                    mask[action_pos["decrease_memory"]] = -10e6
                    mask[action_pos["increase_memory"]] = -10e6
                # Invocation
                else:
                    # Mask out action cpu access
                    if observation[observation_pos["cpu_direction"]] == 1 or observation[observation_pos["cpu"]] == cpu_least_hint:
                        mask[action_pos["decrease_cpu"]] = -10e6
                    elif observation[observation_pos["cpu_direction"]] == -1 or observation[observation_pos["cpu"]] == cpu_cap_per_function:
                        mask[action_pos["increase_cpu"]] = -10e6
                    
                    # Mask out action memory access
                    if observation[observation_pos["memory_direction"]] == 1 or observation[observation_pos["memory"]] == memory_least_hint:
                        mask[action_pos["decrease_memory"]] = -10e6
                    elif observation[observation_pos["memory_direction"]] == -1 or observation[observation_pos["memory"]] == memory_cap_per_function:
                        mask[action_pos["increase_memory"]] = -10e6
        else:
            # Init mask
            mask = torch.zeros(observation.size(0), self.action_dim)

            for i in range(function_num):
                observation_pos, action_pos = self.get_encode_pos(i)

                for j in range(observation.size(0)):
                    observation_j = observation[j, :]
                    # No invocation
                    if observation_j[observation_pos["invoke_num"]] == 0: 
                        mask[j, action_pos["decrease_cpu"]] = -10e6
                        mask[j, action_pos["increase_cpu"]] = -10e6
                        mask[j, action_pos["decrease_memory"]] = -10e6
                        mask[j, action_pos["increase_memory"]] = -10e6
                    # Invocation
                    else:
                        # Mask out action cpu access
                        if observation_j[observation_pos["cpu_direction"]] == 1 or observation_j[observation_pos["cpu"]] == cpu_least_hint:
                            mask[j, action_pos["decrease_cpu"]] = -10e6
                        elif observation_j[observation_pos["cpu_direction"]] == -1 or observation_j[observation_pos["cpu"]] == cpu_cap_per_function:
                            mask[j, action_pos["increase_cpu"]] = -10e6
                        
                        # Mask out action memory access
                        if observation_j[observation_pos["memory_direction"]] == 1 or observation_j[observation_pos["memory"]] == memory_least_hint:
                            mask[j, action_pos["decrease_memory"]] = -10e6
                        elif observation_j[observation_pos["memory_direction"]] == -1 or observation_j[observation_pos["memory"]] == memory_cap_per_function:
                            mask[j, action_pos["increase_memory"]] = -10e6

        return mask
        
    def forward(self, observation):
        x = observation
        for layer in self.layer_module:
            x = layer(x)

        # Apply mask if actor
        if self.is_actor is True:
            # print("input before mask: {}".format(x))
            mask = self.get_mask(observation)
            # print("mask: {}".format(mask))
            x = x + mask
            # print("input after mask: {}".format(x))

        return x

