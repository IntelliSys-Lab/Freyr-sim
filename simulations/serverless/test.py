import numpy as np
import torch


actions = torch.Tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.1])
values = [
    [1.0, 1.2, 1.5, 2.0],
    [1.0, 1.1],
    [0.1],
    [0.1, 1.0, 8.0]
    ]

def zero_padding(values, max_len):
    for value in values:
        for i in range(max_len-len(value)):
            value.append(0)
            
    return values

values_padded = zero_padding(values, 8)
print(values_padded)
print(values)
# print(np.mean(values, axis=0))
