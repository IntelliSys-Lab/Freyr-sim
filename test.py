import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def norm(x):
    x = (x - x.mean()) / (x.std() + np.finfo(np.float32).eps.item())
    return x

a = torch.Tensor([[0], [-1], [-2], [-3], [-4], [-5]])

print(F.normalize(a))
print(norm(a))
