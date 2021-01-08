import numpy as np
import pandas as pd
import torch


a = torch.Tensor([[0, 1, 2]])
b = torch.Tensor([[-10e6, -10e6, -10e6]])

print(a.shape)
print(b.shape)
print(a+b)
