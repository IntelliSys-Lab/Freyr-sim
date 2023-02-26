import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.MultiheadAttention(32, 1, batch_first=True)
q = torch.randn(64, 1, 32)
k = torch.randn(64, 10, 32)
v = torch.randn(64, 10, 32)

output, weights = m(q, k, v)

# print(output.shape)
# print(weights.shape)
