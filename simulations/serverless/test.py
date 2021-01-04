import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as stats
import random 


action = torch.ones(640, 1) * 64

print(action.shape)

action = action.long()
action = F.one_hot(action, 65).float()

print(action.shape)