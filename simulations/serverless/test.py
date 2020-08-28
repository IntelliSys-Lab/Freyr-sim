import numpy as np
import torch
import scipy.stats as stats
import random 


print("poisson: ")
print(stats.poisson.rvs(mu=0.5, size=100))

print("norm: ")
print(random.randint(0, 1))

print("bernoulli: ")
print(stats.bernoulli.rvs(p=0.5, size=100))