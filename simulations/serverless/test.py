import numpy as np
import torch
import scipy.stats as stats
import random 


# l = [[None]] * 3
l = []
for _ in range(3):
    l.append([])

print("before: {}".format(l))

for i in range(len(l)):
    l[i].append("fuck")

    print("during: {}".format(l))

print("after: {}".format(l))
