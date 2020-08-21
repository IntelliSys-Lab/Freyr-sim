import numpy as np

a = [5, 4, 3, 2, 1]
b = []

# print("before: {}, len: {}".format(a, len(a)))

for i in a:
    if i == 3:
        a.remove(i)
    if i == 2:
        a.remove(i)

# print("after: {}, len: {}".format(a, len(a)))

ideal_cpu = 2
cpu = 4

print(np.max([ideal_cpu, cpu])/cpu)

