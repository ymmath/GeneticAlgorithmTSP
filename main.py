import random
# import matplotlib.pyplot as plt
from r0823209 import r0823209
import numpy as np
import csv

a = r0823209()
solutions = []
l = range(0, 1)
for k in l:
    sol = a.optimize("tour29.csv")
    solutions.append(sol)

print(np.mean(solutions), np.min(solutions))

# plt.plot(l, solutions,'ro')
# plt.ylabel("Best fitness value")
# plt.xticks(l)
# plt.savefig("repetition")