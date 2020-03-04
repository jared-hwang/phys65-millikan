import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as sc
import csv

# Data filename
filename = "test_data.csv"

# Experiment conditions
temp = 273   # K
E    = 305.5 # V
p    = 101325 # Pascal
b    = 8.2E-3 # Pa * m

# Resultant consts
g = sc.g
rho = 886 # kg/m^3
eta = (1.827E-5)*(291.15 + 120) / (temp + 120) * ((temp/291.15)**(3/2))

vels = []

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    vels = [row for row in csv_reader][1:]
    vels = [(float(row[0]), float(row[1])) for row in vels]


def calc_q(velpair):
    a = math.sqrt((9 * eta * velpair[0]) / (2 * rho * g))
    result = (sc.pi * 6) / E
    result = result * (velpair[0] + velpair[1]) * math.sqrt(velpair[0])
    result = result * ((1 / (1 + (b / (p * a))))**2)
    result = result * math.sqrt((9 * (eta**3))/ (2 * rho * g))

    return result

qs = [calc_q(pair) for pair in vels]
print(qs)

# Plot Results
num_bins = 50

fig, ax = plt.subplots()
n, bins, patches = ax.hist(qs, bins='auto') #, density=1)
ax.set_xlabel("q")
ax.set_ylabel("Frequency")

plt.show()