import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.constants as sc
import csv

# Data filename
filename = "Millikan V data.csv"

# Experiment conditions
temp = 292.3049   # K
E    = 305.5 / 0.0076 # V / d
p    = 105655.3 # Pascal
b    = 8.2E-3 # Pa * m

# Resultant consts
g = sc.g
rho = 886 # kg/m^3
eta = (1.827E-5)*(291.15 + 120) / (temp + 120) * ((temp/291.15)**(3/2))

vels = []

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    vels = [row for row in csv_reader][1:]
    vels = [(float(row[15]), float(row[16])) for row in vels]


def calc_q(velpair):
    a = math.sqrt((9 * eta * velpair[0]) / (2 * rho * g))
    result = (sc.pi * 6) / E
    result = result * (velpair[0] + velpair[1]) * math.sqrt(velpair[0])
    result = result * ((1 / (1 + (b / (p * a))))**(3/2))
    result = result * math.sqrt((9 * (eta**3))/ (2 * rho * g))

    result = result / sc.e
    return result

# print(vels)

# print(sc.e)

qs = [calc_q(pair) for pair in vels]

# Plot Results
num_bins = 50

fig, ax = plt.subplots()
n, bins, patches = ax.hist(qs, bins=70)
ax.set_xlabel("q (eV)")
ax.set_ylabel("Frequency")

plt.show()