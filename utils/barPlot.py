# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# Values of each group
bars1 = [12, 28, 1]  # detected
bars2 = [28, 7, 16]

# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()

# The position of the bars on the x-axis
r = [0, 1, 2]

# Names of group and bar width
names = ['MNIST', 'CIFAR', 'GTSRB']
barWidth = 1

# Create blue bars
plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# Create red bars, on top of the firs ones
plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)

# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")

# Show graphic
plt.show()