import numpy as np
import matplotlib.pyplot as plt

# Make a fake dataset
num_vehicles = 24
max_speed_limit = 24
min_speed_limit = 22
no_unnecessary_braking = 24
safe_distance = 22
all = 20
height = [max_speed_limit, min_speed_limit, no_unnecessary_braking, safe_distance, all]
bars = ('R1', 'R2', 'R3', 'R4', 'R0')
y_pos = np.arange(len(bars))
plt.rcParams['svg.fonttype'] = 'none'
plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
plt.axhline(y=num_vehicles, color='black')
plt.xticks(y_pos, bars)
plt.show()
