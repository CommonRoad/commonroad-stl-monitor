import numpy as np
import matplotlib.pyplot as plt

# Make a fake dataset
num_vehicles = 2366
max_speed_limit = 2365
min_speed_limit = 1994
no_unnecessary_braking = 2366
safe_distance = 1532
all = 1194
height = [safe_distance, no_unnecessary_braking, max_speed_limit, min_speed_limit, all]
bars = ('R_G1', 'R_G2', 'R_G3', 'R_G4', 'R_G0')
y_pos = np.arange(len(bars))
plt.rcParams['svg.fonttype'] = 'none'
ax = plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.4))
plt.axhline(y=num_vehicles, color='black', linestyle='--')
plt.xticks(y_pos, bars)

totals = []
# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i, el in enumerate(ax.patches):
    # get_x pulls left or right; get_height pushes up or down
    plt.text(el.get_x()+0.35, el.get_height()-700, str("%6.2f" % ((height[i]*100)/num_vehicles))+'%', color='black', rotation=90)
plt.show()
