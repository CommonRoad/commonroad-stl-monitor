import numpy as np
import matplotlib.pyplot as plt

num_vehicles = 1678
R_G1 = 1024
R_G2 = 1673
R_G3 = 1677
R_G4 = 1678
R_G5 = 1662
R_I1 = 1678
R_I2 = 1652
R_I3 = 1678
R_I4 = 1678
R_I5 = 1663
R_0 = 982
height = [R_G1, R_G2, R_G3, R_G5, R_I1, R_I2, R_I3, R_I4, R_0]
bars = ('R_G1', 'R_G2', 'R_G3', 'R_G5', 'R_I1', 'R_I2', 'R_I3', 'R_I4', 'R_G0')
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
