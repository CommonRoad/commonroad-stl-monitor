import numpy as np
import matplotlib.pyplot as plt

# num_vehicles = 3605
# R_G1 = 2161
# R_G2 = 3539
# R_G3 = 2046
# R_G4 = 0
# R_G5 = 3579
# R_I1 = 3605
# R_I2 = 3521
# R_I3 = 3605
# R_I4 = 3605
# R_I5 = 0
# R_0 = 982


num_vehicles = 453 + 594 + 443 + 413 + 573 + 404
R_G1 = 282 + 286 + 262 + 198 + 561 + 223
R_G2 = 438 + 581 + 438 + 409 + 474 + 394
R_G3 = 348 + 496 + 304 + 249 + 570 + 277
R_G4 = 450 + 594 + 443 + 413 + 570 + 404
R_I1 = 453 + 594 + 443 + 413 + 573 + 404
R_I2 = 451 + 592 + 434 + 405 + 571 + 403
R_I3 = num_vehicles
R_I4 = 453 + 594 + 443 + 413 + 573 + 404
R_0 = 187 + 198 + 168 + 103 + 277 + 120

height = [R_G1, R_G2, R_G3, R_G4, R_I1, R_I2, R_I3, R_I4, R_0]
bars = ('R_G1', 'R_G2', 'R_G3', 'R_G4', 'R_I1', 'R_I2', 'R_I3', 'R_I4', 'R_G0')
y_pos = np.arange(len(bars))
plt.rcParams['svg.fonttype'] = 'none'
ax = plt.bar(y_pos, height, color='#c3c3c3ff')
plt.axhline(y=num_vehicles, color='black', linestyle='--')
plt.xticks(y_pos, bars, rotation='vertical')

totals = []
# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i, el in enumerate(ax.patches):
    # get_x pulls left or right; get_height pushes up or down
    plt.text(el.get_x()+0.3, el.get_height()-500, str("%6.2f" % ((height[i]*100)/num_vehicles))+'%', color='black', rotation=90)
plt.show()
