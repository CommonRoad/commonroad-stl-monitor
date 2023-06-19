import matplotlib as mpl
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import chaikins_corner_cutting, resample_polyline
from matplotlib import pyplot as plt

mpl.use("tkagg")

scn = CommonRoadFileReader("DEU_LocationALower-25_11_T-1.xml").open(True)[0]

obs = scn.obstacle_by_id(30)
state = obs.prediction.trajectory.state_at_time_step(126)

ref_path = scn.lanelet_network.find_lanelet_by_id(3).center_vertices
new_ref_path = ref_path
# for i in range(0, 1):
#     new_ref_path = chaikins_corner_cutting(new_ref_path)
new_ref_path = resample_polyline(new_ref_path, 0.5)


clcs = CurvilinearCoordinateSystem(new_ref_path, resample=False, eps2=0.0)
proj = np.array(clcs.projection_domain())


# rnd = MPRenderer()
# scn.draw(rnd)
# rnd.render()
#
# plt.plot(*proj.T, zorder=1000)
plt.plot(*new_ref_path.T, zorder=1000)
# plt.plot(clcs.ref_pos, clcs.ref_curv)
plt.show()

s, d = clcs.convert_to_curvilinear_coords(*state.position)
pass
