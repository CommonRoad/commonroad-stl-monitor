import matplotlib.pyplot as plt

# import functions to read xml file and visualize commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# generate path of the file to be opened
file_path = "../scenarios/test_intersection/DEU_Intersectionwithlightsandsigns-1_1_T-1.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

plt.figure(figsize=(25, 10))
rnd = MPRenderer()
rnd.draw_params.time_begin = 0
scenario.draw(rnd)
planning_problem_set.draw(rnd)
rnd.render()
plt.show()
# plot the scenario for 40 time step, here each time step corresponds to 0.1 second
for i in range(0, 40):
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    # set time step in draw_params
    rnd.draw_params.time_begin = i
    # plot the scenario at different time step
    scenario.draw(rnd)
    # plot the planning problem set
    planning_problem_set.draw(rnd)
    rnd.render()