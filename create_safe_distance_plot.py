from commonroad.common.file_reader import CommonRoadFileReader
from statistical_evaluation import create_vehicle

filename = "./../../../commonroad/scenarios/tum_cps/scenarios/" + "hand-crafted/" + "DEU_A99-1_2_T-1" + ".xml"
    #filename = "highD_generator/scenarios/DEU_LocationB-1_1_T-1.xml"
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()
