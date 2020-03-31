import unittest

from commonroad.common.file_reader import CommonRoadFileReader
from src.common.commonroad_evaluation import CommonRoadObstacleEvaluation


class TestCommonRoadMonitor(unittest.TestCase):
    def setUp(self):
        self.cr_eval = CommonRoadObstacleEvaluation("../src/")
        self.test_scenario_dir = "../scenarios/test/"

    def test_keeps_max_lane_speed_limit(self):
        # one vehicle which always violates speed limit (1002)
        # two vehicles which never violate speed limit (1001, 1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_max_speed_limit.xml").open()
        exp_result = [(1000, {'R_G3': False}), (1001, {'R_G3': True}),
                      (1002, {'R_G3': False}), (1003, {'R_G3': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG3"])
        print("Max Lane Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_fov_speed_limit(self):
        # two vehicles which always violate speed limit (1001, 1002)
        # one vehicle which never violates speed limit (1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = ["SRG3"]
        exp_result = [(1000, {'R_G3': False}), (1001, {'R_G3': False}),
                      (1002, {'R_G3': False}), (1003, {'R_G3': True})]
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 32
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG3"])
        print("Max FOV Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 60

    def test_keeps_braking_speed_limit(self):
        # two vehicles which always violate speed limit (1001, 1002)
        # one vehicle which never violates speed limit (1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_max_speed_limit.xml").open()
        exp_result = [(1000, {'R_G3': False}), (1001, {'R_G3': False}),
                      (1002, {'R_G3': False}), (1003, {'R_G3': True})]
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 32
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG3"])
        print("Max Braking Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 60

    def test_keeps_min_speed_limit(self):
        # two lanes with minimum speed limit sign
        # one vehicle which keeps minimum speed limit based on sign (1000)
        # one vehicle which violates minimum speed limit based on sign (1001)
        # two vehicles which preserves traffic flow (1001 ,1004)
        # two vehicles without following vehicle (1000, 1002)
        # one vehicle which does not preserve traffic flow with leading and following vehicle (1003)
        # one vehicle which drives to alone and slow on single lane -> according rule false (1005)

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_min_speed_limit.xml").open()
        exp_result = [(1000, {'R_G4': False}), (1001, {'R_G4': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG4"])
        print("Min Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_preserve_traffic_flow(self):
        # two vehicles which preserves traffic flow (1001 ,1004)
        # two vehicles without following vehicle (1000, 1002)
        # one vehicle which does not preserve traffic flow with leading and following vehicle (1003)
        # one vehicle which drives to alone and slow on single lane -> according rule false (1005)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_preserve_traffic_flow.xml").open()
        exp_result = [(1000, {'R_G5': True}), (1001, {'R_G5': True}),
                      (1002, {'R_G5': True}), (1003, {'R_G5': False}),
                      (1004, {'R_G5': True}), (1005, {'R_G5': False})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG5"])
        print("Traffic Flow Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_safe_distance(self):
        # three vehicles which have no leading vehicle (1001, 1004, 1006)
        # one vehicle which violates safe distance to directly leading vehicle (1003)
        # one vehicle which violates safe distance to two leading vehicles (1002)
        # one vehicle which violates safe distance partially (1000)
        # one vehicle which always keeps safe distance (1005)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir + "test_safe_distance.xml").open()
        exp_result = [(1000, {'R_G1_veh_1001': False, 'R_G1_veh_1002': True, 'R_G1_veh_1003': True,
                              'R_G1_veh_1004': True, 'R_G1_veh_1005': True, 'R_G1_veh_1006': True}),
                      (1001, {'R_G1_veh_1000': True, 'R_G1_veh_1002': True, 'R_G1_veh_1003': True,
                              'R_G1_veh_1004': True, 'R_G1_veh_1005': True, 'R_G1_veh_1006': True}),
                      (1002, {'R_G1_veh_1000': True, 'R_G1_veh_1001': True, 'R_G1_veh_1003': False,
                              'R_G1_veh_1004': False, 'R_G1_veh_1005': True, 'R_G1_veh_1006': True}),
                      (1003, {'R_G1_veh_1000': True, 'R_G1_veh_1001': True, 'R_G1_veh_1002': True,
                              'R_G1_veh_1004': False, 'R_G1_veh_1005': True, 'R_G1_veh_1006': True}),
                      (1004, {'R_G1_veh_1000': True, 'R_G1_veh_1001': True, 'R_G1_veh_1002': True,
                              'R_G1_veh_1003': True, 'R_G1_veh_1005': True, 'R_G1_veh_1006': True}),
                      (1005, {'R_G1_veh_1000': True, 'R_G1_veh_1001': True, 'R_G1_veh_1002': True,
                              'R_G1_veh_1003': True, 'R_G1_veh_1004': True, 'R_G1_veh_1006': True}),
                      (1006, {'R_G1_veh_1000': True, 'R_G1_veh_1001': True, 'R_G1_veh_1002': True,
                              'R_G1_veh_1003': True, 'R_G1_veh_1004': True, 'R_G1_veh_1005': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG1"])
        print("Safe Distance Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_unnecessary_braking_1(self):
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes normal (1006)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_unnecessary_braking_1.xml").open()
        exp_result = [(1000, {'R_G2': True}), (1001, {'R_G2': True}), (1002, {'R_G2': False}),
                      (1005, {'R_G2': True}), (1006, {'R_G2': True}), (1007, {'R_G2': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG2"])
        print("Unnecessary Braking Test 1:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_unnecessary_braking_2(self):
        # one vehicle which without leading vehicle which brakes very strong, because it is necessary (1000)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_unnecessary_braking_2.xml").open()
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 5
        exp_result = [(1000, {'R_G2': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRG2"])
        print("Unnecessary Braking Test 2:")
        print(result)
        self.assertEqual(exp_result, result)
        self.cr_eval.ego_vehicle_param["fov_speed_limit"] = 60

    def test_standstill(self):
        # one vehicle which is in standstill with a leading vehicle in standstill(1000)
        # one vehicle which is in standstill without a leading vehicle in standstill and which is not
        # part of a congestion a leading vehicle in standstill (1001)
        # one vehicle which drives with higher velocity (1002)
        # seven vehicles which are in a congestion and drive with slow velocity (1003, 1004, 1006, 1007, 1008, 1009,
        # 1010)
        # one vehicle which is in standstill and part of a congestion (1005)

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir + "test_standstill.xml").open()
        exp_result = [(1000, {'R_I1': True}), (1001, {'R_I1': False}), (1002, {'R_I1': True}), (1003, {'R_I1': True}),
                      (1004, {'R_I1': True}), (1005, {'R_I1': True}), (1006, {'R_I1': True}), (1007, {'R_I1': True}),
                      (1008, {'R_I1': True}), (1009, {'R_I1': True}), (1010, {'R_I1': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI1"])
        print("Standstill:")
        print(result)
        self.assertEqual(exp_result, result)
        print(self.cr_eval.eval_dict)

    def test_reversing_and_u_turn(self):
        # one vehicle which drives first in correct direction and than reversely (1000)
        # one vehicle which drives always reversely (1001)
        # one vehicle which drives always in correct direction (1002)
        # one vehicle which makes a u-turn (1003)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_reversing_and_u_turn.xml").open()
        exp_result = [(1000, {'R_I3': False}), (1001, {'R_I3': False}), (1002, {'R_I3': False}), (1003, {'R_I3': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI3"])
        print("Reversing and U-turn:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_overtaking_right_congestion(self):
        # one vehicle which overtakes a congestion slightly faster (1000)
        # one vehicle which overtakes a congestion too fast (1001)
        # all other vehicles a part of a congestion
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_overtaking_right_congestion.xml").open()
        exp_result = [(1000, {'R_I2_veh_1001': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1001, {'R_I2_veh_1000': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': False,
                              'R_I2_veh_1004': False, 'R_I2_veh_1005': False, 'R_I2_veh_1006': False,
                              'R_I2_veh_1007': False, 'R_I2_veh_1008': False}),
                      (1002, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1003, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1004, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1005, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1006, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1007, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1006': True, 'R_I2_veh_1008': True}),
                      (1008, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1006': True, 'R_I2_veh_1007': True}),
                      ]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI2"])
        print("Overtaking right congestion:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_overtaking_right_broad_lane_marking(self):
        # one vehicle right of a broad lane marking which overtakes on the right side one vehicle left of a broad
        # lane marking and one vehicle right of a broad lane marking (1001)
        # three vehicle overtaking a vehicle left of a broad lane marking (1000, 1002, 1003)
        # one vehicle left of a broad lane marking which overtakes on the right side another vehicle also left
        # of a broad lane marking (1004)
        # one vehicle driving on the leftmost lane (1005)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_overtaking_right_broad_lane_marking.xml").open()
        exp_result = [(1000, {'R_I2_veh_1001': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True}),
                      (1001, {'R_I2_veh_1000': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': False,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True}),
                      (1002, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True}),
                      (1003, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True}),
                      (1004, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1005': False}),
                      (1005, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI2"])
        print("Overtaking right broad lane marking:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_overtaking_right_normal_street(self):
        # one vehicle right of a broad lane marking which overtakes (1001)
        # one vehicle right of a broad lane marking which does not overtake (1000)
        # two vehicles driving on the left (1002, 1003)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_overtaking_right_normal.xml").open()
        exp_result = [(1000, {'R_I2_veh_1001': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': True}),
                      (1001, {'R_I2_veh_1000': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': False}),
                      (1002, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1003': True}),
                      (1003, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True}),
                      ]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI2"])
        print("Overtaking right normal road:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_overtaking_access_ramp(self):
        # one vehicle overtaking on access ramp (1002)
        # one vehicle driving exactly left of vehicle on access ramp (1000)
        # one vehicle driving slowly on leftmost lane (1004)
        # one vehicle on main carriage way overtaking vehicle on leftmost lane (1003)
        # one vehicle which is overtaken by vehicle on access ramp (1001)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_overtaking_access_ramp.xml").open()
        exp_result = [(1000, {'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True}),
                      (1001, {'R_I2_veh_1000': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True}),
                      (1002, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True}),
                      (1003, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True,
                              'R_I2_veh_1002': True, 'R_I2_veh_1004': False}),
                      (1004, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True,
                              'R_I2_veh_1002': True, 'R_I2_veh_1003': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI2"])
        print("Overtaking right access ramp:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_overtaking_exit_ramp(self):
        # one vehicle overtaking on exit ramp with high velocity (1001)
        # one vehicle overtaking on access ramp with appropriate velocity (1000)
        # all other vehicles are part of a traffic jam
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_overtaking_exit_ramp.xml").open()
        exp_result = [(1000, {'R_I2_veh_1001': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1001, {'R_I2_veh_1000': True, 'R_I2_veh_1002': True, 'R_I2_veh_1003': False,
                              'R_I2_veh_1004': False, 'R_I2_veh_1005': False, 'R_I2_veh_1006': False,
                              'R_I2_veh_1007': False, 'R_I2_veh_1008': False}),
                      (1002, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1003': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1003, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1004': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1004, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1005': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1005, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1006': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1006, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1007': True, 'R_I2_veh_1008': True}),
                      (1007, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1006': True, 'R_I2_veh_1008': True}),
                      (1008, {'R_I2_veh_1000': True, 'R_I2_veh_1001': True, 'R_I2_veh_1002': True,
                              'R_I2_veh_1003': True, 'R_I2_veh_1004': True, 'R_I2_veh_1005': True,
                              'R_I2_veh_1006': True, 'R_I2_veh_1007': True}),
                      ]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI2"])
        print("Overtaking right exit ramp:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_consider_entering_vehicles(self):
        # one vehicle driving always in the left most lane (1001)
        # one vehicle changing to rightmost main carriage way lane (1000)
        # one vehicle entering main carriage way (1002)
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_consider_entering_vehicles.xml").open()
        exp_result = [(1000, {'R_I5_veh_1001': True, 'R_I5_veh_1002': False}),
                      (1001, {'R_I5_veh_1000': True, 'R_I5_veh_1002': True}),
                      (1002, {'R_I5_veh_1000': True, 'R_I5_veh_1001': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI5"])
        print("Considering entering vehicles:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_emergency_lane_broad_enough_with_shoulder(self):
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_emergency_three_lanes_with_shoulder.xml").open()
        exp_result = [(1000, {'R_I4': False}), (1001, {'R_I4': True}), (1002, {'R_I4': True}), (1003, {'R_I4': True}),
                      (1004, {'R_I4': False}), (1005, {'R_I4': False}), (1006, {'R_I4': True}), (1007, {'R_I4': True}),
                      (1008, {'R_I4': False}), (1009, {'R_I4': True}), (1010, {'R_I4': False}), (1011, {'R_I4': True}),
                      (1012, {'R_I4': True}), (1013, {'R_I4': True}), (1014, {'R_I4': True}), (1015, {'R_I4': True}),
                      (1016, {'R_I4': False}), (1017, {'R_I4': True}), (1018, {'R_I4': True}), (1019, {'R_I4': True}),
                      (1020, {'R_I4': True}), (1021, {'R_I4': False}), (1022, {'R_I4': False}), (1023, {'R_I4': True})]
        result = self.cr_eval.evaluate_scenario(scenario, ["F_SRI4"])
        print("Test emergency lane:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_emergency_lane_not_broad_enough(self):
        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir +
                                                              "test_emergency_two_lanes_not_broad_enough.xml").open()
        exp_result = [(1000, {'R_I4': False}), (1001, {'R_I4': True}), (1002, {'R_I4': True}), (1003, {'R_I4': True}),
                      (1004, {'R_I4': False}), (1005, {'R_I4': False}), (1006, {'R_I4': False}), (1007, {'R_I4': False}),
                      (1008, {'R_I4': False}), (1009, {'R_I4': False}), (1010, {'R_I4': False}), (1011, {'R_I4': True}),
                      (1012, {'R_I4': False}), (1013, {'R_I4': True}), (1014, {'R_I4': False})]
        result = self.cr_eval.evaluate_scenario(scenario, ["SRI4"])
        print("Test emergency lane:")
        print(result)
        self.assertEqual(exp_result, result)


if __name__ == '__main__':
    unittest.main()
