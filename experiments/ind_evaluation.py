import itertools
import logging
import os
import time
from multiprocessing import Process, Queue, Semaphore
import threading
import numpy as np
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.common.vehicle import DynamicObstacleVehicle
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.common.helper import load_yaml
from typing import Iterable, List, Optional, Dict


class IndEvaluator:
    def __init__(
        self,
        scenario_path: "Optional[str]" = None,
        save_filename: "Optional[str]" = None,
        save_filepath: "Optional[str]" = None,
        log_filename: str = "inD_evaluation.log",
        max_scenario_number: int = 400,
        num_threads: int = 8,
    ):
        config_path = os.path.join(os.getcwd(), "../crmonitor/config.yaml")
        self.config = load_yaml(str(config_path))
        self.config["scenario"] = "intersection"
        self.config["intersection_road_network_param"]["map_type"] = "dataset"
        self.max_scenario_number = max_scenario_number
        self.num_threads = num_threads
        self.scenario_path = scenario_path
        # logging
        self.log_filename = log_filename
        self._init_logger()

        self.rules = ["R_IN1", "R_IN3", "R_IN4", "R_IN5"]
        self.use_bool = False

        rules_path = os.path.join(os.getcwd(), "../crmonitor/traffic_rules_rtamt.yaml")
        self.traffic_rules = load_yaml(str(rules_path))
        self.traffic_rules["traffic_rules_param"]["use_mpr"] = False
        self.traffic_rules["traffic_rules_param"]["mpr_scenario"] = "intersection"

        # create data_loader
        self.data_processor = DataProcessor(
            save_filename=save_filename,
            save_filepath=save_filepath,
        )

    @property
    def log_path(self) -> str:
        return os.path.join("./", self.log_filename)

    def _init_logger(self) -> None:
        logger = logging.getLogger("inD_evaluation")
        if not getattr(logger, "is_initialized", False):
            logger.setLevel(logging.DEBUG)
            fileHandler = logging.FileHandler(self.log_path)
            fileHandler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] <%(processName)-11s> {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)

            logger.is_initialized = True

        self.logger = logger

    def evaluation(self):
        for world in self.world_iter():
            self.process_world(world)
        self.data_processor.save_data()

    def evaluation_parallel(self):
        def joined(process: "Process"):
            process.join(1)
            if process.exitcode is not None:
                self.logger.info(f"Process {process.name} joint.")
                return True
            return False

        queue = Queue()
        num_worker = self.num_threads
        semaphore = Semaphore(num_worker)
        list_processes: "List[Process]" = []
        queue.put(self.data_processor.data)
        for world in self.world_iter():
            semaphore.acquire()
            process = Process(
                target=self.process_world_parallel,
                args=(world, queue, semaphore),
            )
            list_processes.append(process)
            process.start()

        while semaphore.get_value() < num_worker:
            time.sleep(1)

        # count_joined = 0
        while len(list_processes) > 1:
            list_processes[:] = [
                process for process in list_processes if not joined(process)
            ]

        self.data_processor.data = queue.get()
        self.data_processor.save_data()

        while len(list_processes) > 0:
            list_processes[:] = [
                process for process in list_processes if not joined(process)
            ]

    def world_iter(self):
        i = 0
        file_names = os.listdir(self.scenario_path)
        for file_name in file_names:
            root, ext = os.path.splitext(file_name)
            if ext == ".xml":
                scenario_file_path = os.path.join(self.scenario_path, file_name)
                scenario, _ = CommonRoadFileReader(scenario_file_path).open(
                    lanelet_assignment=True
                )
                world = World.create_from_scenario(scenario, self.config)
                if str(world.scenario.scenario_id) in self.data_processor.data.index:
                    continue
                if i >= self.max_scenario_number:
                    break
                log_msg = f"PROCESSING {world.scenario.scenario_id}:{i}"
                self.logger.info(log_msg)
                yield world
                i += 1

    def process_world(self, world: "World"):
        for ego_vehicle in world.vehicles:
            self.process_vehicle(world, ego_vehicle)
        self.data_processor.save_data()

    def process_world_parallel(
        self,
        world: "World",
        queue: "Queue",
        semaphore: "threading.Semaphore",
    ):
        self.data_processor.generate_empty_dataframe()
        for ego_vehicle in world.vehicles:
            self.process_vehicle(world, ego_vehicle)
        self.data_processor.append_dataframe(queue.get())
        self.data_processor.save_data()
        queue.put(self.data_processor.data)
        semaphore.release()

    def process_vehicle(self, world: "World", ego_vehicle: DynamicObstacleVehicle):
        rule_violation = {}
        for rule in self.rules:
            try:
                rule_violation[rule] = self.process_rule(world, ego_vehicle, rule)
            except Exception as e:
                raise e
        dict_index = {
            "scenario_id": str(world.scenario.scenario_id),
            "ego_id": ego_vehicle.id,
        }
        self.data_processor.add_reult(dict_index, rule_violation)

    def process_rule(
        self, world: World, ego_vehicle: DynamicObstacleVehicle, rule: str
    ):
        log_msg = f"PROCESSING {world.scenario.scenario_id}:{ego_vehicle.id}:{rule}"
        self.logger.info(log_msg)
        rule_eval = PropositionRuleEvaluator.create_from_config(
            world,
            ego_vehicle,
            rule,
            use_boolean=self.use_bool,
            traffic_rules_config=self.traffic_rules,
        )
        violation = 1.0
        for _ in range(
            rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            if rob < 0:
                violation = -1.0
                break
        dict_evaluation = {"violation_bool": violation}
        return dict_evaluation


class DataProcessor:
    def __init__(
        self,
        save_filename: "Optional[str]" = None,
        save_filepath: "Optional[str]" = None,
    ):
        self.save_filename = save_filename
        self.save_filepath = save_filepath
        self.index = ["scenario_id", "ego_id"]

        self.generate_empty_dataframe()

    def generate_empty_dataframe(self):
        index = pd.MultiIndex.from_tuples([], names=self.index)
        columns = pd.MultiIndex.from_product([["R_IN1"], ["violation_bool"]])
        columns = columns.append(
            pd.MultiIndex.from_product([["R_IN3"], ["violation_bool"]])
        )
        columns = columns.append(
            pd.MultiIndex.from_product([["R_IN4"], ["violation_bool"]])
        )
        columns = columns.append(
            pd.MultiIndex.from_product([["R_IN5"], ["violation_bool"]])
        )
        self.data = pd.DataFrame(index=index, columns=columns)

    def save_data(self, file_name: "Optional[str]" = None):
        if file_name is not None:
            self.save_filename = file_name
        save_path = os.path.join(self.save_filepath, self.save_filename)
        self.data.to_csv(save_path)

    def add_reult(self, dict_index, rule_violation):
        index = tuple(dict_index[f] for f in self.index)
        self.data.loc[index, :] = None
        self.data.loc[index, :].update(self.flatten_dict(rule_violation))

    @staticmethod
    def flatten_dict(nested_dict: "Dict"):
        return {
            (outerKey, innerKey): values
            for outerKey, innerDict in nested_dict.items()
            for innerKey, values in innerDict.items()
        }

    def append_dataframe(self, data: "pd.DataFrame"):
        self.data = pd.concat([self.data, data])
