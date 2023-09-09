import itertools
import logging
import os
import time
from multiprocessing import Process, Queue, Semaphore
import numpy as np
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.common.helper import load_yaml
from typing import Iterable, List, Optional


class IndEvaluator:

    def __init__(
        self,
        scenario_path: "Optional[str]" = None,
        save_filename: "Optional[str]" = None,
        save_filepath: "Optional[str]" = None,
        log_filename: str = "inD_evaluation.log",
        max_scenario_number: int = 400,
    ):
        config_path = os.path.join(os.getcwd(), "../crmonitor/config.yaml")
        self.config = load_yaml(str(config_path))
        self.config["scenario"] = "intersection"
        self.config["intersection_road_network_param"]["map_type"] = "dataset"
        self.max_scenario_number = max_scenario_number
        self.scenario_path = scenario_path
        # logging
        self.log_filename = log_filename
        self._init_logger()

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
        pass

    def world_iter(self):
        i = 0
        file_names = os.listdir(self.scenario_path)
        for file_name in file_names:
            root, ext = os.path.splitext(file_name)
            if ext == ".xml":
                scenario_file_path = os.path.join(self.scenario_path, file_name)
                scenario, _ = CommonRoadFileReader(scenario_file_path).open(lanelet_assignment=True)
                world = World.create_from_scenario(scenario, self.config)
                if str(world.scenario.scenario_id) in self.data_processor.data.index:
                    continue
                if i >= self.max_scenario_number:
                    break
                log_msg = (
                    f"PROCESSING {world.scenario.scenario_id}:{i}"
                )
                self.logger.info(log_msg)
                yield world
                i += 1

    def process_world(self, world: "World"):
        pass


class DataProcessor:
    def __init__(self,
                 save_filename: "Optional[str]" = None,
                 save_filepath: "Optional[str]" = None,
                 ):
        self.save_filename = save_filename
        self.save_filepath = save_filepath
        self.generate_empty_dataframe()

    def generate_empty_dataframe(self):
        index = pd.MultiIndex.from_tuples([], names=["scenario_id", "ego_id"])
        columns = pd.MultiIndex.from_product([["R_IN1"], ["violation_bool"]])
        columns = columns.append(pd.MultiIndex.from_product([["R_IN2"], ["violation_bool"]]))
        columns = columns.append(pd.MultiIndex.from_product([["R_IN3"], ["violation_bool"]]))
        self.data = pd.DataFrame(index=index, columns=columns)

    def save_data(self, file_name: "Optional[str]" = None):
        if file_name is not None:
            self.save_filename = file_name
        save_path = os.path.join(self.save_filepath, self.save_filename)
        self.data.to_csv(save_path)
