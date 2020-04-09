# CommonRoad Traffic Rule Monitor

## Getting Started
This introduction will give you an overview how to install, parametrize, 
and execute the CommonRoad Traffic Rule Monitor.

We recommend Ubuntu 18.04 as operating system. \
For the monitor you need at least Python 3.6 and the following packages:
* *matplotlib* >= 2.5.0
* *numpy* >= 3.1.0
* *metric-temporal-logic* == 0.1.4
* *python-monitors* >= 0.1.1
* *ruamel.yaml* >= 0.16.10 
* *commonroad-io* >= 2020.1

The usage of the Anaconda Python distribution is recommended. \
You can install the required Python packages with the provided *requirements.txt* file (*pip install -r requirements.txt*). 
Additionally, you need the *[CommonRoad vehicle models](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/tree/master/Python)* which must be added to your Python interpreter path.

Additionally, you need the following software:
* *[commonroad-curvilinear-coordinate-system](./external/curvilinear_coordinate_system)* 
which has to be installed according to the readme inside the linked directory.

## Running the CommonRoad Traffic Rule Monitor
There exist several options to run the CommonRoad traffic rule monitor:
* *[Jupyter notebook](./jupyter/scenario_evaluation.ipynb)* : Test scenarios can be visualized and every vehicle within the scenario 
is evaluated based on a selected set of traffic rules.
* *[Test cases](./test/run_test.py)* : Run all test cases which evaluate the formalized rules in different scenarios.
* *[main.py](./src/main.py)* : Traffic rule evaluation for a single scenario or all scenarios which are located within predefined directories.

The temporal logic formulas of the rules and the rule specific parametrization can be found in the file *[traffic_rules.yaml](./src/traffic_rules.yaml)*.  
Simulation and vehicle parameters can be set in the file *[config.yaml](./src/config.yaml)*.