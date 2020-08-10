# CommonRoad Traffic Rule Monitor

**Note:** !!!Robustness mode only works for forward rules!!!

**Note:** Currently, major changes are being made, so runtime errors may occur. 
In addition, not all of the options listed below to execute the software work. 
We try to finish the changes as soon as possible. The test cases should work.
The Develop branch is always most up-to-date

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
* *[main.py](crmonitor/main.py)* : Traffic rule evaluation for a single scenario or all scenarios which are located within predefined directories.

The main file can be started from the commonroad_monitor directory via commandline by executing  
`python -m src.main --max_num_scenarios #Scenarios --scenario_directories /directory_1 /../directory_2`.  
This should be used for the sequential evaluation of scenarios.

In the following the different parameters are explained:
* **max_num_scenarios**: The maximum number scenarios which should be evaluated. The default number is *10*.
* **scenario_directories**: List of scenario directories.

You can list the different parameters by executing `python main.py -h`.

The temporal logic formulas of the rules and the rule specific parametrization can be found in the file *[traffic_rules.yaml](crmonitor/traffic_rules.yaml)*.  
Simulation and vehicle parameters can be set in the file *[config.yaml](crmonitor/config.yaml)*.  
The test scenarios (*[./scenarios/test](scenarios/test_interstate)*) are described in the test script as comment in each test (*[./test/test_rules.py](tests/test_rules.py)*).