# CommonRoad Traffic Rule Monitor

## Getting Started
This introduction will give you an overview how to install, parametrize, 
and execute the CommonRoad Traffic Rule Monitor.

We recommend Ubuntu 18.04 as operating system. \
For the monitor you need at least Python 3.6 and the following packages:
* *matplotlib* >= 2.5.0
* *numpy* >= 3.1.0
* *metric-temporal-logic*  >= 0.1.4

The usage of the Anaconda Python distribution is recommended. \
You can install the required Python packages with the provided *requirements.txt* file (*pip install -r requirements.txt*). 

Additionally, you need the following software:
* *[CommonRoad IO](./commonroad/io)* 
which has to be installed by executing **python setup.py install** within the linked directory. 
* *[CommonRoad curvilinear coordinate system](./commonroad/curvilinear)* 
which has to be installed according to the readme inside the linked directory.

## Running the CommonRoad Traffic Rule Monitor
TODO