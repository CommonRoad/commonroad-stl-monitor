## Installation
- Install [anconda](https://www.anaconda.com/)
- Clone and enter the repository
- Create the environment and install the dependencies by
```
conda env create -f environment.yml
```
- Activate the created environment
```
conda activate py-cr37
```
- Checkout submodules
```
git submodule update --init --recursive
```
- Install commonroad-drivability-checker (press Ctrl-c if asked for root permission)
```
sudo apt install libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev
cd external/commonroad-drivability-checker
bash build.sh -e /home/`whoami`/anaconda3/envs/cr-py37 -v 3.7 -i -j 4
cd ../..
```
- Install rtamt
```
pip install external/rtamt
```
- Install package to development path
```
conda develop .
```

## Run the tests
- Run 
```
cd crmonitor/tests
python -m unittest
```
