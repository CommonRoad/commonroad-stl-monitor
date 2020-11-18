#!/bin/bash

echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
apt-get update -y > /dev/null
apt-get install -y apt-utils > /dev/null
apt-get install -y build-essential > /dev/null
apt-get install -y pkg-config > /dev/null
apt-get install -y libeigen3-dev > /dev/null
apt-get install -y libboost-all-dev
apt-get install -y cmake > /dev/null


#git clone https://github.com/nickovic/rtamt
#cd rtamt/rtamt
cd "${CI_PROJECT_DIR}/external/rtamt/rtamt"
mkdir build
cd build
cmake -DPythonVersion=3 ../
make -j 20
cd ../../
pip install .