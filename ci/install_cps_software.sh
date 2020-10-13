#!/usr/bin/env bash
# Allow permissions to all repositories
git config --global url."https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab.lrz.de/".insteadOf "git@gitlab.lrz.de:"

## drivability-checker
#git clone https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker.git "$HOME"/commonroad-drivability-checker
#cd "$HOME"/commonroad-drivability-checker || exit
#bash build.sh -e "/opt/conda/envs/commonroad/" -v 3.6 --cgal --serializer --no-root -j 4

# commonroad-io
git clone --branch develop git@gitlab.lrz.de:cps/commonroad-io.git "$HOME"/commonroad-io
pip install -e "$HOME"/commonroad-io

# curvilinear-coordinate-system
git clone --branch development git@gitlab.lrz.de:cps/commonroad-curvilinear-coordinate-system.git "$HOME"/commonroad-curvilinear-coordinate-system
cd "$HOME"/commonroad-curvilinear-coordinate-system/ || exit
mkdir -p build
cd build || exit
cmake -DADD_PYTHON_BINDINGS=TRUE -DPATH_TO_PYTHON_ENVIRONMENT="/opt/conda/envs/commonroad/" -DPYTHON_VERSION="3.6" -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
python setup.py install
cd "$CI_PROJECT_DIR" || exit