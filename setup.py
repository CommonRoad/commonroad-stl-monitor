from setuptools import find_packages, setup

setup(
    name="stl-crmonitor",
    packages=find_packages(
        exclude=[
            "external",
            "scenarios",
            "tools",
        ]
    ),
    version="0.0.0.dev4",
    install_requires=[
        "rtamt @ https://github.com/cirrostratus1/rtamt/archive/6ef48e956837f5b60f514011d3d6eabf6307e042.zip",
        "numba==0.51.2",
        "commonroad-io==2022.1",
        "commonroad-vehicle-models==2.0.0",
        "ruamel-yaml>=0.16.12",
        "commonroad-drivability-checker>=2021.4",
        "antlr4-python3-runtime>=4.5,<4.6"
    ],
    package_data={'crmonitor': ['*.yaml']},
)
