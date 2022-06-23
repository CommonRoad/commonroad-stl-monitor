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
        "rtamt==0.3",
        "numba==0.51.2",
        "commonroad-io==2021.1",
        "commonroad-vehicle-models==1.0.0",
        "ruamel-yaml>=0.16.12",
        "commonroad-drivability-checker>=2021.4",
        "antlr4-python3-runtime>=4.6,<=4.8"
    ],
    package_data={'crmonitor': ['*.yaml']},
)
