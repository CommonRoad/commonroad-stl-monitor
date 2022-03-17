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
    version="0.0.0-dev2",
    install_requires=[
        "antlr4-python3-runtime==4.5",
        "rtamt==0.2.5",
        "numba==0.51.2",
        "commonroad-io==2021.1",
        "commonroad-vehicle-models==1.0.0",
        "ruamel-yaml==0.16.12"
    ],  # package_dir={"": "src"},
)
