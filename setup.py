from setuptools import setup

setup(
    name='commonroad-monitor',
    version='0.5.0',
    packages=['crmonitor'],
    url='https://commonroad.in.tum.de/',
    license='',
    author='Sebastian Maierhofer',
    author_email='sebastian.maierhofer@tum.de',
    description='Traffic Rule Monitor for CommonRoad Scenarios',
    install_requires=[
        'python-monitors>=0.1.1',
        'scipy>=1.4.1',
        'numpy>=1.16.4'
        'metric-temporal-logic>=0.1.7'
        'commonroad-io==2020.2'
        'matplotlib>=2.5.0'
        'ruamel.yaml>=0.16.10'
        'bezier>=2020.2.3'
        'antlr4-python3-runtime==4.7.2'
        'jupyter'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
