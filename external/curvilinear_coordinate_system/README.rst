=====================================================
CommonRoad-Curvilinear-Coordinatesystem: Installation
=====================================================

System Requirements
-------------------

The software is written in C++ and Python 3.6 and tested on MacOs and Linux. The usage of the Anaconda_ Python distribution is strongly recommended.

.. _Anaconda: http://www.anaconda.com/download/#download


Dependencies
------------

C++ Libraries:

* Boost
* OpenMP

The following libraries are needed for the Python wrapper (can be found under external/):

* `pybind11 <https://github.com/pybind/pybind11>`_

The required Python dependencies are:

* numpy>=1.13
* matplotlib>=2.2.2
* networkx
* commonroad-io

Full Installation with Anaconda
-------------------------------

It is assumed that you have installed Anaconda_.

#. Install all dependencies.

#. Open your console in the root folder of the repository.

#. Activate your environment with 

	.. code-block:: console

		   $ source activate commonroad-py36

#. Compile the library by running
    
        .. code-block:: console
            
            $ mkdir build
            $ cd build
            $ cmake -DADD_PYTHON_BINDINGS=TRUE -DPATH_TO_PYTHON_ENVIRONMENT="/path/to/anaconda_env/" -DPYTHON_VERSION="X.X" -DCMAKE_BUILD_TYPE=Release ..
            $ make

        Note that you have to replace 

         - *"/path/to/anaconda_env/"* with the path to your Anaconda environment, and
         - *"X.X"* with the Python version of your Anaconda environment (e.g., 3.6)

#. Install commonroad-curvilinear-coordinate-system with

    .. code-block:: bash

            $ cd ..
            $ python setup.py install

    **OR** add the root folder of the commonroad-curvilinear-coordinate-system to your Python-Interpreter.
