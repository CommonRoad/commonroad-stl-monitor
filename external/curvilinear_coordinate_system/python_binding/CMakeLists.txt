pybind11_add_module(pycrccosy src/module_geometry.cc src/py_bind.cc)

set_target_properties(pycrccosy PROPERTIES COMPILE_DEFINITIONS "PY_WRAPPER_MODULE_GEOMETRY=1 ")

target_link_libraries(pycrccosy PRIVATE crccosy)

message(STATUS "${PYTHON_INCLUDE_DIR}")
