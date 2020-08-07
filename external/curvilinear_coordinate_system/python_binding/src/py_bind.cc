#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef PY_WRAPPER_MODULE_GEOMETRY
void init_module_geometry(py::module &m);
#endif

PYBIND11_MODULE(pycrccosy, m) {
#ifdef PY_WRAPPER_MODULE_GEOMETRY
  init_module_geometry(m);
#endif
}
