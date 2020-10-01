#include <pybind11/pybind11.h>
#include <algorithm>
#include <math.h>

float rear_s(float d, float length, float s, float theta, float width) {
    return std::min({(length / 2) * cos(theta) - (width / 2) * sin(theta) + s,
                    (length / 2) * cos(theta) - (-width / 2) * sin(theta) + s,
                    (-length / 2) * cos(theta) - (width / 2) * sin(theta) + s,
                    (-length / 2) * cos(theta) - (-width / 2) * sin(theta) + s});
}

float front_s(float d, float length, float s, float theta, float width) {
    return std::max({(length / 2) * cos(theta) - (width / 2) * sin(theta) + s,
                    (length / 2) * cos(theta) - (-width / 2) * sin(theta) + s,
                    (-length / 2) * cos(theta) - (width / 2) * sin(theta) + s,
                    (-length / 2) * cos(theta) - (-width / 2) * sin(theta) + s});
}

namespace py = pybind11;

PYBIND11_MODULE(crmonitor_cpp, m) {
    m.doc() = "CommonRoad Monitor pybind11 plugin";

    m.def("rear_s", &rear_s, "Calculates rear position of vehicle", py::arg("d"), py::arg("length"), py::arg("s"),
        py::arg("theta"), py::arg("width"));
    m.def("front_s", &front_s, "Calculates front position of vehicle", py::arg("d"), py::arg("length"), py::arg("s"),
        py::arg("theta"), py::arg("width"));

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
