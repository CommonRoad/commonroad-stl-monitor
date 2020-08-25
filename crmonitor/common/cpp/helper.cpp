#include <pybind11/pybind11.h>
#include <algorithm>
#include <math.h>

float rear_s(float d, float l, float s, float theta, float w) {
    return std::min((s - l / 2) * cos(theta) - sin(theta) * (d + w / 2),
    (s - l/2) * cos(theta) - sin(theta) * (d - w / 2));
}

float front_s(float d, float l, float s, float theta, float w) {
    return std::max((s + l / 2) * cos(theta) -sin(theta) * (d + w / 2),
                   (s + l / 2) * cos(theta) -sin(theta) * (d - w / 2));
}

namespace py = pybind11;

PYBIND11_MODULE(cmake_example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("rear_s", &rear_s, "A function which adds two numbers");
    m.def("front_s", &front_s, "A function which adds two numbers");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}
