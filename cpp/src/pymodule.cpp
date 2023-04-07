#include <stdint.h>
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "radon_transform.hpp"
#include "pygil.hpp"
namespace py = pybind11;
typedef py::array_t<uint8_t> uint8_array;


uint8_array radon_transform_with_noise(
    const char* img_path,
    double theta,
    double start_angle=0.,
    double end_angle=180.,
    bool crop=true,
    bool norm=true,
    bool add_noise=true
)
{
    static RadonTransformer radon_transformer;
    PyGilSwitcher pygil;
    cv::Mat radon = radon_transformer.radon_transform_with_noise(
        img_path,
        theta,
        start_angle,
        end_angle,
        crop,
        norm,
        add_noise
    );
    return uint8_array({radon.rows, radon.cols, 1}, radon.data);
}


PYBIND11_MODULE(cpptrans, m)
{
    m.doc() = "Radon Transform Module";
    m.def(
        "radon_transform_with_noise",
        &radon_transform_with_noise,
        py::arg("img_path"),
        py::arg("theta"),
        py::arg("start_angle")=0.,
        py::arg("end_angle")=180.,
        py::arg("crop")=true,
        py::arg("norm")=true,
        py::arg("add_noise")=true
    );
}
