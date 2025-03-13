#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/boards_from_corners.h"


namespace py = pybind11;
using namespace cbdetect;

// Convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    auto buf = input.request();
    if (buf.ndim != 2 && buf.ndim != 3) {
        throw std::runtime_error("Input array must be 2D (grayscale) or 3D (color)");
    }
    
    int type = (buf.ndim == 2) ? CV_8UC1 : CV_8UC3;
    return cv::Mat(buf.shape[0], buf.shape[1], type, (void*)buf.ptr);
}

PYBIND11_MODULE(pycbdetect, m) {
    m.doc() = "Python bindings for cbdetect using pybind11";

    py::enum_<DetectMethod>(m, "DetectMethod")
        .value("TemplateMatchFast", DetectMethod::TemplateMatchFast)
        .value("TemplateMatchSlow", DetectMethod::TemplateMatchSlow)
        .value("HessianResponse", DetectMethod::HessianResponse)
        .value("LocalizedRadonTransform", DetectMethod::LocalizedRadonTransform);

    py::enum_<CornerType>(m, "CornerType")
        .value("SaddlePoint", CornerType::SaddlePoint)
        .value("MonkeySaddlePoint", CornerType::MonkeySaddlePoint);

    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("show_processing", &Params::show_processing)
        .def_readwrite("show_debug_image", &Params::show_debug_image)
        .def_readwrite("show_grow_processing", &Params::show_grow_processing)
        .def_readwrite("norm", &Params::norm)
        .def_readwrite("polynomial_fit", &Params::polynomial_fit)
        .def_readwrite("norm_half_kernel_size", &Params::norm_half_kernel_size)
        .def_readwrite("polynomial_fit_half_kernel_size", &Params::polynomial_fit_half_kernel_size)
        .def_readwrite("init_loc_thr", &Params::init_loc_thr)
        .def_readwrite("score_thr", &Params::score_thr)
        .def_readwrite("strict_grow", &Params::strict_grow)
        .def_readwrite("overlay", &Params::overlay)
        .def_readwrite("occlusion", &Params::occlusion)
        .def_readwrite("detect_method", &Params::detect_method)
        .def_readwrite("corner_type", &Params::corner_type)
        .def_readwrite("radius", &Params::radius);

    py::class_<Corner>(m, "Corner")
        .def(py::init<>())
        .def_readwrite("p", &Corner::p)
        .def_readwrite("r", &Corner::r)
        .def_readwrite("v1", &Corner::v1)
        .def_readwrite("v2", &Corner::v2)
        .def_readwrite("v3", &Corner::v3)
        .def_readwrite("score", &Corner::score);

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def_readwrite("idx", &Board::idx)
        .def_readwrite("energy", &Board::energy)
        .def_readwrite("num", &Board::num);

    m.def("find_corners", [](py::array_t<uint8_t> img_np, Params params) {
        cv::Mat img = numpy_to_mat(img_np);
        Corner corners;
        find_corners(img, corners, params);
        return corners;
    }, "Detect corners in an image");

    m.def("boards_from_corners", [](py::array_t<uint8_t> img_np, const Corner& corners, Params params) {
        cv::Mat img = numpy_to_mat(img_np);
        std::vector<Board> boards;
        boards_from_corners(img, corners, boards, params);
        return boards;
    }, "Detect checkerboard from corners");
}
