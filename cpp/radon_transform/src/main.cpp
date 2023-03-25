#include <iostream>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "pybind11/gil.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;
typedef py::array_t<uint8_t> uint8_array;


class PyGilSwitcher
{
    public:
        PyGilSwitcher()
        {
            py::gil_scoped_release _release;
        }
        ~PyGilSwitcher()
        {
            py::gil_scoped_acquire _acquire;
        }
};


class Transformer
{
    private:
        std::default_random_engine e;
        std::uniform_int_distribution<unsigned int> u;

    public:
        Transformer(): u(3, 9) {}

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
            PyGilSwitcher pygilswitch;
            cv::Mat _srcMat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            CV_Assert(_srcMat.dims == 2);
            CV_Assert(_srcMat.channels() == 1);
            CV_Assert((end_angle - start_angle) * theta > 0);

            int _row_num, _col_num, _out_mat_type;
            _col_num = cvRound((end_angle - start_angle) / theta);
            transpose(_srcMat, _srcMat);
            cv::Mat _masked_src;
            cv::Point _center;

            if (_srcMat.type() == CV_32FC1 || _srcMat.type() == CV_64FC1) {
                _out_mat_type = CV_64FC1;
            }
            else {
                _out_mat_type = CV_32FC1;
            }

            if (crop) {
                // crop the source into square
                _row_num = cv::min(_srcMat.rows, _srcMat.cols);
                cv::Rect _crop_ROI(
                        _srcMat.cols / 2 - _row_num / 2,
                        _srcMat.rows / 2 - _row_num / 2,
                        _row_num, _row_num);
                _srcMat = _srcMat(_crop_ROI);
                // crop the source into circle
                cv::Mat _mask(_srcMat.size(), CV_8UC1, cv::Scalar(0));
                _center = cv::Point(_srcMat.cols / 2, _srcMat.rows / 2);
                circle(_mask, _center, _srcMat.cols / 2, cv::Scalar(255), cv::FILLED);
                _srcMat.copyTo(_masked_src, _mask);
            }
            else {
                // avoid cropping corner when rotating
                _row_num = cvCeil(sqrt(_srcMat.rows * _srcMat.rows + _srcMat.cols * _srcMat.cols));
                _masked_src = cv::Mat(cv::Size(_row_num, _row_num), _srcMat.type(), cv::Scalar(0));
                _center = cv::Point(_masked_src.cols / 2, _masked_src.rows / 2);
                _srcMat.copyTo(_masked_src(cv::Rect(
                                           (_row_num - _srcMat.cols) / 2,
                                           (_row_num - _srcMat.rows) / 2,
                                           _srcMat.cols, _srcMat.rows)));
            }

            double _t;
            cv::Mat _rotated_src;
            cv::Mat _radon(_row_num, _col_num, _out_mat_type);
            float _kernel_data[25];
            for (unsigned int i = 0;i < 25; i++) {
                _kernel_data[i] = 0.1;
            }
            cv::Mat _kernel(25, 1, CV_32F, _kernel_data);
            // cv::Mat _kernel = (cv::Mat_<float>(10, 1) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

            for (int _col = 0; _col < _col_num; _col++) {
                // rotate the source by _t
                _t = (start_angle + _col * theta);
                cv::Mat _r_matrix = cv::getRotationMatrix2D(_center, _t, 1);
                cv::warpAffine(_masked_src, _rotated_src, _r_matrix, _masked_src.size());
                cv::Mat _col_mat = _radon.col(_col);
                // make projection
                cv::reduce(_rotated_src, _col_mat, 1, cv::REDUCE_SUM, _out_mat_type);
                // add noise
                if (add_noise && _col % u(e) == 0) {
                    cv::filter2D(_col_mat, _col_mat, -1, _kernel);
                }
            }

            if (norm) {
                normalize(_radon, _radon, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            }
            if (add_noise) {
                cv::GaussianBlur(_radon, _radon, {5, 5}, 1.5);
            }

            return uint8_array({_radon.rows, _radon.cols, 1}, _radon.data);
        }
};


PYBIND11_MODULE(cpptrans, m)
{
    m.doc() = "Radon Transform Module";
    py::class_<Transformer >(m, "Transform")
        .def(py::init())
        .def(
            "radon_transform_with_noise",
            &Transformer::radon_transform_with_noise,
            py::arg("img_path"),
            py::arg("theta"),
            py::arg("start_angle")=0.,
            py::arg("end_angle")=180.,
            py::arg("crop")=true,
            py::arg("norm")=true,
            py::arg("add_noise")=true
        );
}
