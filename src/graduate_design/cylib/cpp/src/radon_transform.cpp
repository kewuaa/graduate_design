#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "radon_transform.hpp"
#include <vector>


RadonTransformer::RadonTransformer(
    float theta,
    float start_angle,
    float end_angle,
    bool crop,
    bool norm,
    bool add_noise
):
    u(3, 6),
    _theta(theta), _start_angle(start_angle), _end_angle(end_angle),
    _crop(crop), _norm(norm), _add_noise(add_noise) {}


void RadonTransformer::radon_transform_with_noise(
    const char* bytes,
    unsigned int byte_length,
    std::vector<unsigned char>& out_buf
)
{
    CV_Assert((_end_angle - _start_angle) * _theta > 0);
    std::vector<unsigned char> _v(bytes, bytes + byte_length);
    cv::Mat _srcMat = cv::imdecode(_v, cv::IMREAD_GRAYSCALE);

    int _row_num, _col_num, _out_mat_type;
    _col_num = cvRound((_end_angle - _start_angle) / _theta);
    transpose(_srcMat, _srcMat);
    cv::Mat _masked_src;
    cv::Point _center;

    if (_srcMat.type() == CV_32FC1 || _srcMat.type() == CV_64FC1) {
        _out_mat_type = CV_64FC1;
    }
    else {
        _out_mat_type = CV_32FC1;
    }

    if (_crop) {
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
        _t = (_start_angle + _col * _theta);
        cv::Mat _r_matrix = cv::getRotationMatrix2D(_center, _t, 1);
        cv::warpAffine(_masked_src, _rotated_src, _r_matrix, _masked_src.size());
        cv::Mat _col_mat = _radon.col(_col);
        // make projection
        cv::reduce(_rotated_src, _col_mat, 1, cv::REDUCE_SUM, _out_mat_type);
        // add noise
        if (_add_noise && _col % u(e) == 0) {
            cv::filter2D(_col_mat, _col_mat, -1, _kernel);
        }
    }

    if (_norm) {
        normalize(_radon, _radon, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
    if (_add_noise) {
        cv::GaussianBlur(_radon, _radon, {5, 5}, 1.5);
    }
    cv::imencode(".png", _radon, out_buf);
}
