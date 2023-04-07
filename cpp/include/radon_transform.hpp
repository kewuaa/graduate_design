#pragma once
#include <random>
#include "opencv2/opencv.hpp"


class RadonTransformer
{
    private:
        std::default_random_engine e;
        std::uniform_int_distribution<unsigned int> u;
    public:
        RadonTransformer();
        cv::Mat radon_transform_with_noise(
            const char* img_path,
            double theta,
            double start_angle=0.,
            double end_angle=180.,
            bool crop=true,
            bool norm=true,
            bool add_noise=true
        );
};
