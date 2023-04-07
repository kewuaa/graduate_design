#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include "radon_transform.hpp"

int main(int argc, char** argv)
{
    RadonTransformer radon_transformer;
    cv::Mat radon = radon_transformer.radon_transform_with_noise("D:\\Code\\py\\graduate_design\\data\\120x120_100_255_10_3_[0, 180]_1_without_noise\\imgs\\1.png", 1.5);
    cv::imshow("img", radon);
    cv::waitKey();
    return 0;
}
