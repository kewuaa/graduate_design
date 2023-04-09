#include <iostream>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "radon_transform.hpp"

int main(int argc, char** argv)
{
    RadonTransformer radon_transformer(1.5);
    const char* save_path = "D:/Code/py/graduate_design/cython/cpp/data/11.png";
    const char* img_path = "D:/Code/py/graduate_design/cython/cpp/data/1.png";
    //! 以二进制流方式读取图片到内存
    FILE* pFile = fopen(img_path, "rb");
    fseek(pFile, 0, SEEK_END);
    long lSize = ftell(pFile);
    rewind(pFile);
    char* pData = new char[lSize];
    fread(pData, sizeof(char), lSize, pFile);
    fclose(pFile);
    std::vector<unsigned char> out_buf;
    radon_transformer.radon_transform_with_noise(
        pData,
        lSize,
        out_buf
    );
    cv::Mat radon = cv::imdecode(out_buf, cv::IMREAD_GRAYSCALE);
    cv::imshow("img", radon);
    cv::waitKey();
    cv::imwrite(save_path, radon);
    cv::waitKey();
    delete [] pData;
    return 0;
}
