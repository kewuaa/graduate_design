#pragma once
#include <random>


class RadonTransformer
{
    private:
        std::default_random_engine e;
        std::uniform_int_distribution<unsigned int> u;
        float _theta, _start_angle, _end_angle;
        bool _crop, _norm, _add_noise;
    public:
        RadonTransformer(
            float theta=1.,
            float start_angle=0.,
            float end_angle=180.,
            bool crop=true,
            bool norm=true,
            bool add_noise=true
        );
        void radon_transform_with_noise(
            const char* bytes,
            unsigned int byte_length,
            std::vector<unsigned char>& out_buf
        );
};
