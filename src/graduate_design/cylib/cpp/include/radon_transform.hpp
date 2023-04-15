#pragma once
#include <random>


namespace algorithm {
    class RadonTransformer
    {
        private:
            std::default_random_engine e;
            std::uniform_int_distribution<unsigned int> u;
            float _theta, _start_angle, _end_angle;
            bool _crop, _norm, _add_noise;
        public:
            RadonTransformer();
            RadonTransformer(
                float theta,
                float start_angle,
                float end_angle,
                bool crop,
                bool norm,
                bool add_noise
            );
            void radon_transform_with_noise(
                const unsigned char* bytes,
                size_t byte_length,
                std::vector<unsigned char>& out_buf
            );
    };
}
