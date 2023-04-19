#pragma once
#include <random>
#define PI 3.1415926535897932384626


namespace graph {
    struct Area
    {
        uint16_t radius;
        float center[2];
    };

    class Generator
    {
        private:
            std::default_random_engine engine;
            uint16_t img_size;
            bool solid_radius;
            union {
                uint16_t radius; uint16_t radius_range[2];
            };

            uint16_t get_radius();
            static bool overlap(const Area& area1, const Area& area2);
            bool in_circle(float x, float y, uint16_t r);
            void gen_area(Area& area);
        public:
            Generator();
            Generator(uint16_t img_size, uint16_t radius);
            Generator(uint16_t img_size, uint16_t min_radius, uint16_t max_radius);
            void gen_circle(float* points, const Area& area);
            void gen_polygon(float* points, const Area& area, uint16_t n_sides);
            uint16_t gen(uint16_t num, Area* areas);
    };
}
