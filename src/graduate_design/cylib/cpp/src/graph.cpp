#include <algorithm>
#include <random>
#include "graph.hpp"


namespace graph {
    std::random_device seed;

    Generator::Generator() {};

    Generator::Generator(uint16_t img_size, uint16_t radius):
        engine(seed()), img_size(img_size), solid_radius(true), radius(radius) {}

    Generator::Generator(uint16_t img_size, uint16_t min_radius, uint16_t max_radius):
        engine(seed()), img_size(img_size), solid_radius(false)
    {
        radius_range[0] = min_radius;
        radius_range[1] = max_radius;
    }

    uint16_t Generator::get_radius()
    {
        static std::uniform_int_distribution<uint16_t> radius_dist(radius_range[0], radius_range[1]);
        return radius_dist(engine);
    }

    bool Generator::overlap(const Area& area1, const Area& area2)
    {
        bool x_overlap = area1.center[0] + area1.radius > area2.center[0] - area2.radius
            && area1.center[0] - area1.radius < area2.center[0] + area2.radius;
        bool y_overlap = area1.center[1] + area1.radius > area2.center[1] - area2.radius
            && area1.center[1] - area1.radius < area2.center[1] + area2.radius;
        return x_overlap && y_overlap;
    }

    bool Generator::in_circle(float x, float y, uint16_t r)
    {
        static uint16_t half_size = img_size / 2;
        return std::sqrt(
            std::pow(x - half_size, 2) + std::pow(y - half_size, 2)
        ) + r + 1 < half_size;
    }

    void Generator::gen_area(Area& area)
    {
        uint16_t _radius;
        if (solid_radius) {
            _radius = radius;
        } else {
            _radius = get_radius();
        }
        std::uniform_real_distribution<float> pos_dist(_radius, img_size - _radius);
        float x = pos_dist(engine), y = pos_dist(engine);
        while (!in_circle(x, y, _radius)) {
            x = pos_dist(engine);
            y = pos_dist(engine);
        }
        area.radius = _radius;
        area.center[0] = x;
        area.center[1] = y;
    }

    void Generator::gen_polygon(
        float* points,
        const Area& area,
        uint16_t n_sides
    )
    {
        static float available_angles[8] = {
            0., PI / 8, PI / 4, PI / 8 * 3, PI / 2,
            PI / 8 * 5, PI / 4 * 3, PI / 8 * 7
        };
        std::shuffle(
            available_angles,
            available_angles + 8,
            std::default_random_engine(seed())
        );
        std::sort(
            available_angles,
            available_angles + n_sides,
            [](float a, float b) { return a < b; }
        );
        for (uint16_t i = 0; i < n_sides; i++) {
            uint16_t j = 2 * i;
            points[j] = area.center[0] +
                std::cos(available_angles[i]) * area.radius;
            points[j + 1] = area.center[1] +
                std::sin(available_angles[i]) * area.radius;
        }
    }

    void Generator::gen_circle(float* points, const Area& area)
    {
        points[0] = area.center[0] - area.radius;
        points[1] = area.center[1] - area.radius;
        points[2] = area.center[0] + area.radius;
        points[3] = area.center[1] + area.radius;
    }

    uint16_t Generator::gen(uint16_t num, Area* areas)
    {
        uint16_t n = 1, try_num = 0;
        gen_area(areas[0]);
        while (n < num && try_num < 1000) {
            Area& a = areas[n];
            gen_area(a);
            bool overlap_flag = true;
            for (uint16_t i = 0; i < n; i++) {
                if (overlap(a, areas[i])) {
                    overlap_flag = false;
                    try_num++;
                    break;
                }
            }
            if (overlap_flag) {
                n++;
            }
        }
        return n;
    }
}
