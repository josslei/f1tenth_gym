#ifndef F110_ROLLOUT_KERNEL_COLLISION_HPP_
#define F110_ROLLOUT_KERNEL_COLLISION_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "simd.hpp"

namespace f110_rollout_kernel {

using Vertices4 = std::array<std::array<double, 2>, 4>;

inline Vertices4 get_vertices(double x, double y, double theta, double length,
                              double width) {
  double c = std::cos(theta);
  double s = std::sin(theta);

  double hl = length * 0.5;
  double hw = width * 0.5;

  return Vertices4{{
      {x + c * (-hl) - s * hw, y + s * (-hl) + c * hw},
      {x + c * (-hl) - s * (-hw), y + s * (-hl) + c * (-hw)},
      {x + c * hl - s * (-hw), y + s * hl + c * (-hw)},
      {x + c * hl - s * hw, y + s * hl + c * hw},
  }};
}

inline double dot(const std::array<double, 2> &a,
                  const std::array<double, 2> &b) {
  return a[0] * b[0] + a[1] * b[1];
}

inline std::array<double, 2> sub(const std::array<double, 2> &a,
                                 const std::array<double, 2> &b) {
  return {a[0] - b[0], a[1] - b[1]};
}

inline std::array<double, 2> add(const std::array<double, 2> &a,
                                 const std::array<double, 2> &b) {
  return {a[0] + b[0], a[1] + b[1]};
}

inline std::array<double, 2> mul(const std::array<double, 2> &a, double s) {
  return {a[0] * s, a[1] * s};
}

inline std::array<double, 2> avg_point(const Vertices4 &v) {
  return {(v[0][0] + v[1][0] + v[2][0] + v[3][0]) * 0.25,
          (v[0][1] + v[1][1] + v[2][1] + v[3][1]) * 0.25};
}

inline std::array<double, 2> perpendicular(const std::array<double, 2> &v) {
  return {v[1], -v[0]};
}

inline std::array<double, 2> triple_product(const std::array<double, 2> &a,
                                            const std::array<double, 2> &b,
                                            const std::array<double, 2> &c) {
  double ac = dot(a, c);
  double bc = dot(b, c);
  return {b[0] * ac - a[0] * bc, b[1] * ac - a[1] * bc};
}

inline std::array<double, 2> support(const Vertices4 &v1, const Vertices4 &v2,
                                     const std::array<double, 2> &d) {
  int i1 = 0;
  double best1 = dot(v1[0], d);
  for (int i = 1; i < 4; ++i) {
    double p = dot(v1[i], d);
    if (p > best1) {
      best1 = p;
      i1 = i;
    }
  }

  std::array<double, 2> nd = {-d[0], -d[1]};
  int i2 = 0;
  double best2 = dot(v2[0], nd);
  for (int i = 1; i < 4; ++i) {
    double p = dot(v2[i], nd);
    if (p > best2) {
      best2 = p;
      i2 = i;
    }
  }

  return {v1[i1][0] - v2[i2][0], v1[i1][1] - v2[i2][1]};
}

inline bool collision(const Vertices4 &v1, const Vertices4 &v2) {
  std::array<std::array<double, 2>, 3> simplex{};
  int index = 0;

  auto d = sub(avg_point(v1), avg_point(v2));
  if (d[0] == 0.0 && d[1] == 0.0) {
    d[0] = 1.0;
  }

  auto a = support(v1, v2, d);
  simplex[index] = a;

  if (dot(d, a) <= 0.0) {
    return false;
  }

  d = mul(a, -1.0);

  for (int iter = 0; iter < 1000; ++iter) {
    a = support(v1, v2, d);
    index += 1;
    simplex[index] = a;

    if (dot(d, a) <= 0.0) {
      return false;
    }

    auto ao = mul(a, -1.0);

    if (index < 2) {
      auto b = simplex[0];
      auto ab = sub(b, a);
      d = triple_product(ab, ao, ab);
      if (dot(d, d) < 1e-10) {
        d = perpendicular(ab);
      }
      continue;
    }

    auto b = simplex[1];
    auto c = simplex[0];
    auto ab = sub(b, a);
    auto ac = sub(c, a);

    auto acperp = triple_product(ab, ac, ac);
    if (dot(acperp, ao) >= 0.0) {
      d = acperp;
    } else {
      auto abperp = triple_product(ac, ab, ab);
      if (dot(abperp, ao) < 0.0) {
        return true;
      }
      simplex[0] = simplex[1];
      d = abperp;
    }

    simplex[1] = simplex[2];
    index -= 1;
  }

  return false;
}

inline void collision_batch(const double *poses_x, const double *poses_y,
                            const double *poses_theta, int B, double length,
                            double width, bool *out) {
  if (B < 2) {
    for (int b = 0; b < B; ++b) {
      out[b] = false;
    }
    return;
  }

  for_batch(B, [&](int start, int count) {
    for (int b = start; b < start + count; ++b) {
      out[b] = false;
      auto v1 =
          get_vertices(poses_x[b], poses_y[b], poses_theta[b], length, width);
      for (int other = 0; other < B; ++other) {
        if (other == b)
          continue;
        auto v2 = get_vertices(poses_x[other], poses_y[other],
                               poses_theta[other], length, width);
        if (collision(v1, v2)) {
          out[b] = true;
          break;
        }
      }
    }
  });
}

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_COLLISION_HPP_
