#ifndef F110_ROLLOUT_KERNEL_SIMD_HPP_
#define F110_ROLLOUT_KERNEL_SIMD_HPP_

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

namespace f110_rollout_kernel {

inline std::size_t platform_f32_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  return hn::Lanes(d);
}

inline const std::size_t kPlatformF32Lanes = platform_f32_lanes();

template <typename Func> inline void for_batch(int B, Func &&fn) {
  const std::size_t lanes = kPlatformF32Lanes;
  int b = 0;
  for (; b + static_cast<int>(lanes) <= B; b += static_cast<int>(lanes)) {
    fn(b, static_cast<int>(lanes));
  }
  if (b < B) {
    fn(b, B - b);
  }
}

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_SIMD_HPP_
