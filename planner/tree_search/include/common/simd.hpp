#ifndef PLANNER_TREE_SEARCH_COMMON_SIMD_HPP_
#define PLANNER_TREE_SEARCH_COMMON_SIMD_HPP_

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

namespace planner::tree_search {

struct SimdLaneCounts {
  std::size_t f32 = 1;
  std::size_t f16 = 1;
  std::size_t bf16 = 1;
  std::size_t i32 = 1;
  std::size_t u8 = 1;
};

// Highway lane counts may depend on the compiled target and scalar type.
// Keep these as runtime constants instead of constexpr so scalable targets are
// represented correctly.
inline std::size_t platform_f32_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  return hn::Lanes(d);
}

inline std::size_t platform_f16_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<hwy::float16_t> d;
  return hn::Lanes(d);
}

inline std::size_t platform_bf16_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<hwy::bfloat16_t> d;
  return hn::Lanes(d);
}

inline std::size_t platform_i32_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<std::int32_t> d;
  return hn::Lanes(d);
}

inline std::size_t platform_u8_lanes() {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<std::uint8_t> d;
  return hn::Lanes(d);
}

inline SimdLaneCounts platform_simd_lanes() {
  return SimdLaneCounts{platform_f32_lanes(), platform_f16_lanes(),
                        platform_bf16_lanes(), platform_i32_lanes(),
                        platform_u8_lanes()};
}

inline const SimdLaneCounts kPlatformSimdLanes = platform_simd_lanes();
inline const std::size_t kPlatformF32Lanes = platform_f32_lanes();
inline const std::size_t kPlatformF16Lanes = platform_f16_lanes();
inline const std::size_t kPlatformBF16Lanes = platform_bf16_lanes();
inline const std::size_t kPlatformI32Lanes = platform_i32_lanes();
inline const std::size_t kPlatformU8Lanes = platform_u8_lanes();

} // namespace planner::tree_search

#endif // PLANNER_TREE_SEARCH_COMMON_SIMD_HPP_
