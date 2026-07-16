// Minimal stand-in for ROS geometry_msgs/Point.
#pragma once

namespace geometry_msgs {
struct Point {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};
} // namespace geometry_msgs
