// Minimal stand-in for the ROS nav_msgs/OccupancyGrid message so that the
// original LearningMPC headers (track.h, occupancy_grid.h) compile unmodified
// outside ROS. Field names and types mirror the ROS message exactly as used
// by that code: info.resolution, info.width, info.height,
// info.origin.position.{x,y,z}, and data (row-major int8, origin bottom-left).
#pragma once

#include <cstdint>
#include <geometry_msgs/Point.h>
#include <vector>

namespace nav_msgs {

struct MapMetaData {
  double resolution = 0.0;
  uint32_t width = 0;
  uint32_t height = 0;
  struct Origin {
    geometry_msgs::Point position;
    struct Quat {
      double x = 0, y = 0, z = 0, w = 1;
    } orientation;
  } origin;
};

struct OccupancyGrid {
  MapMetaData info;
  std::vector<int8_t> data;
  typedef OccupancyGrid *Ptr;
};

} // namespace nav_msgs
