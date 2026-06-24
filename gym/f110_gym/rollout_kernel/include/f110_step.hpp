#ifndef F110_ROLLOUT_KERNEL_F110_STEP_HPP_
#define F110_ROLLOUT_KERNEL_F110_STEP_HPP_

#include <cmath>
#include <cstdint>
#include <vector>

#include "dynamics.hpp"
#include "f110_collision.hpp"
#include "f110_observation.hpp"
#include "f110_reward.hpp"
#include "f110_scan.hpp"
#include "f110_track_map.hpp"
#include "simd.hpp"
#include "step.hpp"
#include "types.hpp"

namespace f110_rollout_kernel {

using f110_gym::F110ProgressReward;

struct CompleteStepBatchResult {
  std::vector<F110State> states;
  std::vector<float> scans;
  std::vector<float> observations;
  std::vector<double> rewards;
  std::vector<uint8_t> terminals;
  std::vector<uint8_t> collisions;
  std::vector<int> lap_counts;
  std::vector<double> lap_times;
  std::size_t collision_terminations = 0;
  std::size_t backward_terminations = 0;
  int obs_dim = 0;
};

inline void complete_step_batch(
    F110State *states, const F110Action *actions, const float *prev_actions_ptr,
    const TrackMap &track_map, const F110Params &params, Integrator integrator,
    F110ProgressReward *reward_fns, const ObservationConfig &obs_config,
    const double *waypoints_x, const double *waypoints_y, int num_waypoints,
    const double *cum_arc_lengths, int B, double car_length, double car_width,
    CompleteStepBatchResult &result, bool check_pairwise_collisions = false) {
  result.states.resize(static_cast<std::size_t>(B));
  result.scans.assign(static_cast<std::size_t>(B) * 1080, 0.0f);
  int obs_dim = observation_dim(obs_config);
  result.obs_dim = obs_dim;
  result.observations.assign(
      static_cast<std::size_t>(B) * static_cast<std::size_t>(obs_dim), 0.0f);
  result.rewards.assign(static_cast<std::size_t>(B), 0.0);
  result.terminals.assign(static_cast<std::size_t>(B), 0);
  result.collisions.assign(static_cast<std::size_t>(B), 0);
  result.lap_counts.assign(static_cast<std::size_t>(B), 0);
  result.lap_times.assign(static_cast<std::size_t>(B), 0.0);

  std::vector<double> px(static_cast<std::size_t>(B));
  std::vector<double> py(static_cast<std::size_t>(B));
  std::vector<double> pt(static_cast<std::size_t>(B));
  std::vector<double> vl(static_cast<std::size_t>(B));
  std::vector<double> sa(static_cast<std::size_t>(B));
  std::vector<double> yr(static_cast<std::size_t>(B));
  std::vector<double> vx(static_cast<std::size_t>(B));
  std::vector<double> vy(static_cast<std::size_t>(B));
  std::vector<double> poses(static_cast<std::size_t>(B) * 3);

  for (int b = 0; b < B; ++b) {
    F110StepResult sr = step(states[b], actions[b], params, integrator);
    result.states[static_cast<std::size_t>(b)] = sr.state;
    px[static_cast<std::size_t>(b)] = sr.state.x;
    py[static_cast<std::size_t>(b)] = sr.state.y;
    pt[static_cast<std::size_t>(b)] = sr.state.yaw_angle;
    vl[static_cast<std::size_t>(b)] = sr.state.velocity;
    sa[static_cast<std::size_t>(b)] = sr.state.steer_angle;
    yr[static_cast<std::size_t>(b)] = sr.state.yaw_rate;
    vx[static_cast<std::size_t>(b)] =
        sr.state.velocity * std::cos(sr.state.slip_angle);
    vy[static_cast<std::size_t>(b)] =
        sr.state.velocity * std::sin(sr.state.slip_angle);
    poses[static_cast<std::size_t>(b) * 3] = px[static_cast<std::size_t>(b)];
    poses[static_cast<std::size_t>(b) * 3 + 1] =
        py[static_cast<std::size_t>(b)];
    poses[static_cast<std::size_t>(b) * 3 + 2] =
        pt[static_cast<std::size_t>(b)];
  }

  get_scan_batch(poses.data(), B, track_map, result.scans.data());

  for (int b = 0; b < B; ++b) {
    result.collisions[static_cast<std::size_t>(b)] =
        check_ttc_one(result.scans.data() + b * 1080, vl[b], track_map) ? 1 : 0;
  }

  if (check_pairwise_collisions && B >= 2) {
    for (int b = 0; b < B; ++b) {
      if (result.collisions[static_cast<std::size_t>(b)])
        continue;
      auto v1 = get_vertices(
          px[static_cast<std::size_t>(b)], py[static_cast<std::size_t>(b)],
          pt[static_cast<std::size_t>(b)], car_length, car_width);
      for (int o = 0; o < B; ++o) {
        if (o == b)
          continue;
        auto v2 = get_vertices(
            px[static_cast<std::size_t>(o)], py[static_cast<std::size_t>(o)],
            pt[static_cast<std::size_t>(o)], car_length, car_width);
        if (collision(v1, v2)) {
          result.collisions[static_cast<std::size_t>(b)] = 1;
          break;
        }
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    if (result.collisions[static_cast<std::size_t>(b)]) {
      states[b].in_collision = true;
      states[b].velocity = 0.0;
    }
  }

  build_observation_batch(
      result.scans.data(), px.data(), py.data(), pt.data(), vx.data(),
      vy.data(), yr.data(), sa.data(), result.collisions.data(),
      prev_actions_ptr, waypoints_x, waypoints_y, num_waypoints,
      cum_arc_lengths, B, obs_config, result.observations.data());

  for (int b = 0; b < B; ++b) {
    bool coll = result.collisions[static_cast<std::size_t>(b)] != 0;
    bool terminated = reward_fns[b].is_terminal(
        px[static_cast<std::size_t>(b)], py[static_cast<std::size_t>(b)],
        pt[static_cast<std::size_t>(b)], coll);
    double r = reward_fns[b](
        px[static_cast<std::size_t>(b)], py[static_cast<std::size_t>(b)],
        pt[static_cast<std::size_t>(b)], vx[static_cast<std::size_t>(b)],
        vy[static_cast<std::size_t>(b)], actions[b].steer, actions[b].velocity,
        coll, terminated);
    result.rewards[static_cast<std::size_t>(b)] = r;
    result.terminals[static_cast<std::size_t>(b)] = terminated ? 1 : 0;
    if (terminated) {
      if (coll) {
        ++result.collision_terminations;
      } else {
        ++result.backward_terminations;
      }
    }
    if (terminated) {
      reward_fns[b].reset();
    }
  }
}

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_F110_STEP_HPP_
