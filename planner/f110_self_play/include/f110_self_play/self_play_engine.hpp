#ifndef PLANNER_F110_SELF_PLAY_SELF_PLAY_ENGINE_HPP_
#define PLANNER_F110_SELF_PLAY_SELF_PLAY_ENGINE_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "dynamics.hpp"
#include "f110_observation.hpp"
#include "f110_reward.hpp"
#include "f110_scan.hpp"
#include "f110_self_play/action_lattice.hpp"
#include "f110_self_play/muzero_search_adapter.hpp"
#include "f110_step.hpp"
#include "f110_track_map.hpp"
#include "types.hpp"

namespace planner::f110_self_play {

struct TrajectoryStep {
  torch::Tensor obs;
  int32_t action;
  float reward;
  bool done;
  torch::Tensor root_policy;
};

struct Trajectory {
  std::vector<TrajectoryStep> steps;
};

struct SelfPlayResult {
  std::vector<Trajectory> trajectories;
  std::map<std::string, double> metrics;
};

class SelfPlayEngine {
public:
  inline SelfPlayEngine(
      std::shared_ptr<MuZeroSearchAdapter> search,
      const f110_rollout_kernel::TrackMap &track_map,
      const f110_rollout_kernel::ObservationConfig &obs_config,
      ActionLattice action_lattice, float discount, bool sample_actions,
      bool print_metrics, const std::vector<double> &waypoints_x,
      const std::vector<double> &waypoints_y,
      const std::vector<double> &cum_arc_lengths,
      const f110_rollout_kernel::F110Params &dynamics_params, double car_length,
      double car_width, double speed_reward_weight, double progress_weight,
      double steer_smoothness_weight, double collision_penalty,
      double spin_threshold)
      : search(std::move(search)), track_map(track_map), obs_config(obs_config),
        action_lattice(std::move(action_lattice)), discount(discount),
        sample_actions(sample_actions), print_metrics(print_metrics),
        waypoints_x(waypoints_x), waypoints_y(waypoints_y),
        cum_arc_lengths(cum_arc_lengths), dynamics_params(dynamics_params),
        car_length(car_length), car_width(car_width),
        speed_reward_weight(speed_reward_weight),
        progress_weight(progress_weight),
        steer_smoothness_weight(steer_smoothness_weight),
        collision_penalty(collision_penalty), spin_threshold(spin_threshold),
        rng(std::random_device{}()) {}

  inline SelfPlayResult
  generate(int32_t rollout_steps, int32_t batch_size,
           const std::vector<f110_rollout_kernel::F110State> &initial_states) {
    std::map<std::string, double> metrics;
    std::vector<f110_rollout_kernel::F110State> states = initial_states;
    int obs_dim = f110_rollout_kernel::observation_dim(obs_config);
    std::cout << "Self-play engine started: batch_size=" << batch_size
              << ", rollout_steps=" << rollout_steps << std::endl;

    std::vector<f110_gym::F110ProgressReward> reward_fns;
    reward_fns.reserve(static_cast<std::size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
      reward_fns.emplace_back(waypoints_x, waypoints_y, speed_reward_weight,
                              progress_weight, steer_smoothness_weight,
                              collision_penalty, spin_threshold);
    }

    std::vector<float> prev_actions(static_cast<std::size_t>(batch_size) * 2,
                                    0.0f);

    auto obs_tensor = compute_initial_observations(states, batch_size, obs_dim);

    std::vector<Trajectory> active(static_cast<std::size_t>(batch_size));
    std::vector<Trajectory> completed;

    for (int32_t step = 0; step < rollout_steps; ++step) {
      auto prev_obs = obs_tensor;
      auto search_result = search->search_batch(obs_tensor);
      accumulate_metrics(metrics, search_result.metrics);
      metrics["self_play/transitions"] += static_cast<double>(batch_size);

      auto action_probs =
          search_result.action_probs.to(torch::kCPU).contiguous();
      auto action_indices = sample_action_batch(action_probs);
      auto normalized_actions = action_lattice.normalized_batch(action_indices);

      std::vector<f110_rollout_kernel::F110Action> f110_actions(
          static_cast<std::size_t>(batch_size));
      auto norm_np = normalized_actions.detach().cpu().contiguous();
      for (int b = 0; b < batch_size; ++b) {
        f110_actions[static_cast<std::size_t>(b)] = {
            static_cast<double>(norm_np[b][0].item<float>()),
            static_cast<double>(norm_np[b][1].item<float>())};
      }

      f110_rollout_kernel::CompleteStepBatchResult step_result;
      f110_rollout_kernel::complete_step_batch(
          states.data(), f110_actions.data(), prev_actions.data(), track_map,
          dynamics_params, f110_rollout_kernel::Integrator::RK4,
          reward_fns.data(), obs_config, waypoints_x.data(), waypoints_y.data(),
          static_cast<int>(waypoints_x.size()), cum_arc_lengths.data(),
          batch_size, car_length, car_width, step_result);

      obs_tensor = torch::from_blob(
                       step_result.observations.data(), {batch_size, obs_dim},
                       torch::TensorOptions().dtype(torch::kFloat32))
                       .clone();

      for (int b = 0; b < batch_size; ++b) {
        auto obs_row = prev_obs.select(0, b).clone();
        auto policy_row = action_probs.select(0, b).clone();
        int32_t action =
            static_cast<int32_t>(action_indices[b].item<int64_t>());
        float r = static_cast<float>(
            step_result.rewards[static_cast<std::size_t>(b)]);
        bool done = step_result.terminals[static_cast<std::size_t>(b)] != 0;

        active[static_cast<std::size_t>(b)].steps.push_back(
            {obs_row, action, r, done, policy_row});

        if (done) {
          completed.push_back(std::move(active[static_cast<std::size_t>(b)]));
          active[static_cast<std::size_t>(b)] = Trajectory{};
          states[static_cast<std::size_t>(b)] =
              initial_states[static_cast<std::size_t>(b)];
          reward_fns[static_cast<std::size_t>(b)].reset();
          prev_actions[static_cast<std::size_t>(b) * 2] = 0.0f;
          prev_actions[static_cast<std::size_t>(b) * 2 + 1] = 0.0f;
        } else {
          prev_actions[static_cast<std::size_t>(b) * 2] =
              static_cast<float>(norm_np[b][0].item<float>());
          prev_actions[static_cast<std::size_t>(b) * 2 + 1] =
              static_cast<float>(norm_np[b][1].item<float>());
        }
      }
    }

    for (auto &traj : active) {
      if (!traj.steps.empty()) {
        completed.push_back(std::move(traj));
      }
    }

    SelfPlayResult result;
    result.trajectories = std::move(completed);
    result.metrics = std::move(metrics);
    result.metrics["self_play/trajectories"] =
        static_cast<double>(result.trajectories.size());
    result.metrics["self_play/search_steps"] =
        metrics.count("self_play/search_steps")
            ? metrics.at("self_play/search_steps")
            : static_cast<double>(rollout_steps);

    if (print_metrics) {
      print_metrics_summary(result.metrics);
    }

    return result;
  }

private:
  std::shared_ptr<MuZeroSearchAdapter> search;
  f110_rollout_kernel::TrackMap track_map;
  f110_rollout_kernel::ObservationConfig obs_config;
  ActionLattice action_lattice;
  float discount;
  bool sample_actions;
  bool print_metrics;
  std::vector<double> waypoints_x, waypoints_y, cum_arc_lengths;
  f110_rollout_kernel::F110Params dynamics_params;
  double car_length, car_width;
  double speed_reward_weight, progress_weight, steer_smoothness_weight;
  double collision_penalty, spin_threshold;
  std::mt19937 rng;

  inline torch::Tensor compute_initial_observations(
      const std::vector<f110_rollout_kernel::F110State> &states, int batch_size,
      int obs_dim) {
    std::vector<float> scans(static_cast<std::size_t>(batch_size) * 1080);
    std::vector<float> observations(static_cast<std::size_t>(batch_size) *
                                    static_cast<std::size_t>(obs_dim));
    std::vector<double> poses_x(static_cast<std::size_t>(batch_size));
    std::vector<double> poses_y(static_cast<std::size_t>(batch_size));
    std::vector<double> poses_t(static_cast<std::size_t>(batch_size));
    std::vector<double> vx(static_cast<std::size_t>(batch_size));
    std::vector<double> sa(static_cast<std::size_t>(batch_size));
    std::vector<double> yr(static_cast<std::size_t>(batch_size));

    for (int b = 0; b < batch_size; ++b) {
      poses_x[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].x;
      poses_y[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].y;
      poses_t[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].yaw_angle;
      sa[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].steer_angle;
      yr[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].yaw_rate;
      vx[static_cast<std::size_t>(b)] =
          states[static_cast<std::size_t>(b)].velocity *
          std::cos(states[static_cast<std::size_t>(b)].slip_angle);
    }

    f110_rollout_kernel::get_scan_batch(poses_x.data(), batch_size, track_map,
                                        scans.data());

    std::vector<uint8_t> col(static_cast<std::size_t>(batch_size), 0);
    std::vector<double> zvel(static_cast<std::size_t>(batch_size), 0.0);
    std::vector<float> prev_actions(static_cast<std::size_t>(batch_size) * 2,
                                    0.0f);

    f110_rollout_kernel::build_observation_batch(
        scans.data(), poses_x.data(), poses_y.data(), poses_t.data(), vx.data(),
        zvel.data(), yr.data(), sa.data(), col.data(), prev_actions.data(),
        waypoints_x.data(), waypoints_y.data(),
        static_cast<int>(waypoints_x.size()), cum_arc_lengths.data(),
        batch_size, obs_config, observations.data());

    return torch::from_blob(observations.data(), {batch_size, obs_dim},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone();
  }

  inline void accumulate_metrics(std::map<std::string, double> &acc,
                                 const std::map<std::string, double> &step) {
    for (const auto &kv : step) {
      acc[kv.first] += kv.second;
    }
    acc["self_play/search_steps"] += 1.0;
  }

  inline torch::Tensor sample_action_batch(const torch::Tensor &action_probs) {
    auto batch_size = action_probs.size(0);
    auto action_count = action_probs.size(1);
    auto actions = torch::empty({batch_size}, torch::kInt64);
    auto probs = action_probs.contiguous().to(torch::kCPU);
    const float *prob_ptr = probs.data_ptr<float>();
    int64_t *action_ptr = actions.data_ptr<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
      const float *row = prob_ptr + b * action_count;
      double sum = 0.0;
      for (int64_t a = 0; a < action_count; ++a) {
        sum += static_cast<double>(row[a]);
      }

      if (!sample_actions || sum <= 0.0) {
        int64_t best = 0;
        double best_w = static_cast<double>(row[0]);
        for (int64_t a = 1; a < action_count; ++a) {
          if (static_cast<double>(row[a]) > best_w) {
            best_w = static_cast<double>(row[a]);
            best = a;
          }
        }
        action_ptr[b] = best;
        continue;
      }

      std::vector<double> weights(static_cast<std::size_t>(action_count));
      for (int64_t a = 0; a < action_count; ++a) {
        weights[static_cast<std::size_t>(a)] =
            static_cast<double>(row[a]) / sum;
      }
      std::discrete_distribution<int64_t> dist(weights.begin(), weights.end());
      action_ptr[b] = dist(rng);
    }
    return actions;
  }

  inline void
  print_metrics_summary(const std::map<std::string, double> &m) const {
    auto v = [&](const std::string &k) {
      auto it = m.find(k);
      return it != m.end() ? it->second : 0.0;
    };

    std::cout << "---------------- Self-Play Metrics ----------------"
              << std::endl;
    std::cout << "[Rollout]" << std::endl;
    std::cout << "  search steps: " << v("self_play/search_steps")
              << " | transitions: " << v("self_play/transitions")
              << " | samples: " << v("self_play/samples") << std::endl;
    std::cout << "  trajectories: " << v("self_play/trajectories") << std::endl;
    std::cout << "[Search Average]" << std::endl;
    std::cout << "  total: " << v("search/total_time_us") / 1000.0
              << " ms | simulations/sec: "
              << (m.find("throughput/simulations_per_second") != m.end()
                      ? v("throughput/simulations_per_second")
                      : 0.0)
              << std::endl;
  }
};

} // namespace planner::f110_self_play

#endif // PLANNER_F110_SELF_PLAY_SELF_PLAY_ENGINE_HPP_
