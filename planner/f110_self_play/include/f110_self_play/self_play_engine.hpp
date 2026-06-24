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
  using Clock = std::chrono::steady_clock;

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
      double car_width, double q_s_progress, double q_s_alpha,
      double q_s_smooth, double terminal_penalty, double alpha_th,
      double slip_terminal_penalty, double q_offtrack_grad)
      : search(std::move(search)), track_map(track_map), obs_config(obs_config),
        action_lattice(std::move(action_lattice)), discount(discount),
        sample_actions(sample_actions), print_metrics(print_metrics),
        waypoints_x(waypoints_x), waypoints_y(waypoints_y),
        cum_arc_lengths(cum_arc_lengths), dynamics_params(dynamics_params),
        car_length(car_length), car_width(car_width),
        q_s_progress(q_s_progress), q_s_alpha(q_s_alpha),
        q_s_smooth(q_s_smooth), terminal_penalty(terminal_penalty),
        alpha_th(alpha_th), slip_terminal_penalty(slip_terminal_penalty),
        q_offtrack_grad(q_offtrack_grad), rng(std::random_device{}()) {}

  inline SelfPlayResult
  generate(int32_t rollout_steps, int32_t batch_size,
           const std::vector<f110_rollout_kernel::F110State> &initial_states) {
    const auto generate_start = Clock::now();
    std::map<std::string, double> metrics;
    metrics["self_play/batch_size"] = static_cast<double>(batch_size);
    std::vector<f110_rollout_kernel::F110State> states = initial_states;
    int obs_dim = f110_rollout_kernel::observation_dim(obs_config);
    std::cout << "Self-play engine started: batch_size=" << batch_size
              << ", rollout_steps=" << rollout_steps << std::endl;

    std::vector<f110_gym::F110ProgressReward> reward_fns;
    reward_fns.reserve(static_cast<std::size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
      reward_fns.emplace_back(track_map, waypoints_x, waypoints_y, q_s_progress,
                              q_s_alpha, q_s_smooth, terminal_penalty, alpha_th,
                              slip_terminal_penalty, q_offtrack_grad);
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
          batch_size, car_length, car_width, step_result, false);

      metrics["self_play/terminal_collisions"] +=
          static_cast<double>(step_result.collision_terminations);
      metrics["self_play/terminal_backwards"] +=
          static_cast<double>(step_result.backward_terminations);

      obs_tensor = torch::from_blob(
                       step_result.observations.data(), {batch_size, obs_dim},
                       torch::TensorOptions().dtype(torch::kFloat32))
                       .clone();

      for (int b = 0; b < batch_size; ++b) {
        states[static_cast<std::size_t>(b)] =
            step_result.states[static_cast<std::size_t>(b)];
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

          std::vector<f110_rollout_kernel::F110State> reset_state{
              states[static_cast<std::size_t>(b)]};
          auto reset_obs =
              compute_initial_observations(reset_state, 1, obs_dim);
          obs_tensor.select(0, b).copy_(reset_obs.select(0, 0));
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
    result.metrics["self_play/episode_return"] =
        mean_trajectory_return(result.trajectories);
    result.metrics["self_play/discounted_episode_return"] =
        mean_discounted_trajectory_return(result.trajectories, discount);
    result.metrics["self_play/search_steps"] =
        result.metrics.count("self_play/search_steps")
            ? result.metrics.at("self_play/search_steps")
            : static_cast<double>(rollout_steps);
    result.metrics["self_play/samples"] =
        static_cast<double>(trajectory_step_count(result.trajectories));
    result.metrics["self_play/total_time_us"] =
        static_cast<double>(elapsed_us(generate_start, Clock::now()));
    const double search_steps = result.metrics["self_play/search_steps"];
    if (search_steps > 0.0) {
      result.metrics["search/iterations"] /= search_steps;
      result.metrics["search/simulations_per_lane"] /= search_steps;
      result.metrics["tree/nodes_allocated_avg"] /= search_steps;
      result.metrics["tree/search_depth_avg"] /= search_steps;
      result.metrics["tree/root_visit_count_avg"] /= search_steps;
    }
    result.metrics["search/iterations_per_search_call"] =
        result.metrics["search/iterations"];
    result.metrics["search/simulations_per_search_call"] =
        search_steps > 0.0
            ? result.metrics["search/simulations_total"] / search_steps
            : 0.0;
    finalize_throughput_metrics(result.metrics);

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
  double q_s_progress, q_s_alpha, q_s_smooth;
  double terminal_penalty, alpha_th, slip_terminal_penalty, q_offtrack_grad;
  std::mt19937 rng;

  static inline long long elapsed_us(const Clock::time_point &start,
                                     const Clock::time_point &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
  }

  static inline double per_second(double count, double time_us) {
    return time_us > 0.0 ? count / (time_us / 1.0e6) : 0.0;
  }

  static inline std::size_t
  trajectory_step_count(const std::vector<Trajectory> &trajectories) {
    std::size_t steps = 0;
    for (const auto &trajectory : trajectories) {
      steps += trajectory.steps.size();
    }
    return steps;
  }

  static inline double trajectory_return(const Trajectory &trajectory) {
    double total = 0.0;
    for (const auto &step : trajectory.steps) {
      total += static_cast<double>(step.reward);
    }
    return total;
  }

  static inline double
  discounted_trajectory_return(const Trajectory &trajectory, double discount) {
    double total = 0.0;
    double gamma = 1.0;
    for (const auto &step : trajectory.steps) {
      total += gamma * static_cast<double>(step.reward);
      gamma *= discount;
    }
    return total;
  }

  static inline double
  mean_trajectory_return(const std::vector<Trajectory> &trajectories) {
    if (trajectories.empty()) {
      return 0.0;
    }
    double total = 0.0;
    for (const auto &trajectory : trajectories) {
      total += trajectory_return(trajectory);
    }
    return total / static_cast<double>(trajectories.size());
  }

  static inline double
  mean_discounted_trajectory_return(const std::vector<Trajectory> &trajectories,
                                    double discount) {
    if (trajectories.empty()) {
      return 0.0;
    }
    double total = 0.0;
    for (const auto &trajectory : trajectories) {
      total += discounted_trajectory_return(trajectory, discount);
    }
    return total / static_cast<double>(trajectories.size());
  }

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
    std::vector<double> poses(static_cast<std::size_t>(batch_size) * 3);

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
      poses[static_cast<std::size_t>(b) * 3] =
          poses_x[static_cast<std::size_t>(b)];
      poses[static_cast<std::size_t>(b) * 3 + 1] =
          poses_y[static_cast<std::size_t>(b)];
      poses[static_cast<std::size_t>(b) * 3 + 2] =
          poses_t[static_cast<std::size_t>(b)];
    }

    f110_rollout_kernel::get_scan_batch(poses.data(), batch_size, track_map,
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
      if (kv.first == "throughput/simulations_per_second") {
        continue;
      }
      if (kv.first == "tree/nodes_allocated_min" ||
          kv.first == "tree/search_depth_min" ||
          kv.first == "tree/root_visit_count_min") {
        auto it = acc.find(kv.first);
        if (it == acc.end() || kv.second < it->second) {
          acc[kv.first] = kv.second;
        }
        continue;
      }
      if (kv.first == "tree/nodes_allocated_max" ||
          kv.first == "tree/search_depth_max" ||
          kv.first == "tree/root_visit_count_max") {
        auto it = acc.find(kv.first);
        if (it == acc.end() || kv.second > it->second) {
          acc[kv.first] = kv.second;
        }
        continue;
      }
      acc[kv.first] += kv.second;
    }
    acc["self_play/search_steps"] += 1.0;
  }

  static inline void
  finalize_throughput_metrics(std::map<std::string, double> &metrics) {
    const double total_time_us = metrics["self_play/total_time_us"];
    const double search_steps = metrics["self_play/search_steps"];
    const double search_time_us = metrics["search/total_time_us"];

    metrics["self_play/transitions_per_second"] =
        per_second(metrics["self_play/transitions"], total_time_us);
    metrics["self_play/samples_per_second"] =
        per_second(metrics["self_play/samples"], total_time_us);
    metrics["self_play/searches_per_second"] =
        per_second(search_steps, total_time_us);
    metrics["self_play/trajectories_per_second"] =
        per_second(metrics["self_play/trajectories"], total_time_us);
    metrics["search/simulations_per_second"] =
        per_second(metrics["search/simulations"], search_time_us);
    metrics["throughput/simulations_per_second"] =
        metrics["search/simulations_per_second"];
    metrics["search/avg_total_time_us"] =
        search_steps > 0.0 ? search_time_us / search_steps : 0.0;
    metrics["search/avg_initial_time_us"] =
        search_steps > 0.0 ? metrics["inference/initial_time_us"] / search_steps
                           : 0.0;
    metrics["search/avg_recurrent_time_us"] =
        search_steps > 0.0
            ? metrics["inference/recurrent_time_us"] / search_steps
            : 0.0;
    metrics["search/avg_payload_copy_time_us"] =
        search_steps > 0.0
            ? metrics["inference/payload_copy_time_us"] / search_steps
            : 0.0;
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
    std::cout << "  trajectories: " << v("self_play/trajectories")
              << " | avg len: "
              << (v("self_play/trajectories") > 0.0
                      ? v("self_play/samples") / v("self_play/trajectories")
                      : 0.0)
              << " | episode return: " << v("self_play/episode_return")
              << " | discounted return: "
              << v("self_play/discounted_episode_return")
              << " | total: " << v("self_play/total_time_us") / 1000.0 << " ms"
              << std::endl;
    std::cout << "  terminal collisions: " << v("self_play/terminal_collisions")
              << " | terminal backwards: " << v("self_play/terminal_backwards")
              << std::endl;
    std::cout << "[Throughput]" << std::endl;
    std::cout << "  transitions/sec: " << v("self_play/transitions_per_second")
              << " | samples/sec: " << v("self_play/samples_per_second")
              << std::endl;
    std::cout << "  searches/sec: " << v("self_play/searches_per_second")
              << " | trajectories/sec: "
              << v("self_play/trajectories_per_second") << std::endl;
    std::cout << "  simulations/sec: " << v("throughput/simulations_per_second")
              << std::endl;
    std::cout << "[Search Average]" << std::endl;
    std::cout << "  total: " << v("search/avg_total_time_us") / 1000.0
              << " ms | initial: " << v("search/avg_initial_time_us") / 1000.0
              << " ms | recurrent: "
              << v("search/avg_recurrent_time_us") / 1000.0 << " ms"
              << std::endl;
    std::cout << "  copy: " << v("search/avg_payload_copy_time_us") / 1000.0
              << " ms | iterations/search: "
              << v("search/iterations_per_search_call")
              << " | simulations/lane/search: "
              << v("search/simulations_per_lane")
              << " | simulations/search_call: "
              << v("search/simulations_per_search_call") << std::endl;
  }
};

} // namespace planner::f110_self_play

#endif // PLANNER_F110_SELF_PLAY_SELF_PLAY_ENGINE_HPP_
