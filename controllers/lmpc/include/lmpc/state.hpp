#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace f110_gym_lmpc {

struct GymVehicleState {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;
};

struct RacingLmpcState {
  double s = 0.0;
  double e_y = 0.0;
  double e_psi = 0.0;
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;

  std::array<double, 6> to_array() const;
};

struct PaperLmpcState {
  double v_x = 0.0;
  double v_y = 0.0;
  double omega = 0.0;
  double e_psi = 0.0;
  double s = 0.0;
  double e_y = 0.0;

  std::array<double, 6> to_array() const;
};

struct LmpcControlCommand {
  double steering = 0.0;
  double velocity = 0.0;
};

struct LmpcReference {
  double curvature = 0.0;
  double target_speed = 3.0;
  double left_bound = 1.0;
  double right_bound = 1.0;
};

struct LmpcConfig {
  std::size_t horizon = 15;
  double dt = 0.01;
  double target_speed = 3.0;
  double max_cpu_time = 0.08;
  int max_iter = 100;
  double tolerance = 1e-3;
  double track_half_width = 1.0;
  double max_drive_force = 5.0;
  double max_brake_force = -10.0;
  double max_steer = 0.42;
  double wheelbase = 0.33;
};

struct SparseErrorModel {
  std::array<std::array<double, 6>, 6> A{};
  std::array<std::array<double, 2>, 6> B{};
  std::array<double, 6> C{};
};

struct LmpcSample {
  PaperLmpcState x;
  std::array<double, 2> u{};
  PaperLmpcState x_next;
};

struct FrenetProjection {
  double s = 0.0;
  double e_y = 0.0;
  double heading = 0.0;
  std::size_t segment_index = 0;
};

class CenterlineTrack {
public:
  CenterlineTrack(std::vector<double> x, std::vector<double> y,
                  bool closed = true);

  FrenetProjection project(double x, double y) const;
  RacingLmpcState to_racing_state(const GymVehicleState &state) const;
  PaperLmpcState to_paper_state(const GymVehicleState &state) const;

  double total_length() const;
  const std::vector<double> &s() const;

private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> s_;
  bool closed_ = true;
  double total_length_ = 0.0;
};

double normalize_angle(double angle);
PaperLmpcState racing_to_paper(const RacingLmpcState &state);

class NativeLMPCController {
public:
  NativeLMPCController();
  explicit NativeLMPCController(const LmpcConfig &config);
  ~NativeLMPCController();

  void reset();
  void update(const RacingLmpcState &state);
  void set_reference(const LmpcReference &reference);
  LmpcControlCommand control();

  const SparseErrorModel &error_model() const;
  std::size_t sample_count() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace f110_gym_lmpc
