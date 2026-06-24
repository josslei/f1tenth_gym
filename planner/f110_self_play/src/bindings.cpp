#include <torch/script.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "f110_observation.hpp"
#include "f110_self_play.hpp"
#include "f110_track_map.hpp"
#include "types.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
namespace sp = planner::f110_self_play;
namespace rk = f110_rollout_kernel;

namespace {

std::vector<float> array_to_vector(
    py::array_t<float, py::array::c_style | py::array::forcecast> array) {
  auto arr = array.unchecked<1>();
  std::vector<float> out(static_cast<std::size_t>(arr.shape(0)));
  for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
    out[static_cast<std::size_t>(i)] = arr(i);
  }
  return out;
}

py::array_t<float> tensor_row_to_array(const torch::Tensor &tensor) {
  auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  py::array_t<float> out(static_cast<py::ssize_t>(cpu.numel()));
  std::memcpy(out.mutable_data(), cpu.data_ptr<float>(),
              static_cast<std::size_t>(cpu.numel()) * sizeof(float));
  return out;
}

py::object self_play_result_to_python(const sp::SelfPlayResult &result) {
  py::module_ types = py::module_::import("types");
  py::object ns = types.attr("SimpleNamespace");
  py::list trajectories;
  for (const auto &trajectory : result.trajectories) {
    py::list transitions;
    for (const auto &step : trajectory.steps) {
      transitions.append(
          ns("obs"_a = tensor_row_to_array(step.obs), "action"_a = step.action,
             "reward"_a = step.reward, "done"_a = step.done,
             "root_policy"_a = tensor_row_to_array(step.root_policy)));
    }
    trajectories.append(ns("transitions"_a = transitions));
  }
  return ns("trajectories"_a = trajectories, "metrics"_a = result.metrics);
}

} // namespace

PYBIND11_MODULE(f110_self_play_native, m) {
  m.doc() = "Native F110 self-play bindings";

  py::class_<rk::TrackMap>(m, "TrackMap")
      .def(py::init<>())
      .def_readwrite("height", &rk::TrackMap::height)
      .def_readwrite("width", &rk::TrackMap::width)
      .def_readwrite("resolution", &rk::TrackMap::resolution)
      .def_readwrite("orig_x", &rk::TrackMap::orig_x)
      .def_readwrite("orig_y", &rk::TrackMap::orig_y)
      .def_readwrite("orig_c", &rk::TrackMap::orig_c)
      .def_readwrite("orig_s", &rk::TrackMap::orig_s)
      .def_readwrite("dt", &rk::TrackMap::dt)
      .def_readwrite("theta_dis", &rk::TrackMap::theta_dis)
      .def_readwrite("num_beams", &rk::TrackMap::num_beams)
      .def_readwrite("fov", &rk::TrackMap::fov)
      .def_readwrite("max_range", &rk::TrackMap::max_range)
      .def_readwrite("eps", &rk::TrackMap::eps)
      .def_readwrite("ttc_thresh", &rk::TrackMap::ttc_thresh)
      .def_readwrite("sines", &rk::TrackMap::sines)
      .def_readwrite("cosines", &rk::TrackMap::cosines)
      .def_readwrite("side_distances", &rk::TrackMap::side_distances)
      .def("compute_scan_tables", &rk::TrackMap::compute_scan_tables);

  py::class_<rk::ObservationConfig>(m, "ObservationConfig")
      .def(py::init<>())
      .def_readwrite("scan_size", &rk::ObservationConfig::scan_size)
      .def_readwrite("scan_max_m", &rk::ObservationConfig::scan_max_m)
      .def_readwrite("include_ego_state",
                     &rk::ObservationConfig::include_ego_state)
      .def_readwrite("speed_scale", &rk::ObservationConfig::speed_scale)
      .def_readwrite("yaw_rate_scale", &rk::ObservationConfig::yaw_rate_scale)
      .def_readwrite("steer_scale", &rk::ObservationConfig::steer_scale)
      .def_readwrite("include_waypoints",
                     &rk::ObservationConfig::include_waypoints)
      .def_readwrite("lookahead_distances",
                     &rk::ObservationConfig::lookahead_distances)
      .def_readwrite("waypoint_scale", &rk::ObservationConfig::waypoint_scale)
      .def_readwrite("waypoint_resample_spacing",
                     &rk::ObservationConfig::waypoint_resample_spacing);

  py::class_<rk::F110State>(m, "F110State")
      .def(py::init<>())
      .def_readwrite("x", &rk::F110State::x)
      .def_readwrite("y", &rk::F110State::y)
      .def_readwrite("steer_angle", &rk::F110State::steer_angle)
      .def_readwrite("velocity", &rk::F110State::velocity)
      .def_readwrite("yaw_angle", &rk::F110State::yaw_angle)
      .def_readwrite("yaw_rate", &rk::F110State::yaw_rate)
      .def_readwrite("slip_angle", &rk::F110State::slip_angle);

  py::class_<sp::ActionLattice>(m, "ActionLattice")
      .def(py::init<int32_t, int32_t, float, float>(), py::arg("steering_bins"),
           py::arg("velocity_bins"), py::arg("velocity_min") = 1.0f,
           py::arg("velocity_max") = 8.0f)
      .def(py::init(
               [](py::array_t<float, py::array::c_style | py::array::forcecast>
                      steering_bins,
                  py::array_t<float, py::array::c_style | py::array::forcecast>
                      velocity_bins) {
                 return sp::ActionLattice(array_to_vector(steering_bins),
                                          array_to_vector(velocity_bins));
               }),
           py::arg("steering_bins"), py::arg("velocity_bins"))
      .def_property_readonly("action_count", &sp::ActionLattice::action_count)
      .def(
          "normalized_action",
          [](const sp::ActionLattice &self, int32_t action_index) {
            return tensor_row_to_array(self.normalized_action(action_index));
          },
          py::arg("action_index"))
      .def(
          "normalized_batch",
          [](const sp::ActionLattice &self,
             py::array_t<int64_t> action_indices) {
            auto arr = action_indices.unchecked<1>();
            auto indices = torch::empty({arr.shape(0)}, torch::kInt64);
            auto *ptr = indices.data_ptr<int64_t>();
            for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
              ptr[i] = arr(i);
            }
            auto normalized = self.normalized_batch(indices).contiguous();
            py::array_t<float> out({arr.shape(0), py::ssize_t(2)});
            std::memcpy(out.mutable_data(), normalized.data_ptr<float>(),
                        static_cast<std::size_t>(normalized.numel()) *
                            sizeof(float));
            return out;
          },
          py::arg("action_indices"));

  py::class_<sp::MuZeroSearchAdapter, std::shared_ptr<sp::MuZeroSearchAdapter>>(
      m, "MuZeroSearchAdapter")
      .def(py::init([](const std::string &model_path, int32_t num_iters,
                       float temperature, float c_puct, int32_t batch_size,
                       int32_t action_count, int32_t hidden_size,
                       int32_t max_nodes, const std::string &device_name,
                       bool print_metrics) {
             auto torch_mod = py::module_::import("torch");
             torch::Device device = [&]() {
               if (!device_name.empty()) {
                 return torch::Device(device_name);
               }
               if (torch_mod.attr("cuda").attr("is_available")().cast<bool>()) {
                 return torch::Device(torch::kCUDA);
               }
               return torch::Device(torch::kCPU);
             }();
             TORCH_CHECK(
                 device.type() == torch::kCPU || device.type() == torch::kCUDA,
                 "MuZero native backend supports only CPU or CUDA, got ",
                 device.str());
             if (max_nodes <= 0) {
               max_nodes = num_iters + 1;
             }
             return sp::MuZeroSearchAdapter(
                 model_path, num_iters, temperature, c_puct, batch_size,
                 action_count, hidden_size, max_nodes, device, print_metrics);
           }),
           py::arg("model_path"), py::arg("num_iters"), py::arg("temperature"),
           py::arg("c_puct"), py::arg("batch_size"), py::arg("action_count"),
           py::arg("hidden_size"), py::arg("max_nodes") = 0,
           py::arg("device") = "", py::arg("print_metrics") = false);

  py::class_<sp::SelfPlayEngine>(m, "SelfPlayEngine")
      .def(py::init(
               [](std::shared_ptr<sp::MuZeroSearchAdapter> search,
                  const rk::TrackMap &track_map,
                  const rk::ObservationConfig &obs_config,
                  sp::ActionLattice action_lattice, float discount,
                  bool sample_actions, bool print_metrics,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      waypoints_x,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      waypoints_y,
                  py::array_t<double, py::array::c_style | py::array::forcecast>
                      cum_arc_lengths,
                  const rk::F110Params &dynamics_params, double car_length,
                  double car_width, double speed_reward_weight,
                  double progress_weight, double steer_smoothness_weight,
                  double collision_penalty, double spin_threshold) {
                 auto wx = waypoints_x.unchecked<1>();
                 auto wy = waypoints_y.unchecked<1>();
                 auto ca = cum_arc_lengths.unchecked<1>();
                 std::vector<double> wxv(static_cast<std::size_t>(wx.shape(0)));
                 std::vector<double> wyv(static_cast<std::size_t>(wy.shape(0)));
                 std::vector<double> cav(static_cast<std::size_t>(ca.shape(0)));
                 for (py::ssize_t i = 0; i < wx.shape(0); ++i) {
                   wxv[static_cast<std::size_t>(i)] = wx(i);
                 }
                 for (py::ssize_t i = 0; i < wy.shape(0); ++i) {
                   wyv[static_cast<std::size_t>(i)] = wy(i);
                 }
                 for (py::ssize_t i = 0; i < ca.shape(0); ++i) {
                   cav[static_cast<std::size_t>(i)] = ca(i);
                 }
                 return sp::SelfPlayEngine(
                     std::move(search), track_map, obs_config,
                     std::move(action_lattice), discount, sample_actions,
                     print_metrics, std::move(wxv), std::move(wyv),
                     std::move(cav), dynamics_params, car_length, car_width,
                     speed_reward_weight, progress_weight,
                     steer_smoothness_weight, collision_penalty,
                     spin_threshold);
               }),
           py::arg("search"), py::arg("track_map"), py::arg("obs_config"),
           py::arg("action_lattice"), py::arg("discount"),
           py::arg("sample_actions"), py::arg("print_metrics"),
           py::arg("waypoints_x"), py::arg("waypoints_y"),
           py::arg("cum_arc_lengths"), py::arg("dynamics_params"),
           py::arg("car_length"), py::arg("car_width"),
           py::arg("speed_reward_weight"), py::arg("progress_weight"),
           py::arg("steer_smoothness_weight"), py::arg("collision_penalty"),
           py::arg("spin_threshold"))
      .def(
          "generate",
          [](sp::SelfPlayEngine &self, int32_t rollout_steps,
             int32_t batch_size, py::array_t<double> init_states) {
            auto arr = init_states.unchecked<2>();
            std::vector<rk::F110State> states(
                static_cast<std::size_t>(arr.shape(0)));
            for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
              states[static_cast<std::size_t>(i)] = {
                  arr(i, 0), arr(i, 1), arr(i, 2), arr(i, 3),
                  arr(i, 4), arr(i, 5), arr(i, 6), 0,
                  0,         0,         false};
            }
            return self_play_result_to_python(
                self.generate(rollout_steps, batch_size, states));
          },
          py::arg("rollout_steps"), py::arg("batch_size"),
          py::arg("initial_states"));

  py::class_<rk::F110Params>(m, "F110Params")
      .def(py::init<>())
      .def_readwrite("mu", &rk::F110Params::mu)
      .def_readwrite("c_sf", &rk::F110Params::c_sf)
      .def_readwrite("c_sr", &rk::F110Params::c_sr)
      .def_readwrite("lf", &rk::F110Params::lf)
      .def_readwrite("lr", &rk::F110Params::lr)
      .def_readwrite("h", &rk::F110Params::h)
      .def_readwrite("m", &rk::F110Params::m)
      .def_readwrite("inertia", &rk::F110Params::inertia)
      .def_readwrite("s_min", &rk::F110Params::s_min)
      .def_readwrite("s_max", &rk::F110Params::s_max)
      .def_readwrite("sv_min", &rk::F110Params::sv_min)
      .def_readwrite("sv_max", &rk::F110Params::sv_max)
      .def_readwrite("v_switch", &rk::F110Params::v_switch)
      .def_readwrite("a_max", &rk::F110Params::a_max)
      .def_readwrite("v_min", &rk::F110Params::v_min)
      .def_readwrite("v_max", &rk::F110Params::v_max)
      .def_readwrite("timestep", &rk::F110Params::timestep);
}
