#include "f110_collision.hpp"
#include "f110_observation.hpp"
#include "f110_reward.hpp"
#include "f110_scan.hpp"
#include "f110_step.hpp"
#include "f110_track_map.hpp"
#include "step.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>
#include <vector>

namespace py = pybind11;
namespace rk = f110_rollout_kernel;

PYBIND11_MODULE(_f110_rollout_kernel, m) {
  m.doc() = "Native F110 rollout kernel for planner/search rollouts.";

  m.attr("STATE_COLUMNS") =
      py::make_tuple("x", "y", "steer_angle", "velocity", "yaw_angle",
                     "yaw_rate", "slip_angle", "steer_buffer_0",
                     "steer_buffer_1", "steer_buffer_len", "in_collision");

  py::enum_<rk::Integrator>(m, "Integrator")
      .value("Euler", rk::Integrator::Euler)
      .value("RK4", rk::Integrator::RK4)
      .export_values();

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

  py::class_<rk::F110State>(m, "F110State")
      .def(py::init<>())
      .def_readwrite("x", &rk::F110State::x)
      .def_readwrite("y", &rk::F110State::y)
      .def_readwrite("steer_angle", &rk::F110State::steer_angle)
      .def_readwrite("velocity", &rk::F110State::velocity)
      .def_readwrite("yaw_angle", &rk::F110State::yaw_angle)
      .def_readwrite("yaw_rate", &rk::F110State::yaw_rate)
      .def_readwrite("slip_angle", &rk::F110State::slip_angle)
      .def_readwrite("steer_buffer_0", &rk::F110State::steer_buffer_0)
      .def_readwrite("steer_buffer_1", &rk::F110State::steer_buffer_1)
      .def_readwrite("steer_buffer_len", &rk::F110State::steer_buffer_len)
      .def_readwrite("in_collision", &rk::F110State::in_collision);

  py::class_<rk::F110Action>(m, "F110Action")
      .def(py::init([](double steer, double velocity) {
             return rk::F110Action{steer, velocity};
           }),
           py::arg("steer") = 0.0, py::arg("velocity") = 0.0)
      .def_readwrite("steer", &rk::F110Action::steer)
      .def_readwrite("velocity", &rk::F110Action::velocity);

  py::class_<rk::F110StepResult>(m, "F110StepResult")
      .def_readonly("state", &rk::F110StepResult::state)
      .def_readonly("reward", &rk::F110StepResult::reward)
      .def_readonly("discount", &rk::F110StepResult::discount)
      .def_readonly("terminal", &rk::F110StepResult::terminal);

  py::class_<rk::F110ProgressReward>(m, "F110ProgressReward")
      .def(py::init<>())
      .def(py::init<const std::vector<double> &, const std::vector<double> &,
                    double, double, double, double, double>(),
           py::arg("waypoints_x"), py::arg("waypoints_y"),
           py::arg("speed_reward_weight") = 0.1,
           py::arg("progress_weight") = 2.0,
           py::arg("steer_smoothness_weight") = 0.5,
           py::arg("collision_penalty") = 50.0,
           py::arg("spin_threshold") = 100.0)
      .def("__call__", &rk::F110ProgressReward::operator(), py::arg("px"),
           py::arg("py"), py::arg("theta"), py::arg("vx"), py::arg("vy"),
           py::arg("steer"), py::arg("collision"), py::arg("terminated"))
      .def("set_waypoints", &rk::F110ProgressReward::set_waypoints,
           py::arg("waypoints_x"), py::arg("waypoints_y"))
      .def("reset", &rk::F110ProgressReward::reset);

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

  m.def("observation_dim", &rk::observation_dim, py::arg("config"));

  m.def(
      "get_scan",
      [](double px, double py, double theta, const rk::TrackMap &map) {
        py::ssize_t nb = static_cast<py::ssize_t>(map.num_beams);
        py::array_t<float> scan(nb);
        rk::get_scan_one(px, py, theta, map, scan.mutable_data());
        return scan;
      },
      py::arg("px"), py::arg("py"), py::arg("theta"), py::arg("map"));

  m.def(
      "get_scan_batch",
      [](py::array_t<double> poses, const rk::TrackMap &map) {
        auto p = poses.unchecked<2>();
        py::ssize_t B = p.shape(0);
        py::ssize_t nb = static_cast<py::ssize_t>(map.num_beams);
        py::array_t<float> scans({B, nb});
        std::vector<double> pflat(static_cast<std::size_t>(B * 3));
        for (py::ssize_t i = 0; i < B; ++i) {
          pflat[static_cast<std::size_t>(i * 3)] = p(i, 0);
          pflat[static_cast<std::size_t>(i * 3 + 1)] = p(i, 1);
          pflat[static_cast<std::size_t>(i * 3 + 2)] = p(i, 2);
        }
        rk::get_scan_batch(pflat.data(), static_cast<int>(B), map,
                           scans.mutable_data());
        return scans;
      },
      py::arg("poses"), py::arg("map"));

  m.def(
      "check_ttc",
      [](py::array_t<float> scan, double vel, const rk::TrackMap &map) {
        return rk::check_ttc_one(scan.data(), vel, map);
      },
      py::arg("scan"), py::arg("vel"), py::arg("map"));

  m.def(
      "build_observation",
      [](py::array_t<float> scan_1080, double px, double py, double theta,
         double vx, double vy, double yaw_rate, double steer, bool collision,
         py::array_t<float, py::array::c_style | py::array::forcecast>
             prev_action,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             waypoints_x,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             waypoints_y,
         py::array_t<double, py::array::c_style | py::array::forcecast>
             cum_arc_lengths,
         const rk::ObservationConfig &config) {
        int nwp = static_cast<int>(waypoints_x.size());
        int odim = rk::observation_dim(config);
        py::array_t<float> obs(odim);
        rk::build_observation_one(
            scan_1080.data(), px, py, theta, vx, vy, yaw_rate, steer,
            collision ? 1 : 0, prev_action.data(), waypoints_x.data(),
            waypoints_y.data(), nwp, cum_arc_lengths.data(), config,
            obs.mutable_data());
        return obs;
      },
      py::arg("scan"), py::arg("px"), py::arg("py"), py::arg("theta"),
      py::arg("vx"), py::arg("vy"), py::arg("yaw_rate"), py::arg("steer"),
      py::arg("collision"), py::arg("prev_action"), py::arg("waypoints_x"),
      py::arg("waypoints_y"), py::arg("cum_arc_lengths"), py::arg("config"));

  m.def(
      "get_vertices",
      [](double x, double y, double theta, double length, double width) {
        auto v = rk::get_vertices(x, y, theta, length, width);
        py::array_t<double> arr({py::ssize_t(4), py::ssize_t(2)});
        auto a = arr.mutable_unchecked<2>();
        for (int i = 0; i < 4; ++i) {
          a(i, 0) = v[static_cast<std::size_t>(i)][0];
          a(i, 1) = v[static_cast<std::size_t>(i)][1];
        }
        return arr;
      },
      py::arg("x"), py::arg("y"), py::arg("theta"), py::arg("length"),
      py::arg("width"));

  m.def(
      "gjk_collision",
      [](py::array_t<double> v1, py::array_t<double> v2) {
        auto a1 = v1.unchecked<2>();
        auto a2 = v2.unchecked<2>();
        rk::Vertices4 vv1, vv2;
        for (int i = 0; i < 4; ++i) {
          vv1[static_cast<std::size_t>(i)] = {a1(i, 0), a1(i, 1)};
          vv2[static_cast<std::size_t>(i)] = {a2(i, 0), a2(i, 1)};
        }
        return rk::collision(vv1, vv2);
      },
      py::arg("vertices1"), py::arg("vertices2"));

  m.attr("DEFAULT_PARAMS") = rk::F110Params{};
  m.def("step", &rk::step, py::arg("state"), py::arg("action"),
        py::arg("params") = rk::F110Params{},
        py::arg("integrator") = rk::Integrator::RK4);
}
