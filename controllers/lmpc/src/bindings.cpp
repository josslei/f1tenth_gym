#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lmpc_controller.hpp"

namespace py = pybind11;

namespace {

casadi::DM dm_from_array(const py::array_t<double> &array) {
  const auto values = array.unchecked<1>();
  casadi::DM result = casadi::DM::zeros(values.shape(0), 1);
  for (py::ssize_t i = 0; i < values.shape(0); ++i) {
    result(i) = values(i);
  }
  return result;
}

casadi::DM dm_from_matrix(const py::array_t<double> &array) {
  const auto values = array.unchecked<2>();
  casadi::DM result = casadi::DM::zeros(values.shape(0), values.shape(1));
  for (py::ssize_t row = 0; row < values.shape(0); ++row) {
    for (py::ssize_t col = 0; col < values.shape(1); ++col) {
      result(row, col) = values(row, col);
    }
  }
  return result;
}

py::array_t<double> array_from_dm(const casadi::DM &value) {
  if (value.size2() == 1) {
    py::array_t<double> result(value.size1());
    auto output = result.mutable_unchecked<1>();
    for (casadi_int row = 0; row < value.size1(); ++row) {
      output(row) = static_cast<double>(value(row));
    }
    return result;
  }
  py::array_t<double> result({value.size1(), value.size2()});
  auto output = result.mutable_unchecked<2>();
  for (casadi_int row = 0; row < value.size1(); ++row) {
    for (casadi_int col = 0; col < value.size2(); ++col) {
      output(row, col) = static_cast<double>(value(row, col));
    }
  }
  return result;
}

} // namespace

PYBIND11_MODULE(lmpc_native, module) {
  py::class_<lmpc::VehicleParams>(module, "VehicleParams")
      .def(py::init<>())
      .def_readwrite("mu", &lmpc::VehicleParams::mu)
      .def_readwrite("C_Sf", &lmpc::VehicleParams::C_Sf)
      .def_readwrite("C_Sr", &lmpc::VehicleParams::C_Sr)
      .def_readwrite("lf", &lmpc::VehicleParams::lf)
      .def_readwrite("lr", &lmpc::VehicleParams::lr)
      .def_readwrite("h", &lmpc::VehicleParams::h)
      .def_readwrite("m", &lmpc::VehicleParams::m)
      .def_readwrite("I", &lmpc::VehicleParams::I);

  py::class_<lmpc::LmpcConfig>(module, "LmpcConfig")
      .def(py::init<>())
      .def_readwrite("dt", &lmpc::LmpcConfig::dt)
      .def_readwrite("horizon_steps", &lmpc::LmpcConfig::horizon_steps)
      .def_readwrite("centerline_csv_path",
                     &lmpc::LmpcConfig::centerline_csv_path)
      .def_readwrite("seed_lap_csv_path", &lmpc::LmpcConfig::seed_lap_csv_path)
      .def_readwrite("vehicle_params", &lmpc::LmpcConfig::vehicle_params)
      .def_readwrite("K", &lmpc::LmpcConfig::K)
      .def_readwrite("a_min", &lmpc::LmpcConfig::a_min)
      .def_readwrite("a_max", &lmpc::LmpcConfig::a_max)
      .def_readwrite("delta_min", &lmpc::LmpcConfig::delta_min)
      .def_readwrite("delta_max", &lmpc::LmpcConfig::delta_max)
      .def_readwrite("v_max", &lmpc::LmpcConfig::v_max)
      .def_readwrite("velocity_threshold",
                     &lmpc::LmpcConfig::velocity_threshold)
      .def_readwrite("model_mode", &lmpc::LmpcConfig::model_mode)
      .def_readwrite("map_margin", &lmpc::LmpcConfig::map_margin)
      .def_readwrite("waypoint_space", &lmpc::LmpcConfig::waypoint_space)
      .def_readwrite("r_accel", &lmpc::LmpcConfig::r_accel)
      .def_readwrite("r_steer", &lmpc::LmpcConfig::r_steer)
      .def_readwrite("r_d_accel", &lmpc::LmpcConfig::r_d_accel)
      .def_readwrite("r_d_steer", &lmpc::LmpcConfig::r_d_steer)
      .def_readwrite("ey_slack_l2", &lmpc::LmpcConfig::ey_slack_l2)
      .def_readwrite("terminal_slack_weight",
                     &lmpc::LmpcConfig::terminal_slack_weight)
      .def_readwrite("osqp_max_iter", &lmpc::LmpcConfig::osqp_max_iter)
      .def_readwrite("osqp_scaling", &lmpc::LmpcConfig::osqp_scaling)
      .def_readwrite("osqp_eps_prim_inf", &lmpc::LmpcConfig::osqp_eps_prim_inf)
      .def_readwrite("osqp_eps_abs", &lmpc::LmpcConfig::osqp_eps_abs)
      .def_readwrite("osqp_eps_rel", &lmpc::LmpcConfig::osqp_eps_rel)
      .def_readwrite("occupancy_grid", &lmpc::LmpcConfig::occupancy_grid)
      .def_readwrite("map_width", &lmpc::LmpcConfig::map_width)
      .def_readwrite("map_height", &lmpc::LmpcConfig::map_height)
      .def_readwrite("map_resolution", &lmpc::LmpcConfig::map_resolution)
      .def_readwrite("map_origin_x", &lmpc::LmpcConfig::map_origin_x)
      .def_readwrite("map_origin_y", &lmpc::LmpcConfig::map_origin_y)
      .def_readwrite("reference_waypoint_csv_path",
                     &lmpc::LmpcConfig::reference_waypoint_csv_path)
      .def_readwrite("reference_seed_lap_csv_path",
                     &lmpc::LmpcConfig::reference_seed_lap_csv_path)
      .def_readwrite("initial_x", &lmpc::LmpcConfig::initial_x)
      .def_readwrite("initial_y", &lmpc::LmpcConfig::initial_y)
      .def_readwrite("initial_yaw", &lmpc::LmpcConfig::initial_yaw)
      .def_readwrite("regression_enabled",
                     &lmpc::LmpcConfig::regression_enabled)
      .def_readwrite("regression_num_neighbors",
                     &lmpc::LmpcConfig::regression_num_neighbors)
      .def_readwrite("regression_bandwidth",
                     &lmpc::LmpcConfig::regression_bandwidth)
      .def_readwrite("regression_regularization",
                     &lmpc::LmpcConfig::regression_regularization)
      .def_readwrite("regression_Q", &lmpc::LmpcConfig::regression_Q);

  py::class_<lmpc::ControllerTimings>(module, "ControllerTimings")
      .def_readonly("rollout_lin_ms", &lmpc::ControllerTimings::rollout_lin_ms)
      .def_readonly("knn_ms", &lmpc::ControllerTimings::knn_ms)
      .def_readonly("regression_ms", &lmpc::ControllerTimings::regression_ms)
      .def_readonly("set_params_ms", &lmpc::ControllerTimings::set_params_ms)
      .def_readonly("solver_ms", &lmpc::ControllerTimings::solver_ms)
      .def_readonly("postcheck_ms", &lmpc::ControllerTimings::postcheck_ms);

  py::class_<lmpc::LMPCController>(module, "NativeLMPCController")
      .def(py::init<const lmpc::LmpcConfig &>())
      .def("reset", &lmpc::LMPCController::reset)
      .def("update",
           [](lmpc::LMPCController &controller,
              const py::array_t<double> &state, double t, double actual_delta) {
             controller.update(dm_from_array(state), t, actual_delta);
           })
      .def("control",
           [](lmpc::LMPCController &controller) {
             return array_from_dm(controller.control());
           })
      .def("add_lap",
           [](lmpc::LMPCController &controller,
              const py::array_t<double> &states,
              const py::array_t<double> &controls,
              const py::array_t<double> &costs) {
             controller.add_lap(dm_from_matrix(states),
                                dm_from_matrix(controls), dm_from_array(costs));
           })
      .def("predicted_trajectory",
           [](const lmpc::LMPCController &controller) {
             return array_from_dm(controller.predicted_trajectory());
           })
      .def("last_timings", &lmpc::LMPCController::last_timings,
           py::return_value_policy::reference_internal)
      .def("last_terminal_slack",
           [](const lmpc::LMPCController &controller) {
             return array_from_dm(controller.last_terminal_slack_value());
           })
      .def("last_solve_ok", &lmpc::LMPCController::last_solve_ok)
      .def("using_dynamic_model", &lmpc::LMPCController::using_dynamic_model)
      .def("regression_pool_size", &lmpc::LMPCController::regression_pool_size)
      .def("last_regression_correction_norm",
           &lmpc::LMPCController::last_regression_correction_norm);
}
