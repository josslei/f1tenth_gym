#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "lmpc_controller.hpp"

namespace py = pybind11;

namespace {

// The only array/DM conversion in this module lives here, at the Python/C++
// boundary -- lmpc::LMPCController itself speaks casadi::DM end to end.
casadi::DM dm_from_array(const py::array_t<double> &arr) {
  const py::buffer_info info = arr.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a 1-D array");
  }
  const double *data = static_cast<const double *>(info.ptr);
  return casadi::DM(std::vector<double>(data, data + info.shape[0]));
}

// 2-D counterpart for matrix-valued inputs (add_lap's trajectories). Same
// explicit (i, j) loop rationale as array2d_from_dm below: sidesteps numpy
// row-major vs. CasADi column-major layout reasoning, and add_lap runs once
// per LAP, not per control step, so the loop's cost is irrelevant.
casadi::DM dm2d_from_array(const py::array_t<double> &arr) {
  const py::buffer_info info = arr.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("expected a 2-D array");
  }
  const auto rows = static_cast<casadi_int>(info.shape[0]);
  const auto cols = static_cast<casadi_int>(info.shape[1]);
  casadi::DM dm = casadi::DM::zeros(rows, cols);
  const auto view = arr.unchecked<2>();
  for (casadi_int i = 0; i < rows; ++i) {
    for (casadi_int j = 0; j < cols; ++j) {
      dm(i, j) = view(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
    }
  }
  return dm;
}

py::array_t<double> array_from_dm(const casadi::DM &dm) {
  const std::vector<double> values = dm.get_elements();
  py::array_t<double> arr(static_cast<py::ssize_t>(values.size()));
  std::copy(values.begin(), values.end(), arr.mutable_data());
  return arr;
}

// 2-D counterpart for matrix-valued DMs (e.g. predicted_trajectory()). Uses
// an explicit (i, j) loop rather than dm.get_elements()'s flat, column-
// major buffer -- avoids having to reason about numpy's default row-major
// layout vs. CasADi's column-major storage; matrices here are small
// (kStateDim x horizon_steps+1) and this runs at most once per rendered
// frame, so the loop's cost is not a concern.
py::array_t<double> array2d_from_dm(const casadi::DM &dm) {
  const auto rows = static_cast<py::ssize_t>(dm.size1());
  const auto cols = static_cast<py::ssize_t>(dm.size2());
  py::array_t<double> arr({rows, cols});
  auto view = arr.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < rows; ++i) {
    for (py::ssize_t j = 0; j < cols; ++j) {
      view(i, j) = static_cast<double>(
          dm(static_cast<casadi_int>(i), static_cast<casadi_int>(j)));
    }
  }
  return arr;
}

} // namespace

PYBIND11_MODULE(lmpc_native, m) {
  // CasADi loads conic/NLP solvers (e.g. "qrqp", DESIGN.md SS7) as separate
  // plugin .dylibs via its own dlopen-by-name search, which does not
  // consult this module's rpath -- see the comment on LMPC_NATIVE_DIR in
  // CMakeLists.txt. The CMake build always copies plugin .dylibs into that
  // same directory as lmpc_native.so, so this makes them discoverable
  // regardless of the caller's working directory.
  casadi::GlobalOptions::setCasadiPath(LMPC_NATIVE_DIR);

  m.doc() = "Native LMPC controller (casadi-backed FHOCP solver)";

  // Every field here is documented in controllers/lmpc/include/lmpc_config.hpp
  // -- kept in sync with that struct rather than re-explaining each knob here.
  py::class_<lmpc::dynamics::VehicleParams>(m, "VehicleParams")
      .def(py::init<>())
      .def_readwrite("mu", &lmpc::dynamics::VehicleParams::mu)
      .def_readwrite("C_Sf", &lmpc::dynamics::VehicleParams::C_Sf)
      .def_readwrite("C_Sr", &lmpc::dynamics::VehicleParams::C_Sr)
      .def_readwrite("lf", &lmpc::dynamics::VehicleParams::lf)
      .def_readwrite("lr", &lmpc::dynamics::VehicleParams::lr)
      .def_readwrite("h", &lmpc::dynamics::VehicleParams::h)
      .def_readwrite("m", &lmpc::dynamics::VehicleParams::m)
      .def_readwrite("I", &lmpc::dynamics::VehicleParams::I);

  py::class_<lmpc::LmpcConfig>(m, "LmpcConfig")
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
      .def_readwrite("v_min", &lmpc::LmpcConfig::v_min)
      .def_readwrite("v_max", &lmpc::LmpcConfig::v_max)
      .def_readwrite("delta_min", &lmpc::LmpcConfig::delta_min)
      .def_readwrite("delta_max", &lmpc::LmpcConfig::delta_max)
      .def_readwrite("sv_max", &lmpc::LmpcConfig::sv_max)
      .def_readwrite("ey_max", &lmpc::LmpcConfig::ey_max)
      .def_readwrite("cost_to_go_weight", &lmpc::LmpcConfig::cost_to_go_weight)
      .def_readwrite("terminal_slack_weight",
                     &lmpc::LmpcConfig::terminal_slack_weight)
      .def_readwrite("c_u", &lmpc::LmpcConfig::c_u)
      .def_readwrite("c_d_u", &lmpc::LmpcConfig::c_d_u)
      .def_readwrite("ey_slack_l1", &lmpc::LmpcConfig::ey_slack_l1)
      .def_readwrite("ey_slack_l2", &lmpc::LmpcConfig::ey_slack_l2)
      .def_readwrite("scale_x_vy", &lmpc::LmpcConfig::scale_x_vy)
      .def_readwrite("scale_x_omega", &lmpc::LmpcConfig::scale_x_omega)
      .def_readwrite("scale_x_epsi", &lmpc::LmpcConfig::scale_x_epsi)
      .def_readwrite("solver_name", &lmpc::LmpcConfig::solver_name);

  // recom.md's requested profiling breakdown -- lmpc_controller.hpp's
  // ControllerTimings comment has the full per-field rationale.
  py::class_<lmpc::ControllerTimings>(m, "ControllerTimings")
      .def_readonly("rollout_lin_ms", &lmpc::ControllerTimings::rollout_lin_ms)
      .def_readonly("knn_ms", &lmpc::ControllerTimings::knn_ms)
      .def_readonly("set_params_ms", &lmpc::ControllerTimings::set_params_ms)
      .def_readonly("solver_ms", &lmpc::ControllerTimings::solver_ms)
      .def_readonly("postcheck_ms", &lmpc::ControllerTimings::postcheck_ms);

  py::class_<lmpc::LMPCController>(m, "NativeLMPCController")
      .def(py::init<const lmpc::LmpcConfig &>(), py::arg("config"))
      .def("reset", &lmpc::LMPCController::reset)
      .def(
          "update",
          [](lmpc::LMPCController &self, const py::array_t<double> &x, double t,
             double actual_delta) {
            self.update(dm_from_array(x), t, actual_delta);
          },
          py::arg("x"), py::arg("t"), py::arg("actual_delta"))
      .def("control",
           [](lmpc::LMPCController &self) {
             return array_from_dm(self.control());
           })
      .def("predicted_next_state",
           [](const lmpc::LMPCController &self) {
             return array_from_dm(self.predicted_next_state());
           })
      .def("predicted_trajectory",
           [](const lmpc::LMPCController &self) {
             return array2d_from_dm(self.predicted_trajectory());
           })
      .def("last_timings", &lmpc::LMPCController::last_timings)
      .def(
          "add_lap",
          [](lmpc::LMPCController &self, const py::array_t<double> &x_lap,
             const py::array_t<double> &u_lap,
             const py::array_t<double> &J_lap) {
            self.add_lap(dm2d_from_array(x_lap), dm2d_from_array(u_lap),
                         dm_from_array(J_lap));
          },
          py::arg("x_lap"), py::arg("u_lap"), py::arg("J_lap"));
}
