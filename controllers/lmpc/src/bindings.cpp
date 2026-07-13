#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "lmpc_controller.hpp"

namespace py = pybind11;

namespace
{

// The only array/DM conversion in this module lives here, at the Python/C++
// boundary -- lmpc::LMPCController itself speaks casadi::DM end to end.
casadi::DM dm_from_array(const py::array_t<double> & arr)
{
  const py::buffer_info info = arr.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a 1-D array");
  }
  const double * data = static_cast<const double *>(info.ptr);
  return casadi::DM(std::vector<double>(data, data + info.shape[0]));
}

py::array_t<double> array_from_dm(const casadi::DM & dm)
{
  const std::vector<double> values = dm.get_elements();
  py::array_t<double> arr(static_cast<py::ssize_t>(values.size()));
  std::copy(values.begin(), values.end(), arr.mutable_data());
  return arr;
}

}  // namespace

PYBIND11_MODULE(lmpc_native, m)
{
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
  .def_readwrite("centerline_csv_path", &lmpc::LmpcConfig::centerline_csv_path)
  .def_readwrite("seed_lap_csv_path", &lmpc::LmpcConfig::seed_lap_csv_path)
  .def_readwrite("vehicle_params", &lmpc::LmpcConfig::vehicle_params)
  .def_readwrite("K", &lmpc::LmpcConfig::K)
  .def_readwrite("a_min", &lmpc::LmpcConfig::a_min)
  .def_readwrite("a_max", &lmpc::LmpcConfig::a_max)
  .def_readwrite("delta_min", &lmpc::LmpcConfig::delta_min)
  .def_readwrite("delta_max", &lmpc::LmpcConfig::delta_max)
  .def_readwrite("ey_max", &lmpc::LmpcConfig::ey_max)
  .def_readwrite("c_u", &lmpc::LmpcConfig::c_u)
  .def_readwrite("c_du", &lmpc::LmpcConfig::c_du)
  .def_readwrite("solver_name", &lmpc::LmpcConfig::solver_name)
  .def_readwrite("linearization_speed_floor", &lmpc::LmpcConfig::linearization_speed_floor);

  py::class_<lmpc::LMPCController>(m, "NativeLMPCController")
  .def(py::init<const lmpc::LmpcConfig &>(), py::arg("config"))
  .def("reset", &lmpc::LMPCController::reset)
  .def(
    "update", [](lmpc::LMPCController & self, const py::array_t<double> & x, double t) {
      self.update(dm_from_array(x), t);
    },
    py::arg("x"), py::arg("t"))
  .def(
    "control", [](lmpc::LMPCController & self) {
      return array_from_dm(self.control());
    })
  .def(
    "predicted_next_state", [](const lmpc::LMPCController & self) {
      return array_from_dm(self.predicted_next_state());
    });
}
