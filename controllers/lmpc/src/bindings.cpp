#include "lmpc/state.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace lmpc = f110_gym_lmpc;

PYBIND11_MODULE(lmpc_native, m) {
  m.doc() = "Native C++ utilities for Gym-facing LMPC integration.";

  m.attr("RACING_STATE_COLUMNS") =
      py::make_tuple("s", "e_y", "e_psi", "v_x", "v_y", "omega");
  m.attr("PAPER_STATE_COLUMNS") =
      py::make_tuple("v_x", "v_y", "omega", "e_psi", "s", "e_y");

  py::class_<lmpc::GymVehicleState>(m, "GymVehicleState")
      .def(py::init<double, double, double, double, double, double>(),
           py::arg("x"), py::arg("y"), py::arg("yaw"), py::arg("v_x"),
           py::arg("v_y"), py::arg("omega"))
      .def_readwrite("x", &lmpc::GymVehicleState::x)
      .def_readwrite("y", &lmpc::GymVehicleState::y)
      .def_readwrite("yaw", &lmpc::GymVehicleState::yaw)
      .def_readwrite("v_x", &lmpc::GymVehicleState::v_x)
      .def_readwrite("v_y", &lmpc::GymVehicleState::v_y)
      .def_readwrite("omega", &lmpc::GymVehicleState::omega);

  py::class_<lmpc::RacingLmpcState>(m, "RacingLmpcState")
      .def(py::init<>())
      .def_readwrite("s", &lmpc::RacingLmpcState::s)
      .def_readwrite("e_y", &lmpc::RacingLmpcState::e_y)
      .def_readwrite("e_psi", &lmpc::RacingLmpcState::e_psi)
      .def_readwrite("v_x", &lmpc::RacingLmpcState::v_x)
      .def_readwrite("v_y", &lmpc::RacingLmpcState::v_y)
      .def_readwrite("omega", &lmpc::RacingLmpcState::omega)
      .def("to_array", &lmpc::RacingLmpcState::to_array);

  py::class_<lmpc::PaperLmpcState>(m, "PaperLmpcState")
      .def(py::init<>())
      .def_readwrite("v_x", &lmpc::PaperLmpcState::v_x)
      .def_readwrite("v_y", &lmpc::PaperLmpcState::v_y)
      .def_readwrite("omega", &lmpc::PaperLmpcState::omega)
      .def_readwrite("e_psi", &lmpc::PaperLmpcState::e_psi)
      .def_readwrite("s", &lmpc::PaperLmpcState::s)
      .def_readwrite("e_y", &lmpc::PaperLmpcState::e_y)
      .def("to_array", &lmpc::PaperLmpcState::to_array);

  py::class_<lmpc::LmpcControlCommand>(m, "LmpcControlCommand")
      .def(py::init<>())
      .def_readwrite("steering", &lmpc::LmpcControlCommand::steering)
      .def_readwrite("velocity", &lmpc::LmpcControlCommand::velocity);

  py::class_<lmpc::LmpcReference>(m, "LmpcReference")
      .def(py::init<>())
      .def_readwrite("curvature", &lmpc::LmpcReference::curvature)
      .def_readwrite("target_speed", &lmpc::LmpcReference::target_speed)
      .def_readwrite("left_bound", &lmpc::LmpcReference::left_bound)
      .def_readwrite("right_bound", &lmpc::LmpcReference::right_bound);

  py::class_<lmpc::LmpcConfig>(m, "LmpcConfig")
      .def(py::init<>())
      .def_readwrite("horizon", &lmpc::LmpcConfig::horizon)
      .def_readwrite("dt", &lmpc::LmpcConfig::dt)
      .def_readwrite("target_speed", &lmpc::LmpcConfig::target_speed)
      .def_readwrite("max_cpu_time", &lmpc::LmpcConfig::max_cpu_time)
      .def_readwrite("max_iter", &lmpc::LmpcConfig::max_iter)
      .def_readwrite("tolerance", &lmpc::LmpcConfig::tolerance)
      .def_readwrite("track_half_width", &lmpc::LmpcConfig::track_half_width)
      .def_readwrite("max_drive_force", &lmpc::LmpcConfig::max_drive_force)
      .def_readwrite("max_brake_force", &lmpc::LmpcConfig::max_brake_force)
      .def_readwrite("max_steer", &lmpc::LmpcConfig::max_steer)
      .def_readwrite("wheelbase", &lmpc::LmpcConfig::wheelbase)
      .def_readwrite("track_length", &lmpc::LmpcConfig::track_length)
      .def_readwrite("max_lap_stored", &lmpc::LmpcConfig::max_lap_stored)
      .def_readwrite("reg_dist_max", &lmpc::LmpcConfig::reg_dist_max)
      .def_readwrite("reg_max_points", &lmpc::LmpcConfig::reg_max_points)
      .def_readwrite("reg_max_points_per_lap",
                     &lmpc::LmpcConfig::reg_max_points_per_lap);

  py::class_<lmpc::SparseErrorModel>(m, "SparseErrorModel")
      .def_readonly("A", &lmpc::SparseErrorModel::A)
      .def_readonly("B", &lmpc::SparseErrorModel::B)
      .def_readonly("C", &lmpc::SparseErrorModel::C);

  py::class_<lmpc::FrenetProjection>(m, "FrenetProjection")
      .def_readonly("s", &lmpc::FrenetProjection::s)
      .def_readonly("e_y", &lmpc::FrenetProjection::e_y)
      .def_readonly("heading", &lmpc::FrenetProjection::heading)
      .def_readonly("segment_index", &lmpc::FrenetProjection::segment_index);

  py::class_<lmpc::CenterlineTrack>(m, "CenterlineTrack")
      .def(py::init<std::vector<double>, std::vector<double>, bool>(),
           py::arg("x"), py::arg("y"), py::arg("closed") = true)
      .def("project", &lmpc::CenterlineTrack::project, py::arg("x"),
           py::arg("y"))
      .def("to_racing_state", &lmpc::CenterlineTrack::to_racing_state,
           py::arg("state"))
      .def("to_paper_state", &lmpc::CenterlineTrack::to_paper_state,
           py::arg("state"))
      .def("total_length", &lmpc::CenterlineTrack::total_length)
      .def("s", &lmpc::CenterlineTrack::s);

  m.def("normalize_angle", &lmpc::normalize_angle, py::arg("angle"));
  m.def("racing_to_paper", &lmpc::racing_to_paper, py::arg("state"));

  py::class_<lmpc::NativeLMPCController>(m, "NativeLMPCController")
      .def(py::init<>())
      .def(py::init<const lmpc::LmpcConfig &>(), py::arg("config"))
      .def("reset", &lmpc::NativeLMPCController::reset)
      .def("update", &lmpc::NativeLMPCController::update, py::arg("state"))
      .def("set_reference", &lmpc::NativeLMPCController::set_reference,
           py::arg("reference"))
      .def("control", &lmpc::NativeLMPCController::control)
      .def("error_model", &lmpc::NativeLMPCController::error_model,
           py::return_value_policy::reference_internal)
      .def("sample_count", &lmpc::NativeLMPCController::sample_count);
}
