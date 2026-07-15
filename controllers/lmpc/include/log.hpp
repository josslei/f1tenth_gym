#pragma once

#include <memory>

#include <casadi/casadi.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// This vendored fmt (bundled inside spdlog) requires an explicit opt-in per
// type for its operator<<-based fallback formatter -- spdlog/fmt/ostr.h
// alone does NOT make casadi::DM (== casadi::Matrix<double>, which already
// defines operator<<) formattable via "{}"; every SPDLOG_LOGGER_*(logger,
// "...{}...", some_DM) call fails to compile without this specialization
// (confirmed directly: it compiles fine when SPDLOG_ACTIVE_LEVEL strips the
// call, e.g. Release builds, and fails the moment a non-Release build
// actually instantiates it). One specialization here covers every call site
// across the module.
template <>
struct fmt::formatter<casadi::Matrix<double>> : fmt::ostream_formatter {};

namespace lmpc {

// Process-wide logger for the native LMPC module (debug/perf visibility
// only -- actual solver failures are still reported via thrown exceptions,
// e.g. LMPCController::control()/QpBuilder::solve(), never through this
// logger). Replaces the old ad hoc LMPC_DEBUG_TERMINAL/LMPC_DEBUG_STAGES
// getenv()-gated std::cerr prints with spdlog's own two-tier cost model:
//  - Compile time: SPDLOG_ACTIVE_LEVEL (CMakeLists.txt, per build type)
//    strips SPDLOG_LOGGER_TRACE/_DEBUG calls below that level out of the
//    binary entirely -- a Release build pays nothing for them, not even a
//    runtime branch.
//  - Run time (whatever level survived compilation): the standard
//    SPDLOG_LEVEL env var, e.g. SPDLOG_LEVEL=lmpc=trace, read once via
//    spdlog::cfg::load_env_levels() below.
// spdlog/fmt/ostr.h pulls in fmt's operator<<-based formatter, so casadi::DM
// (which already defines operator<<) can be logged directly via "{}", the
// same way it used to stream into std::cerr.
inline const std::shared_ptr<spdlog::logger> &log() {
  static const std::shared_ptr<spdlog::logger> logger = [] {
    auto result = spdlog::stderr_color_mt("lmpc");
    result->set_pattern("[%n] [%^%l%$] %v");
    spdlog::cfg::load_env_levels();
    return result;
  }();
  return logger;
}

} // namespace lmpc
