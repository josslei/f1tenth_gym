cmake_minimum_required(VERSION 3.17)
project(lmpc_controller LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(LMPC_THIRDPARTY_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../thirdparty"
  CACHE PATH "Path to the repository thirdparty directory"
)
set(LMPC_CASADI_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/casadi"
  CACHE PATH "Path to the CasADi source checkout"
)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(casadi CONFIG QUIET
  NO_CMAKE_PACKAGE_REGISTRY
  HINTS
    "${LMPC_CASADI_SOURCE_DIR}/build"
    "${LMPC_CASADI_SOURCE_DIR}/build/casadi"
    "${LMPC_CASADI_SOURCE_DIR}/cmake"
)
if(NOT TARGET casadi AND NOT TARGET casadi::casadi)
  # DESIGN.md SS7: qrqp is CasADi's own built-in dense active-set QP
  # solver -- part of casadi core, no external plugin/build flag needed.
  # ipopt is ALSO available (LmpcConfig::solver_name, 2026-07-14): CasADi's
  # own superbuild git-clones and builds IPOPT (jgillis/Ipopt-1, upstream's
  # own easier-to-build fork) plus its MUMPS linear-solver dependency from
  # source via ExternalProject_Add -- no system package manager install
  # needed. WITH_BUILD_REQUIRED covers whichever of IPOPT's OWN transitive
  # deps (LAPACK, METIS) aren't already found on this system, building
  # those from source too rather than requiring them pre-installed.
  set(WITH_IPOPT ON CACHE BOOL "Enable the IPOPT NLP solver interface" FORCE)
  set(WITH_BUILD_REQUIRED ON CACHE BOOL "Build IPOPT's Fortran-based transitive deps (LAPACK/METIS) from source if not found on the system" FORCE)
  set(WITH_BUILD_IPOPT ON CACHE BOOL "Build IPOPT itself from source (git clone + configure/make, no brew/apt install)" FORCE)
  set(WITH_PYTHON OFF CACHE BOOL "Do not build CasADi's own Python frontend here" FORCE)
  set(WITH_EXAMPLES OFF CACHE BOOL "Do not build CasADi examples here" FORCE)
  set(WITH_DOC OFF CACHE BOOL "Do not build CasADi docs here" FORCE)
  add_subdirectory(
    "${LMPC_CASADI_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/casadi"
    EXCLUDE_FROM_ALL
  )
endif()
if(TARGET casadi)
  set(LMPC_CASADI_TARGET casadi)
elseif(TARGET casadi::casadi)
  set(LMPC_CASADI_TARGET casadi::casadi)
else()
  message(FATAL_ERROR "CasADi source/package did not provide a usable CMake target")
endif()

add_subdirectory(
  "${LMPC_THIRDPARTY_DIR}/pybind11"
  "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  EXCLUDE_FROM_ALL
)

# spdlog (2026-07-15): debug/perf logging (include/log.hpp), header-only so
# no extra plugin dylib to copy next to lmpc_native.so (unlike the CasADi
# solver plugins below). Its own tests/examples aren't needed here.
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Do not build spdlog's own example" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Do not build spdlog's own tests" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "Do not generate an spdlog install target" FORCE)
add_subdirectory(
  "${LMPC_THIRDPARTY_DIR}/spdlog"
  "${CMAKE_CURRENT_BINARY_DIR}/spdlog"
  EXCLUDE_FROM_ALL
)

add_library(lmpc_core STATIC
  src/lmpc_controller.cpp
  src/track.cpp
  src/safe_set.cpp
  src/qp_builder.cpp
)
target_include_directories(lmpc_core PUBLIC include)
target_compile_features(lmpc_core PUBLIC cxx_std_17)
target_compile_options(lmpc_core PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3>
)
target_link_libraries(lmpc_core PUBLIC ${LMPC_CASADI_TARGET} spdlog::spdlog_header_only)
# SPDLOG_ACTIVE_LEVEL is spdlog's COMPILE-TIME cost knob (include/log.hpp's
# comment has the full rationale): below this level, SPDLOG_LOGGER_TRACE/
# _DEBUG calls are stripped from the binary entirely, not just
# runtime-skipped. Release (this project's default CMAKE_BUILD_TYPE, set
# above if the caller didn't choose one) strips both debug/perf logging
# tiers out completely; any other build type keeps everything compiled in,
# gated at runtime instead via the standard SPDLOG_LEVEL env var.
target_compile_definitions(lmpc_core PUBLIC
  $<IF:$<CONFIG:Release>,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO,SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>
)

pybind11_add_module(lmpc_native MODULE
  src/bindings.cpp
)
target_link_libraries(lmpc_native PRIVATE lmpc_core)
target_include_directories(lmpc_native PRIVATE include)
target_compile_features(lmpc_native PRIVATE cxx_std_17)
target_compile_options(lmpc_native PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3 -fvisibility=hidden>
)
# CasADi's own plugin dlopen (see LMPC_PLUGIN_TARGETS below) resolves bare
# filenames itself rather than deferring to the loading library's rpath, so
# it never finds a plugin sitting right next to lmpc_native.so unless told
# to look there. src/bindings.cpp calls
# casadi::GlobalOptions::setCasadiPath(LMPC_NATIVE_DIR) using this absolute,
# build-time-known path -- this project always places lmpc_native.so
# directly in CMAKE_CURRENT_SOURCE_DIR (see LIBRARY_OUTPUT_DIRECTORY below),
# so the path is stable regardless of the caller's working directory.
target_compile_definitions(lmpc_native PRIVATE
  LMPC_NATIVE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

set_target_properties(lmpc_native PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# CasADi loads conic/NLP solvers as separate plugin .dylibs at runtime
# (dlopen'd by name, e.g. "libcasadi_conic_qrqp.dylib"), NOT linked into
# libcasadi itself -- and add_subdirectory(casadi ... EXCLUDE_FROM_ALL)
# means plugin targets aren't built by a plain `cmake --build .` unless
# something depends on them. DESIGN.md SS7 pins qrqp for correctness-first
# bring-up; casadi_nlpsol_ipopt is the same story for LmpcConfig::
# solver_name = "ipopt" (2026-07-14) -- without it here, `cmake --build .`
# silently skips both the ipopt-external ExternalProject_Add (git clone +
# build IPOPT/MUMPS from source) and casadi's own ipopt interface plugin,
# even with WITH_IPOPT/WITH_BUILD_IPOPT ON above, since nothing else in
# this project's own (non-EXCLUDE_FROM_ALL) build graph depends on it.
set(LMPC_PLUGIN_TARGETS casadi_conic_qrqp casadi_nlpsol_ipopt)
foreach(plugin_target ${LMPC_PLUGIN_TARGETS})
  add_dependencies(lmpc_native ${plugin_target})
endforeach()
if(LMPC_PLUGIN_TARGETS)
  # CasADi's plugin loader looks up the UNVERSIONED name
  # ("libcasadi_conic_qrqp.dylib"), but $<TARGET_FILE:...> resolves to the
  # versioned file (e.g. "...3.7.dylib") -- copy_directory instead, so the
  # unversioned symlink CMake also generates in the build tree comes along
  # with it (same approach as this project's previous native LMPC port).
  add_custom_command(TARGET lmpc_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/casadi/lib"
      "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Copying CasADi plugin libraries next to lmpc_native"
  )
endif()

if(APPLE)
  target_link_options(lmpc_native PRIVATE "-Wl,-dead_strip")
elseif(UNIX AND NOT MSVC)
  target_compile_options(lmpc_native PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(lmpc_native PRIVATE "-Wl,--gc-sections")
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(lmpc_terminal_tests tests/test_terminal_safe_set.cpp)
  target_link_libraries(lmpc_terminal_tests PRIVATE lmpc_core)
  add_dependencies(lmpc_terminal_tests casadi_conic_qrqp)
  add_test(NAME lmpc_terminal_tests COMMAND lmpc_terminal_tests)
  set_tests_properties(lmpc_terminal_tests PROPERTIES
    ENVIRONMENT "CASADIPATH=${CMAKE_CURRENT_BINARY_DIR}/casadi/lib"
  )
endif()
