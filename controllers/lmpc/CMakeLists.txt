cmake_minimum_required(VERSION 3.17)
project(lmpc_controller LANGUAGES CXX)

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
set(LMPC_RACING_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/Racing-LMPC-ROS2/src"
  CACHE PATH "Path to Racing-LMPC-ROS2/src"
)
set(LMPC_CASADI_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/casadi"
  CACHE PATH "Path to the CasADi source checkout"
)
set(LMPC_EIGEN_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/eigen"
  CACHE PATH "Path to the Eigen source checkout"
)
set(LMPC_CGAL_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/cgal"
  CACHE PATH "Path to the CGAL source checkout"
)
set(LMPC_BOOST_SOURCE_DIR
  "${LMPC_THIRDPARTY_DIR}/boost"
  CACHE PATH "Path to the Boost source checkout"
)
file(GLOB LMPC_CGAL_INCLUDE_DIRS CONFIGURE_DEPENDS
  "${LMPC_CGAL_SOURCE_DIR}/*/include"
)
file(GLOB LMPC_BOOST_INCLUDE_DIRS CONFIGURE_DEPENDS
  "${LMPC_BOOST_SOURCE_DIR}/libs/*/include"
)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(casadi CONFIG QUIET
  HINTS
    "${LMPC_CASADI_SOURCE_DIR}/build"
    "${LMPC_CASADI_SOURCE_DIR}/build/casadi"
    "${LMPC_CASADI_SOURCE_DIR}/cmake"
)
if(NOT TARGET casadi AND NOT TARGET casadi::casadi)
  set(WITH_IPOPT OFF CACHE BOOL "Do not require IPOPT for the first Gym MPC build" FORCE)
  set(WITH_BUILD_REQUIRED OFF CACHE BOOL "Do not build Fortran-based CasADi dependencies" FORCE)
  set(WITH_BUILD_IPOPT OFF CACHE BOOL "Do not build IPOPT from source" FORCE)
  set(WITH_QPOASES OFF CACHE BOOL "Do not require LAPACK-backed qpOASES for Gym MPC" FORCE)
  set(WITH_PYTHON OFF CACHE BOOL "Do not build CasADi Python frontend here" FORCE)
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

add_library(lmpc_state_core STATIC
  src/state.cpp
  src/native_controller.cpp
  "${LMPC_RACING_SOURCE_DIR}/tools/lmpc_utils/src/lookup.cpp"
  "${LMPC_RACING_SOURCE_DIR}/tools/lmpc_utils/src/utils.cpp"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/base_vehicle_model/src/base_vehicle_model.cpp"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/kinematic_bicycle_model/src/kinematic_bicycle_model.cpp"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/racing_trajectory/src/safe_set.cpp"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/racing_trajectory/src/trajectory_kd_tree.cpp"
)
target_include_directories(lmpc_state_core PUBLIC
  include
  "${LMPC_EIGEN_SOURCE_DIR}"
  "${LMPC_CGAL_SOURCE_DIR}"
  "${LMPC_BOOST_SOURCE_DIR}"
  ${LMPC_CGAL_INCLUDE_DIRS}
  ${LMPC_BOOST_INCLUDE_DIRS}
  "${LMPC_RACING_SOURCE_DIR}/tools/lmpc_utils/include"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/base_vehicle_model/include"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/kinematic_bicycle_model/include"
  "${LMPC_RACING_SOURCE_DIR}/vehicle_dynamics_models/racing_trajectory/include"
)
target_compile_features(lmpc_state_core PUBLIC cxx_std_17)
target_compile_options(lmpc_state_core PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3>
)
target_link_libraries(lmpc_state_core PUBLIC ${LMPC_CASADI_TARGET})

pybind11_add_module(lmpc_native MODULE
  src/bindings.cpp
)
target_link_libraries(lmpc_native PRIVATE lmpc_state_core)
target_include_directories(lmpc_native PRIVATE include)
target_compile_features(lmpc_native PRIVATE cxx_std_17)
target_compile_options(lmpc_native PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3 -fvisibility=hidden>
)

set_target_properties(lmpc_native PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set(LMPC_CASADI_PLUGIN_TARGETS)
foreach(plugin_target IN ITEMS casadi_nlpsol_sqpmethod casadi_conic_qrqp)
  if(TARGET ${plugin_target})
    list(APPEND LMPC_CASADI_PLUGIN_TARGETS ${plugin_target})
  endif()
endforeach()

if(LMPC_CASADI_PLUGIN_TARGETS)
  add_dependencies(lmpc_native ${LMPC_CASADI_PLUGIN_TARGETS})
  add_custom_command(TARGET lmpc_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/casadi/lib"
      $<TARGET_FILE_DIR:lmpc_native>
  )
endif()

if(APPLE)
  target_link_options(lmpc_native PRIVATE "-Wl,-dead_strip")
elseif(UNIX AND NOT MSVC)
  target_compile_options(lmpc_native PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(lmpc_native PRIVATE "-Wl,--gc-sections")
endif()
