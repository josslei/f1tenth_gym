cmake_minimum_required(VERSION 3.17)
project(lmpc_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(LMPC_DEPS_DIR "${CMAKE_CURRENT_LIST_DIR}/deps" CACHE PATH
  "Install prefix for Eigen, OSQP, and OsqpEigen")
list(PREPEND CMAKE_PREFIX_PATH "${LMPC_DEPS_DIR}")

find_package(Eigen3 3.4 CONFIG REQUIRED)
find_package(osqp CONFIG REQUIRED)
find_package(OsqpEigen CONFIG REQUIRED)

set(LMPC_CASADI_SOURCE_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/casadi"
  CACHE PATH "CasADi source directory"
)
find_package(casadi CONFIG QUIET HINTS
  "${LMPC_CASADI_SOURCE_DIR}/build"
  "${LMPC_CASADI_SOURCE_DIR}/build/casadi"
)
if(NOT TARGET casadi AND NOT TARGET casadi::casadi)
  set(WITH_PYTHON OFF CACHE BOOL "" FORCE)
  set(WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WITH_DOC OFF CACHE BOOL "" FORCE)
  add_subdirectory(
    "${LMPC_CASADI_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/casadi"
    EXCLUDE_FROM_ALL
  )
endif()
if(TARGET casadi)
  set(LMPC_CASADI_TARGET casadi)
else()
  set(LMPC_CASADI_TARGET casadi::casadi)
endif()

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/pybind11"
  "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  EXCLUDE_FROM_ALL
)

# The reference headers remain byte-identical and therefore retain their
# LearningMPC/... includes. Keep the source tree flat while providing that
# original include layout only inside the generated build tree.
set(LMPC_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/reference_include")
file(MAKE_DIRECTORY "${LMPC_COMPAT_INCLUDE_DIR}/LearningMPC")
foreach(header track.h occupancy_grid.h spline.h CSVReader.h car_params.h)
  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/include/${header}"
    "${LMPC_COMPAT_INCLUDE_DIR}/LearningMPC/${header}"
    COPYONLY
  )
endforeach()

pybind11_add_module(lmpc_native MODULE
  src/bindings.cpp
  src/lmpc_controller.cpp
)
target_include_directories(lmpc_native PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
  "${CMAKE_CURRENT_LIST_DIR}/include/ros_shim"
  "${LMPC_COMPAT_INCLUDE_DIR}"
)
target_link_libraries(lmpc_native PRIVATE
  OsqpEigen::OsqpEigen
  osqp::osqp
  Eigen3::Eigen
  ${LMPC_CASADI_TARGET}
)
target_compile_options(lmpc_native PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3>
)
set_target_properties(lmpc_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  BUILD_RPATH "${LMPC_DEPS_DIR}/lib"
)
