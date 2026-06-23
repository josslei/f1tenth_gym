cmake_minimum_required(VERSION 3.17)
project(f110_rollout_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(F110_ROLLOUT_THIRDPARTY_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../../thirdparty"
  CACHE PATH "Path to the repository thirdparty directory"
)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(
  "${F110_ROLLOUT_THIRDPARTY_DIR}/pybind11"
  "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  EXCLUDE_FROM_ALL
)

pybind11_add_module(
  _f110_rollout_kernel
  src/bindings.cpp
  src/dynamics.cpp
  src/step.cpp
)

target_include_directories(
  _f110_rollout_kernel
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${F110_ROLLOUT_THIRDPARTY_DIR}/highway
)

target_compile_features(_f110_rollout_kernel PRIVATE cxx_std_17)
target_compile_options(_f110_rollout_kernel PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3 -fvisibility=hidden>
)
if(APPLE)
  target_compile_options(_f110_rollout_kernel PRIVATE -mcpu=native)
else()
  target_compile_options(_f110_rollout_kernel PRIVATE -march=native)
endif()

set_target_properties(_f110_rollout_kernel PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/natives
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/natives
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/natives
)
if(APPLE)
  target_link_options(_f110_rollout_kernel PRIVATE "-Wl,-dead_strip")
elseif(UNIX AND NOT MSVC)
  target_compile_options(_f110_rollout_kernel PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(_f110_rollout_kernel PRIVATE "-Wl,--gc-sections")
endif()
