cmake_minimum_required(VERSION 3.17)
project(f110_self_play LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(F110_SELF_PLAY_THIRDPARTY_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../thirdparty"
  CACHE PATH "Path to the repository thirdparty directory"
)
set(F110_SELF_PLAY_TREE_SEARCH_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../tree_search"
  CACHE PATH "Path to the tree_search package"
)
set(F110_SELF_PLAY_ROLLOUT_KERNEL_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../gym/f110_gym/rollout_kernel"
  CACHE PATH "Path to the rollout kernel package"
)

add_library(f110_self_play_headers INTERFACE)
target_include_directories(
  f110_self_play_headers
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${F110_SELF_PLAY_THIRDPARTY_DIR}/highway
    ${F110_SELF_PLAY_TREE_SEARCH_DIR}/include
    ${F110_SELF_PLAY_ROLLOUT_KERNEL_DIR}/include
)
target_compile_features(f110_self_play_headers INTERFACE cxx_std_17)

add_subdirectory(
  "${F110_SELF_PLAY_THIRDPARTY_DIR}/pybind11"
  "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  EXCLUDE_FROM_ALL
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Torch REQUIRED CONFIG PATHS "${F110_SELF_PLAY_THIRDPARTY_DIR}/libtorch" NO_DEFAULT_PATH)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

if(UNIX AND NOT APPLE)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
    OUTPUT_VARIABLE PYTHON_TORCH_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PYTHON_TORCH_LIBRARY_ERROR
    RESULT_VARIABLE PYTHON_TORCH_LIBRARY_RESULT
  )
  if(NOT PYTHON_TORCH_LIBRARY_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the Python torch package: ${PYTHON_TORCH_LIBRARY_ERROR}")
  endif()
  find_library(
    TORCH_PYTHON_LIBRARY
    torch_python
    PATHS "${PYTHON_TORCH_LIBRARY_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Could not find libtorch_python in ${PYTHON_TORCH_LIBRARY_DIR}")
  endif()
endif()

pybind11_add_module(
  f110_self_play_native
  MODULE
    src/bindings.cpp
    ${F110_SELF_PLAY_ROLLOUT_KERNEL_DIR}/src/dynamics.cpp
    ${F110_SELF_PLAY_ROLLOUT_KERNEL_DIR}/src/step.cpp
)
target_link_libraries(
  f110_self_play_native
  PRIVATE
    f110_self_play_headers
    ${TORCH_LIBRARIES}
)
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(f110_self_play_native PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
target_include_directories(
  f110_self_play_native
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${F110_SELF_PLAY_THIRDPARTY_DIR}/highway
    ${F110_SELF_PLAY_TREE_SEARCH_DIR}/include
    ${F110_SELF_PLAY_ROLLOUT_KERNEL_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
target_compile_features(f110_self_play_native PRIVATE cxx_std_17)
target_compile_options(f110_self_play_native PRIVATE ${TORCH_CXX_FLAGS_LIST})
if(APPLE)
  target_compile_options(f110_self_play_native PRIVATE -mcpu=native)
else()
  target_compile_options(f110_self_play_native PRIVATE -march=native)
endif()
set_target_properties(f110_self_play_native PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(APPLE)
  target_link_options(f110_self_play_native PRIVATE "-Wl,-dead_strip")
elseif(UNIX AND NOT MSVC)
  target_compile_options(f110_self_play_native PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(f110_self_play_native PRIVATE "-Wl,--gc-sections")
endif()
