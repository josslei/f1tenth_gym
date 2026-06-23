cmake_minimum_required(VERSION 3.17)
project(tree_search LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TREE_SEARCH_THIRDPARTY_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../thirdparty"
  CACHE PATH "Path to the repository thirdparty directory"
)
set(TREE_SEARCH_LIBTORCH_DIR
  "${TREE_SEARCH_THIRDPARTY_DIR}/libtorch"
  CACHE PATH "Path to the local libtorch distribution root"
)

option(TREE_SEARCH_NATIVE_OPTIMIZATIONS "Enable CPU-native tuning" ON)

add_library(tree_search_headers INTERFACE)
target_include_directories(
  tree_search_headers
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TREE_SEARCH_THIRDPARTY_DIR}/highway
)
target_compile_features(tree_search_headers INTERFACE cxx_std_17)

add_subdirectory(
  "${TREE_SEARCH_THIRDPARTY_DIR}/pybind11"
  "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
  EXCLUDE_FROM_ALL
)

find_package(Torch REQUIRED CONFIG PATHS "${TREE_SEARCH_LIBTORCH_DIR}" NO_DEFAULT_PATH)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

pybind11_add_module(tree_search_native MODULE src/bindings.cpp)
target_link_libraries(
  tree_search_native
  PRIVATE
    tree_search_headers
    ${TORCH_LIBRARIES}
)
target_include_directories(
  tree_search_native
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TREE_SEARCH_THIRDPARTY_DIR}/highway
    ${TORCH_INCLUDE_DIRS}
)
target_compile_features(tree_search_native PRIVATE cxx_std_17)
target_compile_options(tree_search_native PRIVATE ${TORCH_CXX_FLAGS_LIST})
if(TREE_SEARCH_NATIVE_OPTIMIZATIONS)
  if(APPLE)
    target_compile_options(tree_search_native PRIVATE -mcpu=native)
  else()
    target_compile_options(tree_search_native PRIVATE -march=native)
  endif()
endif()
set_target_properties(tree_search_native PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(APPLE)
  target_link_options(tree_search_native PRIVATE "-Wl,-dead_strip")
elseif(UNIX AND NOT MSVC)
  target_compile_options(tree_search_native PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(tree_search_native PRIVATE "-Wl,--gc-sections")
endif()
set_target_properties(tree_search_native PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
