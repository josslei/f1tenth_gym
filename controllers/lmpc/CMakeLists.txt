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
  # Revisit these WITH_* flags if a later solver choice (OSQP/qpOASES/
  # HPIPM) needs one.
  set(WITH_IPOPT OFF CACHE BOOL "Do not require IPOPT for the initial LMPC build" FORCE)
  set(WITH_BUILD_REQUIRED OFF CACHE BOOL "Do not build Fortran-based CasADi dependencies" FORCE)
  set(WITH_BUILD_IPOPT OFF CACHE BOOL "Do not build IPOPT from source" FORCE)
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
target_link_libraries(lmpc_core PUBLIC ${LMPC_CASADI_TARGET})

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
# bring-up, so it's the one plugin required here; add more to this list
# (and PLUGIN_TARGETS below) if SS7's later solver choices are enabled.
set(LMPC_PLUGIN_TARGETS casadi_conic_qrqp)
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
