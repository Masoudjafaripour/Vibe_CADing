cmake_minimum_required(VERSION 3.16)
project(vibecad_geometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O3: the whole point of this module is speed
endif()

find_package(OpenMP)  # optional: parallel SDF sampling

add_library(vibecad_geometry src/marching_cubes.cpp src/mesh_stats.cpp src/stl.cpp
            src/printability.cpp src/stl_read.cpp src/simplify.cpp src/mesh_ops.cpp
            src/json.cpp src/csg.cpp src/families.cpp)
target_include_directories(vibecad_geometry PUBLIC include)
target_compile_options(vibecad_geometry PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vibecad_geometry PUBLIC OpenMP::OpenMP_CXX)
endif()

# Skeletal motion: FK, foot locking (IK), metrics, BVH export
add_library(vibecad_motion_core src/motion.cpp src/bvh.cpp)
target_include_directories(vibecad_motion_core PUBLIC include)
target_compile_options(vibecad_motion_core PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

# Python module (built by `pip install .` via scikit-build-core, or -DVIBECAD_PYTHON=ON)
if(SKBUILD)  # set by scikit-build-core (to a version number, so normalise to ON/OFF)
  set(_vibecad_python_default ON)
else()
  set(_vibecad_python_default OFF)
endif()
option(VIBECAD_PYTHON "Build the vibecad_geom Python module" ${_vibecad_python_default})
if(VIBECAD_PYTHON)
  set_target_properties(vibecad_geometry PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(vibecad_geom src/bindings.cpp)
  target_link_libraries(vibecad_geom PRIVATE vibecad_geometry)
  install(TARGETS vibecad_geom LIBRARY DESTINATION .)

  set_target_properties(vibecad_motion_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  pybind11_add_module(vibecad_motion src/motion_bindings.cpp)
  target_link_libraries(vibecad_motion PRIVATE vibecad_motion_core)
  install(TARGETS vibecad_motion LIBRARY DESTINATION .)
endif()
if(SKBUILD)
  return()  # pip only needs the module
endif()

add_executable(mc_demo src/mc_demo.cpp)
target_link_libraries(mc_demo PRIVATE vibecad_geometry)

enable_testing()
add_executable(test_mc tests/test_mc.cpp)
target_link_libraries(test_mc PRIVATE vibecad_geometry)
add_test(NAME marching_cubes COMMAND test_mc)
add_executable(test_motion tests/test_motion.cpp)
target_link_libraries(test_motion PRIVATE vibecad_motion_core)
add_test(NAME motion COMMAND test_motion)
add_executable(test_print tests/test_print.cpp)
target_link_libraries(test_print PRIVATE vibecad_geometry)
add_test(NAME printability COMMAND test_print)
add_executable(test_mesh_ops tests/test_mesh_ops.cpp)
target_link_libraries(test_mesh_ops PRIVATE vibecad_geometry)
add_test(NAME mesh_ops COMMAND test_mesh_ops)
add_executable(test_csg tests/test_csg.cpp)
target_link_libraries(test_csg PRIVATE vibecad_geometry)
add_test(NAME csg COMMAND test_csg)
add_executable(test_families tests/test_families.cpp)
target_link_libraries(test_families PRIVATE vibecad_geometry)
add_test(NAME families COMMAND test_families)
