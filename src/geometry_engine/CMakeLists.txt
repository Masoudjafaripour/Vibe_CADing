cmake_minimum_required(VERSION 3.16)
project(vibecad_geometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O3: the whole point of this module is speed
endif()

find_package(OpenMP)  # optional: parallel SDF sampling

add_library(vibecad_geometry src/marching_cubes.cpp src/mesh_stats.cpp src/stl.cpp)
target_include_directories(vibecad_geometry PUBLIC include)
target_compile_options(vibecad_geometry PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vibecad_geometry PUBLIC OpenMP::OpenMP_CXX)
endif()

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
