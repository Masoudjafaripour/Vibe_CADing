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

add_executable(mc_demo src/mc_demo.cpp)
target_link_libraries(mc_demo PRIVATE vibecad_geometry)

enable_testing()
add_executable(test_mc tests/test_mc.cpp)
target_link_libraries(test_mc PRIVATE vibecad_geometry)
add_test(NAME marching_cubes COMMAND test_mc)
