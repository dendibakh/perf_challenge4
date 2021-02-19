cmake_minimum_required(VERSION 2.8)
project(canny)

set(CMAKE_CXX_FLAGS "-g -O3 -ffast-math -march=core-avx2 ${CMAKE_CXX_FLAGS} -Rpass-missed=loop-vectorize -Wall")
set(CMAKE_C_FLAGS "-g -O3 -ffast-math -march=core-avx2 ${CMAKE_C_FLAGS} -Rpass-missed=loop-vectorize -Wall")

add_executable(canny canny_source.c)
target_link_libraries(canny m)

add_executable(validate validate.cpp)
