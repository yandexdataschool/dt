cmake_minimum_required(VERSION 2.8)
project(decision_train C CXX)

add_library(dtrain dtrain.cpp)
add_library(dt_pid dt_pid.cpp)
target_link_libraries(dt_pid dtrain)

add_subdirectory(./googletest)
enable_testing()
include_directories(googletest/include/)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -Wall -Werror -O2")
add_executable(run_tests unit_tests.cpp)
target_link_libraries(run_tests gtest gtest_main dtrain)

add_executable(run_model run_model.cpp)
target_link_libraries(run_model dt_pid)
