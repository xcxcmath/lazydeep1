cmake_minimum_required(VERSION 3.13)
project(lazydeep1)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

include_directories(Eigen)

set(lz lazy/Operand.hpp lazy/Operator.hpp lazy/Variable.hpp lazy/Placeholder.hpp lazy/Constant.hpp lazy/train/Optimizer.hpp lazy/train/AdamOptimizer.hpp)

add_executable(lazydeep1 main.cpp ${lz})