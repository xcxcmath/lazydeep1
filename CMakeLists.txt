cmake_minimum_required(VERSION 3.13)
project(lazydeep1)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -ffast-math -fopenmp -march=native")

include_directories(Eigen)

set(lazy_operand lazy/Operand.hpp
        lazy/Variable.hpp
        lazy/Placeholder.hpp
        lazy/Constant.hpp)

set(lazy_ops lazy/ops/Operator.hpp
        lazy/ops/Math.hpp
        lazy/ops/NN.hpp)

set(lazy_train lazy/train/Optimizer.hpp
        lazy/train/AdamOptimizer.hpp
        lazy/train/MomentumOptimizer.hpp)

set(lazy ${lazy_operand} ${lazy_ops} ${lazy_train})

add_executable(lazydeep1 main.cpp ${lazy})