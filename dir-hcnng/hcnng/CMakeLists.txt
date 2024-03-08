cmake_minimum_required(VERSION 3.25)
project(hcnng)

set(CMAKE_CXX_STANDARD 14)

add_executable(hcnng
        common.h
        hcnng.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(hcnng PRIVATE OpenMP::OpenMP_CXX)