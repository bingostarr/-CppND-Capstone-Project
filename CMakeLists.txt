cmake_minimum_required(VERSION 3.11.3)

# Set the C++ standard we will use
set(CMAKE_CXX_STANDARD 17)

# Add the path of the cmake files to the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

project(capstone_first)

# Set library output path to /lib
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")

# Add project executable
add_executable(capstone_first src/main.cpp src/matrix.cpp src/matrix3d.cpp src/dataset.cpp src/layer.cpp src/cnn.cpp)

target_link_libraries(capstone_first PUBLIC pthread)
target_include_directories(capstone_first PRIVATE inc)