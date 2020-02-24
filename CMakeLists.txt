cmake_minimum_required(VERSION 3.11.3)

# Set the C++ standard we will use
set(CMAKE_CXX_STANDARD 17)

# Add the path of the cmake files to the CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

project(capstone_first)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Set library output path to /lib
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")

# Add project executable
add_executable(capstone_first src/main.cpp src/matrix.cpp src/matrix3d.cpp src/dataset.cpp src/layer.cpp src/cnn.cpp)

target_link_libraries(capstone_first PUBLIC pthread ${OpenCV_LIBRARIES})
target_include_directories(capstone_first PRIVATE inc)

###################################
# Locate project prerequisites
#find_package(io2d REQUIRED)
#find_package(Cairo)
#find_package(GraphicsMagick)

# Set IO2D flags
#set(IO2D_WITHOUT_SAMPLES 1)
#set(IO2D_WITHOUT_TESTS 1)

# Add the pugixml and GoogleTest library subdirectories
#add_subdirectory(thirdparty/pugixml)
#add_subdirectory(thirdparty/googletest)

#target_link_libraries(capstone_first PRIVATE io2d::io2d PUBLIC pugixml)

# Add the testing executable
#add_executable(capstone_first_test test/utest_rp_a_star_search.cpp src/route_planner.cpp src/model.cpp src/route_model.cpp)
#target_link_libraries(capstone_first_test gtest_main pugixml)