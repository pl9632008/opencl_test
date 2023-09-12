cmake_minimum_required(VERSION 3.0)
project(testopencl)

find_package(OpenCV REQUIRED)
# find_package(OpenCLHeaders REQUIRED )


aux_source_directory(./src src)

include_directories(${OpenCV_INCLUDE_DIRS}  )

add_executable(test ${src})

target_link_libraries(test  ${OpenCV_LIBS} )

#target_link_libraries(test  ${OpenCV_LIBS} OpenCL::Headers)
