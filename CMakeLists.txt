cmake_minimum_required (VERSION 3.8)

set(OpenCV_DIR C:/opencv/build/x64/vc15)
set(OpenCV_INCLUDE_DIRS C:/opencv/build/include)

set(OpenCV_LIBD C:/opencv/build/x64/vc15/lib/opencv_world342d.lib)
set(OpenCV_LIB C:/opencv/build/x64/vc15/lib/opencv_world342.lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


project ("ComputerVision")
add_subdirectory ("ComputerVision")

file(COPY background.png DESTINATION ${PROJECT_BINARY_DIR})
file(COPY train.png DESTINATION ${PROJECT_BINARY_DIR})
file(COPY barrier.png DESTINATION ${PROJECT_BINARY_DIR})
file(COPY levelcrossing.mpg DESTINATION ${PROJECT_BINARY_DIR})


