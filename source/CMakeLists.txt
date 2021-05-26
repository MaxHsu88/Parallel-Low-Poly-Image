# CMakeList.txt : CMake project for LowPoly, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 2.8)
set(EXECUTABLE_OUTPUT_PATH ../)

set( CMAKE_PREFIX_PATH "/home/ipc21/ipc21s05/opencv/" )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Add source to this project's executable.
file(GLOB src
    "*.h"
    "*.cpp"
    "*.cu"
)

cuda_add_executable( ParaLowPoly ${src})
TARGET_LINK_LIBRARIES( ParaLowPoly opencv_core opencv_imgcodecs opencv_imgproc)