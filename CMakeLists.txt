# cmake_minimum_required(VERSION 3.20)

# find_package(OpenCV CONFIG REQUIRED)
# project(Test)

# add_executable(Test test.cpp)
# target_link_libraries(Test PRIVATE opencv_ml opencv_dnn opencv_core opencv_flann opencv_calib3d opencv_features2d opencv_highgui opencv_imgcodecs opencv_imgproc opencv_objdetect opencv_photo opencv_stitching opencv_video opencv_videoio)

# cmake_minimum_required( VERSION 3.0 )
# # set( CMAKE_TOOLCHAIN_FILE "C:\\Users\\athyn\\Documents\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" )
# # include(C:\\Users\\athyn\\Documents\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)

# project(Test)
# set(OpenCV_DIR C:\\opencv\\build)
# set(NumCpp_DIR C:\\Users\\athyn\\Documents\\vcpkg\\packages\\numcpp_x86-windows\\share\\NumCpp)


# find_package( OpenCV REQUIRED )
# set(OpenCV_FOUND 1)
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# add_executable(Test test.cpp)

# target_link_libraries(Test PRIVATE NumCpp::NumCpp ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.2)
set( CMAKE_TOOLCHAIN_FILE C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake )
project( Test )
set(OpenCV_DIR C:\\opencv\\build)
set(NumCpp_DIR C:\\vcpkg\\packages\\numcpp_x64-windows\\share\\NumCpp)

find_package( OpenCV REQUIRED )
find_package(NumCpp CONFIG REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Test CameraPlacementRayTraced.cpp )
target_link_libraries(Test PRIVATE NumCpp::NumCpp ${OpenCV_LIBS})