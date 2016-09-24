cmake_minimum_required(VERSION 2.6)
project(pardiso_example)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-parameter -Wno-deprecated-register -Wno-return-type-c-linkage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-unused-parameter -Wno-deprecated-register -Wno-return-type-c-linkage")

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.11" CACHE STRING "Deployment target for OSX" FORCE)
endif(APPLE)

set(LIBIGL_DIR /Users/olkido/Dropbox/Work/code/other/libigl/)
set(LIBIGL_INCLUDE_DIR ${LIBIGL_DIR}/include)
include_directories(${LIBIGL_INCLUDE_DIR})

set(EIGEN_INCLUDE_DIRS  ${LIBIGL_DIR}/external/nanogui/ext/eigen)
include_directories(${EIGEN_INCLUDE_DIRS})

set(PARDISO_LIB /Users/olkido/Dropbox/Work/code/other/pardiso/libpardiso500-MACOS-X86-64.dylib)
list(APPEND SHARED_LIBRARIES ${PARDISO_LIB})

# This is necessary to work around the static linking order dependencies on linux
if(UNIX AND NOT APPLE)
  set(SHARED_LIBRARIES ${SHARED_LIBRARIES} ${SHARED_LIBRARIES})
endif(UNIX AND NOT APPLE)


file(GLOB SOURCES
"${PROJECT_SOURCE_DIR}/*.h"
"${PROJECT_SOURCE_DIR}/*.hpp"
"${PROJECT_SOURCE_DIR}/*.cpp"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${SHARED_SOURCES})
target_link_libraries(${PROJECT_NAME} ${SHARED_LIBRARIES})
