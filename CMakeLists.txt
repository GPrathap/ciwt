cmake_minimum_required(VERSION 2.8)
###########################################################################
# Add include path' for cmake pera_copter_libraries
###########################################################################
message( STATUS "Include cmake modules" ${CMAKE_SOURCE_DIR} )
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-config/libraries.cmake")
project(tracy_node)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport message_generation sensor_msgs)
generate_messages(DEPENDENCIES sensor_msgs)
catkin_package(CATKIN_DEPENDS cv_bridge image_transport message_runtime sensor_msgs)

#set(CMAKE_BUILD_TYPE Debug)
#add_definitions("-O3 -w -Wall -c -fmessage-length=0 -MMD -MP -std=c++11")

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Boost COMPONENTS program_options regex atomic date_time system chrono filesystem thread graph REQUIRED)
include_directories(${BOOST_INCLUDE_DIRS})
message("boost lib: ${Boost_LIBRARIES}")
message("boost inc:${Boost_INCLUDE_DIR}")

SET(VERSION_REQ_OCV "3.2.0")
FIND_PACKAGE( OpenCV ${VERSION_REQ_OCV} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


include_directories(${catkin_INCLUDE_DIRS})

#set(VTK_DIR "/root/software/VTK-7.1.0/build")
#find_package(VTK REQUIRED)
#include(${VTK_USE_FILE})

set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${Boost_LIBRARIES})
set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${PCL_LIBRARIES})
#set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${VTK_LIBRARIES})
set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${OpenCV_LIBRARIES})
set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${catkin_LIBRARIES})

message(  "Include directories for ${PROJECT_NAME}:" ${${PROJECT_NAME}_Headers})
message(  "Include directories for ${PROJECT_NAME}:"${PCL_LIBRARIES})


foreach( DIR  connected_components libelas libviso2 scene_segmentation tracking )
    message( STATUS "  - include/" ${DIR} )
    include_directories( "include/${DIR}" )
    link_directories("include/${DIR}")
endforeach()

message( STATUS "Include sub-directories into build process:" )
foreach( DIR CIWT connected_components external libelas libviso2 scene_segmentation sun_utils tracking )
    message( STATUS "  - include/" ${DIR} )
    include_directories( "src/${DIR}" )
    link_directories("src/${DIR}")
    add_subdirectory( src/${DIR} )
endforeach()

add_subdirectory(src)

set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} CIWT libelas libviso2  scene_segmentation  connected_components tracking sun_utils external)
message( STATUS "Linked libries: " ${${PROJECT_NAME}_LIB} )

set(${PROJECT_NAME}_LIB ${${PROJECT_NAME}_LIB} ${${PROJECT_NAME}_libraries})

add_executable(${PROJECT_NAME} Tracy.cpp Tracy.h)
target_link_libraries (${PROJECT_NAME} -Wl,--start-group  ${${PROJECT_NAME}_LIB} -Wl,--end-group)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

#add_executable(try0 try0)
#target_link_libraries (try0 -Wl,--start-group  ${${PROJECT_NAME}_LIB} -Wl,--end-group)


#add_executable(Tracy Tracy.cpp)
#target_link_libraries (Tracy -Wl,--start-group  ${${PROJECT_NAME}_LIB} -Wl,--end-group)



install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# Native or Linux build

#target_link_libraries(ciwt ${OpenCV_LIBS} ${PCL_LIBRARIES} $ ${Boost_LIBRARIES} elas viso2  scene_segmentation  connected_components tracking)
#add_executable(${PROJECT_NAME} ${SUN_UTILS_SRC_FILES} ${CIWT_SRC_FILES} ${EXTERNAL_SRC_FILES} "CIWT.cpp")
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} $ ${Boost_LIBRARIES} )
#install(TARGETS ${PROJECT_NAME} DESTINATION /root/ci)
#
#install(TARGETS ${PROJECT_NAME} DESTINATION /root/ciwt/bin)
#install(TARGETS try0 DESTINATION /root/ciwt/bin)
