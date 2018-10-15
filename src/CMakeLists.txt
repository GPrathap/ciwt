cmake_minimum_required(VERSION 2.8)

message( STATUS "Include  library directories" )
include_directories ( ${src} )

message( STATUS "Include  libraries into build process:" )
foreach( DIR ${src} )
    message( STATUS "  - " ${DIR} )
    add_subdirectory( ${DIR} )
endforeach()