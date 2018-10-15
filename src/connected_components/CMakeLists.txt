cmake_minimum_required(VERSION 2.8)
project(CIWTApp)


AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} SUB_SOURCES)
SET( connected_components ${SUB_SOURCES})

ADD_LIBRARY ( connected_components  ${connected_components} )