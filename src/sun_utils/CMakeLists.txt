cmake_minimum_required(VERSION 2.8)
project(CIWTApp)


AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} SUB_SOURCES)
SET( sun_utils ${SUB_SOURCES})

ADD_LIBRARY ( sun_utils  ${sun_utils} )