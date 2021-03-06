cmake_minimum_required(VERSION 3.0.0)

project(parker VERSION 0.1.0)

SET(CMAKE_SHARED_LINKER_FLAGS ${Enclave_Link_Flags}) 
# Find source files
file(GLOB SOURCES src/*.cpp)

# Include header files
include_directories(include)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

# Install library headers
file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})