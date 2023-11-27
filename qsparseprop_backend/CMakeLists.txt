cmake_minimum_required(VERSION 3.24)
project(QSparseProp)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Include various tools, flags, libraries etc.
if (CMAKE_C_COMPILER_ID MATCHES "Clang\$" AND TESTING)
    include(address-sanitizer)
    include(undefined-sanitizer)
endif()
include(clang-tidy)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Intel\$"))
    include(openmp)

    find_package(GTest)

    if (NOT GTEST_FOUND AND TESTING)
        message(STATUS "Adding GTest as external project")
        include(googletest)
    endif ()
endif()

if (STOCHASTIC_ROUNDING_DISABLED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQSPARSEPROP_STOCHASTIC_ROUNDING_DISABLED=\"1\"")
endif(STOCHASTIC_ROUNDING_DISABLED)

if (TESTING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQSPARSEPROP_TESTING=\"1\"")
endif(TESTING)

add_custom_target(lint)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(python)

#add_executable(QSparseProp main.cpp)
#if (NOT (CMAKE_C_COMPILER_ID MATCHES "Intel\$"))
#    target_link_libraries(QSparseProp LINK_PUBLIC ${OPEN_MP_PATH})
#endif()