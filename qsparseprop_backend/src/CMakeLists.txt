set(QSPARSEPROP_SOURCES main.cpp)

# Add subdirectories with code to use here
# TODO: Add subdirectories for sparse and conf directories
add_subdirectory(conf)
add_subdirectory(quantization)
add_subdirectory(sparse)
add_subdirectory(utils)

add_library(qsparse_core STATIC ${QSPARSEPROP_SOURCES})
target_include_directories(qsparse_core PUBLIC ${CMAKE_SOURCE_DIR})

# Intel c/cpp compiler comes with its OpenMP library.
if (NOT (CMAKE_C_COMPILER_ID MATCHES "Intel\$"))
    target_link_libraries(qsparse_core LINK_PRIVATE ${OPEN_MP_PATH})
endif()

add_clang_tidy_target(lint_qsparse_core ${QSPARSEPROP_SOURCES})
add_dependencies(lint lint_qsparse_core)

add_executable(qsparse main.cpp)
target_link_libraries(qsparse PUBLIC qsparse_core)