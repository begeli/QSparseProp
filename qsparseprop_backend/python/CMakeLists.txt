SET(
        PYBIND11_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_backend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_conv2d_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_linear_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_conv2d_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_conv2d_over_on_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_conv2d_over_on_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_linear_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_ditheredq_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_luq_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_sawbq_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_stdq_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils_wrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_conv2d_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_linear_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_conv2d_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_conv2d_over_on_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_conv2d_over_on_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparseprop_linear_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_ditheredq_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_sawbq_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qsparseprop_stdq_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils_wrapper.cpp
)

#IF(APPLE)
#    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
#    set(CMAKE_HAVE_THREADS_LIBRARY 1)
#    set(CMAKE_USE_WIN32_THREADS_INIT 0)
#    set(CMAKE_USE_PTHREADS_INIT 1)
#    set(THREADS_PREFER_PTHREAD_FLAG ON)
#ENDIF()

#find_package(Torch 1.9 REQUIRED)
#find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
#message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(qsparseprop_backend ${PYBIND11_SOURCES})
target_link_libraries(
        qsparseprop_backend
        PRIVATE qsparse_core
#        PRIVATE ${TORCH_LIBRARIES}
#        PRIVATE ${TORCH_PYTHON_LIBRARY}
)
#set_property(TARGET qsparseprop_backend PROPERTY CXX_STANDARD 17)
