#message(STATUS "set env var")
#set(ENV{LDFLAGS} "-L/usr/local/opt/llvm/lib")
#set(ENV{CPPFLAGS} "-I/usr/local/opt/llvm/include")

#if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
#    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
#    set(OpenMP_C_LIB_NAMES "omp")
#    set(OpenMP_omp_LIBRARY omp)
#endif()

#if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
#    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
#    set(OpenMP_CXX_LIB_NAMES "omp")
#    set(OpenMP_omp_LIBRARY omp)
#endif()

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    message(STATUS "found openmp")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# OpenMP linking - This is where the OpenMP library is located in my computer.
include_directories(/usr/local/opt/libomp/include)
find_library(OPENMP_LIB libomp.dylib /usr/local/opt/libomp/lib)
set(OPEN_MP_PATH ${OPENMP_LIB} CACHE INTERNAL "")