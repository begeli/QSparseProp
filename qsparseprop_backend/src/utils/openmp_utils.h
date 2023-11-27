#ifndef QSPARSEPROP_OPENMP_UTILS_H
#define QSPARSEPROP_OPENMP_UTILS_H

#if defined(_OPENMP)
#include <omp.h>
#endif

int get_OpenMP_threads();

int get_OpenMP_thread();

#endif //QSPARSEPROP_OPENMP_UTILS_H
