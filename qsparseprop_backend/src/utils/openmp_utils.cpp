#include "openmp_utils.h"

int get_OpenMP_threads()
{
    static int nthr = -1;
    if ( nthr == -1 ) {
        #if defined(_OPENMP)
            #pragma omp parallel default(none) shared(nthr)
            {
                nthr = omp_get_num_threads();
            }
        #else
            nthr = 1;
        #endif
    }
    return nthr;
}

int get_OpenMP_thread() {
    static int thread = -1;

#if defined(_OPENMP)
    thread = omp_get_thread_num();
#else
    thread = 0;
#endif

    return thread;
}
