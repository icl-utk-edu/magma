#ifndef PAPI_SDE_MAGMA_H
#define PAPI_SDE_MAGMA_H

#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
  #define MAGMA_INTEGER PAPI_SDE_long_long
#else
  #define MAGMA_INTEGER PAPI_SDE_int
#endif

#define MAGMA_REAL_DOUBLE PAPI_SDE_double

#endif

