#ifndef PAPI_SDE_MAGMA_H
#define PAPI_SDE_MAGMA_H

#include "papi_sde_interface.h"

#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
  #define MAGMA_INTEGER PAPI_SDE_long_long
#else
  #define MAGMA_INTEGER PAPI_SDE_int
#endif

#define MAGMA_REAL_DOUBLE PAPI_SDE_double


/* Struct for SDE handle and SDE counters */
typedef struct sde_t
{
    papi_handle_t handle;      // PAPI SDE handle
    /* Add new SDE counters HERE */
} sde_t;


/* Struct for SDE names */
typedef struct sde_name_t
{
    const char *numiter = "numiter_I";
    const char *InitialResidual = "InitialResidual_D";
    const char *FinalResidual = "FinalResidual_D";
    const char *IterativeResidual = "IterativeResidual_D";
    const char *SolverRuntime = "SolverRuntime_D";
    const char *IterativeResidual_RCRD = "IterativeResidual_RCRD_D";
} sde_name_t;


/* Struct for SDE descriptions */
typedef struct sde_desc_t
{
    const char *numiter = "Number of iterations until convergence attained (I=integer)";
    const char *InitialResidual = "Initial residual (D=double)";
    const char *FinalResidual = "Final residual (D=double)";
    const char *IterativeResidual = "Iterative residual (D=double)";
    const char *SolverRuntime = "Total run-time of the solver (D=double)";
    const char *IterativeResidual_RCRD = "Array of all residuals until convergence (RCRD=recorder) (D=double)";
} sde_desc_t;


/* Struct for SDE Recorders */
typedef struct sde_rcrd_t
{
    const char *magma_env_on;      // Check if PAPI_SDE_MAGMA env is set
    void* handle_iter_res;         // Handle for Iterative Residual recorder
   /* Add handle for new SDE recorders HERE */
} sde_rcrd_t;


#endif

