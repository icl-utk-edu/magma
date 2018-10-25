#ifndef PAPI_SDE_HOOK_H
#define PAPI_SDE_HOOK_H

#include "magma_types.h"
#include "magmasparse_types.h"
#include "papi_sde_magma.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Register all SDE counters and recorders */
int magma_z_papi_sde_hook(magma_z_solver_par *solver_par);
int magma_d_papi_sde_hook(magma_d_solver_par *solver_par);
int magma_s_papi_sde_hook(magma_s_solver_par *solver_par);
int magma_c_papi_sde_hook(magma_c_solver_par *solver_par);


#ifdef __cplusplus
}
#endif

#endif
