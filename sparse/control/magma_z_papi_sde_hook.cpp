/*
       @precisions normal z -> s d c
*/

#include "papi_sde_hook.h"

int magma_z_papi_sde_hook(magma_z_solver_par *solver_par)
{
    // Check in PAPI SDE for MAGMA is on   
    solver_par->sde_rcrd.magma_env_on = getenv("PAPI_SDE_MAGMA");

    if ( solver_par->sde_rcrd.magma_env_on != NULL ) {
        sde_t sde;
        sde_name_t sde_name;
        sde_desc_t sde_desc;

        // PAPI SDE handler
         sde.handle = papi_sde_init("MAGMA");

        // PAPI SDE Counters
        papi_sde_register_counter( sde.handle, sde_name.numiter,
                                   PAPI_SDE_RO|PAPI_SDE_INSTANT, MAGMA_INTEGER,
                                   &(solver_par->numiter) );
        papi_sde_register_counter( sde.handle, sde_name.InitialResidual,
                                   PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double,
                                   &(solver_par->init_res) );
        papi_sde_register_counter( sde.handle, sde_name.FinalResidual,
                                   PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double,
                                   &(solver_par->final_res) );
        papi_sde_register_counter( sde.handle, sde_name.IterativeResidual,
                                   PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double,
                                   &(solver_par->iter_res) );
        papi_sde_register_counter( sde.handle, sde_name.SolverRuntime,
                                   PAPI_SDE_RO|PAPI_SDE_INSTANT, MAGMA_REAL_DOUBLE,
                                   &(solver_par->runtime) );
        papi_sde_create_recorder( sde.handle, sde_name.IterativeResidual_RCRD,
                                  sizeof(double), papi_sde_compare_double,
                                  &solver_par->sde_rcrd.handle_iter_res );


        // PAPI SDE Counters' Description 
        papi_sde_describe_counter( sde.handle, sde_name.numiter,
                                   sde_desc.numiter );
        papi_sde_describe_counter( sde.handle, sde_name.InitialResidual,
                                   sde_desc.InitialResidual );
        papi_sde_describe_counter( sde.handle, sde_name.FinalResidual,
                                   sde_desc.FinalResidual );
        papi_sde_describe_counter( sde.handle, sde_name.IterativeResidual,
                                   sde_desc.IterativeResidual );
        papi_sde_describe_counter( sde.handle, sde_name.SolverRuntime,
                                   sde_desc.SolverRuntime );
        papi_sde_describe_counter( sde.handle, sde_name.IterativeResidual_RCRD,
                                   sde_desc.IterativeResidual_RCRD );
    }

    return 0;
}

