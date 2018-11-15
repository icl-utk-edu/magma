/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, support
#include <papi.h>
#define NUM_PAPI_SDE_EVTS 8

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    
    // magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_matrix A={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};
    int i=1;

    // PAPI declaration
    int j, event_set, retval;
    long long int values[NUM_PAPI_SDE_EVTS];
    int event_codes[NUM_PAPI_SDE_EVTS];
    const char *sde_event_name[] = {"sde:::MAGMA::numiter_I",
                                    "sde:::MAGMA::InitialResidual_D",
                                    "sde:::MAGMA::FinalResidual_D",
                                    "sde:::MAGMA::IterativeResidual_D",
                                    "sde:::MAGMA::SolverRuntime_D",
                                    "sde:::MAGMA::SpmvCount_I",
                                    "sde:::MAGMA::IterativeResidual_RCRD_D:CNT",
                                    "sde:::MAGMA::IterativeResidual_RCRD_D"}; 
    const char *is_papi_sde_on = getenv("PAPI_SDE_MAGMA");


    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    TESTING_CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    // PAPI initialization stuff
    if ( is_papi_sde_on != NULL ) {
        retval = PAPI_library_init( PAPI_VER_CURRENT );
        if( retval != PAPI_VER_CURRENT ){
            fprintf( stderr, "PAPI_library_init failed: %d. Ensure that the PAPI library used at run time is the same version used during linking\n",PAPI_VER_CURRENT );
        }
        event_set = PAPI_NULL;
        retval = PAPI_create_eventset( &event_set );
        if( retval != PAPI_OK ){
            fprintf( stderr, "PAPI_create_eventset failed\n" );
        } 

        for (j=0; j<NUM_PAPI_SDE_EVTS; j++) {
            retval = PAPI_event_name_to_code( (char *)sde_event_name[j], &event_codes[j] );
            if( retval != PAPI_OK ) {
                fprintf( stderr, "PAPI_event_name_to_code(%s) failed.\n",sde_event_name[j]);
            }
        }

        retval = PAPI_add_events( event_set, event_codes, NUM_PAPI_SDE_EVTS );
        if( retval != PAPI_OK ){
            fprintf( stderr, "PAPI_add_events failed\n" );
        }
    }

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
        }

        if ( is_papi_sde_on != NULL ) {
            retval = PAPI_start( event_set );
            if( retval != PAPI_OK ){
                fprintf( stderr, "PAPI_start failed\n" );
            }
        }

        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        TESTING_CHECK( magma_zeigensolverinfo_init( &zopts.solver_par, queue ));


        // scale matrix
        TESTING_CHECK( magma_zmscale( &A, zopts.scaling, queue ));
        
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }

        TESTING_CHECK( magma_zmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        
        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                            (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        
        printf("matrixinfo = [\n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m   ||   stored nnz\n");
        printf("%%============================================================================%%\n");
        printf("  %8lld  %8lld      %10lld             %4lld        %10lld\n",
               (long long) B.num_rows, (long long) B.num_cols, (long long) B.true_nnz,
               (long long) (B.true_nnz/B.num_rows), (long long) B.nnz );
        printf("%%============================================================================%%\n");
        printf("];\n");

        TESTING_CHECK( magma_zmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        TESTING_CHECK( magma_zvinit_rand( &b, Magma_DEV, A.num_rows, 1, queue ));
        //magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue );
        //magma_z_spmv( one, dB, x, zero, b, queue );                 //  b = A x
        //magma_zmfree(&x, queue );
        TESTING_CHECK( magma_zvinit_rand( &x, Magma_DEV, A.num_cols, 1, queue ));
        
        info = magma_z_solver( dB, b, &x, &zopts, queue );
        if( info != 0 ) {
            printf("%%error: solver returned: %s (%lld).\n",
                    magma_strerror( info ), (long long) info );
        }


        printf("convergence = [\n");
        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
       
        if ( is_papi_sde_on != NULL ) {
            retval = PAPI_stop( event_set, values );
            if( retval != PAPI_OK ){
                fprintf( stderr, "PAPI_stop failed\n" );
            }
            printf(">>>> PAPI counter report:\n");
            printf("    %s: %lld\n",sde_event_name[0], values[0]);
            printf("    %s: %.4e\n",sde_event_name[1], GET_DOUBLE_SDE(values[1]));
            printf("    %s: %.4e\n",sde_event_name[2], GET_DOUBLE_SDE(values[2]));
            printf("    %s: %.4e\n",sde_event_name[3], GET_DOUBLE_SDE(values[3]));
            printf("    %s: %.4e\n",sde_event_name[4], GET_DOUBLE_SDE(values[4]));
            printf("    %s: %lld\n",sde_event_name[5], values[5]);
            printf("    %s: %lld\n",sde_event_name[6], values[6]);

            double *sde_ptr = GET_SDE_RECORDER_ADDRESS(values[7], double);
            for (j=0; j<values[6]; j++){
                printf("    %d: %.4e\n",j, sde_ptr[j]);
            }
            printf("<<<< PAPI counter report END\n");
        }
 
        zopts.solver_par.verbose = 0;
        printf("solverinfo = [\n");
        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
        
        printf("precondinfo = [\n");
        printf("%%   setup  runtime\n");        
        printf("  %.6f  %.6f\n",
           zopts.precond_par.setuptime, zopts.precond_par.runtime );
        printf("];\n\n");
        magma_zmfree(&dB, queue );
        magma_zmfree(&B, queue );
        magma_zmfree(&A, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
