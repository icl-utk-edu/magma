/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt
       @suthor Stephen Wood
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"

// includes, papi
#ifdef HAVE_PAPI
#include "papi.h"
#define NUM_EVENTS 1         /* # of PAPI events */
#define PAPI                 /* PAPI test variable */
//#define NAME_TO_CODE
#endif  // HAVE_PAPI

/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    
#ifdef PAPI
    int retval, i;
    int EventSet = PAPI_NULL;
    long long values[NUM_EVENTS];
#ifdef NAME_TO_CODE
    char *EventName[] = { "PAPI_FP_OPS" };
    int events[NUM_EVENTS];
#else    
    int events[NUM_EVENTS] = { PAPI_FP_OPS };
    char event_name[PAPI_MAX_STR_LEN];
#endif
#endif    

    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_matrix A={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};
    
    int inp=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &inp, queue ));
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    TESTING_CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( inp < argc ) {
        if ( strcmp("LAPLACE2D", argv[inp]) == 0 && inp+1 < argc ) {   // Laplace test
            inp++;
            magma_int_t laplace_size = atoi( argv[inp] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[inp], queue ));
        }

        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        TESTING_CHECK( magma_zeigensolverinfo_init( &zopts.solver_par, queue ));

        // scale matrix
        TESTING_CHECK( magma_zmscale( &A, zopts.scaling, queue ));

        /**************************** START PAPI **********************************/
    
#ifdef PAPI
    /* PAPI Initialization */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT )
        fprintf( stderr, "PAPI_library_init failed\n" );
    
    printf( "PAPI_VERSION     : %4d %6d %7d\n",
            PAPI_VERSION_MAJOR( PAPI_VERSION ),
            PAPI_VERSION_MINOR( PAPI_VERSION ),
            PAPI_VERSION_REVISION( PAPI_VERSION ) );
    
    
#ifdef NAME_TO_CODE
    /* convert PAPI native events to PAPI code */
    for( i = 0; i < NUM_EVENTS; i++ )
    {
        retval = PAPI_event_name_to_code( EventName[i], &events[i] );
        if( retval != PAPI_OK )
            fprintf( stderr, "PAPI_event_name_to_code failed\n" );
        else
            printf( "Name %s --- Code: %x\n", EventName[i], events[i] );
    }
#endif
    
    retval = PAPI_create_eventset( &EventSet );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_create_eventset failed\n" );
    
    retval = PAPI_add_events( EventSet, events, NUM_EVENTS );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_add_events failed\n" );
    
    retval = PAPI_start( EventSet );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_start failed\n" );
#endif
    
    /**************************** END PAPI **********************************/ 
    
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }

    /**************************** START PAPI **********************************/
    
#ifdef PAPI
    retval = PAPI_stop( EventSet, values );
    
    printf("\n%% PAPI preconditioner info:");
    printf("\nPAPIprecondinfo = [\n");
    printf("%%   values        event_name\n");
    printf("%%============================================================================%%\n");
        
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_stop failed\n" );
    
    for( i = 0; i < NUM_EVENTS; i++ )
    {
#ifdef NAME_TO_CODE
        printf( "%% %s \n", EventName[i] );
        printf( "  %12lld \n", values[i] );
#else
        retval = PAPI_event_code_to_name( events[i], event_name );
        if( retval != PAPI_OK )
            fprintf( stderr, "PAPI_event_code_to_name failed\n" );
        
        printf( "%% %s \n", event_name );
        printf( "  %12lld \n", values[i] );
#endif
    printf("%%============================================================================%%\n");
    printf("];\n");
    }
#endif
    
    /**************************** END PAPI **********************************/       
        
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
        TESTING_CHECK( magma_zvinit( &b, Magma_DEV, A.num_rows, 1, one, queue ));
        //magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue );
        //magma_z_spmv( one, dB, x, zero, b, queue );                 //  b = A x
        //magma_zmfree(&x, queue );
        TESTING_CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));

    /**************************** START PAPI **********************************/
    
#ifdef PAPI
    
    retval = PAPI_start( EventSet );
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_start failed\n" );
#endif
    
    /**************************** END PAPI **********************************/        
        
        info = magma_z_solver( dB, b, &x, &zopts, queue );
        
    /**************************** START PAPI **********************************/
    
#ifdef PAPI
    retval = PAPI_stop( EventSet, values );
    
    printf("\n%% PAPI solver info:");
    printf("\nPAPIsolverinfo = [\n");
    printf("%%   values        event_name\n");
    printf("%%============================================================================%%\n");
        
    if( retval != PAPI_OK )
        fprintf( stderr, "PAPI_stop failed\n" );
    
    for( i = 0; i < NUM_EVENTS; i++ )
    {
#ifdef NAME_TO_CODE
        printf( "%% %s \n", EventName[i] );
        printf( "  %12lld \n", values[i] );
#else
        retval = PAPI_event_code_to_name( events[i], event_name );
        if( retval != PAPI_OK )
            fprintf( stderr, "PAPI_event_code_to_name failed\n" );
        
        printf( "%% %s \n", event_name );
        printf( "  %12lld \n", values[i] );
#endif
    printf("%%============================================================================%%\n");
    printf("];\n");
    }
#endif
    
    /**************************** END PAPI **********************************/        
        
        if( info != 0 ) {
            printf("%%error: solver returned: %s (%lld).\n",
                    magma_strerror( info ), (long long) info );
        }
        printf("convergence = [\n");
        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
        
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
        inp++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
