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
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    TESTING_CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
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
