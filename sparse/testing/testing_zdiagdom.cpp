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
    
    magma_z_matrix A={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};
    double min_dd, max_dd, avg_dd;
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

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

        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                            (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        
        printf("matrixinfo = [\n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m   ||   stored nnz\n");
        printf("%%============================================================================%%\n");
        printf("  %8lld  %8lld      %10lld             %4lld        %10lld\n",
               (long long) A.num_rows, (long long) A.num_cols, (long long) A.true_nnz,
               (long long) (A.true_nnz/A.num_rows), (long long) A.nnz );
        printf("%%============================================================================%%\n");
        printf("];\n");
        
        TESTING_CHECK( magma_zmdiagdom( zopts.precond_par.L, &min_dd, &max_dd, &avg_dd, queue ) );
        printf("diagonaldominance = [\n");
        printf("%%   min     ||   max   ||   avg\n");
        printf("%%============================================================================%%\n");
        printf("  %4.4f      %4.4f      %4.4f\n",
                  min_dd, max_dd, avg_dd );
        printf("%%============================================================================%%\n");
        printf("];\n");

        magma_zmfree(&A, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
