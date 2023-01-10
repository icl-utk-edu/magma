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
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_matrix A={Magma_CSR}, dB={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};

    int i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );

        magma_int_t n = A.num_rows;
        TESTING_CHECK( magma_zmtransfer( A, &dB, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        TESTING_CHECK( magma_zvinit( &b, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue ));
        
        TESTING_CHECK( magma_zprint_vector( b, 90, 10, queue ));
        
        TESTING_CHECK( magma_zprint_matrix( A, queue ));
        printf("\n\n\n");
        TESTING_CHECK( magma_zprint_matrix( dB, queue ));
        
        double res;
        res = magma_dznrm2( n, b.dval, 1, queue );
        printf("norm0: %f\n", res);
        
        TESTING_CHECK( magma_z_spmv( one, dB, x, zero, b, queue ));         //  b = A x

        TESTING_CHECK( magma_zprint_vector( b, 0, 100, queue ));
        TESTING_CHECK( magma_zprint_vector( b, b.num_rows-10, 10, queue ));

        res = magma_dznrm2( n, b.dval, 1, queue );
        printf("norm: %f\n", res);

        
        TESTING_CHECK( magma_zresidual( dB, x, b, &res, queue ));
        printf("res: %f\n", res);


        magma_zmfree(&dB, queue );

        magma_zmfree(&A, queue );
        
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );

        i++;
    }

    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
