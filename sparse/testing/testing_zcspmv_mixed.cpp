/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
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
   -- testing the mixed precision SpMV
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
    magma_c_matrix cA={Magma_CSR}, dcB={Magma_CSR};
    magma_z_matrix diag={Magma_CSR}, ddiag={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};
    real_Double_t start, end;

    int i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
            TESTING_CHECK( magma_cm_5stencil(  laplace_size, &cA, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
            TESTING_CHECK( magma_c_csr_mtx( &cA,  argv[i], queue ));
        }

        printf( "\n# matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        magma_int_t n = A.num_rows;
        real_Double_t FLOPS = 2.0*A.nnz/1e9;

        
        // reference:
        printf("reference run:\n");
        
        // vectors and matrix copy
        TESTING_CHECK( magma_zvinit( &b, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue ));
        TESTING_CHECK( magma_zmtransfer( A, &dB, Magma_CPU, Magma_DEV, queue ));
        TESTING_CHECK( magma_zprint_matrix( dB, queue ));
        
        
        // warmup
        for (int z=0; z<10; z++) {
            TESTING_CHECK( magma_z_spmv( one, dB, x, zero, b, queue ));         //  b = A x
        }
        
        start = magma_wtime();
        for (int z=0; z<10; z++) {
            TESTING_CHECK( magma_z_spmv( one, dB, x, zero, b, queue ));         //  b = A x
        }
        end = magma_wtime();
        TESTING_CHECK( magma_zprint_vector( b, 0, 10, queue ));
        TESTING_CHECK( magma_zprint_vector( b, b.num_rows-10, 10, queue ));
        printf( "\n > cuSPARSE CSR SpMV : %.2e seconds %.2e GFLOP/s.\n\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        
        magma_zmfree(&dB, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );  

        
        // now the mixed precision SpMV
        printf("\n\nmixed precsion SpMV run:\n");
        
        TESTING_CHECK( magma_zvinit( &b, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue ));
        
        magma_zvinit(&diag, Magma_CPU,  n, 1, zero, queue );
        // now artifically extract the diagonal values into a seperate vector
        // and set the values in A to zero
        for (magma_int_t k=0; k<cA.num_rows; k++) {
            for (magma_int_t j=cA.row[k]; j<cA.row[k+1]; j++) {
                if (cA.col[j] == k ){
                    diag.val[k] = MAGMA_Z_MAKE( 
                          (double) MAGMA_C_REAL(cA.val[ j ]),
                          (double) MAGMA_C_IMAG(cA.val[ j ])  );
                    cA.val[j] = MAGMA_C_ZERO;
                }
            }
        }

        TESTING_CHECK( magma_cmtransfer( cA, &dcB, Magma_CPU, Magma_DEV, queue ));
        TESTING_CHECK( magma_zmtransfer( diag, &ddiag, Magma_CPU, Magma_DEV, queue ));
        TESTING_CHECK( magma_cprint_matrix( dcB, queue ));
        
        // mixed precision SpMV kernel
        start = magma_wtime();
        for (int z=0; z<10; z++) {
            magma_zcgecsrmv_mixed_prec( 
                MagmaNoTrans, dcB.num_rows, dcB.num_cols, MAGMA_Z_ONE, 
                ddiag.dval, dcB.dval, dcB.drow, dcB.dcol, 
                x.dval, MAGMA_Z_ZERO, b.dval, queue);
        }
        end = magma_wtime();
        TESTING_CHECK( magma_zprint_vector( b, 0, 10, queue ));
        TESTING_CHECK( magma_zprint_vector( b, b.num_rows-10, 10, queue ));
        printf( "\n > MAGMA mixed precision SpMV : %.2e seconds %.2e GFLOP/s.\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        
        magma_cmfree(&dcB, queue );

        magma_cmfree(&cA, queue );
        
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );

        i++;
    }

    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
