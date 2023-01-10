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
   -- testing Jaccard weights
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    real_Double_t res;
    double resd;
    magma_z_matrix Z={Magma_CSR}, dZ={Magma_CSR}, 
    A={Magma_CSR}, A2={Magma_CSR}, dA={Magma_CSR};
    
    magma_index_t *comm_i=NULL;
    magmaDoubleComplex *comm_v=NULL;
    real_Double_t start, end;
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
                (long long) Z.num_rows, (long long) Z.num_cols, (long long) Z.nnz );
        
        // convert to COO
        TESTING_CHECK( magma_zmconvert( Z, &A, Magma_CSR, Magma_CSRCOO, queue ) );
        // transfer to GPU
        TESTING_CHECK( magma_zmtransfer( Z, &dZ, Magma_CPU, Magma_DEV, queue ));
        TESTING_CHECK( magma_zmtransfer( A, &dA, Magma_CPU, Magma_DEV, queue ));
        
        start = magma_sync_wtime( queue );
        for(int i=0; i<10; i++)
            TESTING_CHECK( magma_zjaccard_weights( dZ, &dA, queue ));
        end = magma_sync_wtime( queue );
        printf( " > %.2e seconds.\n", (end-start)/10 );
        
        
        // transfer back
        TESTING_CHECK( magma_zmtransfer( dA, &A2, Magma_DEV, Magma_CPU, queue ));
        
        magma_zprint_matrix(A2, queue );
        
        
        magma_zmfree(&A, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&Z, queue );
        magma_zmfree(&dA, queue );
        magma_zmfree(&dZ, queue );

        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
