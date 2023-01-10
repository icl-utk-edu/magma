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
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    real_Double_t res;
    double resd;
    magma_z_matrix Z={Magma_CSR}, A={Magma_CSR}, AT={Magma_CSR}, 
    A2={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    
    magma_index_t *comm_i=NULL;
    magmaDoubleComplex *comm_v=NULL;
    magma_int_t start, end;
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

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
        
        // slice matrix
        TESTING_CHECK( magma_index_malloc_cpu( &comm_i, Z.num_rows ) );
        TESTING_CHECK( magma_zmalloc_cpu( &comm_v, Z.num_rows ) );
        
        TESTING_CHECK( magma_zmslice( 1, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_zprint_matrix( A2, queue );
        magma_zprint_matrix( AT, queue );
        magma_zprint_matrix( B, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&AT, queue );
        magma_zmfree(&B, queue );

        TESTING_CHECK( magma_zmslice( 9, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_zprint_matrix( A2, queue );
        magma_zprint_matrix( AT, queue );
        magma_zprint_matrix( B, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&AT, queue );
        magma_zmfree(&B, queue );
        
        TESTING_CHECK( magma_zmslice( 9, 1, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_zprint_matrix( A2, queue );
        magma_zprint_matrix( AT, queue );
        magma_zprint_matrix( B, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&AT, queue );
        magma_zmfree(&B, queue );

        TESTING_CHECK( magma_zmslice( 9, 8, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_zprint_matrix( A2, queue );
        magma_zprint_matrix( AT, queue );
        magma_zprint_matrix( B, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&AT, queue );
        magma_zmfree(&B, queue );
        
        
        // scale matrix
        TESTING_CHECK( magma_zmscale( &Z, zopts.scaling, queue ));

        // remove nonzeros in matrix
        TESTING_CHECK( magma_zmcsrcompressor( &Z, queue ));
        
        // convert to be non-symmetric
        TESTING_CHECK( magma_zmconvert( Z, &A, Magma_CSR, Magma_CSRL, queue ));
        
        // transpose
        TESTING_CHECK( magma_zmtranspose( A, &AT, queue ));

        // convert, copy back and forth to check everything works

        TESTING_CHECK( magma_zmconvert( AT, &B, Magma_CSR, zopts.output_format, queue ));
        magma_zmfree(&AT, queue );
        TESTING_CHECK( magma_zmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));
        magma_zmfree(&B, queue );
        TESTING_CHECK( magma_zmcsrcompressor_gpu( &dB, queue ));
        TESTING_CHECK( magma_zmtransfer( dB, &B, Magma_DEV, Magma_CPU, queue ));
        magma_zmfree(&dB, queue );
        TESTING_CHECK( magma_zmconvert( B, &AT, zopts.output_format,Magma_CSR, queue ));
        magma_zmfree(&B, queue );
        
        TESTING_CHECK( magma_zmfrobenius( A, A, A, &resd, queue));
        printf("%% Frobenius on pattern: ||A-B||_S = %8.2e\n", resd);

        // transpose back
        TESTING_CHECK( magma_zmtranspose( AT, &A2, queue ));
        magma_zmfree(&AT, queue );
        TESTING_CHECK( magma_zmdiff( A, A2, &res, queue));
        printf("%% ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester:  ok\n");
        else
            printf("%% tester:  failed\n");
        
        magma_free_cpu( comm_i );
        magma_free_cpu( comm_v );
        comm_i=NULL;
        comm_v=NULL;
        magma_zmfree(&A, queue );
        magma_zmfree(&A2, queue );
        magma_zmfree(&Z, queue );

        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
