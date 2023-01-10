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
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t j, n=1000000, FLOPS;
    magma_int_t count = 100;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );
    magmaDoubleComplex two = MAGMA_Z_MAKE( 2.0, 0.0 );

    magma_z_matrix a={Magma_CSR}, ad={Magma_CSR}, bd={Magma_CSR}, cd={Magma_CSR};
    TESTING_CHECK( magma_zvinit( &a, Magma_CPU, n, 1, one, queue ));
    TESTING_CHECK( magma_zvinit( &bd, Magma_DEV, n, 1, two, queue ));
    TESTING_CHECK( magma_zvinit( &cd, Magma_DEV, n, 1, one, queue ));
    
    TESTING_CHECK( magma_zmtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ));

    real_Double_t start, end, res;
    
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = magma_dznrm2( n, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA nrm2: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_zscal( n, two, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA scal: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_zaxpy( n, one, ad.dval, 1, bd.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA axpy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_zcopy( n, bd.dval, 1, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA copy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = MAGMA_Z_REAL( magma_zdotc( n, ad.dval, 1, bd.dval, 1, queue ));
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA dotc: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );

    printf("%% tester BLAS:  ok\n");
    
    // use res to silence compiler warnings
    if ( magma_d_isnan( (double) real( res ))) {
        info = -1;
    }

    magma_zmfree( &a, queue);
    magma_zmfree(&ad, queue);
    magma_zmfree(&bd, queue);
    magma_zmfree(&cd, queue);
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
