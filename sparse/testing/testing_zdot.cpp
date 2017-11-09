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
   -- testing zdot
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    const magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
    const magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex alpha;

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_z_matrix a={Magma_CSR}, b={Magma_CSR}, x={Magma_CSR}, y={Magma_CSR}, skp={Magma_CSR};

    printf("%%=======================================================================================================================================================================\n");
    printf("\n");
    printf("            |                            runtime                                            |                              GFLOPS\n");
    printf("%% n num_vecs |  CUDOT       CUGEMV       MAGMAGEMV       MDOT       MDGM    MDGM_SHFL      |      CUDOT       CUGEMV      MAGMAGEMV       MDOT       MDGM      MDGM_SHFL\n");
    printf("%%------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n");

    for( magma_int_t num_vecs=1; num_vecs <= 32; num_vecs += 1 ) {
        for( magma_int_t n=500000; n < 500001; n += 10000 ) {
            int iters = 10;
            double computations = (2.* n * iters * num_vecs);

            #ifndef ENABLE_TIMER
            #define ENABLE_TIMER
            #endif
            
            #ifdef ENABLE_TIMER
            real_Double_t mdot1, mdot2, mdgm1, mdgm2, magmagemv1, magmagemv2, cugemv1, cugemv2, cudot1, cudot2;
            real_Double_t mdot_time, mdgm_time, mdgmshf_time, magmagemv_time, cugemv_time, cudot_time;
            #endif

            TESTING_CHECK( magma_zvinit( &a, Magma_DEV, n, num_vecs, one, queue ));
            TESTING_CHECK( magma_zvinit( &b, Magma_DEV, n, 1, one, queue ));
            TESTING_CHECK( magma_zvinit( &x, Magma_DEV, n, 8, one, queue ));
            TESTING_CHECK( magma_zvinit( &y, Magma_DEV, n, 8, one, queue ));
            TESTING_CHECK( magma_zvinit( &skp, Magma_DEV, 1, num_vecs, zero, queue ));

            // warm up
            TESTING_CHECK( magma_zgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));

            // CUDOT
            #ifdef ENABLE_TIMER
            cudot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                for( int l=0; l < num_vecs; l++) {
                    alpha = magma_zdotc( n, a.dval+l*a.num_rows, 1, b.dval, 1, queue );
                }
            }
            #ifdef ENABLE_TIMER
            cudot2 = magma_sync_wtime( queue );
            cudot_time=cudot2-cudot1;
            #endif
            // CUGeMV
            #ifdef ENABLE_TIMER
            cugemv1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                magma_zgemv( MagmaTrans, n, num_vecs, one, a.dval, n, b.dval, 1, zero, skp.dval, 1, queue );
            }
            #ifdef ENABLE_TIMER
            cugemv2 = magma_sync_wtime( queue );
            cugemv_time=cugemv2-cugemv1;
            #endif
            // MAGMAGeMV
            #ifdef ENABLE_TIMER
            magmagemv1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                magmablas_zgemv( MagmaTrans, n, num_vecs, one, a.dval, n, b.dval, 1, zero, skp.dval, 1, queue );
            }
            #ifdef ENABLE_TIMER
            magmagemv2 = magma_sync_wtime( queue );
            magmagemv_time = magmagemv2 - magmagemv1;
            #endif
            // MDOT
            #ifdef ENABLE_TIMER
            mdot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                for( int c = 0; c < num_vecs/2; c++ ) {
                    TESTING_CHECK( magma_zmdotc( n, 2, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));
                }
                for( int c = 0; c < num_vecs % 2; c++ ){
                    TESTING_CHECK( magma_zmdotc( n, 1, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));
                }
            }
            #ifdef ENABLE_TIMER
            mdot2 = magma_sync_wtime( queue );
            mdot_time=mdot2-mdot1;
            #endif
            // MDGM
            #ifdef ENABLE_TIMER
            mdgm1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                TESTING_CHECK( magma_zgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));
                //h++;
            }
            #ifdef ENABLE_TIMER
            mdgm2 = magma_sync_wtime( queue );
            mdgm_time=mdgm2-mdgm1;
            #endif
            // MDGM_shfl
            
            #ifdef ENABLE_TIMER
            mdgm1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h < iters; h++) {
                TESTING_CHECK( magma_zgemvmdot_shfl( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));
            }
            #ifdef ENABLE_TIMER
            mdgm2 = magma_sync_wtime( queue );
            mdgmshf_time=mdgm2-mdgm1;
            #endif
                
                
            //magma_zprint_gpu( num_vecs, 1, skp.dval, num_vecs, opts.queue );

            //Chronometry
            #ifdef ENABLE_TIMER
            printf("%lld  %lld  %e  %e  %e  %e  %e  %e  || %e  %e  %e  %e  %e  %e\n",
                    (long long) n, (long long) num_vecs,
                    cudot_time/iters,
                    (cugemv_time)/iters,
                    (magmagemv_time)/iters,
                    (mdot_time)/iters,
                    (mdgm_time)/iters,
                    (mdgmshf_time)/iters,
                    computations/(cudot_time*1e9),
                    computations/(cugemv_time*1e9),
                    computations/(magmagemv_time*1e9),
                    computations/(mdot_time*1e9),
                    computations/(mdgm_time*1e9),
                    computations/(mdgmshf_time*1e9) );
            #endif

            magma_zmfree(&a, queue );
            magma_zmfree(&b, queue );
            magma_zmfree(&x, queue );
            magma_zmfree(&y, queue );
            magma_zmfree(&skp, queue );
        }

        //printf("%%================================================================================================================================================\n");
        //printf("\n");
        //printf("\n");
    }
    
    // use alpha to silence compiler warnings
    if ( magma_d_isnan( (double) real( alpha ))) {
        info = -1;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
