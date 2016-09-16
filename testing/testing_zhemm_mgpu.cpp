/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Mark Gates
       @author Azzam Haidar
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#include "../control/trace.h"  // internal header

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zhemm_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha     = MAGMA_Z_MAKE( 3.456, 5.678 );
    magmaDoubleComplex beta      = MAGMA_Z_MAKE( 1.234, 2.456 );
    
    real_Double_t    gflops, gpu_perf=0., cpu_perf=0., gpu_time=0., cpu_time=0.;
    real_Double_t    gpu_perf2=0., gpu_time2=0.;
    double           Anorm, error, work[1];
    magmaDoubleComplex *hA, *hB, *hC, *hR;
    magmaDoubleComplex_ptr dA[MagmaMaxGPUs], dB[MagmaMaxGPUs], dC[MagmaMaxGPUs], dwork[MagmaMaxGPUs];
    magmaDoubleComplex_ptr dA2;
    magma_int_t i, j, dev, M, N, size, lda, ldb, ldc, ldda, lddb, lddc, msize, nb;
    magma_int_t ione     = 1;
    magma_int_t iseed[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    // default values
    nb = (opts.nb > 0 ? opts.nb : 64);
    
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2];
    magma_int_t ncmplx = 0;
    magma_buildconnection_mgpu( gnode, &ncmplx, opts.ngpu );
    
    printf("%% Initializing communication pattern... GPU-ncmplx %lld\n", (long long) ncmplx);
    for (i=0; i < ncmplx; ++i) {
        magma_int_t myngpu = gnode[i][MagmaMaxGPUs];
        printf("%% cmplx %lld has %lld GPUs:", (long long) i, (long long) myngpu );
        for (j=0; j < myngpu; ++j) {
            printf(" %lld", (long long) gnode[i][j] );
            if (j < myngpu-1) {
                printf(",");
            }
        }
        printf("\n");
    }

    // number of queues per GPU. Requires ngpu.
    magma_int_t nqueue  = opts.ngpu;
    // number of events per GPU. Require ngpu*ngpu.
    magma_int_t nevents = opts.ngpu*opts.ngpu;
    magma_queue_t queues[MagmaMaxGPUs][20], queues0[MagmaMaxGPUs];
    magma_event_t events[MagmaMaxGPUs][MagmaMaxGPUs*MagmaMaxGPUs + 10];
    for( dev = 0; dev < opts.ngpu; ++dev ) {
        magma_setdevice( dev );
        for( i = 0; i < nqueue; ++i ) {
            magma_queue_create( dev, &queues[dev][i] );
        }
        queues0[dev] = queues[dev][0];
        for( i = 0; i < nevents; ++i ) {
            magma_event_create( &events[dev][i] );
        }
    }

    printf("%% nb %lld, ngpu %lld, version %lld\n", (long long) nb, (long long) opts.ngpu, (long long) opts.version );
    printf("%%   M     N    nb offset  CPU Gflop/s (sec)   GPU Gflop/s (sec)   CUBLAS hemm (sec)   ||R|| / ||A||*||B||\n");
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      M = opts.msize[itest];
      N = opts.nsize[itest];
      for( int offset = 0; offset < N; offset += min(N,nb) ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            msize = M - offset;
            lda   = M;  // TODO depends on side
            ldb   = M;
            ldc   = M;
            ldda  = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb  = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc  = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZHEMM( MagmaLeft, (double)msize, (double)N ) / 1e9;
            
            magma_int_t dworksiz = lddc*N + (M*N)*opts.ngpu;
            
            TESTING_CHECK( magma_zmalloc_cpu( &hA, lda*M ));
            TESTING_CHECK( magma_zmalloc_cpu( &hB, ldb*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &hC, ldc*N ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &hR, ldc*N ));

            for( dev = 0; dev < opts.ngpu; ++dev ) {
                magma_int_t mlocal = ((M / nb) / opts.ngpu + 1) * nb;
                magma_setdevice( dev );
                TESTING_CHECK( magma_zmalloc( &dA[dev],    ldda*mlocal ));
                TESTING_CHECK( magma_zmalloc( &dB[dev],    lddb*N      ));
                TESTING_CHECK( magma_zmalloc( &dC[dev],    lddc*N      ));
                TESTING_CHECK( magma_zmalloc( &dwork[dev], dworksiz    ));
            }
            
            if ( opts.check ) {
                magma_setdevice( 0 );
                TESTING_CHECK( magma_zmalloc( &dA2, ldda*M ));
            }

            size = lda*M;
            lapackf77_zlarnv( &ione, iseed, &size, hA );
            magma_zmake_hermitian( M, hA, lda );
            
            size = ldb*N;
            lapackf77_zlarnv( &ione, iseed, &size, hB );
            size = ldc*N;
            lapackf77_zlarnv( &ione, iseed, &size, hC );
            lapackf77_zlacpy( "Full", &M, &N, hC, &ldc, hR, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix_1D_col_bcyclic( opts.ngpu, M, M, nb, hA, lda, dA, ldda, queues0 );
            for( dev = 0; dev < opts.ngpu; ++dev ) {
                magma_setdevice( dev );
                magma_zsetmatrix( M, N, hB, lda, dB[dev], ldda, opts.queue );
                // since when offset != 0, the GPU that does beta*C may not be 0,
                // send initial hC to all GPUs.
                magma_zsetmatrix( M, N, hC, lda, dC[dev], ldda, opts.queue );
            }
            
            trace_init( 1, opts.ngpu, nqueue, (magma_queue_t*) queues );
            
            gpu_time = magma_sync_wtime(0);
            
            magmablas_zhemm_mgpu(
                MagmaLeft, MagmaLower, msize, N,
                alpha, dA, ldda, offset,
                       dB, ldda,
                beta,  dC, ldda, dwork, dworksiz,
                opts.ngpu, nb, queues, nqueue, events, nevents, gnode, ncmplx);
            
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            #ifdef TRACING
            char buf[80];
            snprintf( buf, sizeof(buf), "zhemm-m%lld-n%lld-nb%lld-ngpu%lld-run%lld.svg",
                      (long long) M, (long long) N, (long long) nb, (long long) opts.ngpu, (long long) iter );
            trace_finalize( buf, "trace.css" );
            #endif
            
            /* ====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            if ( opts.check && iter == 0 ) {
                magma_setdevice( 0 );
                magma_zsetmatrix( M, M, hA, lda, dA2, ldda, opts.queue );
                magma_zsetmatrix( M, N, hB, lda, dB[0], ldda, opts.queue );
                magma_zsetmatrix( M, N, hC, lda, dwork[0], ldda, opts.queue );
                
                gpu_time2 = magma_sync_wtime(0);
                magma_zhemm(
                    MagmaLeft, MagmaLower, msize, N,
                    alpha, dA2 + offset + offset*ldda, ldda,
                           dB[0],    ldda,
                    beta,  dwork[0], ldda, opts.queue );
                gpu_time2 = magma_sync_wtime(0) - gpu_time2;
                gpu_perf2 = gflops / gpu_time2;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.check ) {
                // store ||A||*||B||
                Anorm  = lapackf77_zlange("fro", &msize, &msize, hA + offset + offset*lda, &lda, work );
                Anorm *= lapackf77_zlange("fro", &msize, &N, hB, &lda, work );
                
                //printf( "A =" ); magma_zprint( M, M, hA, lda );
                //printf( "B =" ); magma_zprint( M, N, hB, lda );
                //printf( "C =" ); magma_zprint( M, N, hC, lda );
                
                cpu_time = magma_wtime();
                blasf77_zhemm( "Left", "Lower", &msize, &N,
                                &alpha, hA + offset + offset*lda, &lda,
                                        hB, &lda,
                                &beta,  hC, &lda );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                
                for (dev=0; dev < opts.ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_zgetmatrix( M, N, dC[dev], ldda, hR, lda, opts.queue );
                    
                    // compute relative error ||R||/||A||*||B||, where R := C_magma - C_lapack = R - C
                    size = ldc*N;
                    blasf77_zaxpy( &size, &c_neg_one, hC, &ione, hR, &ione );
                    error = lapackf77_zlange("fro", &msize, &N, hR, &lda, work) / Anorm;
                    
                    //printf( "R ="  ); magma_zprint( M, N, hR, lda );
                    bool okay = (error < tol);
                    status += ! okay;
                    if (dev == 0) {
                        printf( "%5lld %5lld %5lld %5lld   %7.1f (%7.4f)   %7.1f (%7.4f)   %7.1f (%7.4f)   %8.2e   %s\n",
                                (long long) M, (long long) N, (long long) nb, (long long) offset,
                                cpu_perf, cpu_time,
                                gpu_perf, gpu_time,
                                gpu_perf2, gpu_time2,
                                error, (okay ? "ok" : "failed") );
                    }
                    else {
                        printf( "    dev %lld %74s  %8.2e   %s\n",
                                (long long) dev, "",
                                error, (okay ? "ok" : "failed") );
                    }
                }
            } else {
                printf( "%5lld %5lld %5lld %5lld     ---   (  ---  )   %7.1f (%7.4f)     ---   (  ---  )   ---\n",
                        (long long) M, (long long) N, (long long) nb, (long long) offset,
                        gpu_perf, gpu_time );
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            
            magma_free_pinned( hR );
            
            for( dev = 0; dev < opts.ngpu; ++dev ) {
                magma_setdevice( dev );
                magma_free( dA[dev]    );
                magma_free( dB[dev]    );
                magma_free( dC[dev]    );
                magma_free( dwork[dev] );
            }
            
            if ( opts.check ) {
                magma_setdevice( 0 );
                magma_free( dA2 );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }  // offset
      printf( "\n" );
    }

    for( dev = 0; dev < opts.ngpu; ++dev ) {
        magma_setdevice( dev );
        for( i = 0; i < nqueue; ++i ) {
            magma_queue_destroy( queues[dev][i] );
        }
        for( i = 0; i < nevents; ++i ) {
            magma_event_destroy( events[dev][i] );
        }
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
