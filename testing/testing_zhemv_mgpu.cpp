/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


// --------------------
int main(int argc, char **argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dX(i_)      dX, ((i_))
    #define dY(i_)      dY, ((i_))
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dX(i_)     (dX + (i_))
    #define dY(i_)     (dY + (i_))
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t gflops, cpu_time=0, cpu_perf=0, gpu_time, gpu_perf, mgpu_time, mgpu_perf, dev_time, dev_perf;
    double      Ynorm, error=0, error2=0, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t n_local[MagmaMaxGPUs];

    magma_int_t N, Noffset, lda, ldda, blocks, lhwork, ldwork, matsize, vecsize;
    magma_int_t incx = 1;

    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  1.5, -2.3 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.6,  0.8 );
    magmaDoubleComplex *A, *X, *Y, *Ylapack, *Ydev, *Ymagma, *Ymagma1, *hwork;
    magmaDoubleComplex_ptr dA, dX, dY;
    magmaDoubleComplex_ptr d_lA[MagmaMaxGPUs], dwork[MagmaMaxGPUs];

    magma_device_t dev;
    magma_queue_t queues[MagmaMaxGPUs];
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    magma_int_t nb = 64;  // required by magmablas_zhemv_mgpu implementation

    for( dev=0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    // currently, tests all offsets in the offsets array;
    // comment out loop below to test a specific offset.
    magma_int_t offset = opts.offset;
    magma_int_t offsets[] = { 0, 1, 31, 32, 33, 63, 64, 65, 100, 200 };
    magma_int_t noffsets = sizeof(offsets) / sizeof(*offsets);
    
    printf("%% uplo = %s, ngpu %lld, block size = %lld, offset %lld\n",
            lapack_uplo_const(opts.uplo), (long long) opts.ngpu, (long long) nb, (long long) offset );
    printf( "%%                 BLAS                CUBLAS              MAGMA 1 GPU         MAGMA MGPU       Error rel  Error rel\n"
            "%%   N  offset     Gflop/s (msec)      Gflop/s (msec)      Gflop/s (msec)      Gflop/s (msec)   to CUBLAS  to LAPACK\n"
            "%%==================================================================================================================\n" );
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      
      // comment out these two lines & end of loop to test a specific offset
      for( int ioffset=0; ioffset < noffsets; ioffset += 1 ) {
        offset = offsets[ioffset];
        
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N       = opts.nsize[itest];
            Noffset = N + offset;
            lda     = Noffset;
            ldda    = magma_roundup( Noffset, opts.align );  // multiple of 32 by default
            matsize = Noffset*ldda;
            vecsize = (Noffset-1)*incx + 1;
            gflops  = FLOPS_ZHEMV( N ) / 1e9;
            
            blocks = magma_ceildiv( N + (offset % nb), nb );
            lhwork = N*opts.ngpu;
            ldwork = ldda*(blocks + 1);

            TESTING_CHECK( magma_zmalloc_cpu( &A,       matsize ));
            TESTING_CHECK( magma_zmalloc_cpu( &Y,       vecsize ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ydev,    vecsize ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ymagma,  vecsize ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ymagma1, vecsize ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ylapack, vecsize ));

            TESTING_CHECK( magma_zmalloc_pinned( &X,       vecsize ));
            TESTING_CHECK( magma_zmalloc_pinned( &hwork,   lhwork  ));
            
            magma_setdevice( opts.device );
            TESTING_CHECK( magma_zmalloc( &dA, matsize ));
            TESTING_CHECK( magma_zmalloc( &dX, vecsize ));
            TESTING_CHECK( magma_zmalloc( &dY, vecsize ));
            
            // TODO make magma_zmalloc_bcyclic helper function?
            for( dev=0; dev < opts.ngpu; dev++ ) {
                n_local[dev] = ((Noffset/nb)/opts.ngpu)*nb;
                if (dev < (Noffset/nb) % opts.ngpu)
                    n_local[dev] += nb;
                else if (dev == (Noffset/nb) % opts.ngpu)
                    n_local[dev] += Noffset % nb;
                
                magma_setdevice( dev );
                TESTING_CHECK( magma_zmalloc( &d_lA[dev],  ldda*n_local[dev] ));
                TESTING_CHECK( magma_zmalloc( &dwork[dev], ldwork ));
            }
            
            //////////////////////////////////////////////////////////////////////////
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &matsize, A );
            magma_zmake_hermitian( Noffset, A, lda );
            
            lapackf77_zlarnv( &ione, ISEED, &vecsize, X );
            lapackf77_zlarnv( &ione, ISEED, &vecsize, Y );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_setdevice( opts.device );
            magma_zsetmatrix( Noffset, Noffset, A, lda, dA, ldda, opts.queue );
            magma_zsetvector( Noffset, X, incx, dX, incx, opts.queue );
            magma_zsetvector( Noffset, Y, incx, dY, incx, opts.queue );
            
            dev_time = magma_sync_wtime(0);
            magma_zhemv( opts.uplo, N,
                         alpha, dA(offset, offset), ldda,
                                dX(offset), incx,
                         beta,  dY(offset), incx, opts.queue );
            dev_time = magma_sync_wtime(0) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetvector( Noffset, dY, incx, Ydev, incx, opts.queue );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (1 GPU)
               =================================================================== */
            magma_setdevice( opts.device );
            magma_zsetvector( Noffset, Y, incx, dY, incx, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            
            magmablas_zhemv_work( opts.uplo, N,
                                  alpha, dA(offset, offset), ldda,
                                         dX(offset), incx,
                                  beta,  dY(offset), incx, dwork[ opts.device ], ldwork,
                                  opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            magma_zgetvector( Noffset, dY, incx, Ymagma1, incx, opts.queue );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (multi-GPU)
               =================================================================== */
            magma_zsetmatrix_1D_col_bcyclic( opts.ngpu, Noffset, Noffset, nb, A, lda, d_lA, ldda, queues );
            blasf77_zcopy( &Noffset, Y, &incx, Ymagma, &incx );
            
            // workspaces do NOT need to be zero -- set to NAN to prove
            for( dev=0; dev < opts.ngpu; ++dev ) {
                magma_setdevice( dev );
                magmablas_zlaset( MagmaFull, ldwork, 1, MAGMA_Z_NAN, MAGMA_Z_NAN, dwork[dev], ldwork, opts.queue );
            }
            lapackf77_zlaset( "Full", &lhwork, &ione, &MAGMA_Z_NAN, &MAGMA_Z_NAN, hwork, &lhwork );
            
            mgpu_time = magma_sync_wtime(0);
            
            magma_int_t info;
            info = magmablas_zhemv_mgpu(
                opts.uplo, N,
                alpha,
                d_lA, ldda, offset,
                X + offset, incx,
                beta,
                Ymagma + offset, incx,
                hwork, lhwork,
                dwork, ldwork,
                opts.ngpu, nb, queues );
            if (info != 0) {
                printf("magmablas_zhemv_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            info = magmablas_zhemv_mgpu_sync(
                opts.uplo, N,
                alpha,
                d_lA, ldda, offset,
                X + offset, incx,
                beta,
                Ymagma + offset, incx,
                hwork, lhwork,
                dwork, ldwork,
                opts.ngpu, nb, queues );
            if (info != 0) {
                printf("magmablas_zhemv_sync returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            mgpu_time = magma_sync_wtime(0) - mgpu_time;
            mgpu_perf = gflops / mgpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                blasf77_zcopy( &Noffset, Y, &incx, Ylapack, &incx );
                
                cpu_time = magma_wtime();
                blasf77_zhemv( lapack_uplo_const(opts.uplo), &N,
                               &alpha, A + offset + offset*lda, &lda,
                                       X + offset, &incx,
                               &beta,  Ylapack + offset, &incx );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
    
                /* =====================================================================
                   Compute the Difference LAPACK vs. Magma
                   =================================================================== */
                Ynorm  = lapackf77_zlange( "F", &Noffset, &ione, Ylapack, &Noffset, work );
                blasf77_zaxpy( &Noffset, &c_neg_one, Ymagma, &incx, Ylapack, &incx );
                error2 = lapackf77_zlange( "F", &Noffset, &ione, Ylapack, &Noffset, work ) / Ynorm;
            }
            
            /* =====================================================================
               Compute the Difference cuBLAS vs. Magma
               =================================================================== */
            Ynorm = lapackf77_zlange( "F", &Noffset, &ione, Ydev, &Noffset, work );
            blasf77_zaxpy( &Noffset, &c_neg_one, Ymagma, &incx, Ydev, &incx );
            error = lapackf77_zlange( "F", &Noffset, &ione, Ydev, &Noffset, work ) / Ynorm;
            
            bool okay = (error < tol && error2 < tol);
            status += ! okay;
            if ( opts.lapack ) {
                printf( "%5lld  %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                        (long long) N, (long long) offset,
                        cpu_perf, cpu_time*1000.,
                        dev_perf, dev_time*1000.,
                        gpu_perf, gpu_time*1000.,
                        mgpu_perf, mgpu_time*1000.,
                        error, error2, (okay ? "ok" : "failed") );
            }
            else {
                printf( "%5lld  %5lld     ---   (  ---  )   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e     ---      %s\n",
                        (long long) N, (long long) offset,
                        dev_perf, dev_time*1000.,
                        gpu_perf, gpu_time*1000.,
                        mgpu_perf, mgpu_time*1000.,
                        error, (okay ? "ok" : "failed") );
            }
            
            /* Free Memory */
            magma_free_cpu( A );
            magma_free_cpu( Y );
            magma_free_cpu( Ydev );
            magma_free_cpu( Ymagma  );
            magma_free_cpu( Ymagma1 );
            magma_free_cpu( Ylapack );

            magma_free_pinned( X );
            magma_free_pinned( hwork   );
            
            magma_setdevice( opts.device );
            magma_free( dA );
            magma_free( dX );
            magma_free( dY );
            
            for( dev=0; dev < opts.ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev]  );
                magma_free( dwork[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
        
      // comment out these two lines line & top of loop test a specific offset
      }  // end for ioffset
      printf( "\n" );
    }
    
    for( dev=0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
