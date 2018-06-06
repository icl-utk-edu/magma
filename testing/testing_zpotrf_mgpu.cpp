/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double           Anorm, error, work[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t N, n2, lda, ldda, max_size, ngpu;
    magma_int_t info, nb;
    magma_int_t ione     = 1;
    int status = 0;
    
    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    magma_queue_t queues[ MagmaMaxAccelerators ] = { NULL };
    for( int dev=0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    printf("%% ngpu = %lld, uplo = %s\n", (long long) opts.ngpu, lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R||_F / ||A||_F\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_zpotrf_nb( N );
            gflops = FLOPS_ZPOTRF( N ) / 1e9;
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %lld GPUs\n", (long long) ngpu );
            }
            
            // Allocate host memory for the matrix
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));
            
            // Allocate device memory
            // matrix is distributed by block-rows or block-columns
            // this is maximum size that any GPU stores;
            // size is rounded up to full blocks in both rows and columns
            max_size = (1+N/(nb*ngpu))*nb * magma_roundup( N, nb );
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                TESTING_CHECK( magma_zmalloc( &d_lA[dev], max_size ));
            }
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zpotrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if ( opts.uplo == MagmaUpper ) {
                ldda = magma_roundup( N, nb );
                magma_zsetmatrix_1D_col_bcyclic( ngpu, N, N, nb, h_R, lda, d_lA, ldda, queues );
            }
            else {
                ldda = (1+N/(nb*ngpu))*nb;
                magma_zsetmatrix_1D_row_bcyclic( ngpu, N, N, nb, h_R, lda, d_lA, ldda, queues );
            }

            gpu_time = magma_wtime();
            magma_zpotrf_mgpu( ngpu, opts.uplo, N, d_lA, ldda, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zpotrf_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if ( opts.uplo == MagmaUpper ) {
                magma_zgetmatrix_1D_col_bcyclic( ngpu, N, N, nb, d_lA, ldda, h_R, lda, queues );
            }
            else {
                magma_zgetmatrix_1D_row_bcyclic( ngpu, N, N, nb, d_lA, ldda, h_R, lda, queues );
            }
            
            /* =====================================================================
               Check the result compared to LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = lapackf77_zlange("f", &N, &N, h_A, &lda, work );
                error = lapackf77_zlange("f", &N, &N, h_R, &lda, work ) / Anorm;
                
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (long long) N, gpu_perf, gpu_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_pinned( h_R );
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int dev=0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
