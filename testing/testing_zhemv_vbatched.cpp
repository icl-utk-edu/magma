/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah
       
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

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhemv_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t N;
    magma_int_t total_size_A_cpu = 0, total_size_X = 0, total_size_Y = 0;
    magma_int_t total_size_A_dev = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;
    magma_int_t max_N;

    magmaDoubleComplex *h_A, *h_X, *h_Y, *h_Ymagma;
    magmaDoubleComplex *d_A, *d_X, *d_Y;
    magmaDoubleComplex **h_A_array = NULL;
    magmaDoubleComplex **h_X_array = NULL;
    magmaDoubleComplex **h_Y_array = NULL;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_X_array = NULL;
    magmaDoubleComplex **d_Y_array = NULL;
    magmaDoubleComplex *h_A_tmp, *h_X_tmp, *h_Y_tmp, *h_Ymagma_tmp;
    magma_int_t *h_N, *d_N;
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_incx, *d_incx;
    magma_int_t *h_incy, *d_incy;
    magma_int_t max_inc = 1;
    
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    // allocate space for the sizes/leading dim.
    TESTING_CHECK( magma_imalloc_cpu(&h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incx, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incy, batchCount) );
    
    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_incx, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_incy, batchCount+1) );
    
    double *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Xnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Ynorm, batchCount ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_X_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_Y_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_X_array, batchCount*sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_Y_array, batchCount*sizeof(magmaDoubleComplex*)) );
    
    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n"
           "%% uplo = %s\n",
           lapack_uplo_const(opts.uplo));
    
    printf("%%              max\n");
    printf("%% BatchCount     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=======================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            h_lda = h_N;
            
            // guarantee reproducible sizes
            srand(1000);
            
            gflops = 0;
            max_N = 0;
            total_size_A_cpu = total_size_A_dev = 0;
            total_size_X = total_size_Y = 0;
            for (int i = 0; i < batchCount; i++) {
                h_N[i]    = 1 + (rand() % N);
                h_incx[i] = 1 + (rand() % max_inc);
                h_incy[i] = 1 + (rand() % max_inc);
                
                max_N = max( max_N, h_N[i] );
                
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default
                
                total_size_A_cpu += h_N[i] * h_lda[i];
                total_size_A_dev += h_N[i] * h_ldda[i];
                
                total_size_X += h_N[i] * h_incx[i];
                total_size_Y += h_N[i] * h_incy[i];
                
                gflops += FLOPS_ZHEMV( h_N[i] ) / 1e9;
            }
            
            TESTING_CHECK( magma_zmalloc_cpu(&h_A, total_size_A_cpu) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_X,   total_size_X) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Y,   total_size_Y) );
            TESTING_CHECK( magma_zmalloc_cpu(&h_Ymagma, total_size_Y) );
            
            TESTING_CHECK( magma_zmalloc(&d_A, total_size_A_dev) );
            TESTING_CHECK( magma_zmalloc(&d_X, total_size_X) );
            TESTING_CHECK( magma_zmalloc(&d_Y, total_size_Y) );

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &total_size_A_cpu, h_A );
            lapackf77_zlarnv( &ione, ISEED, &total_size_X, h_X );
            lapackf77_zlarnv( &ione, ISEED, &total_size_Y, h_Y );
            
            /* set the opposite triangular part to NAN to check */
            magmaDoubleComplex* hAi = h_A;
            for (magma_int_t i = 0; i < batchCount; i++) {
                magma_int_t Ni = h_N[i]-1;
                if ( opts.uplo == MagmaUpper ) {
                    lapackf77_zlaset( "Lower", &Ni, &Ni, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hAi[1], &h_lda[i] );
                }
                else {
                    lapackf77_zlaset( "Upper", &Ni, &Ni, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &hAi[ h_lda[i] ], &h_lda[i] );
                }
                hAi += h_lda[i] * h_N[i];
            }

            // Compute norms for error computation
            h_A_tmp = h_A;
            h_X_tmp = h_X;
            h_Y_tmp = h_Y;
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &h_N[s], h_A_tmp, &h_lda[s], work );
                Xnorm[s] = lapackf77_zlange( "F", &ione, &h_N[s], h_X_tmp, &h_incx[s], work );
                Ynorm[s] = lapackf77_zlange( "F", &ione, &h_N[s], h_Y_tmp, &h_incy[s], work );
                h_A_tmp += h_N[s] * h_lda[s];
                h_X_tmp += h_N[s] * h_incx[s];
                h_Y_tmp += h_N[s] * h_incy[s];
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_incx, 1, d_incx, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_incy, 1, d_incy, 1, opts.queue );
            
            h_A_array[0] = d_A;
            h_X_array[0] = d_X;
            h_Y_array[0] = d_Y;
            for (int i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + h_N[i-1] * h_ldda[i-1];
                h_X_array[i] = h_X_array[i-1] + h_N[i-1] * h_incx[i-1];
                h_Y_array[i] = h_Y_array[i-1] + h_N[i-1] * h_incy[i-1];
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_X_array, 1, d_X_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_Y_array, 1, d_Y_array, 1, opts.queue );
            
            h_A_tmp = h_A;
            for (int i = 0; i < batchCount; i++) {
                magma_zsetmatrix( h_N[i], h_N[i], h_A_tmp, h_lda[i], h_A_array[i], h_ldda[i], opts.queue );
                h_A_tmp += h_N[i] * h_lda[i];
            }
            magma_zsetvector( total_size_X, h_X, 1, d_X, 1, opts.queue );
            magma_zsetvector( total_size_Y, h_Y, 1, d_Y, 1, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            #if 0
            magmablas_zhemv_vbatched_max_nocheck(opts.uplo, d_N,
                             alpha, d_A_array, d_ldda,
                                    d_X_array, d_incx,
                             beta,  d_Y_array, d_incy,
                             batchCount, max_N, opts.queue);
            #else
            magmablas_zhemv_vbatched(opts.uplo, d_N,
                             alpha, d_A_array, d_ldda,
                                    d_X_array, d_incx,
                             beta,  d_Y_array, d_incy,
                             batchCount, opts.queue);
            #endif
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_zgetvector(total_size_Y, d_Y, 1, h_Ymagma, 1, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                // displace pointers for the cpu, reuse h_A_array
                h_A_array[0] = h_A;
                h_X_array[0] = h_X;
                h_Y_array[0] = h_Y;
                for (int i = 1; i < batchCount; i++) {
                    h_A_array[i] = h_A_array[i-1] + h_N[i-1] * h_lda[i-1];
                    h_X_array[i] = h_X_array[i-1] + h_N[i-1] * h_incx[i-1];
                    h_Y_array[i] = h_Y_array[i-1] + h_N[i-1] * h_incy[i-1];
                }
                cpu_time = magma_wtime();
                //#define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    blasf77_zhemv( lapack_uplo_const(opts.uplo), &h_N[s],
                                   &alpha, h_A_array[s], &h_lda[s],
                                           h_X_array[s], &h_incx[s],
                                   &beta,  h_Y_array[s], &h_incy[s] );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = m
                magma_error = 0;
                
                h_Y_tmp = h_Y;
                h_Ymagma_tmp = h_Ymagma;
                for (int s=0; s < batchCount; s++){
                    normalize = sqrt(double(h_N[s]+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    blasf77_zaxpy( &h_N[s], &c_neg_one, h_Y_tmp, &h_incy[s], h_Ymagma_tmp, &h_incy[s] );
                    error = lapackf77_zlange( "F", &ione, &h_N[s], h_Ymagma_tmp, &h_incy[s], work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    
                    h_Y_tmp      += h_N[s] * h_incy[s];
                    h_Ymagma_tmp += h_N[s] * h_incy[s];
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) max_N,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount,(long long) max_N,
                       magma_perf,  1000.*magma_time);
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Y );
            magma_free_cpu( h_Ymagma );

            magma_free( d_A );
            magma_free( d_X );
            magma_free( d_Y );
            
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    // free resources
    magma_free_cpu( h_N );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_incx );
    magma_free_cpu( h_incy );

    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    magma_free_cpu( h_A_array );
    magma_free_cpu( h_X_array );
    magma_free_cpu( h_Y_array );
    
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_incx );
    magma_free( d_incy );
    magma_free( d_A_array );
    magma_free( d_X_array );
    magma_free( d_Y_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
