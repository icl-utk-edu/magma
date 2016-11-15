/*
   -- MAGMA (version 1.5) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
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
   -- Testing zpotrf_vbatched
*/

int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *d_A;
    magma_int_t N, total_size_cpu, total_size_dev, info;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], Anorm, error, magma_error;
    int status = 0;
    magmaDoubleComplex **h_A_array=NULL, **d_A_array = NULL;
    magma_int_t *dinfo_magma;
    magma_int_t *hinfo_magma;
    magma_int_t max_N, batchCount;

    magma_int_t *h_N, *d_N;
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magmaDoubleComplex *h_A_tmp, *h_R_tmp, *d_A_tmp; 
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv);
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    double tol = opts.tolerance * lapackf77_dlamch("E");    

    TESTING_CHECK( magma_imalloc_cpu(&h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&hinfo_magma, batchCount) );
    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&dinfo_magma,  batchCount) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount * sizeof(magmaDoubleComplex*)) );

    h_lda = h_N;
    printf("%%              max\n");
    printf("%% BatchCount     N   CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%=====================================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            srand(1000); // guarantee reproducible sizes
            N   = opts.nsize[i];
            total_size_cpu = total_size_dev = 0;
            gflops = 0;
            max_N = 0;
            for(int k = 0; k < batchCount; k++){
                h_N[k] = 1 + (rand() % N);
                max_N = max( max_N, h_N[k] );
                h_ldda[k] = magma_roundup( h_N[k], opts.align );  // multiple of 32 by default
                total_size_cpu += h_N[k] * h_lda[k]; 
                total_size_dev += h_N[k] * h_ldda[k]; 
                gflops += FLOPS_ZPOTRF( h_N[k] ) / 1e9;
            }

            TESTING_CHECK( magma_zmalloc_cpu(&h_A, total_size_cpu) );
            TESTING_CHECK( magma_zmalloc_pinned(&h_R, total_size_cpu) );
            TESTING_CHECK( magma_zmalloc(&d_A, total_size_dev) );

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &total_size_cpu, h_A );
            h_A_tmp = h_A; 
            for (int s=0; s < batchCount; s++){
                magma_zmake_hpd( h_N[s], h_A_tmp, h_lda[s]); // need modification
                h_A_tmp += h_N[s] * h_lda[s]; 
            }
            
            h_A_tmp = h_A;
            h_R_tmp = h_R;
            d_A_tmp = d_A;
            for(int s = 0; s < batchCount; s++){
                lapackf77_zlacpy( MagmaFullStr, &h_N[s], &h_N[s], h_A_tmp, &h_lda[s], h_R_tmp, &h_lda[s] );
                magma_zsetmatrix( h_N[s], h_N[s], h_A_tmp, h_lda[s], d_A_tmp, h_ldda[s], opts.queue );
                h_A_tmp += h_N[s] * h_lda[s]; 
                h_R_tmp += h_N[s] * h_lda[s]; 
                d_A_tmp += h_N[s] * h_ldda[s];
            }

            h_A_array[0] = d_A; 
            for(int s = 1; s < batchCount; s++){
                h_A_array[s] = h_A_array[s-1] + h_N[s-1] * h_ldda[s-1]; 
            }
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), h_A_array, 1, d_A_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue);

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            memset(hinfo_magma, 0, batchCount * sizeof(magma_int_t));
            magma_setvector(batchCount, sizeof(magma_int_t), hinfo_magma, 1, dinfo_magma, 1, opts.queue);
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_zpotrf_vbatched_max_nocheck( 
                       opts.uplo, d_N, 
                       d_A_array, d_ldda, 
                       dinfo_magma, batchCount, max_N, 
                       opts.queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, hinfo_magma, 1, opts.queue);
            for (int s=0; s < batchCount; s++){
                if (hinfo_magma[s] != 0 ) {
                    printf("magma_zpotrf_vbatched matrix %d returned internal error %d\n", s, (int)hinfo_magma[s] );
                    status = -1;
                    goto cleanup;
                }
            }
            if (info != 0){
                status = -1;
                goto cleanup;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ){
                h_A_array[0] = h_A; 
                for(int s = 1; s < batchCount; s++){
                    h_A_array[s] = h_A_array[s-1] + h_N[s-1] * h_lda[s-1]; 
                }
                cpu_time = magma_wtime();
                //#define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads); 
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++){
                    lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &h_N[s], h_A_array[s], &h_lda[s], &info );
                    if (info != 0)
                        printf("lapackf77_zpotrf matrix %d returned err %d: %s.\n", (int) s, (int) info, magma_strerror( info ));
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                h_R_tmp = h_R;
                d_A_tmp = d_A;
                for(int s = 0; s < batchCount; s++){
                    magma_zgetmatrix(h_N[s], h_N[s], d_A_tmp, h_ldda[s], h_R_tmp, h_lda[s], opts.queue);
                    h_R_tmp += h_N[s] * h_lda[s]; 
                    d_A_tmp += h_N[s] * h_ldda[s];
                }
                magma_error = 0.0;
                h_A_tmp = h_A;
                h_R_tmp = h_R;
                for (int s=0; s < batchCount; s++)
                {
                    magma_int_t Asize = h_lda[s] * h_N[s];
                    Anorm = lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &h_N[s], h_A_tmp, &h_lda[s], work);
                    blasf77_zaxpy(&Asize, &c_neg_one, h_A_tmp, &ione, h_R_tmp, &ione);
                    error = lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &h_N[s], h_R_tmp, &h_lda[s], work) / Anorm;
                    magma_error = magma_max_nan( magma_error, error );
                    
                    h_A_tmp += h_N[s] * h_lda[s];
                    h_R_tmp += h_N[s] * h_lda[s];
                }
                bool okay = (magma_error < tol);
                status += ! okay;
                
                printf("  %10lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long)batchCount, (long long)max_N, 
                       cpu_perf, cpu_time*1000.,  
                       gpu_perf, gpu_time*1000., 
                       magma_error,  (magma_error < tol ? "ok" : "failed"));
            }            
            else {
                printf("  %10lld %5lld     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (long long)batchCount, (long long)max_N, 
                       gpu_perf, gpu_time*1000. );
            }
cleanup:
            magma_free_pinned( h_R );
            magma_free_cpu( h_A );
            magma_free( d_A );
            if (status == -1)
                break;
        }
        if (status == -1)
            break;

        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_A_array );
    magma_free( dinfo_magma );
    magma_free_cpu( h_N );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_A_array );
    magma_free_cpu( hinfo_magma );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
