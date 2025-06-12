/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

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

#define PRECISION_z

// uncomment to use mkl's group batch interface
//#define USE_MKL_GETRF_BATCH

// uncomment to introduce singularity in one matrix
// by setting two different columns to zeros
// (edit MTX_ID, COL1, and COL2 accordingly)
//#define SINGULARITY_CHECK
#ifdef SINGULARITY_CHECK
#define MTX_ID (10)    // checked against batchCount
#define COL1   (1)     // checked against #columns
#define COL2   (10)    // checked against #columns
#endif


////////////////////////////////////////////////////////////////////////////////
double get_LU_error(magma_int_t M, magma_int_t N,
                    magmaDoubleComplex *A,  magma_int_t lda,
                    magmaDoubleComplex *LU)
{
    magma_int_t min_mn = min(M, N);
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *L, *U;
    double work[1], matnorm, residual;

    TESTING_CHECK( magma_zmalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_zmalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );

    matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_Z_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops=0, magma_perf=0, magma_time=0, cpu_perf=0, cpu_time=0, replacement_tol=0;
    real_Double_t   NbyM;
    double          error;
    magma_int_t     hA_size = 0, dA_size = 0;
    magma_int_t     seed = 0;
    magmaDoubleComplex *hA, *hR, *hA_magma, *hTmp;
    magmaDoubleComplex *dA;
    magmaDoubleComplex **dA_array = NULL, **hA_array = NULL, **hR_array = NULL, **hdA_array = NULL;

    magma_int_t     *hinfo;
    magma_int_t     *dinfo;

    magma_int_t *h_M = NULL, *h_N = NULL, *h_lda  = NULL, *h_ldda = NULL, *h_min_mn = NULL;
    magma_int_t *d_M = NULL, *d_N = NULL, *d_ldda = NULL, *d_min_mn;
    magma_int_t iM, iN, max_M=0, max_N=0, max_minMN=0, max_MxN=0, replacements = 0, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= (opts.check == 2);
    double tol = opts.tolerance * lapackf77_dlamch("E");

    batchCount = opts.batchcount;

    TESTING_CHECK( magma_imalloc_cpu(&h_M,      batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_N,      batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_lda,    batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda,   batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_min_mn, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&hinfo,    batchCount ));

    TESTING_CHECK( magma_imalloc(&d_M,      batchCount) );
    TESTING_CHECK( magma_imalloc(&d_N,      batchCount) );
    TESTING_CHECK( magma_imalloc(&d_ldda,   batchCount) );
    TESTING_CHECK( magma_imalloc(&d_min_mn, batchCount) );
    TESTING_CHECK( magma_imalloc(&dinfo,    batchCount ));

    TESTING_CHECK( magma_malloc_cpu((void**)&hA_array,  batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hR_array,  batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hdA_array, batchCount * sizeof(magmaDoubleComplex*)) );
    TESTING_CHECK( magma_malloc(    (void**)&dA_array,  batchCount * sizeof(magmaDoubleComplex*)) );

    printf("%%             max   max\n");
    printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   Replacements   ||A-LU||/(||A||*N) \n");
    printf("%%==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        seed = rand();
        for( int iter = 0; iter < opts.niter; ++iter ) {
            srand(seed);    // necessary to have the same sizes across different iterations

            iM = opts.msize[itest];
            iN = opts.nsize[itest];
            NbyM  = (real_Double_t)iN / (real_Double_t)iM;

            hA_size  = 0;
            dA_size  = 0;
            gflops   = 0;

            for(int s = 0; s < batchCount; s++) {
                h_M[s]      = 1 + (rand() % iM);
                h_N[s]      = max(1, (magma_int_t) round(NbyM * real_Double_t(h_M[s])) ); // try to keep the M/N ratio
                max_M       = (s == 0) ? h_M[s] : max(h_M[s], max_M);
                max_N       = (s == 0) ? h_N[s] : max(h_N[s], max_N);
                h_lda[s]    = h_M[s];
                h_ldda[s]   = h_lda[s]; //magma_roundup( h_M[s], opts.align );  // multiple of 32 by default
                h_min_mn[s] = min( h_M[s], h_N[s] );
                max_minMN   = (s == 0) ? h_min_mn[s] : max(h_min_mn[s], max_minMN);
                max_MxN     = (s == 0) ? h_M[s] * h_N[s] : max(h_M[s] * h_N[s], max_MxN);
                hA_size    += h_lda[s]  * h_N[s];
                dA_size    += h_ldda[s] * h_N[s];
                gflops     += FLOPS_ZGETRF( h_M[s], h_N[s] ) / 1e9;
            }

            TESTING_CHECK( magma_zmalloc_cpu( &hA,       hA_size  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hA_magma, hA_size  ));
            TESTING_CHECK( magma_zmalloc_pinned( &hR,    hA_size  ));

            TESTING_CHECK( magma_zmalloc( &dA,       dA_size ));

            /* Initialize ptr arrays */
            hA_array [0]    = hA;
            hR_array [0]    = hR;
            hdA_array[0]    = dA;
            for(int s = 1; s < batchCount; s++) {
                hA_array[s]     = hA_array[s-1]  + h_lda[s-1]  * h_N[s-1];
                hR_array[s]     = hR_array[s-1]  + h_lda[s-1]  * h_N[s-1];
                hdA_array[s]    = hdA_array[s-1] + h_ldda[s-1] * h_N[s-1];
            }

            /* Initialize hA and copy to hR */
            lapackf77_zlarnv( &ione, ISEED, &hA_size, hA );

            #ifdef SINGULARITY_CHECK
            // introduce singularity -- for debugging purpose only
            magma_int_t id   = min(MTX_ID, batchCount-1);
            magma_int_t col1 = min(COL1, h_N[id]-1);
            magma_int_t col2 = min(COL2, h_N[id]-1);
            printf("singularity in matrix %lld of size (%lld, %lld) : col. %lld & %lld set to zeros\n",
                   (long long)id, (long long)h_M[id], (long long)h_N[id],
                   (long long)col1, (long long)col2);
            memset(hA_array[id] + col1 * h_lda[id], 0, h_M[id] * sizeof(magmaDoubleComplex));
            memset(hA_array[id] + col2 * h_lda[id], 0, h_M[id] * sizeof(magmaDoubleComplex));
            #endif

            memcpy(hR, hA, hA_size * sizeof(magmaDoubleComplex));

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_setvector(batchCount, sizeof(magmaDoubleComplex*), hdA_array, 1, dA_array, 1, opts.queue);
            magma_isetvector(batchCount, h_M,    1, d_M,    1, opts.queue);
            magma_isetvector(batchCount, h_N,    1, d_N,    1, opts.queue);
            magma_isetvector(batchCount, h_ldda, 1, d_ldda, 1, opts.queue);
            magma_isetvector(batchCount, h_min_mn, 1, d_min_mn, 1, opts.queue);

            for(int s = 0; s < batchCount; s++) {
                magma_zsetmatrix( h_M[s], h_N[s],
                                  hR_array[s],  h_lda[s],
                                  hdA_array[s], h_ldda[s], opts.queue );
            }

            if(opts.version == 1) {
                // main API, with error checking and
                // workspace allocation
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_zgetrf_nopiv_vbatched(
                        d_M, d_N,
                        dA_array, d_ldda, dinfo,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            else {
                // advanced API, totally asynchronous,
                // but requires some setup
                magma_time = magma_sync_wtime( opts.queue );
                magma_int_t nb, recnb;
                magma_get_zgetrf_vbatched_nbparam(max_M, max_N, &nb, &recnb);
                replacement_tol=1e-7;
                info = magma_zgetrf_nopiv_vbatched_max_nocheck(
                        d_M, d_N, d_min_mn,
                        max_M, max_N, max_minMN, max_MxN, nb, 32,
                        dA_array, d_ldda,
                        NULL, replacement_tol, dinfo,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            magma_perf = gflops / magma_time;

            hTmp = hA_magma;
            for(int s = 0; s < batchCount; s++) {
                magma_zgetmatrix( h_M[s], h_N[s], hdA_array[s], h_ldda[s], hTmp, h_lda[s], opts.queue );
                hTmp += h_lda[s] * h_N[s];
            }

            // check info
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo, 1, hinfo, 1, opts.queue );
            replacements = 0;
            for (int i=0; i < batchCount; i++) {
                if(replacement_tol != 0) {
                    if(hinfo[i] >= 0) {
                        replacements += hinfo[i];
                    }
                    else {
                        printf("magma_zgetrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) hinfo[i] );
                    }
                }
                else if (hinfo[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) hinfo[i] );
                }
            }

            if (info != 0) {
                printf("magma_zgetrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)    %10d  ",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000., replacements);
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)   %10d  ",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       magma_perf, magma_time*1000., replacements );
            }

            if ( opts.check == 1 ) {
                hA_array[0] = hA_magma;
                for(int s = 1; s < batchCount; s++) {
                    hA_array[s] = hA_array[s-1] + h_lda[s-1] * h_N[s-1];
                }

                error = 0;
                #pragma omp parallel for reduction(max:error)
                for (int s=0; s < batchCount; s++) {
                    double err = get_LU_error( h_M[s], h_N[s], hR_array[s], h_lda[s], hA_array[s]);
                    error = magma_max_nan( err, error );
                }

                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( hA );
            magma_free_cpu( hA_magma );
            magma_free_pinned( hR );

            magma_free( dA );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_min_mn );
    magma_free( dA_array );
    magma_free( dinfo );

    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_lda );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_min_mn );
    magma_free_cpu( hA_array );
    magma_free_cpu( hR_array );
    magma_free_cpu( hdA_array );
    magma_free_cpu( hinfo );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
