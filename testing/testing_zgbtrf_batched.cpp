/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

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

double get_band_LU_error(
            magma_int_t M, magma_int_t N,
            magma_int_t KL, magma_int_t KU,
            magmaDoubleComplex *AB,  magma_int_t ldab,
            magmaDoubleComplex *LUB, magma_int_t *IPIV)
{
#define   A(i,j)   A[(j)*M    + (i)]
#define  LU(i,j)  LU[(j)*M    + (i)]
#define  AB(i,j)  AB[(j)*ldab + (i)]
#define LUB(i,j) LUB[(j)*ldab + (i)]

    magma_int_t min_mn = min(M, N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *A, *LU, *L, *U;
    double work[1], matnorm, residual;

    TESTING_CHECK( magma_zmalloc_cpu( &A,  M*N ));
    TESTING_CHECK( magma_zmalloc_cpu( &LU, M*N ));
    TESTING_CHECK( magma_zmalloc_cpu( &L,  M*min_mn ));
    TESTING_CHECK( magma_zmalloc_cpu( &U,  min_mn*N ));
    memset( A,  0, M*N*sizeof(magmaDoubleComplex) );
    memset( LU, 0, M*N*sizeof(magmaDoubleComplex) );
    memset( L,  0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U,  0, min_mn*N*sizeof(magmaDoubleComplex) );

    // recover A in dense form, account for extra KL super-diagonals
    for(j = 0; j < N; j++) {
        int col_start      = max(0, j-KU);
        int col_end        = min(j+KL,M-1);
        int col_length     = col_end - col_start + 1;
        int col_start_band = KL + max(KU-j,0);
        memcpy( &A(col_start,j), &AB(col_start_band,j), col_length * sizeof(magmaDoubleComplex));
    }
    // end of converting AB to dense in A

    // recover LU in dense form
    magma_int_t KV = KL + KU;
    for(j = 0; j < N; j++) {
        magma_int_t col_start      = max(0, j-KV);
        magma_int_t col_end        = min(j+KL,M-1);
        magma_int_t col_length     = col_end - col_start + 1;
        magma_int_t col_start_band = max(KV-j,0);
        memcpy( &LU(col_start,j), &LUB(col_start_band,j), col_length * sizeof(magmaDoubleComplex));
    }

    // swapping to recover L
    for(j = 0; j < N-2; j++) {
        const magma_int_t k1 = j+2;
        const magma_int_t k2 = N;
        lapackf77_zlaswp(&ione, &LU(0,j), &M, &k1, &k2, IPIV, &ione );
    }
    // end of converting LUB to dense in LU

    #if 0
    magma_int_t Mband  = KL + 1 + (KL+KU); // need extra KL for the upper factor
    magma_int_t Nband  = N;
    for(j = 0; j < min_mn; j++) {printf("%2d ", IPIV[j]);} printf("\n");
    printf("AB = ");
    magma_zprint(Mband, Nband,  AB, ldab);
    printf("A = ");
    magma_zprint(M, N,  A, M);
    printf("LUB = ");
    magma_zprint(Mband, Nband, LUB, ldab);
    printf("LU = ");
    magma_zprint(M, N,  LU, M);
    #endif

    lapackf77_zlaswp( &N, A, &M, &ione, &min_mn, IPIV, &ione);
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &M, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &M, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );

    matnorm = lapackf77_zlange("f", &M, &N, A, &M, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &M);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*M] = MAGMA_Z_SUB( LU[i+j*M], A[i+j*M] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &M, work);

    magma_free_cpu( A );
    magma_free_cpu( LU );
    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);

#undef A
#undef LU
#undef AB
#undef LUB
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf=0, cpu_time=0;
    double          error;
    magmaDoubleComplex *h_A, *h_R, *h_Amagma;
    magmaDoubleComplex *dA;
    magmaDoubleComplex **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;

    magma_int_t M, N, Mband, Nband, KL, KU, n2, ldab, lddab, min_mn, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    batchCount = opts.batchcount;
    KL         = opts.kl;
    KU         = opts.ku;
    magma_int_t columns;

    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   ||PA-LU||/(||A||*N)\n");
    printf("%%=======================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);

            Mband  = KL + 1 + (KL+KU); // need extra KL for the upper factor
            Nband  = N;
            ldab   = Mband;
            n2     = ldab * Nband * batchCount;
            lddab  = magma_roundup( Mband, opts.align );  // multiple of 32 by default
            gflops = 0.;    // TODO: gflop formula for gbtrf

            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn * batchCount ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,  n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Amagma,  n2     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,  n2     ));

            TESTING_CHECK( magma_zmalloc( &dA,  lddab * Nband * batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv_magma,  min_mn * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            columns = Nband * batchCount;
            lapackf77_zlacpy( MagmaFullStr, &Mband, &columns, h_A, &ldab, h_R, &ldab );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( Mband, columns, h_R, ldab, dA, lddab, opts.queue );
            magma_zset_pointer( dA_array, dA, lddab, 0, 0, lddab*Nband, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, dipiv_magma, 1, 0, 0, min_mn, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            if(opts.version == 1) {
                magma_zgbtrf_batched_small(
                    M,  N, KL, KU,
                    dA_array, lddab,
                    dipiv_magma, dinfo_magma,
                    batchCount, opts.queue );
            }
            else if (opts.version == 2) {
                memcpy(h_Amagma, h_R, Mband * columns * sizeof(magmaDoubleComplex));
                for (magma_int_t s=0; s < batchCount; s++) {
                        magma_int_t locinfo;
                        lapackf77_zgbtrf(&M, &N, &KL, &KU, h_Amagma + s * ldab * Nband, &ldab, ipiv + s * min_mn, &locinfo );
                }
                magma_zsetmatrix( Mband, columns, h_Amagma, ldab, dA, lddab, opts.queue );
                magma_setvector( min_mn * batchCount, sizeof(magma_int_t), ipiv, 1, dipiv_magma, 1, opts.queue );
                magma_memset_async(dinfo_magma, 0, batchCount * sizeof(magma_int_t), opts.queue);
                info = 0;
            }
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_zgetmatrix( Mband, Nband*batchCount, dA, lddab, h_Amagma, ldab, opts.queue );

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );

            for (int i=0; i < batchCount; i++) {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgbtrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }

            if (info != 0) {
                printf("magma_zgbtrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_zgbtrf(&M, &N, &KL, &KU, h_A + s * ldab * Nband, &ldab, ipiv + s * min_mn, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zgbtrf matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000  );
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf, magma_time*1000. );
            }

            if ( opts.check ) {
                magma_getvector( min_mn * batchCount, sizeof(magma_int_t), dipiv_magma, 1, ipiv, 1, opts.queue );
                error = 0;
                for (int i=0; i < batchCount; i++) {
                    for (int k=0; k < min_mn; k++) {
                        if (ipiv[i*min_mn+k] < 1 || ipiv[i*min_mn+k] > M ) {
                            printf("error for matrix %lld ipiv @ %lld = %lld\n",
                                    (long long) i, (long long) k, (long long) ipiv[i*min_mn+k] );
                            error = -1;
                        }
                    }
                    if (error == -1) {
                        break;
                    }

                    double err = get_band_LU_error(M, N, KL, KU, h_R + i * ldab*N,  ldab, h_Amagma + i * ldab*N, ipiv + i * min_mn);
                    if (std::isnan(err) || std::isinf(err)) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( cpu_info );
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free_cpu( h_Amagma );
            magma_free_pinned( h_R );

            magma_free( dA );
            magma_free( dinfo_magma );
            magma_free( dipiv_magma );
            magma_free( dipiv_array );
            magma_free( dA_array );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
