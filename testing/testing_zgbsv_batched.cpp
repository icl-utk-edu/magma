/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Mark gates
   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
 */

// includes, system
#include <stdio.h>
#include <stdlib.h>
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

#define cond (N <= 32 && batchCount == 1)

// multiplies a square band matrix ( Nband x Nband ) with
// a dense matrix (Nband x N)
static void
sqband_x_dense(
    magma_int_t Nband, magma_int_t N,
    magma_int_t KL, magma_int_t KU,
    magmaDoubleComplex* hA, magma_int_t lda,
    magmaDoubleComplex* hB, magma_int_t ldb,
    magmaDoubleComplex* hC, magma_int_t ldc)
{
#if 0
#define   hA(i,j)   hA[(j)*lda + (i)]
#define   hB(i,j)   hB[(j)*ldb + (i)]
#define   hC(i,j)   hC[(j)*ldc + (i)]
#define   hW(i,j)   hW[(j)*ldw + (i)]

    // blocking size
    magma_int_t nb = 32;

    // set hC to zero
    memset(hC, 0, ldc * N * sizeof(magmaDoubleComplex));

    // workspace
    magmaDoubleComplex* hW = NULL;
    magma_int_t ldw = 2 * (KL + KU + 1);
    magma_int_t lwork = nb * ldw;
    magma_zmalloc_cpu(&hW, lwork);

    for(magma_int_t j = 0; j < Nband, j += nb) {
        magma_int_t ib = min(nb, Nband-j);
        magma_int_t j1 = j;
        magma_int_t j2 = j1 + ib - 1;

        magma_int_t hW_offset = 0;
        magma_int_t hB_offset = 0;
        memset(hW, 0, lwork*sizeof(magmaDoubleComplex));
        for(magma_int_t c = 0; c < ib; c++) {
            magma_int_t jc      = c + j;
            magma_int_t c_start = max(0, jc - KU);
            magma_int_t c_end   = min(jc + KL, Nband-1);
            magma_int_t c_len   = c_end - c_start + 1;
            memcpy( &hA(c_start, jc), &hW(hW_offset,c), col_length * sizeof(magmaDoubleComplex));
            hW_offset += ( jc < KU ) ? 0 : 1;
        }


        // gemm
        lapackf77_zgemm(
            MagmaNoTrans, MagmaNoTrans,
            ldw, N, ib,
            MAGMA_Z_ONE,     hW(0,         0), lda,
                             hB(hB_offset, 0), ldb,
            MAGMA_Z_NEG_ONE, hC(hB_offset, 0), ldc);

    }

    magma_free_cpu( hW );
#endif
}
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesv_batched
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t *dipiv, *dinfo_array;
    magma_int_t *ipiv, *cpu_info;
    magma_int_t N, Nband, KL, KU, KV, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dB_array = NULL;
    magma_int_t     **dipiv_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs       = opts.nrhs;
    KL         = opts.kl;
    KU         = opts.ku;
    KV         = KL + KU;
    batchCount = opts.batchcount;

    printf("%% ## INFO ##: Gflop/s calculation is not available\n");
    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% BatchCount   N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            Nband  = KL + 1 + KV; // need extra KL superdiagonals for the upper factor
            lda    = Nband;
            ldb    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( ldb, opts.align );
            gflops = 0.;  // TODO: gflop formula for gbsv?

            sizeA = lda*N*batchCount;
            sizeB = ldb*nrhs*batchCount;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, sizeB ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, sizeB ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, batchCount*N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));

            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N*batchCount    ));
            TESTING_CHECK( magma_zmalloc( &d_B, lddb*nrhs*batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv, N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array,    batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );

            magma_zsetmatrix( Nband, N*batchCount,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N,     nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N,    batchCount, opts.queue );
            magma_zset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, dipiv, 1, 0, 0, N, batchCount, opts.queue );

            if(cond) {
                printf("a = ");
                magma_zprint_gpu(Nband, N, d_A, ldda, opts.queue);
                printf("b = ");
                magma_zprint_gpu(N, nrhs, d_B, lddb, opts.queue);
            }

            // testing MAGMA
            gpu_time = magma_sync_wtime( opts.queue );
            if(opts.version == 1) {
                info = magma_zgbsv_batched(
                        N, KL, KU, nrhs,
                        dA_array, ldda, dipiv_array,
                        dB_array, lddb, dinfo_array,
                        batchCount, opts.queue);
            }
            else if(opts.version == 2) {
                magma_int_t nthreads = max(opts.nb, KL+1);
                info = magma_zgbsv_batched_fused_sm(
                        N, KL, KU, nrhs,
                        dA_array, ldda, dipiv_array,
                        dB_array, lddb, dinfo_array,
                        nthreads, 1, batchCount, opts.queue );
            }
            else{
                // --------------------------------------------------
                magma_int_t linfo = 0, columns = nrhs * batchCount, n2 = N * batchCount;
                info = 0;
                magmaDoubleComplex* hTmp = NULL;
                TESTING_CHECK( magma_zmalloc_cpu( &hTmp, sizeA ));
                lapackf77_zlacpy( MagmaFullStr, &Nband, &n2, h_A, &lda, hTmp, &lda );
                lapackf77_zlacpy( MagmaFullStr, &N, &columns, h_B, &ldb, h_X, &ldb );
                for(magma_int_t s = 0; s < batchCount; s++) {
                    lapackf77_zgbsv( &N, &KL, &KU, &nrhs,
                                     hTmp + s * lda * N,    &lda,
                                     ipiv + s * N,
                                     h_X  + s * ldb * nrhs, &ldb,
                                     &linfo );
                    info += linfo;
                }
                magma_memset(dinfo_array, 0, batchCount * sizeof(magma_int_t));
                magma_zsetmatrix( N, nrhs*batchCount, h_X, ldb, d_B, lddb, opts.queue );
                TESTING_CHECK( magma_free_cpu( hTmp ));
                // --------------------------------------------------
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;

            if(cond) {
                printf("Af = ");
                magma_zprint_gpu(Nband, N, d_A, ldda, opts.queue);
                printf("bm = ");
                magma_zprint_gpu(N, nrhs, d_B, lddb, opts.queue);
            }

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );
            if (info != 0) {
                printf("magma_zgbsv_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }
            else {
                for (int i=0; i < batchCount; i++) {
                    if (cpu_info[i] != 0 ) {
                        printf("magma_zgbsv_batched matrix %lld returned internal error %lld\n",
                                (long long) i, (long long) cpu_info[i] );
                    }
                }
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );

            error = 0;
            for (magma_int_t s=0; s < batchCount; s++) {
                magmaDoubleComplex* hA = h_A + s * lda * N + KL;
                magmaDoubleComplex* hX = h_X + s * ldb * nrhs;
                magmaDoubleComplex* hB = h_B + s * ldb * nrhs;

                Anorm = lapackf77_zlangb("I", &N, &KL, &KU, hA, &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, hX, &ldb, work);

                for(magma_int_t j = 0; j < nrhs; j++) {
                    blasf77_zgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                                   &c_one, hA           , &lda,
                                           hX  + j * ldb, &ione,
                               &c_neg_one, hB  + j * ldb, &ione);
                }

                Rnorm = lapackf77_zlange("I", &N, &nrhs, hB, &ldb, work);

                double err = Rnorm/(N*Anorm*Xnorm);

                if (std::isnan(err) || std::isinf(err)) {
                    error = err;
                    break;
                }
                error = max( err, error );
            }
            bool okay = (error < tol);
            status += ! okay;

            /* ====================================================================
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
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_zgbsv(
                        &N, &KL, &KU, &nrhs,
                        h_A  + s*lda*N,    &lda, ipiv + s*N,
                        h_B  + s*ldb*nrhs, &ldb, &locinfo );

                    if (locinfo != 0) {
                        printf("lapackf77_zgesv matrix %lld returned error %lld: %s.\n",
                                (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                printf( "%10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        gpu_perf, 1000.*gpu_time,
                        error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );
            magma_free_cpu( cpu_info );

            magma_free( d_A );
            magma_free( d_B );

            magma_free( dipiv );
            magma_free( dinfo_array );

            magma_free( dA_array );
            magma_free( dB_array );
            magma_free( dipiv_array );
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
