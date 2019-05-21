/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX 
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          error, Rnorm, Anorm;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B, d_X, d_WORKD;
    magmaFloatComplex  *d_As, *d_Bs, *d_WORKS;
    double          *h_workd;
    magma_int_t *h_ipiv, *d_ipiv;
    magma_int_t lda, ldb, ldx;
    magma_int_t ldda, lddb, lddx;
    magma_int_t N, nrhs, gesv_iter, info, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    printf("%% Epsilon(double): %8.6e\n"
           "%% Epsilon(single): %8.6e\n\n",
           lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon") );
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    nrhs = opts.nrhs;
    
    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    #ifdef REAL
    if ( opts.version == 3 ) {
        printf("%%   N  NRHS   DP-Factor  DP-Solve  HP-Factor  HP-Solve  MP: FP16->FP64-Solve  Iter   |b-Ax|/N|A|\n");
    }
    else{
        printf("%%   N  NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP: FP32->FP64-Solve  Iter   |b-Ax|/N|A|\n");
    }
    #else
    printf("%%   N  NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  Iter   |b-Ax|/N|A|\n");
    #endif
    printf("%%=========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb  = ldx = lda = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb = lddx = ldda;
            
            gflopsF = FLOPS_ZGETRF( N, N ) / 1e9;
            gflopsS = gflopsF + FLOPS_ZGETRS( N, nrhs ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,     lda*N    ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,     ldb*nrhs ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X,     ldx*nrhs ));
            TESTING_CHECK( magma_imalloc_cpu( &h_ipiv,  N        ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_workd, N        ));
            
            TESTING_CHECK( magma_zmalloc( &d_A,     ldda*N        ));
            TESTING_CHECK( magma_zmalloc( &d_B,     lddb*nrhs     ));
            TESTING_CHECK( magma_zmalloc( &d_X,     lddx*nrhs     ));
            TESTING_CHECK( magma_imalloc( &d_ipiv,  N             ));
            TESTING_CHECK( magma_cmalloc( &d_WORKS, ldda*(N+nrhs) ));
            TESTING_CHECK( magma_zmalloc( &d_WORKD, N*nrhs        ));
            
            /* Initialize matrices */
            magma_generate_matrix( opts, N, N, h_A, lda );
            size = ldb * nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaFullStr, &N, &nrhs, h_B, &ldb, h_X, &ldx);
            
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            //=====================================================================
            //              MIXED - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                magma_zcgesv_gpu( opts.transA, N, nrhs,
                        d_A, ldda, h_ipiv, d_ipiv,
                        d_B, lddb, d_X, lddx,
                        d_WORKD, d_WORKS, &gesv_iter, &info);
            }
            #ifdef REAL
            else if ( opts.version == 2 ) {
                magma_dsgesv_iteref_gpu( opts.transA, N, nrhs,
                        d_A, ldda, h_ipiv, d_ipiv,
                        d_B, lddb, d_X, lddx,
                        d_WORKD, d_WORKS, &gesv_iter, &info);
            }
            else if ( opts.version == 3 ) {
                magma_dhgesv_iteref_gpu( opts.transA, N, nrhs,
                        d_A, ldda, h_ipiv, d_ipiv,
                        d_B, lddb, d_X, lddx,
                        d_WORKD, d_WORKS, &gesv_iter, &info);
            }
            #endif
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_zcgesv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //              ERROR DP vs MIXED  - GPU
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_X, lddx, h_X, ldx, opts.queue );
            
            Anorm = lapackf77_zlange("I", &N, &N, h_A, &lda, h_workd);
            blasf77_zgemm( lapack_trans_const(opts.transA), MagmaNoTransStr,
                           &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, h_workd);
            error = Rnorm / (N*Anorm);
            
            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_zgetrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            magma_zgetrs_gpu( opts.transA, N, nrhs, d_A, ldda, h_ipiv, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_zgetrs returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_WORKS;
            d_Bs = d_WORKS + ldda*N;
            magma_zsetmatrix( N, N,    h_A, lda,  d_A,  ldda, opts.queue );
            magma_zsetmatrix( N, nrhs, h_B, ldb,  d_B,  lddb, opts.queue );
            magmablas_zlag2c( N, N,    d_A, ldda, d_As, ldda, opts.queue, &info );
            magmablas_zlag2c( N, nrhs, d_B, lddb, d_Bs, lddb, opts.queue, &info );
            
            gpu_time = magma_wtime();
            #ifdef REAL
            if ( opts.version == 3 ) {
                magma_htgetrf_gpu( N, N,    d_As, ldda, h_ipiv, &info);
            }
            else
            #endif
            {
                magma_cgetrf_gpu(N, N, d_As, ldda, h_ipiv, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_cgetrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_zlag2c(N, N,    d_A, ldda, d_As, ldda, opts.queue, &info );
            magmablas_zlag2c(N, nrhs, d_B, lddb, d_Bs, lddb, opts.queue, &info );
            
            gpu_time = magma_wtime();
            #ifdef REAL
            if ( opts.version == 3 ) {
                magma_htgetrf_gpu( N, N,    d_As, ldda, h_ipiv, &info);
            }
            else
            #endif
            {
                magma_cgetrf_gpu(N, N, d_As, ldda, h_ipiv, &info);
            }
            magma_cgetrs_gpu( opts.transA, N, nrhs, d_As, ldda, h_ipiv,
                              d_Bs, lddb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_cgetrs returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            printf("%5lld %5lld   %7.2f    %7.2f   %7.2f    %7.2f   %7.2f            %4lld   %8.2e   %s\n",
                   (long long) N, (long long) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   (long long) gesv_iter, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            magma_free_cpu( h_A     );
            magma_free_cpu( h_B     );
            magma_free_cpu( h_X     );
            magma_free_cpu( h_ipiv  );
            magma_free_cpu( h_workd );
            
            magma_free( d_A     );
            magma_free( d_B     );
            magma_free( d_X     );
            magma_free( d_ipiv  );
            magma_free( d_WORKS );
            magma_free( d_WORKD );
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
