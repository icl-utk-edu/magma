/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
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

// version 1
static magma_int_t
magma_sgemm_fp16_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha, float* dA, magmaHalf* dhA, magma_int_t ldda,
                 float* dB, magmaHalf* dhB, magma_int_t lddb,
    float beta,  float* dC, magma_int_t lddc,
    magma_queue_t queue )
{
    #ifdef MAGMA_HAVE_CUDA
    cublasGemmEx( magma_queue_get_cublas_handle( queue ),
                  cublas_trans_const( transA ), cublas_trans_const( transB ),
                  (int)m, (int)n, (int)k,
                  (const void*) &alpha, (const void*) dA, CUDA_R_32F, (int)ldda,
                                        (const void*) dB, CUDA_R_32F, (int)lddb,
                  (const void*) &beta,  (      void*) dC, CUDA_R_32F, (int)lddc,
                  CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
    #elif MAGMA_HAVE_HIP
    magma_int_t hinfo = 0;
    magma_int_t Am = (transA == MagmaNoTrans) ? m : k;
    magma_int_t An = (transA == MagmaNoTrans) ? k : m;
    magma_int_t Bm = (transB == MagmaNoTrans) ? k : n;
    magma_int_t Bn = (transB == MagmaNoTrans) ? n : k;
    magmablas_slag2h(Am, An, dA, ldda, dhA, ldda, &hinfo, queue);
    magmablas_slag2h(Bm, Bn, dB, lddb, dhB, lddb, &hinfo, queue);

    hipblasGemmEx( magma_queue_get_hipblas_handle( queue ),
		           hipblas_trans_const( transA ), hipblas_trans_const( transB ),
		           int(m), int(n), int(k),
		           (void*)&alpha, (void*)dhA, HIPBLAS_R_16F, (int)ldda,
                                  (void*)dhB, HIPBLAS_R_16F, (int)lddb,
		           (void*)&beta,  (void*)dC,  HIPBLAS_R_32F, (int)lddc,
		           HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
    #endif
    return 0;
}

// version 2 -- no change for hip
static magma_int_t
magma_sgemm_fp16_v2(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha, float* dA, magmaHalf* dhA, magma_int_t ldda,
                 float* dB, magmaHalf* dhB, magma_int_t lddb,
    float beta,  float* dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t hinfo = 0;
    magma_int_t Am = (transA == MagmaNoTrans) ? m : k;
    magma_int_t An = (transA == MagmaNoTrans) ? k : m;
    magma_int_t Bm = (transB == MagmaNoTrans) ? k : n;
    magma_int_t Bn = (transB == MagmaNoTrans) ? n : k;
    magmablas_slag2h(Am, An, dA, ldda, dhA, ldda, &hinfo, queue);
    magmablas_slag2h(Bm, Bn, dB, lddb, dhB, lddb, &hinfo, queue);

    #ifdef MAGMA_HAVE_CUDA
    cublasGemmEx( magma_queue_get_cublas_handle( queue ),
                  cublas_trans_const( transA ), cublas_trans_const( transB ),
                  (int)m, (int)n, (int)k,
                  (const void*) &alpha, (const void*) dhA, CUDA_R_16F, (int)ldda,
                                        (const void*) dhB, CUDA_R_16F, (int)lddb,
                  (const void*) &beta,  (      void*) dC, CUDA_R_32F, (int)lddc,
                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
    #elif MAGMA_HAVE_HIP
    hipblasGemmEx( magma_queue_get_hipblas_handle( queue ),
		           hipblas_trans_const( transA ), hipblas_trans_const( transB ),
		           int(m), int(n), int(k),
		           (void*)&alpha, (void*)dhA, HIPBLAS_R_16F, (int)ldda,
                                  (void*)dhB, HIPBLAS_R_16F, (int)lddb,
		           (void*)&beta,  (void*)dC,  HIPBLAS_R_32F, (int)lddc,
		           HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
    #endif
    return 0;
}



/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgemm
*/
int main( int argc, char** argv)
{
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    float          magma_error, dev_error, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    float *hA, *hB, *hC, *hCmagma, *hCdev;
    magmaFloat_ptr dA, dB, dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    magmaHalf *dhA=NULL, *dhB=NULL;

    // used only with CUDA
    MAGMA_UNUSED( magma_perf );
    MAGMA_UNUSED( magma_time );
    MAGMA_UNUSED( magma_error );

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // Allow 3*eps; real needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;

    #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
        // for CUDA/HIP, we can check MAGMA vs. CUBLAS/hipBLAS, without running LAPACK
        printf("%% If running lapack (option --lapack), MAGMA and %s error are both computed\n"
               "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
                g_platform_str, g_platform_str );
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  %s error\n",
                g_platform_str, g_platform_str );
    #else
        // for others, we need LAPACK for check
        opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   %s Gflop/s (ms)   CPU Gflop/s (ms)  %s error\n",
                g_platform_str, g_platform_str );
    #endif
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }

            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;

            TESTING_CHECK( magma_smalloc_cpu( &hA,       lda*An ));
            TESTING_CHECK( magma_smalloc_cpu( &hB,       ldb*Bn ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,       ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCmagma,  ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCdev,    ldc*N  ));

            TESTING_CHECK( magma_smalloc( &dA, ldda*An ));
            TESTING_CHECK( magma_smalloc( &dB, lddb*Bn ));
            TESTING_CHECK( magma_smalloc( &dC, lddc*N  ));

            TESTING_CHECK( magma_malloc( (void**)&dhA, ldda*An*sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**)&dhB, lddb*Bn*sizeof(magmaHalf) ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );

            magma_ssetmatrix( Am, An, hA, lda, dA(0,0), ldda, opts.queue );
            magma_ssetmatrix( Bm, Bn, hB, ldb, dB(0,0), lddb, opts.queue );

            // for error checks
            float Anorm = lapackf77_slange( "F", &Am, &An, hA, &lda, work );
            float Bnorm = lapackf77_slange( "F", &Bm, &Bn, hB, &ldb, work );
            float Cnorm = lapackf77_slange( "F", &M,  &N,  hC, &ldc, work );

            /* =====================================================================
               Performs operation using MAGMABLAS (currently only with CUDA)
               =================================================================== */
            #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
                magma_ssetmatrix( M, N, hC, ldc, dC, lddc, opts.queue );

                magma_flush_cache( opts.cache );
                magma_time = magma_sync_wtime( opts.queue );
                if(opts.version == 1) {
                    magma_sgemm_fp16_v1( opts.transA, opts.transB, M, N, K,
                                 alpha, dA, dhA, ldda,
                                        dB, dhB, lddb,
                                 beta,  dC,      lddc,
                                 opts.queue );
                }
                else {
                    magma_sgemm_fp16_v2( opts.transA, opts.transB, M, N, K,
                                 alpha, dA, dhA, ldda,
                                        dB, dhB, lddb,
                                 beta,  dC,      lddc,
                                 opts.queue );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;

                magma_sgetmatrix( M, N, dC, lddc, hCmagma, ldc, opts.queue );
            #endif

            /* =====================================================================
               Performs operation using CUBLAS / hipBLAS
               =================================================================== */
            magma_ssetmatrix( M, N, hC, ldc, dC(0,0), lddc, opts.queue );

            magma_flush_cache( opts.cache );
            dev_time = magma_sync_wtime( opts.queue );
            magma_sgemm( opts.transA, opts.transB, M, N, K,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb,
                         beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;

            magma_sgetmatrix( M, N, dC(0,0), lddc, hCdev, ldc, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                magma_flush_cache( opts.cache );
                cpu_time = magma_wtime();
                blasf77_sgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // Compute forward error bound (see Higham, 2002, sec. 3.5),
                // modified to include alpha, beta, and input C.
                // ||R_magma - R_ref||_p / (gamma_{K+2} |alpha| ||A||_p ||B||_p + 2 |beta| ||C||_p ) < eps/2.
                // This should work with p = 1, inf, fro, but numerical tests
                // show p = 1, inf are very spiky and sometimes exceed eps.
                // We use gamma_n = sqrt(n)*u instead of n*u/(1-n*u), since the
                // former accurately represents statistical average rounding.
                // We allow a slightly looser tolerance.

                // use LAPACK for R_ref
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = lapackf77_slange( "F", &M, &N, hCdev, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
                    blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCmagma, &ione );
                    magma_error = lapackf77_slange( "F", &M, &N, hCmagma, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                    bool okay = (magma_error < tol && dev_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e   %s\n",
                           (long long) M, (long long) N, (long long) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           magma_error, dev_error,
                           (okay ? "ok" : "failed"));
                #else
                    bool okay = (dev_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           dev_error,
                           (okay ? "ok" : "failed"));
                #endif
            }
            else {
                #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)

                    // use cuBLAS for R_ref (currently only with CUDA)
                    blasf77_saxpy( &sizeC, &c_neg_one, hCdev, &ione, hCmagma, &ione );
                    magma_error = lapackf77_slange( "F", &M, &N, hCmagma, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

                    bool okay = (magma_error < tol);
                    status += ! okay;
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )    %8.2e        ---    %s\n",
                           (long long) M, (long long) N, (long long) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           magma_error,
                           (okay ? "ok" : "failed"));
                #else
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )       ---\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time );
                #endif
            }

            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCmagma  );
            magma_free_cpu( hCdev    );

            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            magma_free( dhA );
            magma_free( dhB );
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
