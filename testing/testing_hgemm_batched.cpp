/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
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
   (1) converts a matrix from float to half on the GPU 
   (2) convert back to float and sent it to the CPU to compute the correct norm
*/
void preprocess_matrix( 
            magma_int_t M, magma_int_t N, 
            float     *hA, magma_int_t lda, 
            magmaHalf *dA, magma_int_t ldda, 
            magma_queue_t queue )
{
    float *dwork;
    magma_int_t info = 0; 

    
    TESTING_CHECK( magma_smalloc(&dwork, lda*N) );               // alloc. dwork on GPU
    magma_ssetmatrix(M, N, hA, lda, dwork, lda, queue);          // send to the GPU
    magmablas_slag2h(M, N, dwork, lda, dA, ldda, &info, queue);  // convert: s -> h
    if(info != 0)printf("preprocess_matrix: error at slag2h\n"); // check
    magmablas_hlag2s(M, N, dA, ldda, dwork, lda, queue );        // convert back: h -> hc
    magma_sgetmatrix(M, N, dwork, lda, hA, lda, queue);          // send to the CPU after conversion

    // free workspace
    magma_free( dwork );
}

/* ////////////////////////////////////////////////////////////////////////////
   (1) converts a matrix from half to float on the GPU 
   (2) send the converted matrix to the CPU
*/
void postprocess_matrix( 
            magma_int_t M, magma_int_t N, 
            magmaHalf *dA, magma_int_t ldda, 
            float     *hA, magma_int_t lda, 
            magma_queue_t queue )
{
    float *dwork;

    TESTING_CHECK( magma_smalloc(&dwork, lda*N) );
    magmablas_hlag2s(M, N, dA, ldda, dwork, lda, queue ); // convert h -> s
    magma_sgetmatrix(M, N, dwork, lda, hA, lda, queue);   // send to CPU 

    magma_free( dwork );
}

/* ////////////////////////////////////////////////////////////////////////////
   (1) converts a matrix from float to half on the GPU 
   (2) convert back to float and sent it to the CPU to compute the correct norm
   (3) matrices in the batch are assumed to be within a fixed stride of 
       lda*N (cpu) or ldda*N (gpu)
*/
void preprocess_matrix_batched( 
            magma_int_t M, magma_int_t N, 
            float     *hA, magma_int_t lda, 
            magmaHalf *dA, magma_int_t ldda, 
            magma_int_t batchCount, magma_queue_t queue )
{
    float *dwork;
    magma_int_t info = 0; 

    TESTING_CHECK( magma_smalloc(&dwork, batchCount*lda*N) );               // alloc. dwork on GPU
    magma_ssetmatrix(M, batchCount*N, hA, lda, dwork, lda, queue);          // send to the GPU
    magmablas_slag2h(M, batchCount*N, dwork, lda, dA, ldda, &info, queue);  // convert: s -> h
    if(info != 0)printf("preprocess_matrix: error at slag2h\n");            // check
    magmablas_hlag2s(M, batchCount*N, dA, ldda, dwork, lda, queue );        // convert back: h -> hc
    magma_sgetmatrix(M, batchCount*N, dwork, lda, hA, lda, queue);          // send to the CPU after conversion

    // free workspace
    magma_free( dwork );
}

/* ////////////////////////////////////////////////////////////////////////////
   (1) converts a matrix from half to float on the GPU 
   (2) send the converted matrix to the CPU
   (3) matrices in the batch are assumed to be within a fixed stride of 
       lda*N (cpu) or ldda*N (gpu)
*/
void postprocess_matrix_batched( 
            magma_int_t M, magma_int_t N, 
            magmaHalf *dA, magma_int_t ldda, 
            float     *hA, magma_int_t lda, 
            magma_int_t batchCount, magma_queue_t queue )
{
    float *dwork;

    TESTING_CHECK( magma_smalloc(&dwork, batchCount*lda*N) );
    magmablas_hlag2s(M, batchCount*N, dA, ldda, dwork, lda, queue ); // convert h -> s
    magma_sgetmatrix(M, batchCount*N, dwork, lda, hA, lda, queue);   // send to CPU 

    magma_free( dwork );
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing hgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          error, cublas_error, magma_error, normalize, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    float *hA, *hB, *hC, *hC_magma, *hC_cublas;
    magmaHalf *dA, *dB, *dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magmaHalf alpha = 0.29; 
    magmaHalf beta  = -0.48;

    magmaHalf **dA_array = NULL;
    magmaHalf **dB_array = NULL;
    magmaHalf **dC_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check; // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    
    float *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Cnorm, batchCount ));

    // See testing_sgemm about tolerance.
    // also see: https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic
    float eps = (float)(0.00097656);
    float tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "%% transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB));
    printf("%% BatchCount     M     N     K   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)   CPU Gflop/s   (ms)     MAGMA error   CUBLAS error\n");
    printf("%%                                (Half Precision)      (Half Precision)     (Single Precision)                               \n");
    printf("%%============================================================================================================================\n");
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            }
            else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            }
            else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An*batchCount;
            sizeB = ldb*Bn*batchCount;
            sizeC = ldc*N*batchCount;

            TESTING_CHECK( magma_smalloc_cpu( &hA,  sizeA ));
            TESTING_CHECK( magma_smalloc_cpu( &hB,  sizeB ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,  sizeC  ));
            TESTING_CHECK( magma_smalloc_cpu( &hC_magma,  sizeC  ));
            TESTING_CHECK( magma_smalloc_cpu( &hC_cublas, sizeC  ));

            TESTING_CHECK( magma_malloc( (void**) &dA, ldda*An*batchCount * sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**) &dB, lddb*Bn*batchCount * sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**) &dC, lddc*N*batchCount  * sizeof(magmaHalf) ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array, batchCount * sizeof(magmaHalf*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array, batchCount * sizeof(magmaHalf*) ));
            TESTING_CHECK( magma_malloc( (void**) &dC_array, batchCount * sizeof(magmaHalf*) ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );

            /* preprocessing assumes one big matrix */
            preprocess_matrix(Am, batchCount*An, hA, lda, dA, ldda, opts.queue );
            preprocess_matrix(Bm, batchCount*Bn, hB, ldb, dB, lddb, opts.queue );
            preprocess_matrix( M, batchCount*N,  hC, ldc, dC, lddc, opts.queue );

            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_slange( "F", &Am, &An, &hA[s*lda*An], &lda, work );
                Bnorm[s] = lapackf77_slange( "F", &Bm, &Bn, &hB[s*ldb*Bn], &ldb, work );
                Cnorm[s] = lapackf77_slange( "F", &M,  &N,  &hC[s*ldc*N],  &ldc, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_hset_pointer( dA_array, dA, ldda, 0, 0, ldda*An, batchCount, opts.queue );
            magma_hset_pointer( dB_array, dB, lddb, 0, 0, lddb*Bn, batchCount, opts.queue );
            magma_hset_pointer( dC_array, dC, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_hgemm_batched( 
                        opts.transA, opts.transB, 
                        M, N, K, 
                        alpha, dA_array, ldda, 
                               dB_array, lddb, 
                        beta,  dC_array, lddc, batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            postprocess_matrix(M, batchCount*N, dC, lddc, hC_magma, ldc, opts.queue );

            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            preprocess_matrix( M, batchCount*N,  hC, ldc, dC, lddc, opts.queue );
            cublasSetMathMode(opts.handle, CUBLAS_TENSOR_OP_MATH);
            
            cublas_time = magma_sync_wtime( opts.queue );
            cublasHgemmBatched(opts.handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB),
                               int(M), int(N), int(K),
                               &alpha, (const magmaHalf**)dA_array, int(ldda),
                                       (const magmaHalf**)dB_array, int(lddb),
                               &beta,                     dC_array, int(lddc), int(batchCount) );
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            postprocess_matrix(M, batchCount*N, dC, lddc, hC_cublas, ldc, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                float alpha_r32 = (float)alpha;
                float beta_r32  = (float)beta;
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++) {
                    blasf77_sgemm( lapack_trans_const(opts.transA),
                                   lapack_trans_const(opts.transB),
                                   &M, &N, &K,
                                   &alpha_r32, hA + i*lda*An, &lda,
                                           hB + i*ldb*Bn, &ldb,
                                   &beta_r32,  hC + i*ldc*N, &ldc );
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
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|)
                error = 0;
                magma_error = 0;
                cublas_error = 0;

                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(float(K+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc*N;
                    blasf77_saxpy( &Csize, &c_neg_one, &hC[s*ldc*N], &ione, &hC_magma[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &hC_magma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    
                    // cublas error
                    blasf77_saxpy( &Csize, &c_neg_one, &hC[s*ldc*N], &ione, &hC_cublas[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &hC_cublas[s*ldc*N], &ldc, work )
                          / normalize;
                    cublas_error = magma_max_nan( error, cublas_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld %5lld    %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)       %8.2e      %8.2e   %s\n",
                       (long long) batchCount, (long long) M, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error, (okay ? "ok" : "failed") );
            }
            else {
                // compute error compared cublas
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|)
                error = 0;
                magma_error = 0;

                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(float(K+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc*N;
                    blasf77_saxpy( &Csize, &c_neg_one, &hC_cublas[s*ldc*N], &ione, &hC_magma[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &hC_magma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }
                magma_error = magma_max_nan( error, magma_error );

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld %5lld    %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )       %8.2e        ---      %s\n",
                       (long long) batchCount, (long long) M, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       magma_error, (okay ? "ok" : "failed") );
            }
            
            magma_free_cpu( hA  );
            magma_free_cpu( hB  );
            magma_free_cpu( hC  );
            magma_free_cpu( hC_magma  );
            magma_free_cpu( hC_cublas );

            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            magma_free( dA_array );
            magma_free( dB_array );
            magma_free( dC_array );

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );
    magma_free_cpu( Cnorm );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
