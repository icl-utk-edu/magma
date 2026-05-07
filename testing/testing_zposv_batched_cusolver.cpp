/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Mark gates
   @author Azzam Haidar
   @author Tingxing Dong

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

#ifdef MAGMA_HAVE_CUDA
#include <cusolverDn.h>
#else
#include<rocblas/rocblas.h>
#include<rocsolver/rocsolver.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define PRECISION_z

#ifdef MAGMA_HAVE_CUDA
#define devsolver_handle_t  cusolverDnHandle_t
#define devsolver_create    cusolverDnCreate
#define devsolver_setstream cusolverDnSetStream
#define devsolver_destroy   cusolverDnDestroy

// =====> cusolver interface
#if   defined(PRECISION_z)
#define magma_zpotrf_batched_vendor           cusolverDnZpotrfBatched
#define magma_zpotrs_batched_vendor           cusolverDnZpotrsBatched
#define magma_zpotrf_vendor                   cusolverDnZpotrf
#define magma_zpotrs_vendor                   cusolverDnZpotrs
#define magma_zpotrf_buffer_size_vendor       cusolverDnZpotrf_bufferSize

#elif defined(PRECISION_c)
#define magma_cpotrf_batched_vendor           cusolverDnCpotrfBatched
#define magma_cpotrs_batched_vendor           cusolverDnCpotrsBatched
#define magma_cpotrf_vendor                   cusolverDnCpotrf
#define magma_cpotrs_vendor                   cusolverDnCpotrs
#define magma_cpotrf_buffer_size_vendor       cusolverDnCpotrf_bufferSize

#elif defined(PRECISION_d)
#define magma_dpotrf_batched_vendor           cusolverDnDpotrfBatched
#define magma_dpotrs_batched_vendor           cusolverDnDpotrsBatched
#define magma_dpotrf_vendor                   cusolverDnDpotrf
#define magma_dpotrs_vendor                   cusolverDnDpotrs
#define magma_dpotrf_buffer_size_vendor       cusolverDnDpotrf_bufferSize

#elif defined(PRECISION_s)
#define magma_spotrf_batched_vendor           cusolverDnSpotrfBatched
#define magma_spotrs_batched_vendor           cusolverDnSpotrsBatched
#define magma_spotrf_vendor                   cusolverDnSpotrf
#define magma_spotrs_vendor                   cusolverDnSpotrs
#define magma_spotrf_buffer_size_vendor       cusolverDnSpotrf_bufferSize

#else
#error "One of PRECISION_{s,d,c,z} must be defined."
#endif
#else
// =====> rocsolver interface
#define devsolver_handle_t  rocblas_handle
#define devsolver_create    rocblas_create_handle
#define devsolver_setstream rocblas_set_stream
#define devsolver_destroy   rocblas_destroy_handle

#if   defined(PRECISION_z)
#define magma_zpotrf_batched_vendor       rocsolver_zpotrf_batched
#define magma_zpotrs_batched_vendor       rocsolver_zpotrs_batched
#define magma_zposv_batched_vendor        rocsolver_zposv_batched
#define magma_zpotrf_vendor               rocsolver_zpotrf
#define magma_zpotrs_vendor               rocsolver_zpotrs
#define magma_zposv_vendor                rocsolver_zposv
#define magma_zrocblas_scalar             rocblas_double_complex
#define magma_zpotrf_buffer_size_vendor(...)

#elif defined(PRECISION_c)
#define magma_cpotrf_batched_vendor       rocsolver_cpotrf_batched
#define magma_cpotrs_batched_vendor       rocsolver_cpotrs_batched
#define magma_cposv_batched_vendor        rocsolver_cposv_batched
#define magma_cpotrf_vendor               rocsolver_cpotrf
#define magma_cpotrs_vendor               rocsolver_cpotrs
#define magma_cposv_vendor                rocsolver_cposv
#define magma_crocblas_scalar             rocblas_float_complex
#define magma_cpotrf_buffer_size_vendor(...)

#elif defined(PRECISION_d)
#define magma_dpotrf_batched_vendor       rocsolver_dpotrf_batched
#define magma_dpotrs_batched_vendor       rocsolver_dpotrs_batched
#define magma_dposv_batched_vendor        rocsolver_dposv_batched
#define magma_dpotrf_vendor               rocsolver_dpotrf
#define magma_dpotrs_vendor               rocsolver_dpotrs
#define magma_dposv_vendor                rocsolver_dposv
#define magma_drocblas_scalar             double
#define magma_dpotrf_buffer_size_vendor(...)

#elif defined(PRECISION_s)
#define magma_spotrf_batched_vendor       rocsolver_spotrf_batched
#define magma_spotrs_batched_vendor       rocsolver_spotrs_batched
#define magma_sposv_batched_vendor        rocsolver_sposv_batched
#define magma_spotrf_vendor               rocsolver_spotrf
#define magma_spotrs_vendor               rocsolver_spotrs
#define magma_sposv_vendor                rocsolver_sposv
#define magma_srocblas_scalar             float
#define magma_spotrf_buffer_size_vendor(...)

#else
#error "One of PRECISION_{s,d,c,z} must be defined."
#endif
#endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BATCH POTRF, POTRS, POSV
static void
magma_zpotrf_batched_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    int *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverStatus_t s = magma_zpotrf_batched_vendor( handle, uplo_, (int)n,
                                 (cuDoubleComplex**)dA_array,  (int)ldda,
                                 (int*)dinfo_array, (int)batchCount );
    if(s != CUSOLVER_STATUS_SUCCESS) printf("error in cusolver");
    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zpotrf_batched_vendor( handle, uplo_, (int)n,
                                 (magma_zrocblas_scalar *const *)dA_array, (const int)ldda,
                                 (int*)dinfo_array, (int)batchCount );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zpotrs_batched_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex               **dB_array, magma_int_t lddb,
    int *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    // cusolver batch POTRS works only for single RHS
    if( nrhs < 2 ) {
        cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        magma_zpotrs_batched_vendor( handle, uplo_, (int)n, (int)nrhs,
                                     (cuDoubleComplex**)dA_array,  (int)ldda,
                                     (cuDoubleComplex**)dB_array,  (int)lddb,
                                     (int*)dinfo_array, (int)batchCount );
    }
    else {
        magmaDoubleComplex alpha = MAGMA_Z_ONE;
        cublasZtrsmBatched(
                magma_queue_get_cublas_handle( queue ), cublas_side_const(MagmaLeft), cublas_uplo_const(uplo),
                cublas_trans_const(MagmaNoTrans), cublas_diag_const(MagmaNonUnit),
                int(n), int(nrhs), (const cuDoubleComplex*)&alpha,
                (const cuDoubleComplex**) dA_array, int(ldda),
                (      cuDoubleComplex**) dB_array, int(lddb), int(batchCount) );
        cublasZtrsmBatched(
                magma_queue_get_cublas_handle( queue ), cublas_side_const(MagmaLeft), cublas_uplo_const(uplo),
                cublas_trans_const(MagmaConjTrans), cublas_diag_const(MagmaNonUnit),
                int(n), int(nrhs), (const cuDoubleComplex*)&alpha,
                (const cuDoubleComplex**) dA_array, int(ldda),
                (      cuDoubleComplex**) dB_array, int(lddb), int(batchCount) );
    }
    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zpotrs_batched_vendor( handle, uplo_, (int)n, (int)nrhs,
                                 (magma_zrocblas_scalar *const *)dA_array, (const int)ldda,
                                 (magma_zrocblas_scalar       **)dB_array, (const int)lddb,
                                 (int)batchCount );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zposvs_batched_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex               **dB_array, magma_int_t lddb,
    int *dinfo_array, magma_int_t batchCount,
    magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    magma_zpotrf_batched_vendor_wrapper(uplo, n, dA_array, ldda, dinfo_array, batchCount, queue, handle);
    magma_zpotrs_batched_vendor_wrapper(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue, handle);

    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zposv_batched_vendor( handle, uplo_, (int)n, (int)nrhs,
                                 (magma_zrocblas_scalar *const *)dA_array, (const int)ldda,
                                 (magma_zrocblas_scalar       **)dB_array, (const int)lddb,
                                 (int *)dinfo_array, (int)batchCount );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// NON-BATCH POTRF, POTRS, POSV
static void
magma_zpotrf_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex const *dA, magma_int_t ldda,
    magmaDoubleComplex       *dW, magma_int_t lwork,
    int *dinfo, magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverStatus_t s = magma_zpotrf_vendor( handle, uplo_, (int)n,
                            (cuDoubleComplex*)dA,  (int)ldda,
                            (cuDoubleComplex*)dW,  (int)lwork,
                            (int*)dinfo );
    if(s != CUSOLVER_STATUS_SUCCESS) printf("error in cusolver");

    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zpotrf_vendor( handle, uplo_, (int)n,
                        (magma_zrocblas_scalar *)dA, (const int)ldda, (int*)dinfo );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zpotrs_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex const *dA, magma_int_t ldda,
    magmaDoubleComplex       *dB, magma_int_t lddb,
    int *dinfo, magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    magma_zpotrs_vendor( handle, uplo_, (int)n, (int)nrhs,
                         (cuDoubleComplex*)dA,  (int)ldda,
                         (cuDoubleComplex*)dB,  (int)lddb, (int*)dinfo );
    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zpotrs_vendor( handle, uplo_, (int)n, (int)nrhs,
                         (magma_zrocblas_scalar *)dA, (const int)ldda,
                         (magma_zrocblas_scalar *)dB, (const int)lddb );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zposvs_vendor_wrapper(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex const *dA, magma_int_t ldda,
    magmaDoubleComplex       *dB, magma_int_t lddb,
    magmaDoubleComplex       *dW, magma_int_t lwork,
    int *dinfo, magma_queue_t queue, devsolver_handle_t handle)
{

    #ifdef MAGMA_HAVE_CUDA
    magma_zpotrf_vendor_wrapper(uplo, n, dA, ldda, dW, lwork, dinfo, queue, handle);
    magma_zpotrs_vendor_wrapper(uplo, n, nrhs, dA, ldda, dB, lddb, dinfo, queue, handle);

    #else
    const rocblas_fill uplo_ = (uplo == MagmaLower) ? rocblas_fill_lower : rocblas_fill_upper;
    magma_zposv_vendor( handle, uplo_, (int)n, (int)nrhs,
                        (magma_zrocblas_scalar *)dA, (const int)ldda,
                        (magma_zrocblas_scalar *)dB, (const int)lddb, (int *)dinfo );
    #endif
}

////////////////////////////////////////////////////////////////////////////////
static void
magma_zpotrf_ws_vendor(
    devsolver_handle_t handle, magma_uplo_t uplo,
    magma_int_t n, magmaDoubleComplex *A, magma_int_t ldda, magma_int_t *lwork )
{
    #ifdef MAGMA_HAVE_CUDA
    cublasFillMode_t uplo_ = (uplo == MagmaLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    magma_zpotrf_buffer_size_vendor(handle, uplo_, (int)n, A, (int)ldda, (int*)lwork );
    #else
    lwork = 0;
    #endif
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zposv_batched
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
    magma_int_t *cpu_info;
    magma_int_t *dinfo_array;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info;
    size_t sizeA, sizeB, sizeA_dev, sizeB_dev;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dB_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    magma_queue_t queue = opts.queue;

    nrhs = opts.nrhs;
    batchCount = opts.batchcount;

    const magma_int_t nqueues = 16;
    magma_queue_t      queues[nqueues];
    devsolver_handle_t handles[nqueues];
    if(opts.version == 3) {
        printf("Creating cusolver/rocsolver handles ... ");
        magma_device_t cdev;
        magma_getdevice( &cdev );
        for(int s=0; s<nqueues; s++){
            magma_queue_create( cdev, &queues[s] );
            devsolver_create(&handles[s]);
            #ifdef MAGMA_HAVE_CUDA
            devsolver_setstream(handles[s], magma_queue_get_cuda_stream(queues[s]));
            #else
            devsolver_setstream(handles[s], magma_queue_get_hip_stream(queues[s]));
            #endif
        }
        printf("done\n");
    }

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% BatchCount   N  NRHS   CPU Gflop/s (ms)   GPU Gflop/s (ms)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%==========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb   = ldda;
            gflops = ( FLOPS_ZPOTRF( N) + FLOPS_ZPOTRS( N, nrhs ) ) / 1e9 * batchCount;

            sizeA = (size_t)lda * (size_t)N    * (size_t)batchCount;
            sizeB = (size_t)ldb * (size_t)nrhs * (size_t)batchCount;

            sizeA_dev = (size_t)ldda * (size_t)N    * (size_t)batchCount;
            sizeB_dev = (size_t)lddb * (size_t)nrhs * (size_t)batchCount;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A, sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B, sizeB ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_X, sizeB ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));

            TESTING_CHECK( magma_zmalloc( &d_A, sizeA_dev ));
            TESTING_CHECK( magma_zmalloc( &d_B, sizeB_dev ));
            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array, batchCount * sizeof(magmaDoubleComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array, batchCount * sizeof(magmaDoubleComplex*) ));

            /* Initialize the matrices */
            size_t size_stride = 2e9;
            for(size_t isize = 0; isize < sizeA; isize += size_stride) {
                magma_int_t size_a = (magma_int_t)min(size_stride, sizeA-isize);
                lapackf77_zlarnv( &ione, ISEED, &size_a, h_A + isize );
            }

            for(size_t isize = 0; isize < sizeB; isize += size_stride) {
                magma_int_t size_b = (magma_int_t)min(size_stride, sizeB-isize);
                lapackf77_zlarnv( &ione, ISEED, &size_b, h_B + isize );
            }

            for (int i=0; i < batchCount; i++) {
                magmaDoubleComplex *hAtmp = h_A + (size_t)i * (size_t)lda * (size_t)N;
                magma_zmake_hpd( N, hAtmp, lda ); // need modification
            }

            for(magma_int_t s = 0; s < batchCount; s++) {
                magmaDoubleComplex *Asrc = h_A + (size_t)s * (size_t)lda  * (size_t)N;
                magmaDoubleComplex *Adst = d_A + (size_t)s * (size_t)ldda * (size_t)N;
                magma_zsetmatrix( N, N, Asrc, lda, Adst, ldda, opts.queue );
            }
            magma_zsetmatrix( N, nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N, batchCount, queue );
            magma_zset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, queue );

            if(opts.version == 1) {
                gpu_time = magma_sync_wtime( opts.queue );
                info = magma_zposv_batched(opts.uplo, N, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if(opts.version == 2){
                info = 0;
                devsolver_handle_t handle;
                devsolver_create(&handle);
                #ifdef MAGMA_HAVE_CUDA
                devsolver_setstream(handle, magma_queue_get_cuda_stream(opts.queue));
                #else
                devsolver_setstream(handle, magma_queue_get_hip_stream(opts.queue));
                #endif

                gpu_time = magma_sync_wtime( opts.queue );
                magma_zposvs_batched_vendor_wrapper( opts.uplo, N, nrhs, dA_array,ldda, dB_array, lddb, dinfo_array, batchCount, opts.queue, handle);
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                devsolver_destroy(handle);
            }
            else if(opts.version == 3) {
                info = 0;
                magmaDoubleComplex *dW = NULL;
                magma_int_t lwork      = 0;
                magma_zpotrf_ws_vendor( handles[0], opts.uplo, N, d_A, ldda, &lwork );
                if(lwork > 0) {
                    TESTING_CHECK( magma_zmalloc(&dW, (size_t)batchCount*(size_t)lwork) );
                }

                gpu_time = magma_sync_wtime( queues[0] );
                for(int s=0; s<batchCount; s++){
                    magma_zposvs_vendor_wrapper(
                        opts.uplo, N, nrhs,
                        d_A + (size_t)s*(size_t)ldda*(size_t)N,    ldda,
                        d_B + (size_t)s*(size_t)lddb*(size_t)nrhs, lddb,
                        dW  + (size_t)s*(size_t)lwork, lwork,
                        dinfo_array + s,
                        queues[s%nqueues], handles[s%nqueues]);
                }

                for(int s=0; s<nqueues; s++){
                    magma_queue_sync( queues[s] );
                }
                gpu_time = magma_sync_wtime( queues[0] ) - gpu_time;

                if(dW != NULL) magma_free(dW);
            }
            gpu_perf = gflops / gpu_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zposv_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_zposv_batched returned argument error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );

            error = 0;
            for (magma_int_t s=0; s < batchCount; s++)
            {
                Anorm = lapackf77_zlange("I", &N, &N,    h_A + (size_t)s * (size_t)lda * (size_t)N,    &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X + (size_t)s * (size_t)ldb * (size_t)nrhs, &ldb, work);

                blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A + (size_t)s * (size_t)lda * (size_t)N, &lda,
                                       h_X + (size_t)s * (size_t)ldb * (size_t)nrhs, &ldb,
                           &c_neg_one, h_B + (size_t)s * (size_t)ldb * (size_t)nrhs, &ldb);

                Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B + (size_t)s * (size_t)ldb * (size_t)nrhs, &ldb, work);
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
            if(opts.niter > 1 && iter == 0) printf("# ");

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
                    lapackf77_zposv( lapack_uplo_const(opts.uplo), &N, &nrhs, h_A + (size_t)s * (size_t)lda * (size_t)N,    &lda,
                                                                              h_B + (size_t)s * (size_t)ldb * (size_t)nrhs, &ldb, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_zposv matrix %lld returned error %lld: %s.\n",
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
                        cpu_perf, cpu_time*1000.,
                        gpu_perf, gpu_time*1000.,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        gpu_perf, gpu_time*1000.,
                        error, (okay ? "ok" : "failed"));
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( cpu_info );

            magma_free( d_A );
            magma_free( d_B );

            magma_free( dinfo_array );

            magma_free( dA_array );
            magma_free( dB_array );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "#\n" );
        }
    }

    if(opts.version == 3) {
        for(int i=0; i<nqueues; i++){
            magma_queue_destroy( queues[i] );
            devsolver_destroy(  handles[i] );
        }
    }


    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
