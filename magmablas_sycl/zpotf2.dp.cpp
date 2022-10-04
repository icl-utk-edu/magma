/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @precisions normal z -> s d c
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define COMPLEX

#define zdotc_max_bs 512  // 512 is max threads for 1.x cards

void zpotf2_zdscal( magma_int_t n, magmaDoubleComplex *x, magma_int_t incx, magma_queue_t queue );
void zpotf2_zdotc(  magma_int_t n, magmaDoubleComplex *x, magma_int_t incx, magma_queue_t queue );

#ifdef COMPLEX
void magmablas_zlacgv( magma_int_t n, magmaDoubleComplex *x, magma_int_t incx, magma_queue_t queue );
#endif


// TODO: this function could be in .cpp file -- it has no CUDA code in it.
/***************************************************************************//**
    Purpose
    -------

    zpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U**H * U,  if UPLO = MagmaUpper, or
        A = L  * L**H, if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0 and N <= 512.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U  or A = L * L**H.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    @ingroup magma_potf2
*******************************************************************************/
extern "C" magma_int_t
magma_zpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
#define dA(i_, j_)  (dA + (i_) + (j_)*ldda)

    magma_int_t j;

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0 || n > zdotc_max_bs) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0) {
        return *info;
    }

    /*
    DPCT1064:1283: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;

    if (uplo == MagmaUpper) {
        for (j = 0; j < n; j++) {
            zpotf2_zdotc( j, dA(0,j), 1, queue ); // including zdotc product and update a(j,j)
            if (j < n) {
                #ifdef COMPLEX
                magmablas_zlacgv( j, dA(0, j), 1, queue );
                #endif
                magma_zgemv( MagmaTrans, j, n-j-1,
                             alpha, dA(0, j+1), ldda,
                                    dA(0, j),   1,
                             beta,  dA(j, j+1), ldda, queue );

                #ifdef COMPLEX
                magmablas_zlacgv( j, dA(0, j), 1, queue );
                #endif
                zpotf2_zdscal( n-j, dA(j,j), ldda, queue );
            }
        }
    }
    else {
        for (j = 0; j < n; j++) {
            zpotf2_zdotc( j, dA(j,0), ldda, queue ); // including zdotc product and update a(j,j)
            if (j < n) {
                #ifdef COMPLEX
                magmablas_zlacgv( j, dA(j, 0), ldda, queue );
                #endif
                magma_zgemv( MagmaNoTrans, n-j-1, j,
                             alpha, dA(j+1, 0), ldda,
                                    dA(j,0),    ldda,
                             beta,  dA(j+1, j), 1, queue );

                #ifdef COMPLEX
                magmablas_zlacgv( j, dA(j, 0), ldda, queue );
                #endif
                zpotf2_zdscal( n-j, dA(j,j), 1, queue );
            }
        }
    }

    return *info;
}

#define zdscal_bs  32
#define zdotc_bs  512
#define zlacgv_bs 512


void kernel_zdotc(int n, magmaDoubleComplex *x, int incx, int threadSize,
                  sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{

    // dynamically allocated shared memory, set to size number of threads when the kernel is launched.
    // See CUDA Guide B.2.3
    auto shared_data = (double *)dpct_local;

    int tx = item_ct1.get_local_id(2);

    double *sdata = shared_data;

    /*
    DPCT1064:1285: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex res = MAGMA_Z_ZERO;

    if (tx < n) {
        res = x[tx*incx];
    }

    sdata[tx] = MAGMA_Z_REAL(res * MAGMA_Z_CONJ(res));

    /*
    DPCT1065:1284: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int s = item_ct1.get_local_range(2) / 2; s > 32; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx+s];
        }
        /*
        DPCT1065:1286: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (tx < 32) {
        volatile double* smem = sdata;
        smem[tx] += smem[tx+32];
        smem[tx] += smem[tx+16];
        smem[tx] += smem[tx+8];
        smem[tx] += smem[tx+4];
        smem[tx] += smem[tx+2];
        smem[tx] += smem[tx+1];
    }

    if (tx == 0) {
        double xreal = MAGMA_Z_REAL(x[n*incx]);
        x[n * incx] = MAGMA_Z_MAKE(sycl::sqrt(xreal - sdata[0]), 0);
    }
}

void zpotf2_zdotc(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t incx,
    magma_queue_t queue )
{
    /*
    Specialized Zdotc
    1) performs zdotc sum = x[0:n-1]*conj(x[0:n-1])
    2) updates x[n] = sqrt(x[n]-sum);

    */
    if (n > zdotc_max_bs) {
        fprintf( stderr, "n = %lld > %lld is not supported in zpotf2_zdotc\n",
                 (long long) n, (long long) zdotc_max_bs );
        return;
    }
    int threadSize;

    if (n <= 1024 && n > 512) {
        threadSize = 1024;
    }
    else if (n <= 512 && n > 256 ) {
        threadSize = 512;
    }
    else if (n <= 256 && n > 128) {
        threadSize = 256;
    }
    else if (n <= 128 && n > 64) {
        threadSize = 128;
    }
    else {
        threadSize = 64;
    }

    /*
    DPCT1083:1288: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    size_t shmem = threadSize * sizeof(double);
    /*
    DPCT1049:1287: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, threadSize),
                                           sycl::range<3>(1, 1, threadSize)),
                         [=](sycl::nd_item<3> item_ct1) {
                             kernel_zdotc(n, x, incx, threadSize, item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                         });
    });
}

void kernel_zdscal(int n, magmaDoubleComplex *x, int incx,
                   sycl::nd_item<3> item_ct1, magmaDoubleComplex *factor)
{
    int id = item_ct1.get_group(2) * zdscal_bs + item_ct1.get_local_id(2);

    if (item_ct1.get_local_id(2) == 0) {
        /*
        DPCT1064:1290: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        *factor = MAGMA_Z_MAKE(1.0 / x()(x[0]), 0.0);
    }

    /*
    DPCT1065:1289: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if ( id < n && id > 0) {
        x[id * incx] = x[id * incx] * *factor;
    }
}


void zpotf2_zdscal(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t incx,
    magma_queue_t queue )
{
    /* Specialized zdscal perform x[1:n-1] / x[0] */
    sycl::range<3> threads(1, 1, zdscal_bs);
    int num_blocks = magma_ceildiv( n, zdscal_bs );
    sycl::range<3> grid(1, 1, num_blocks);
    /*
    DPCT1049:1291: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<magmaDoubleComplex, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            factor_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             kernel_zdscal(n, x, incx, item_ct1,
                                           factor_acc_ct1.get_pointer());
                         });
    });
}


#ifdef COMPLEX

void kernel_zlacgv(int n, magmaDoubleComplex *x, int incx,
                   sycl::nd_item<3> item_ct1)
{
    int id = item_ct1.get_group(2) * zlacgv_bs + item_ct1.get_local_id(2);

    if ( id < n ) {
        x[id*incx] = MAGMA_Z_CONJ(x[id*incx]);
    }
}


/***************************************************************************//**
    Purpose
    -------

    ZLACGV conjugates a complex vector of length N.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            The length of the vector X.  N >= 0.

    @param[in,out]
    x       COMPLEX*16 array, dimension (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    @param[in]
    incx    INTEGER
            The spacing between successive elements of X.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacgv
*******************************************************************************/
void magmablas_zlacgv(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t incx,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, zlacgv_bs);
    int num_blocks = magma_ceildiv( n, zlacgv_bs );
    sycl::range<3> grid(1, 1, num_blocks);
    /*
    DPCT1049:1292: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           kernel_zlacgv(n, x, incx, item_ct1);
                       });
}

#endif // COMPLEX
