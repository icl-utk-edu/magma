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

#define zgeru_bs 512  // 512 is max threads for 1.x cards

void magma_zgetf2_swap(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx,
    magma_queue_t queue );

void magma_zscal_zgeru(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda,
    magma_queue_t );


// TODO: this function could be in .cpp file -- it has no CUDA code in it.
/***************************************************************************//**
    ZGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    ---------

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    @ingroup magma_getf2
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_queue_t queue,
    magma_int_t *info )
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > zgeru_bs) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return *info;
    }

    magma_int_t min_mn = min(m, n);
    magma_int_t j, jp;
    
    for (j=0; j < min_mn; j++) {
        /*
        DPCT1026:559: The call to cudaDeviceSetCacheConfig was removed because
        DPC++ currently does not support setting cache config on devices.
        */

        // Find pivot and test for singularity.
        jp = j - 1 + magma_izamax( m-j, dA(j,j), 1, queue );
        ipiv[j] = jp + 1;  // ipiv uses Fortran one-based index
        // Can't check value of dA since it is on GPU
        //if ( dA(jp, j) != 0.0) {
            /*
            DPCT1026:560: The call to cudaDeviceSetCacheConfig was removed
            because DPC++ currently does not support setting cache config on
            devices.
            */

            // Apply the interchange to columns 1:N.
            if (jp != j) {
                magma_zgetf2_swap( n, dA, j, jp, ldda, queue );
            }
            
            // Compute elements J+1:M of J-th column.
            if (j < m) {
                magma_zscal_zgeru( m-j, n-j, dA(j, j), ldda, queue );
            }
        //}
        //else if (*info == 0) {
        //    *info = j;
        //}
    }

    return *info;
}


// ===========================================================================
// TODO: use standard BLAS magma_zswap?
#define zswap_bs 64

/******************************************************************************/

void kernel_zswap(int n, magmaDoubleComplex *x, int i, int j, int incx,
                  sycl::nd_item<3> item_ct1)
{
    int id = item_ct1.get_group(2) * zswap_bs + item_ct1.get_local_id(2);

    if (id < n) {
        magmaDoubleComplex tmp = x[i + incx*id];
        x[i + incx*id] = x[j + incx*id];
        x[j + incx*id] = tmp;
    }
}


/******************************************************************************/
void magma_zgetf2_swap(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx,
    magma_queue_t queue )
{
    /* zswap two row vectors: ith and jth */
    sycl::range<3> threads(1, 1, zswap_bs);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, zswap_bs));
    /*
    DPCT1049:561: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           kernel_zswap(n, x, i, j, incx, item_ct1);
                       });
}


/******************************************************************************/
// dynamically allocated shared memory, set to size n when the kernel is launched.
// See CUDA Guide B.2.3
//extern __shared__ magmaDoubleComplex shared_data[];


/******************************************************************************/

void kernel_zscal_zgeru(int m, int n, magmaDoubleComplex *A, int lda,
                        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    magmaDoubleComplex *shared_y = shared_data;

    int tid = item_ct1.get_group(2) * zgeru_bs + item_ct1.get_local_id(2);

    /*
    DPCT1064:563: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex reg = MAGMA_Z_ZERO;

    if (item_ct1.get_local_id(2) < n) {
        shared_y[item_ct1.get_local_id(2)] = A[lda * item_ct1.get_local_id(2)];
    }

    /*
    DPCT1065:562: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for (int i=1; i < n; i++) {
            /*
            DPCT1064:564: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            A[tid + i * lda] += (MAGMA_Z_NEG_ONE)*shared_y[i] * reg;
        }
    }
}


/******************************************************************************/
void magma_zscal_zgeru(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue )
{
    /*
    Specialized kernel that merges zscal and zgeru
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    sycl::range<3> threads(1, 1, zgeru_bs);
    sycl::range<3> grid(1, 1, magma_ceildiv(m, zgeru_bs));
    /*
    DPCT1083:566: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_size = sizeof(magmaDoubleComplex) * (n);
    /*
    DPCT1049:565: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             kernel_zscal_zgeru(
                                 m, n, dA, ldda, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
    });
}
