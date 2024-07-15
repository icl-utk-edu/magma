/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/
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
        cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );

        // Find pivot and test for singularity.
        jp = j - 1 + magma_izamax( m-j, dA(j,j), 1, queue );
        ipiv[j] = jp + 1;  // ipiv uses Fortran one-based index
        // Can't check value of dA since it is on GPU
        //if ( dA(jp, j) != 0.0) {
            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
            
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
__global__
void kernel_zswap(int n, magmaDoubleComplex *x, int i, int j, int incx)
{
    int id = blockIdx.x * zswap_bs + threadIdx.x;

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
    dim3 threads( zswap_bs );
    dim3 grid( magma_ceildiv( n, zswap_bs ) );
    kernel_zswap
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x, i, j, incx);
}


/******************************************************************************/
// dynamically allocated shared memory, set to size n when the kernel is launched.
// See CUDA Guide B.2.3
//extern __shared__ magmaDoubleComplex shared_data[];


/******************************************************************************/
__global__
void kernel_zscal_zgeru(int m, int n, magmaDoubleComplex *A, int lda)
{
    extern __shared__ magmaDoubleComplex shared_data[];

    magmaDoubleComplex *shared_y = shared_data;

    int tid = blockIdx.x * zgeru_bs + threadIdx.x;

    magmaDoubleComplex reg = MAGMA_Z_ZERO;

    if (threadIdx.x < n) {
        shared_y[threadIdx.x] = A[lda * threadIdx.x];
    }

    __syncthreads();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for (int i=1; i < n; i++) {
            A[tid + i*lda] += (MAGMA_Z_NEG_ONE) * shared_y[i] * reg;
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
    dim3 threads( zgeru_bs );
    dim3 grid( magma_ceildiv( m, zgeru_bs ) );
    size_t shared_size = sizeof(magmaDoubleComplex)*(n);
    kernel_zscal_zgeru
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (m, n, dA, ldda);
}
