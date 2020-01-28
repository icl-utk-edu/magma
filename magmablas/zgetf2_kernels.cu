/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"

/*
    Purpose
    -------
    These are internal routines that might have many assumption.
    They are used in zgetf2_batched.cpp
*/


#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

/******************************************************************************/
extern __shared__ magmaDoubleComplex shared_data[];
extern __shared__ double sdata[];
extern __shared__ int int_sdata[];


/******************************************************************************/
__device__ int
izamax_devfunc(int length, const magmaDoubleComplex *x, int incx, double *shared_x, int *shared_idx)
{
    int tx = threadIdx.x;
    magmaDoubleComplex res;
    double  res1;
    int nchunk = magma_ceildiv( length, zamax );

    if ( tx < zamax ) {
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx; //-1; // -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    __syncthreads();

    for (int s =0; s < nchunk; s++)
    {
        if ( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];
            res1 = fabs(MAGMA_Z_REAL(res)) + fabs(MAGMA_Z_IMAG(res));

            if ( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;
            }
        }
    }
    __syncthreads();

    if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
        magma_getidmax<zamax>(tx, shared_x, shared_idx);
    else
        magma_getidmax_n(min(zamax,length), tx, shared_x, shared_idx);
    return shared_idx[0];
}

/******************************************************************************/
__global__ void
izamax_kernel_batched(int length, int chunk, magmaDoubleComplex **x_array, int xi, int xj, int incx,
                   int step, int lda, magma_int_t** ipiv_array, magma_int_t *info_array, int gbstep)
{
    magmaDoubleComplex *x_start = x_array[blockIdx.z] + xj * lda + xi;
    const magmaDoubleComplex *x = &(x_start[step + step * lda]);

    magma_int_t *ipiv = ipiv_array[blockIdx.z] + xi;
    int tx = threadIdx.x;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(length, x, incx, shared_x, shared_idx);

    if (tx == 0) {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[blockIdx.z] = shared_idx[0] + step + gbstep + 1;
        }
    }
}


/******************************************************************************/
__global__ void
izamax_kernel_native(int length, int chunk, magmaDoubleComplex_ptr x, int incx, 
                     int step, int lda, magma_int_t* ipiv, magma_int_t *info, int gbstep)
{
    const int tx = threadIdx.x;
    x += step * lda + step;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(length, x, incx, shared_x, shared_idx);
    if (tx == 0) {
        ipiv[step]  = shared_idx[0] + step + 1; // Fortran Indexing
        if (shared_x[0] == MAGMA_D_ZERO) {
            (*info) = shared_idx[0] + step + gbstep + 1;
        }
    }
}


/***************************************************************************//**
    Purpose
    -------

    IZAMAX find the index of max absolute value of elements in x and store the index in ipiv

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    length       INTEGER
            On entry, length specifies the size of vector x. length >= 0.


    @param[in]
    x_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension


    @param[in]
    xi      INTEGER
            Row offset, internal use

    @param[in]
    xj      INTEGER
            Column offset, internal use

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            the offset of ipiv

    @param[in]
    lda    INTEGER
            The leading dimension of each array A, internal use to find the starting position of x.

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep    INTEGER
            the offset of info, internal use

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_iamax_batched
*******************************************************************************/
extern "C" magma_int_t
magma_izamax_batched(magma_int_t length,
                     magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx,
                     magma_int_t step,  magma_int_t lda,
                     magma_int_t** ipiv_array, magma_int_t *info_array,
                     magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    dim3 grid(1, 1, batchCount);
    dim3 threads(zamax, 1, 1);

    int chunk = magma_ceildiv( length, zamax );
    izamax_kernel_batched<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
        (length, chunk, x_array, xi, xj, incx, step, lda, ipiv_array, info_array, gbstep);

    return 0;
}


/******************************************************************************/
// For use in magma_izamax_native only
// cublasIzamax always writes 32bit pivots, so make sure it is magma_int_t
__global__ void magma_zpivcast(magma_int_t* dipiv)
{
    // uses only 1 thread
    int* address = (int*)dipiv; 
    int pivot = *address;          // read the value written by cuBLAS (int)
    *dipiv = (magma_int_t)pivot;    // write it back in the same address as dipiv
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_native( magma_int_t length, 
                     magmaDoubleComplex_ptr x, magma_int_t incx, 
                     magma_int_t step,  magma_int_t lda, 
                     magma_int_t* ipiv, magma_int_t *info, 
                     magma_int_t gbstep, magma_queue_t queue)
{
    if (length == 0 ) return 0;

    // TODO: decide the best izamax for all precisions
    if( length <= 15360 ) {
        dim3 grid(1, 1, 1);
        dim3 threads(zamax, 1, 1);

        int chunk = magma_ceildiv( length, zamax );
        izamax_kernel_native<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
            (length, chunk, x, incx, step, lda, ipiv, info, gbstep);
    }
    else {
        cublasPointerMode_t ptr_mode;
        cublasGetPointerMode(queue->cublas_handle(), &ptr_mode);
        cublasSetPointerMode(queue->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);

        cublasIzamax(queue->cublas_handle(), length, x + step * lda + step, 1, (int*)(ipiv+step));
        magma_zpivcast<<< 1, 1, 0, queue->cuda_stream() >>>( ipiv+step );

        cublasSetPointerMode(queue->cublas_handle(), ptr_mode);
        adjust_ipiv( ipiv+step, 1, step, queue);
    }
    return 0;
}


/******************************************************************************/
__device__
void zswap_device( magma_int_t n, 
                   magmaDoubleComplex_ptr x, magma_int_t incx, 
                   magma_int_t step, magma_int_t* ipiv)
{
    const int tx = threadIdx.x; 
    
    __shared__ int jp;

    if (tx == 0){
        jp = ipiv[step] - 1;
    }
    __syncthreads();

    if (jp == step) return; // no pivot

    if (tx < n) {
        magmaDoubleComplex tmp = x[jp + tx * incx];
        x[jp + tx * incx] = x[step + tx * incx];
        x[step + tx * incx] = tmp;
    }
}


/******************************************************************************/
__global__
void zswap_kernel_batched(
        magma_int_t n, 
        magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx, 
        magma_int_t step, magma_int_t** ipiv_array)
{
    magmaDoubleComplex *x = x_array[blockIdx.z] + xj * incx + xi;
    magma_int_t *ipiv = ipiv_array[blockIdx.z] + xi;

    zswap_device(n, x, incx, step, ipiv);
}


/******************************************************************************/
__global__
void zswap_kernel_native( magma_int_t n, 
                          magmaDoubleComplex_ptr x, magma_int_t incx, 
                          magma_int_t step, magma_int_t* ipiv)
{
    zswap_device(n, x, incx, step, ipiv);
}


/***************************************************************************//**
    Purpose
    -------

    zswap two row in x.  index (ipiv[step]-1)-th and index step -th

    This is an internal routine that might have many assumption.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            On entry, n specifies the size of vector x. n >= 0.


    @param[in]
    dA_array  Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension


    @param[in]
    ai      INTEGER
            Row offset, internal use.

    @param[in]
    aj      INTEGER
            Column offset, internal use.

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zswap_batched( magma_int_t n, 
                     magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t incx,
                     magma_int_t step, magma_int_t** ipiv_array,
                     magma_int_t batchCount, magma_queue_t queue)
{
    /*
    zswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    dim3 grid(1, 1, batchCount);
    dim3 threads(zamax, 1, 1);

    zswap_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, dA_array, ai, aj, incx, step, ipiv_array);
    return 0;
}


/******************************************************************************/
extern "C" void
magma_zswap_native( magma_int_t n, magmaDoubleComplex_ptr x, magma_int_t incx, 
                    magma_int_t step, magma_int_t* ipiv,
                    magma_queue_t queue)
{
    /*
    zswap two row: (ipiv[step]-1)th and step th
    */
    if ( n  > MAX_NTHREADS){
        fprintf( stderr, "%s nb=%lld > %lld, not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }
    dim3 grid(1, 1, 1);
    dim3 threads(zamax, 1, 1);

    zswap_kernel_native
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x, incx, step, ipiv);
}


/******************************************************************************/
template<int N>
__device__
void zscal_zgeru_device( int m, int step, 
                         magmaDoubleComplex_ptr dA, int lda, 
                         magma_int_t *info, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    // checkinfo to avoid computation of the singular matrix
    if( (*info) != 0 ) return;

    magmaDoubleComplex_ptr A = dA + step + step * lda;
    magmaDoubleComplex rA[N], reg; 
    __shared__ magmaDoubleComplex shared_y[N];
    
    if (tx < N) {
        shared_y[tx] = A[lda * tx];
    }
    __syncthreads();
    
    if (shared_y[0] == MAGMA_Z_ZERO) {
        (*info) = step + gbstep + 1;
        return;
    }

    // terminate threads that are out of the range
    if (gtx == 0 || gtx >= m) return; 
    
    reg = MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);
    #pragma unroll 
    for(int i = 0; i < N; i++)
        rA[i] = A[ i* lda + gtx ];
        
    rA[0] *= reg;
    
    #pragma unroll 
    for(int i = 1; i < N; i++)
        rA[i] -= rA[0] * shared_y[i];

    #pragma unroll
    for(int i = 0; i < N; i++)
        A[gtx + i * lda] = rA[i];
}


/******************************************************************************/
__device__
void zscal_zgeru_generic_device( int m, int n, int step, 
                         magmaDoubleComplex_ptr dA, int lda, 
                         magma_int_t *info, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    // checkinfo to avoid computation of the singular matrix
    if( (*info) != 0 ) return;
    if (gtx == 0 || gtx >= m) return; 

    magmaDoubleComplex_ptr A = dA + step + step * lda;
    magmaDoubleComplex rA, reg; 
    
    if (A[0] == MAGMA_Z_ZERO) {
        (*info) = step + gbstep + 1;
        return;
    }

    reg = MAGMA_Z_DIV(MAGMA_Z_ONE, A[0]);
    rA  = A[ gtx ];
    rA *= reg;
    
    A[ gtx ] = rA;
    #pragma unroll 
    for(int i = 1; i < n; i++)
        A[i * lda + gtx] -= rA * A[i * lda + 0];

}


/******************************************************************************/
template<int N>
__global__
void zscal_zgeru_1d_kernel_native( int m, int step, 
                                magmaDoubleComplex_ptr dA, int lda, 
                                magma_int_t *info, int gbstep)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the 
    // return statement should be converted into an if-statement 
    zscal_zgeru_device<N>(m, step, dA, lda, info, gbstep);
}


/******************************************************************************/
__global__
void zscal_zgeru_1d_generic_kernel_native( int m, int n, int step, 
                                magmaDoubleComplex_ptr dA, int lda, 
                                magma_int_t *info, int gbstep)
{
    // This dev function has a return statement inside, be sure
    // not to merge it with another dev function. Otherwise, the 
    // return statement should be converted into an if-statement 
    zscal_zgeru_generic_device(m, n, step, dA, lda, info, gbstep);
}


/******************************************************************************/
template<int N>
__global__
void zscal_zgeru_1d_kernel_batched(int m, int step, magmaDoubleComplex **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int gbstep)
{
    const int batchid = blockIdx.z;
    magmaDoubleComplex* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_device<N>(m, step, dA, lda, info, gbstep);
}


/******************************************************************************/
__global__
void zscal_zgeru_1d_generic_kernel_batched(int m, int n, int step, magmaDoubleComplex **dA_array, int ai, int aj, int lda, magma_int_t *info_array, int gbstep)
{
    const int batchid = blockIdx.z;
    magmaDoubleComplex* dA = dA_array[batchid] + aj * lda + ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_generic_device(m, n, step, dA, lda, info, gbstep);
}


/******************************************************************************/
extern "C"
magma_int_t magma_zscal_zgeru_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep,
                                      magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    const int tbx = MAX_NTHREADS / 2;
    dim3 grid(magma_ceildiv(m,tbx), 1, batchCount);
    dim3 threads(tbx, 1, 1);
    switch(n){
        case  1:zscal_zgeru_1d_kernel_batched< 1><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  2:zscal_zgeru_1d_kernel_batched< 2><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  3:zscal_zgeru_1d_kernel_batched< 3><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  4:zscal_zgeru_1d_kernel_batched< 4><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  5:zscal_zgeru_1d_kernel_batched< 5><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  6:zscal_zgeru_1d_kernel_batched< 6><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  7:zscal_zgeru_1d_kernel_batched< 7><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        case  8:zscal_zgeru_1d_kernel_batched< 8><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA_array, ai, aj, lda, info_array, gbstep);break;
        default:zscal_zgeru_1d_generic_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>(m, n, step, dA_array, ai, aj, lda, info_array, gbstep); 
    }
    return 0;
}


/******************************************************************************/
extern "C"
magma_int_t 
magma_zscal_zgeru_native( 
    magma_int_t m, magma_int_t n, magma_int_t step,
    magmaDoubleComplex_ptr dA, magma_int_t lda,
    magma_int_t *info, magma_int_t gbstep,
    magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( n == 0) return 0;
    if ( n > MAX_NTHREADS ) {
        fprintf( stderr, "%s nb=%lld, > %lld, not supported\n", __func__, (long long) n, (long long) MAX_NTHREADS );
        return -15;
    }
    const int tbx = MAX_NTHREADS / 2;
    dim3 grid(magma_ceildiv(m,tbx), 1, 1);
    dim3 threads(tbx, 1, 1);
    switch(n){
        case 1:zscal_zgeru_1d_kernel_native<1><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 2:zscal_zgeru_1d_kernel_native<2><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 3:zscal_zgeru_1d_kernel_native<3><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 4:zscal_zgeru_1d_kernel_native<4><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 5:zscal_zgeru_1d_kernel_native<5><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 6:zscal_zgeru_1d_kernel_native<6><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 7:zscal_zgeru_1d_kernel_native<7><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        case 8:zscal_zgeru_1d_kernel_native<8><<<grid, threads, 0, queue->cuda_stream()>>>( m, step, dA, lda, info, gbstep);break;
        default:zscal_zgeru_1d_generic_kernel_native<<<grid, threads, 0, queue->cuda_stream()>>>( m, n, step, dA, lda, info, gbstep);
    }
    return 0;
}


/******************************************************************************/
__global__
void zgetf2trsm_kernel_batched(int ib, int n, magmaDoubleComplex **dA_array, int step, int lda)
{
    /*
        this kernel does the safe nonblocked TRSM operation
        B = A^-1 * B
    */

    magmaDoubleComplex *A_start = dA_array[blockIdx.z];
    magmaDoubleComplex *A = &(A_start[step + step * lda]);
    magmaDoubleComplex *B = &(A_start[step + (step+ib) * lda]);
    magmaDoubleComplex *shared_a = shared_data;
    magmaDoubleComplex *shared_b = shared_data+ib*ib;

    int tid = threadIdx.x;
    int i,d;


    // Read A and B at the same time to the shared memory (shared_a shared_b)
    // note that shared_b = shared_a+ib*ib so its contiguous
    // I can make it in one loop reading
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n+ib; i++) {
            shared_a[tid + i*ib] = A[tid + i*lda];
        }
    }
    __syncthreads();

    if (tid < n) {
        #pragma unroll
        for (d=0;  d < ib-1; d++) {
            for (i=d+1; i < ib; i++) {
                shared_b[i+tid*ib] += (MAGMA_Z_NEG_ONE) * shared_a[i+d*ib] * shared_b[d+tid*ib];
            }
        }
    }
    __syncthreads();

    // write back B
    if ( tid < ib) {
        #pragma unroll
        for (i=0; i < n; i++) {
            B[tid + i*lda] = shared_b[tid + i*ib];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------

    zgetf2trsm solves one of the matrix equations on gpu

     B = C^-1 * B

    where C, B are part of the matrix A in dA_array,

    This version load C, B into shared memory and solve it
    and copy back to GPU device memory.
    This is an internal routine that might have many assumption.

    Arguments
    ---------
    @param[in]
    ib       INTEGER
            The number of rows/columns of each matrix C, and rows of B.  ib >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix B.  n >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[in]
    step    INTEGER
            The starting address of matrix C in A.  LDDA >= max(1,M).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" void
magma_zgetf2trsm_batched(magma_int_t ib, magma_int_t n, magmaDoubleComplex **dA_array,
                         magma_int_t step, magma_int_t ldda,
                         magma_int_t batchCount, magma_queue_t queue)
{
    if ( n == 0 || ib == 0 ) return;
    size_t shared_size = sizeof(magmaDoubleComplex)*(ib*(ib+n));

    // TODO TODO TODO
    if ( shared_size > (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 46K leaving 2K for extra
    {
        fprintf( stderr, "%s: error out of shared memory\n", __func__ );
        return;
    }

    dim3 grid(1, 1, batchCount);
    dim3 threads(max(n,ib), 1, 1);

    zgetf2trsm_kernel_batched
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (ib, n, dA_array, step, ldda);
}


/******************************************************************************/
template<int NB>
__global__ void 
zgetf2trsm_2d_kernel( int m, int n, 
                           magmaDoubleComplex_ptr dA, int ldda, 
                           magmaDoubleComplex_ptr dB, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    __shared__ magmaDoubleComplex sA[NB * NB];
    __shared__ magmaDoubleComplex sB[NB * NB];
    
    // init sA & sB
    sA[ ty * NB + tx ] = MAGMA_Z_ZERO;
    sB[ ty * NB + tx ] = MAGMA_Z_ZERO;
    
    const int nblocks = magma_ceildiv(n, NB);
    const int n_ = n - (nblocks-1) * NB;
    
    // load A
    if( ty < m && tx < m && tx > ty){
        sA[ty * NB + tx] = dA[ty * ldda + tx];
    }
    
    if( ty == tx ){
        // ignore diagonal elements
        sA[tx * NB + tx] = MAGMA_Z_ONE;
    }
    __syncthreads(); 
    
    #pragma  unroll
    for(int s = 0; s < nblocks-1; s++){
        // load B
        if( tx < m ){
            sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
        }

        // no need to sync because each thread column is less than 32
        // solve
        #pragma unroll
        for(int i = 0; i < NB; i++){
            if(tx >  i){
                 sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
            }
        }
        
        // write B
        if( tx < m){
            dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
        }
        dB += NB * lddb;
    }
    
    // last, possible partial, block
    if( ty < n_ && tx < m){
        sB[ ty * NB + tx] = dB[ ty * lddb + tx ];
    }

    #pragma unroll
    for(int i = 0; i < NB; i++){
        if(tx >  i){
             sB[ ty * NB + tx ] -= sA[ i * NB + tx ] * sB[ ty * NB + i ];
        }
    }
        
    if( ty < n_ && tx < m){
        dB[ ty * lddb + tx ] = sB[ ty * NB + tx ];
    }
}


/******************************************************************************/
extern"C" void 
magma_zgetf2trsm_2d_native( 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDoubleComplex_ptr dB, magma_int_t lddb, 
    magma_queue_t queue)
{
    if( m > 32 ){
        magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
                     m, n, MAGMA_Z_ONE, 
                     dA, ldda, 
                     dB, lddb, queue );
        return;
    }

    const int m8 = magma_roundup(m, 8);
    dim3 grid(1, 1, 1);
    dim3 threads(m8, m8, 1);
    
    switch(m8){
        case  8:zgetf2trsm_2d_kernel< 8><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 16:zgetf2trsm_2d_kernel<16><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 24:zgetf2trsm_2d_kernel<24><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        case 32:zgetf2trsm_2d_kernel<32><<<grid, threads, 0, queue->cuda_stream() >>>( m, n, dA, ldda, dB, lddb ); break;
        default:;
    }
}


/******************************************************************************/
static __device__ void
zupdate_device(int m, int step, magmaDoubleComplex* x, int ldx,  magmaDoubleComplex *A, int lda)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    int indx;
    //magmaDoubleComplex reg = MAGMA_Z_ZERO;

    // update the current column by all the previous one
    #pragma unroll
    for (int i=0; i < step; i++) {
        for (int s=0; s < nchunk; s++)
        {
            indx = tid + s * MAX_NTHREADS;
            if ( indx > i  && indx < m ) {
                A[indx] -=  A[i] * x[indx + i*ldx];
                //printf("         @ step %d tid %d updating x[tid]*y[i]=A %5.3f %5.3f = %5.3f  at i %d\n", step, tid, x[tid + i*ldx], A[i], A[tid],i);
            }
        }
        __syncthreads();
    }

    //printf("         @ step %d tid %d adding %5.3f to A %5.3f make it %5.3f\n",step,tid,-reg,A[tid],A[tid]-reg);
}


/******************************************************************************/
static __device__ void
zscal5_device(int m, magmaDoubleComplex* x, magmaDoubleComplex alpha)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );

    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) {
            #if 0
            x[tid + s * MAX_NTHREADS] *= MAGMA_Z_DIV(MAGMA_Z_ONE, alpha);
            #else
            x[tid + s * MAX_NTHREADS] = x[tid + s * MAX_NTHREADS]/alpha;
            #endif
        }
    }
    __syncthreads();
}


/******************************************************************************/
__global__ void
zcomputecolumn_kernel_shared_batched( int m, int paneloffset, int step, 
                                      magmaDoubleComplex **dA_array, int ai, int aj, 
                                      int lda, magma_int_t **ipiv_array, magma_int_t *info_array, int gbstep)
{
    
    int gboff = paneloffset+step;
    magma_int_t *ipiv           = ipiv_array[blockIdx.z] + ai;
    magmaDoubleComplex *A_start = dA_array[blockIdx.z] + aj * lda + ai;
    magmaDoubleComplex *A0j     = &(A_start[paneloffset + (paneloffset+step) * lda]);
    magmaDoubleComplex *A00     = &(A_start[paneloffset + paneloffset * lda]);

    magmaDoubleComplex *shared_A = shared_data;
    __shared__ double  shared_x[zamax];
    __shared__ int     shared_idx[zamax];
    __shared__ magmaDoubleComplex alpha;
    int tid = threadIdx.x;

    // checkinfo to avoid computation of the singular matrix
    if (info_array[blockIdx.z] != 0 ) return;


    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    // read the current column from dev to shared memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) shared_A[tid + s * MAX_NTHREADS] = A0j[tid + s * MAX_NTHREADS];
    }
    __syncthreads();

    // update this column
    if ( step > 0 ) {
        zupdate_device( m, step, A00, lda, shared_A, 1);
        __syncthreads();
    }

    // if ( tid < (m-step) ) // DO NO TPUT THE IF CONDITION HERE SINCE izamax_devfunc HAS __syncthreads INSIDE.
    // So let all htreads call this routine it will handle correctly based on the size
    // note that izamax need only 128 threads, s
    izamax_devfunc(m-step, shared_A+step, 1, shared_x, shared_idx);
    if (tid == 0) {
        ipiv[gboff]  = shared_idx[0] + gboff + 1; // Fortran Indexing
        alpha = shared_A[shared_idx[0]+step];
        //printf("@ step %d ipiv=%d where gboff=%d  shared_idx %d alpha %5.3f\n",step,ipiv[gboff],gboff,shared_idx[0],alpha);
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[blockIdx.z] = shared_idx[0] + gboff + gbstep + 1;
        }
    }
    __syncthreads();
    if (shared_x[0] == MAGMA_D_ZERO) return;
    __syncthreads();

    // DO NO PUT THE IF CONDITION HERE SINCE izamax_devfunc HAS __syncthreads INSIDE.
    zscal5_device( m-step, shared_A+step, alpha);

    // put back the pivot that has been scaled with itself menaing =1
    if (tid == 0)  shared_A[shared_idx[0] + step] = alpha;
    __syncthreads();

    // write back from shared to dev memory
    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m )
        {
            A0j[tid + s * MAX_NTHREADS] = shared_A[tid + s * MAX_NTHREADS];
            //printf("@ step %d tid %d updating A=x*alpha after A= %5.3f\n",step,tid,shared_A[tid]);
        }
    }
    __syncthreads();
}


/******************************************************************************/
extern "C"
magma_int_t magma_zcomputecolumn_batched( magma_int_t m, magma_int_t paneloffset, magma_int_t step,
                                          magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda,
                                          magma_int_t **ipiv_array,
                                          magma_int_t *info_array, magma_int_t gbstep,
                                          magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */
    if ( m == 0) return 0;

    size_t all_shmem_size = zamax*(sizeof(double)+sizeof(int)) + (m+2)*sizeof(magmaDoubleComplex);
    if ( all_shmem_size >  (MAX_SHARED_ALLOWED*1024) ) // limit the shared memory to 44K leaving 4K for extra
    {
        fprintf( stderr, "%s error out of shared memory\n", __func__ );
        return -20;
    }

    size_t shared_size = sizeof(magmaDoubleComplex)*m;
    dim3 grid(1, 1, batchCount);
    dim3 threads(min(m, MAX_NTHREADS), 1, 1);

    zcomputecolumn_kernel_shared_batched
        <<< grid, threads, shared_size, queue->cuda_stream() >>>
        (m, paneloffset, step, dA_array, ai, aj, lda, ipiv_array, info_array, gbstep);

    return 0;
}


/******************************************************************************/
template<int WIDTH>
__device__ void
zgetf2_fused_device( int m, magmaDoubleComplex* dA, int ldda, magma_int_t* dipiv,
                   magmaDoubleComplex* swork, magma_int_t *info, int gbstep)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    magmaDoubleComplex rA[WIDTH] = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg       = MAGMA_Z_ZERO; 
    magmaDoubleComplex update    = MAGMA_Z_ZERO;
    
    int max_id, rowid = tx;
    int linfo = (gbstep == 0) ? 0 : *info;
    double rx_abs_max = MAGMA_D_ZERO;
    // check from previous calls if the panel factorization failed previously
    // this is necessary to report the correct info value 
    //if(gbstep > 0 && *info != 0) return;
    
    magmaDoubleComplex *sx = (magmaDoubleComplex*)(swork);
    double* dsx = (double*)(sx + blockDim.y * WIDTH);
    int* isx    = (int*)(dsx + blockDim.y * m);
    int* sipiv  = (int*)(isx + blockDim.y * m);
    sx    += ty * WIDTH;
    dsx   += ty * m;
    isx   += ty * m;
    sipiv += ty * WIDTH;

    // init sipiv
    if(tx < WIDTH){
        sipiv[tx] = 0;
    }

    // read 
    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        rA[i] = dA[ i * ldda + tx ];
    }

    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        // izamax and find pivot
        dsx[ rowid ] = fabs(MAGMA_Z_REAL( rA[i] )) + fabs(MAGMA_Z_IMAG( rA[i] ));
        isx[ tx ] = tx;
        __syncthreads();
        magma_getidmax_n(m-i, tx, dsx+i, isx+i); // this devfunc has syncthreads at the end
        rx_abs_max = dsx[i];
        max_id = isx[i];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        __syncthreads();

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = 0; j < WIDTH; j++){
                sx[j] = update * rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id; 
        }
        __syncthreads();
        
        reg = (linfo == 0 ) ? MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] ) : MAGMA_Z_ONE;
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < WIDTH; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }
    // write
    if(tx < WIDTH){
        dipiv[tx] = (magma_int_t)(sipiv[tx] + 1); // fortran indexing
        //printf("--- ipiv[%d] --- = %d\n", tx, dipiv[tx]);
    }
    
    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        dA[ i * ldda + rowid ] = rA[i];
    }
}

/******************************************************************************/
extern __shared__ magmaDoubleComplex zdata[];
template<int WIDTH>
__global__ void
zgetf2_fused_batched_kernel( int m, 
                           magmaDoubleComplex** dA_array, int ai, int aj, int ldda, 
                           magma_int_t** dipiv_array, magma_int_t* info_array, int batchCount)
{
     magmaDoubleComplex* swork = (magmaDoubleComplex*)zdata;
     const int batchid = blockIdx.z * blockDim.y + threadIdx.y;
     if(batchid >= batchCount)return;
     zgetf2_fused_device<WIDTH>(
             m, dA_array[batchid] + aj * ldda + ai, ldda, 
             dipiv_array[batchid] + ai,  
             swork, &info_array[batchid], aj);
}


/***************************************************************************//**
    Purpose
    -------
    magma_zgetf2_reg_batched computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. This routine is used for batch LU panel 
    factorization, and has specific assumption about the value of N 

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is a right-looking unblocked version of the algorithm. The routine is a batched 
    version that factors batchCount M-by-N matrices in parallel.

    This version load an entire matrix (m*n) into registers and factorize it with pivoting 
    and copy back to GPU device memory. 

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  ib >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_fused_batched(
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **dipiv_array, 
    magma_int_t *info_array, magma_int_t batchCount, 
    magma_queue_t queue)
{
    if(m < 0 || m > ZGETF2_FUSED_BATCHED_MAX_ROWS) {
        fprintf( stderr, "%s: m = %4lld not supported, must be between 0 and %4lld\n",
                 __func__, (long long) m, (long long) ZGETF2_FUSED_BATCHED_MAX_ROWS);
        return -1;
    }
    else if(n < 0 || n > 32){
        fprintf( stderr, "%s: n = %4lld not supported, must be between 0 and %4lld\n",
                 __func__, (long long) m, (long long) 32);
        return -2;
    }
    magma_int_t ntcol = (m > 32)? 1 : (2 * (32/m));

    magma_int_t shared_size = 0;
    shared_size += n * sizeof(magmaDoubleComplex);
    shared_size += m * sizeof(double);
    shared_size += m * sizeof(int);    // not magma_int_t
    shared_size += n * sizeof(int);    // not magma_int_t
    shared_size *= ntcol;
    
    dim3 grid(1,1, magma_ceildiv(batchCount,ntcol));
    dim3 threads(m, ntcol, 1);

    switch(n)
    {
        case  1: zgetf2_fused_batched_kernel< 1><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  2: zgetf2_fused_batched_kernel< 2><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  3: zgetf2_fused_batched_kernel< 3><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  4: zgetf2_fused_batched_kernel< 4><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  5: zgetf2_fused_batched_kernel< 5><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  6: zgetf2_fused_batched_kernel< 6><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  7: zgetf2_fused_batched_kernel< 7><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  8: zgetf2_fused_batched_kernel< 8><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case  9: zgetf2_fused_batched_kernel< 9><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 10: zgetf2_fused_batched_kernel<10><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 11: zgetf2_fused_batched_kernel<11><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 12: zgetf2_fused_batched_kernel<12><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 13: zgetf2_fused_batched_kernel<13><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 14: zgetf2_fused_batched_kernel<14><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 15: zgetf2_fused_batched_kernel<15><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 16: zgetf2_fused_batched_kernel<16><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 17: zgetf2_fused_batched_kernel<17><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 18: zgetf2_fused_batched_kernel<18><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 19: zgetf2_fused_batched_kernel<19><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 20: zgetf2_fused_batched_kernel<20><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 21: zgetf2_fused_batched_kernel<21><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 22: zgetf2_fused_batched_kernel<22><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 23: zgetf2_fused_batched_kernel<23><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 24: zgetf2_fused_batched_kernel<24><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 25: zgetf2_fused_batched_kernel<25><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 26: zgetf2_fused_batched_kernel<26><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 27: zgetf2_fused_batched_kernel<27><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 28: zgetf2_fused_batched_kernel<28><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 29: zgetf2_fused_batched_kernel<29><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 30: zgetf2_fused_batched_kernel<30><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 31: zgetf2_fused_batched_kernel<31><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        case 32: zgetf2_fused_batched_kernel<32><<<grid, threads, shared_size, queue->cuda_stream()>>>(m, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount); break;
        default: fprintf( stderr, "%s: n = %4lld is not supported \n", __func__, (long long) n);
    }
    return 0;
}