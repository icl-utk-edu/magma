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

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

#define COMPLEX


#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column


// dynamically allocated shared memory, set to size number of threads when the kernel is launched.
// See CUDA Guide B.2.3
//extern __shared__ magmaDoubleComplex shared_data[];


// dynamically allocated shared memory, set to size number of threads when the kernel is launched.
// See CUDA Guide B.2.3
//extern __shared__ double dble_shared_data[];


/******************************************************************************/
__global__ void zdotc_kernel_batched(
    int n, magmaDoubleComplex **x_array, int incx, int offset,
    magma_int_t *info_array, int gbstep)
{
    extern __shared__ double dble_shared_data[];

    int tx = threadIdx.x;

    magmaDoubleComplex *x = x_array[blockIdx.z]+offset;

    double *sdata = dble_shared_data;

    magmaDoubleComplex res = MAGMA_Z_ZERO;

    if (tx < n) {
        res = x[tx*incx];
    }

    sdata[tx] = MAGMA_Z_REAL(res * MAGMA_Z_CONJ(res));

    __syncthreads();

    for (int s = blockDim.x/2; s > 32; s >>= 1 ) {
        if (tx < s) {
            sdata[tx] += sdata[tx+s];
        }
        __syncthreads();
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
        x[n*incx] = MAGMA_Z_MAKE(sqrt(xreal - sdata[0]), 0);
        if (xreal <= MAGMA_D_ZERO) {
            info_array[blockIdx.z] = offset + gbstep + 1;
        }
    }
}


/******************************************************************************/
void magma_zpotf2_zdotc_batched(magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized Zdotc
    1) performs zdotc sum = x[0:n-1]*conj(x[0:n-1])
    2) updates x[n] = sqrt(x[n]-sum);
    */
    if (n > MAX_NTHREADS) {
        fprintf( stderr, "%s: n = %lld > %lld is not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
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
    
    dim3 grid(1, 1, batchCount);
    size_t shmem = threadSize * sizeof(double);
    zdotc_kernel_batched
        <<< grid, threadSize, shmem, queue->cuda_stream() >>>
        (n, x_array, incx, offset, info_array, gbstep);
}


/******************************************************************************/
__global__ void zdscal_kernel_batched(
    int n, magmaDoubleComplex **x_array, int incx, int offset,
    magma_int_t *info_array)
{
    // checkinfo to avoid computation of the singular matrix
    if (info_array[blockIdx.z] != 0 ) return;

    int id = threadIdx.x;
    magmaDoubleComplex *x = x_array[blockIdx.z]+offset;

    __shared__ magmaDoubleComplex factor;

    if (threadIdx.x == 0) {
        factor = MAGMA_Z_MAKE(1.0/MAGMA_Z_REAL(x[0]), 0.0);
    }

    __syncthreads();

    if ( id < n && id > 0) {
        x[id*incx] = x[id*incx] * factor;
        //printf("x=%f", x[id*incx]);
    }
}


/******************************************************************************/
void magma_zpotf2_zdscal_batched(
    magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx,
    magma_int_t offset, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized Zdscal perform x[1:n-1]/x[0]
    */
    dim3 grid(1, 1, batchCount);
    dim3 threads(n, 1, 1); 

    zdscal_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x_array, incx, offset, info_array);
}


/******************************************************************************/
__global__ void zlacgv_kernel_batched(int n, magmaDoubleComplex **x_array, int incx, int offset)
{
    int id = threadIdx.x;

    magmaDoubleComplex *x = x_array[blockIdx.z]+offset;

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
    N       (input) INTEGER
            The length of the vector X.  N >= 0.

    X       (input/output) COMPLEX*16 array, dimension
                           (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    INCX    (input) INTEGER
            The spacing between successive elements of X.

    @ingroup magma_lacgv_batched
*******************************************************************************/
void magma_zlacgv_batched(
    magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx,
    magma_int_t offset, magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(1, 1, batchCount);
    dim3 threads(n, 1, 1);
   
    zlacgv_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, x_array, incx, offset);
}


/******************************************************************************/
static __device__ void zpotf2_device(int m, int n, 
                              magmaDoubleComplex *A, int lda, 
                              magmaDoubleComplex alpha, 
                              magmaDoubleComplex beta, magma_int_t *info, int gbstep)
{
    extern __shared__ magmaDoubleComplex shared_data[];

    /*
    Each thread block load entire A into shared memory
    factorize it and copy back. n must be small enough to fit shared memory.
    n is checked by a macro POTF2_TILE_SIZE before the kernel. 
    */
    // checkinfo to avoid computation of the singular matrix
    if (*info != 0 ) return;

    int tx = threadIdx.x;
    magmaDoubleComplex *sdata_A = shared_data;
    __shared__ magmaDoubleComplex factor;
    __shared__ double sum[POTF2_TILE_SIZE];

    // load A into sdata_A
    if (tx < m)
    {
        for (int i=0; i < n; i++)
        {  
             sdata_A[tx + i * m] =  A[tx + i * lda];
        }
    }
    __syncthreads();

    for (int iter=0; iter < n; iter++)
    {
        double res = MAGMA_D_ZERO;
        magmaDoubleComplex res1 = MAGMA_Z_ZERO;

        //1) performs zdotc sum = A[iter, 0:iter-1]*conj(A[iter, 0:iter-1])
        //2) updates A[iter,iter] = sqrt(A[iter,iter]-sum);
        if (tx < iter)
        {
            res = MAGMA_Z_REAL (sdata_A[iter + tx * m] * MAGMA_Z_CONJ(sdata_A[iter + tx * m]));         
            sum[tx] = res;
        }
        else
        {
            sum[tx] = 0.0;
        }
        __syncthreads();
        magma_sum_reduce<POTF2_TILE_SIZE>(tx, sum); //tried on K40: if m=32 n=32 the overall zpotf2_device routine time is 60ms n=16 time=25 n=8 time=20ms 
        //magma_sum_reduce_n(iter, tx, sum); //tried on K40: if m=32 n=32 the time went from 61ms to 70ms when switching to reduce_n. n=16 time=28.
        //magma_sum_reduce_inlined(iter, tx, sum); //tried on K40: similar to magma_sum_reduce<POTF2_TILE_SIZE>(tx, sum);
        
        __shared__ double xreal;
        if (tx == 0) {
            xreal = MAGMA_Z_REAL(sdata_A[iter + iter * m]);        
            sdata_A[iter + iter * m] = MAGMA_Z_MAKE(sqrt(xreal - sum[0]), 0);
            if (xreal <= MAGMA_D_ZERO) {
                *info = iter + gbstep + 1;
            }
        }
        __syncthreads();
        if (xreal <= MAGMA_D_ZERO) return;
        __syncthreads();

        //zlacgv conjugates a complex vector of length iter. //TODO
        #ifdef COMPLEX
        if (tx < iter)
        {
            sdata_A[iter + tx * m] = MAGMA_Z_CONJ(sdata_A[iter + tx * m]);
        }
        __syncthreads();  
        #endif
  
        // zgemv  
        // Compute elements iter:n-1 of column iter = A(iter:n,0:iter-1) * A(iter-1,0:iter-1) (row).
        if (tx < m && tx > iter)
        {
            for (int j=0; j < iter; j++)
            {
                res1 += sdata_A[tx + j * m]  *  sdata_A[iter + j * m]; // TODO move the zlacgv conj to be done automatically here implicitly.
            }   
            sdata_A [tx + iter * m] = alpha * res1 + sdata_A [tx + iter * m] * beta;   
        }
        __syncthreads();  

        //zlacgv conjugates a complex vector of length iter.
        #ifdef COMPLEX
        if (tx < iter)
        {
            sdata_A[iter + tx * m] = MAGMA_Z_CONJ(sdata_A[iter + tx * m]);
        }
        __syncthreads();  
        #endif

        // zdscal perform A[iter:n-1, iter]/A[iter,iter];
        if (tx == 0) {
            factor = MAGMA_Z_MAKE(1.0/MAGMA_Z_REAL(sdata_A[iter + iter * m]), 0.0);
        }
        __syncthreads();

        if ( tx < m && tx > iter) {
            sdata_A[ tx + iter * m ]  *= factor;
        }
        __syncthreads();
    }// end of iter

    //copy sdata_A to A
    if (tx < m)
    {
        for (int i=0; i < n; i++)
        {  
             A[tx + i * lda] = sdata_A[tx + i * m];
        }
    }
}


/******************************************************************************/
__global__ void zpotf2_kernel_batched(int m, int n, 
                              magmaDoubleComplex **dA_array, int lda, 
                              magmaDoubleComplex alpha, 
                              magmaDoubleComplex beta, 
                              magma_int_t *info_array, int gbstep)
{
    /*
    Each thread block load entire dA_array[blockIdx.z] into shared memory
    factorize it and copy back. n must be small enough to fit shared memory.
    n is checked by a macro POTF2_TILE_SIZE before the kernel. 
    */
    int batchid = blockIdx.z;
    zpotf2_device(m, n, dA_array[batchid], lda, alpha, beta, &(info_array[batchid]), gbstep);
}


/******************************************************************************/
__global__ void zpotf2_kernel(int m, int n, 
                              magmaDoubleComplex *dA, int lda, 
                              magmaDoubleComplex alpha, 
                              magmaDoubleComplex beta,
                              magma_int_t *info)
{
    zpotf2_device(m, n, dA, lda, alpha, beta, info, 0);
}


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
    m       INTEGER
            The number of rows of the matrix A.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0 and N <= 512.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount). 
             Each is a COMPLEX_16 array A, dimension (lda,n)
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
    lda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    info_array INTEGER array, dimension (batchCount).
               Each is the info parameter for the corresponding matrix A
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.
    
    @param[in]
    gbstep  INTEGER
            Internal use, global step.
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_potf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 || m > POTF2_TILE_SIZE || n > POTF2_TILE_SIZE) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    
    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;

    dim3 dimGrid(1, 1, batchCount);
    dim3 threads(POTF2_TILE_SIZE, 1);
    size_t shmem = sizeof(magmaDoubleComplex)*m*n; // + sizeof(double)*(POTF2_TILE_SIZE+1);

    zpotf2_kernel_batched
        <<< dimGrid, threads, shmem, queue->cuda_stream() >>>
        (m, n, dA_array, lda, alpha, beta, info_array, gbstep);

    return arginfo;
}
