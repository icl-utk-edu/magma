/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <vector>
#include "magma_internal.h"
#include "magma_templates.h"
#include "swap_scalar.cuh"
#include "sync.cuh"

#define COMPLEX
#define PRECISION_z

#define SLDA(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )
#define sA(i,j)    (sA[(j)*slda+(i)])
#define sV(i,j)    (sV[(j)*sldv+(i)])

////////////////////////////////////////////////////////////////////////////////
static __device__ __inline__ int
get_next_index(int n, int col)
{
    int newcol = 0;
    newcol = (col % 2 == 0          ) ? -2 : 2;
    newcol = (col == n-1 || col == 2) ? -1 : newcol;
    newcol = (col == 0              ) ?  0 : newcol;
    return (col + newcol);
}

////////////////////////////////////////////////////////////////////////////////
__device__ static __noinline__ void
magmablas_dmin_key_device(const int n, const int i, double* x, int* ind)
{
    #pragma unroll
    for(int step = 1024; step > 0; step >>= 1) {
        if ( n > step ) {
            if ( i < step && i + step < n ) {
                if ( x[i] > x[i+step] ) {
                    magmablas_iswap_scalar_device(ind[i], ind[i+step]);
                    magmablas_dswap_scalar_device(x[i], x[i+step]);
                }
            }
             __syncthreads();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__device__ static __noinline__ void
magmablas_zsort_eig(const int n, const int tx, double* dW, int* windex)
{
    #pragma unroll
    for(int iN = 0; iN < n-1; iN++) {
        magmablas_dmin_key_device(n-iN, tx, dW+iN, windex+iN);
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void
zheevj_batched_parallel_small_sm_kernel(
    int jobz, magma_uplo_t uplo, int n,
    magmaDoubleComplex** dA_array, magmaDoubleComplex *dA_base, int ldda, int strideA,
    double **dW_array, double *dW_base, int strideW,
    magma_int_t *info_array, int* batch_mask, int *num_sweeps,
    double eps, double heevj_tol, int sort_values, int max_sweeps, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    const int  tx = threadIdx.x;
    const int  ty = threadIdx.y;
    const int ntx = blockDim.x;
    const int nty = blockDim.y;
    const int gtx = (ty * ntx) + tx;
    const int total_threads = nty * ntx;
    const int batchid = blockIdx.x;
    const int n2 = 2 * ( (n+1)/2 );

    magma_int_t imask = (batch_mask == NULL) ? 1 : batch_mask[batchid];
    if(imask == 0) return;

    magmaDoubleComplex *dA = (dA_array == NULL) ? dA_base + batchid * strideA : dA_array[batchid];
    double             *dW = (dW_array == NULL) ? dW_base + batchid * strideW : dW_array[batchid];
    magma_int_t* info = &info_array[batchid];
    const int slda = SLDA(n2);
    const int sldv = SLDA(n2);

    magmaDoubleComplex *sA      = (magmaDoubleComplex*)(zdata);                  // input matrix
    magmaDoubleComplex *sV      = ( jobz == 1 ) ? sA + slda * n2 : NULL;         // eigenvectors if required
    magmaDoubleComplex *sJo     = ( jobz == 1 ) ? sV + sldv*n2   : sA + slda*n2; // j10, j01 (off-diagonal entries of jacobi rotations)
    double             *sW      = (double*)(sJo + 2 * nty);                      // eigenvalues
    double             *sJd     = sW  + n2;                                      // j00, j11 (diagonal     entries of jacobi rotations)
    int                *sorder  = (int*)(sJd + 2 * nty);
    int                *noconv  = sorder + n2;

    // advance pointers w.r.t ty
    sJo += ty * 2;
    sJd += ty * 2;

    // init sorder and sV
    if( gtx < n2 ) {
        sorder[gtx] = gtx;
        if(jobz == 1) {
            for(int i = 0; i < n2; i++) {
                sV(gtx,i) = MAGMA_Z_ZERO;
            }
            sV(gtx,gtx) = MAGMA_Z_ONE;
        }
    }

    // init noconv
    if( gtx < nty ) {
        noconv[ gtx ] = 0;
    }

    // read A
    if(uplo == MagmaLower) {
        for(int j = ty; j < n; j+=nty) {
            for(int i = (tx+j); i < n; i+=ntx) {
                sA(i, j) = dA[ j * ldda + i];
            }
        }
    }
    else{
        for(int j = ty; j < n; j+=nty) {
            for(int i = tx; i < j+1; i+=ntx) {
                sA(i, j) = dA[j * ldda + i];
            }
        }
    }

    // if n is odd, init the last row and column in sA to zero
    if( !(n == n2) ) {
        for(int i = gtx; i < n; i+=total_threads) {
            sA(i,n2-1) = MAGMA_Z_ZERO;
            sA(n2-1,i) = MAGMA_Z_ZERO;
        }
    }
    __syncthreads();


    // make A Hermitian in shared memory
    // first ignore the imaginary parts on the diagonal
    if(gtx < n) {
        sA(gtx, gtx) = MAGMA_Z_MAKE(MAGMA_Z_REAL(sA(gtx, gtx)), MAGMA_D_ZERO);
    }

    if(uplo == MagmaLower) {
        for(int j = ty; j < n-1; j+=nty) {
            for(int i = (tx+j+1); i < n; i+=ntx) {
                sA(j, i) = MAGMA_Z_CONJ( sA(i,j) );
            }
        }
    }
    else{
        for(int j = ty+1; j < n; j+=nty) {
            for(int i = tx; i < j; i+=ntx) {
                sA(j, i) = MAGMA_Z_CONJ( sA(i, j) );
            }
        }
    }
    __syncthreads();

    int sweep      = 0;
    int converge   = 0;
    int j, k;
    while ( sweep < max_sweeps && converge == 0 ) {
        j = 2*ty;
        k = j + 1;
        converge = 1;
        sweep++;

        for(int ipair = 0; ipair < n2-1; ++ipair) {
            if(tx == 0) {
                noconv[ty] = 0;
                double ajj = MAGMA_Z_REAL( sA(j, j) );
                double akk = MAGMA_Z_REAL( sA(k, k) );
                magmaDoubleComplex ajk =   sA(j, k);

                double tol         = eps * heevj_tol * sqrt(MAGMA_D_ABS(ajj) * MAGMA_D_ABS(akk));//eps * 1000;
                double abs_ajk     = MAGMA_Z_ABS( ajk );
                double abs_ajj_akk = MAGMA_D_ABS(ajj - akk);

                if( abs_ajk > tol) {
                    noconv[ty] = 1;

                    double t, cs_theta, sn_theta, cs_alpha;
                    #ifdef COMPLEX
                    double sn_alpha;
                    #endif

                    t = (2 * copysign( abs_ajk, (ajj-akk) )) / (abs_ajj_akk + sqrt(abs_ajj_akk*abs_ajj_akk + 4*abs_ajk*abs_ajk));

                    cs_theta = MAGMA_D_DIV(1.0, sqrt(1.0 + t*t));
                    sn_theta = cs_theta * t;
                    cs_alpha = MAGMA_Z_REAL( ajk ) / abs_ajk;
                    #ifdef COMPLEX
                    sn_alpha = MAGMA_Z_IMAG( ajk ) / abs_ajk;
                    #endif

                    sJd[0] = cs_theta; // j00
                    sJd[1] = cs_theta; // j11
                    #ifdef COMPLEX
                    sJo[0] = -sn_theta * MAGMA_Z_MAKE(cs_alpha,  sn_alpha); // j01
                    sJo[1] =  sn_theta * MAGMA_Z_MAKE(cs_alpha, -sn_alpha); // j10
                    #else
                    sJo[0] = -sn_theta * cs_alpha; // J01
                    sJo[1] =  sn_theta * cs_alpha; // J10
                    #endif
                }
            }
            __syncthreads();

            if( noconv[ty] == 1 ) {
                double j00 = sJd[0];
                double j11 = sJd[1];
                magmaDoubleComplex j01 = sJo[0];
                magmaDoubleComplex j10 = sJo[1];

                // update rows & columns by the jacobi rotations (A = J' * A * J)
                // A = A * J (col. update)
                for(int in = tx; in < n; in+=ntx) {
                    magmaDoubleComplex aj_new = j00 * sA(in,j) + j10 * sA(in,k);
                    magmaDoubleComplex ak_new = j01 * sA(in,j) + j11 * sA(in,k);
                    sA(in,j) = aj_new;
                    sA(in,k) = ak_new;
                }
            }
            __syncthreads();

            if( noconv[ty] == 1 ) {
                double j00 = sJd[0];
                double j11 = sJd[1];
                magmaDoubleComplex j01 = sJo[0];
                magmaDoubleComplex j10 = sJo[1];
                // A = J' * A (row update)
                for(int in = tx; in < n; in+=ntx) {
                    magmaDoubleComplex aj_new = j00 * sA(j, in) + MAGMA_Z_CONJ(j10) * sA(k,in);
                    magmaDoubleComplex ak_new = MAGMA_Z_CONJ(j01) * sA(j,in) + j11  * sA(k,in);
                    sA(j,in) = aj_new;
                    sA(k,in) = ak_new;
                }

                // update vectors if needed
                if(jobz == 1) {
                    for(int in = tx; in < n; in+=ntx) {
                        magmaDoubleComplex vj_new = j00 * sV(in,j) + j10 * sV(in,k);
                        magmaDoubleComplex vk_new = j01 * sV(in,j) + j11 * sV(in,k);
                        sV(in,j) = vj_new;
                        sV(in,k) = vk_new;
                    }
                }
            }
            __syncthreads();

            converge &= !noconv[ty];
            __syncthreads();

            // get the next j, k
            j = get_next_index(n2, j);
            k = get_next_index(n2, k);
        } // loop over ipair

        // test for convergence
        if(tx == 0) noconv[ty] = !converge;
         __syncthreads();
        converge = 1;
        for(int iconv = 0; iconv < nty; ++iconv) {
            converge &= !noconv[iconv];
        }
    } // sweep
    __syncthreads();

    // sort eigenvalues
    if(gtx < n) {
        sW[gtx] = MAGMA_Z_REAL( sA(gtx, gtx) );
    }
    __syncthreads();

    if(sort_values == 1) {
        magmablas_zsort_eig(n, gtx, sW, sorder);
        __syncthreads();    // this is safe because sort_values is the same across all threads
    }

    // write eigenvalues, and optionally eigenvectors
    if(dW != NULL) {
        if(gtx < n) {
            dW[gtx] = sW[gtx]; // ascending order
        }
    }

    if(jobz == 1) {
        if( gtx < n) {
            for(int i = 0; i < n; i++){
                int ii = sorder[i];
                dA[i * ldda + gtx] = sV(gtx,ii);
            }
        }
    }

    if(gtx == 0) {
        *info = (converge == 1) ? 0 : sweep;

        // write #sweeps if required
        if(num_sweeps != NULL) {
            num_sweeps[batchid] = sweep;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zheevj_batched_small_sm_driver(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dA_array, magmaDoubleComplex *dA_base, magma_int_t ldda, magma_int_t strideA,
    double **dW_array, double *dW_base, magma_int_t strideW,
    magma_int_t* info_array, int* batch_mask, int* num_sweeps,
    double heevj_tol, magma_int_t sort_flag, magma_int_t max_sweeps,
    magma_int_t nthreads, magma_int_t ppairs,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    const bool vectors_required  = (jobz == MagmaVec);

    // TODO: choose tolerance
    double eps = lapackf77_dlamch("E");
    if(n == 0 || batchCount == 0) return 0;

    magma_int_t ijobz = vectors_required ? 1 : 0;

    const int n2    = 2 * ( (n+1)/2 );
    nthreads        = max(2, nthreads);
    int sort_values = (int)sort_flag;

    dim3 threads(nthreads, ppairs, 1);
    dim3 grid(batchCount, 1, 1);

    assert( (nthreads * ppairs) >= n2 );
    const magma_int_t slda  = SLDA(n2);
    magma_int_t shmem  = (slda  * n2 * sizeof(magmaDoubleComplex) );                   // the input matrix
                shmem += (ijobz == 0) ? 0 : (slda  * n2 * sizeof(magmaDoubleComplex)); // eigenvectors if required
                shmem += ppairs * (2 * sizeof(magmaDoubleComplex));            // j01, j10
                shmem += n2     * sizeof(double);                              // eigenvalues
                shmem += ppairs * (2 * sizeof(double));                        // j00, j11
                shmem += (n2)       * sizeof(int);   // used internally, no need to make it magma_int_t

                // the shared memory space below should be an integer array of length = parallel-pairs (ppairs * 1 * sizeof(int))
                // however, this shows an error in compute-sanitizer for an out-of-bound access
                // setting it to (ppairs * 2 * sizeof(int)) resolves the problem, but the reason is unclear
                // TODO: investigate why using (ppairs * 1 * sizeof(int)) causes an out-of-bound access
                shmem += ppairs * 2 * sizeof(int);   // used internally, no need to make it magma_int_t

    int nthreads_max, shmem_max; // need to be int for cudaDeviceGetAttribute
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zheevj_batched_parallel_small_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000
    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        printf("error: function %s requires too many threads or too much shared memory\n", __func__);
        return arginfo;
    }

    void *kernel_args[] = {&ijobz, &uplo, &n, &dA_array, &dA_base, &ldda, &strideA, &dW_array, &dW_base, &strideW, &info_array, &batch_mask, &num_sweeps, &eps, &heevj_tol, &sort_values, &max_sweeps, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zheevj_batched_parallel_small_sm_kernel, grid, threads, kernel_args, shmem, queue->cuda_stream());

    if( e != cudaSuccess ) {
        printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
    }

    return arginfo;
}


/***************************************************************************//**
    Purpose
    -------
    ZHEEVJ computes all eigenvalues and, optionally, eigenvectors of a
    Hermitian matrix A based on the Jacobi method

    NOTES:

    ** This is the batch version of the routine, which accepts pointer arrays.
       It solves the eigenvalue problem on a batch of matrices having the same
       dimensions.

    ** This is an experimental routine, currently supporting matrices that
       fit in the shared memory of the GPU. If the eigenvectors are required,
       additional shared memory workspace is required.

    ** This is an expert interface, through which the user can control several
       aspects of the eigensolver

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, length (batchCount).
             Each is a COMPLEX_16 array, dimension (LDA, N)
             On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
             leading N-by-N upper triangular part of A contains the
             upper triangular part of the matrix A.  If UPLO = MagmaLower,
             the leading N-by-N lower triangular part of A contains
             the lower triangular part of the matrix A.
             On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
             orthonormal eigenvectors of the matrix A.
             If JOBZ = MagmaNoVec, then A is unchanged upon completion

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[out]
    dW_array Array of pointers, length (batchCount)
             Each is a DOUBLE PRECISION array, dimension (N)
             The eigenvalues of each matrix A, sorted so that S(i) >= S(i+1).

    @param[out]
    info_array  INTEGER array, dimension(batchCount)
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  the algorithm failed to converge;

    @param[in]
    heevj_tol   DOUBLE PRECISION
                A scalar controlling the tolerance upon which Jacobi rotations
                are generated for an off-diagonal element ajk. During a Jacobi sweep,
                Jacobi rotations are generated for an off-diagonal element ajk if the
                condition below is true:
                |ajk| > heevj_tol * eps * sqrt( |ajj| * |akk| ),
                where eps is the machine epsilon.

    @param[in]
    sort_flag   INTEGER
                = 0: Do not sort the eigenvalues/eigenvectors
                = 1: Sort eigenvalues/eigenvectors in ascending order

    @param[in]
    nthreads    INTEGER
                Number of threads assigned for each off-diagonal element during a parallel
                Jacobi sweep. Minimum is 2 threads.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_heevd
*******************************************************************************/
extern "C" magma_int_t
magma_zheevj_batched_expert_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda,
    double **dW_array, magma_int_t* info_array,
    int* batch_mask, int *num_sweeps,
    double heevj_tol, magma_int_t sort_flag, magma_int_t max_sweeps,
    magma_int_t nthreads, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    // Test the input arguments
    if (! (jobz == MagmaNoVec || jobz == MagmaVec) ) {
        arginfo = -1;
    } else if (! (uplo == MagmaLower || uplo == MagmaUpper) ) {
        arginfo = -2;
    } else if (n < 0 ) {
        arginfo = -3;
    } else if (ldda < max(1,n)) {
        arginfo = -5;
    } else if (heevj_tol < 0) {
        arginfo = -8;
    } else if (sort_flag != 1 && sort_flag != 0) {
        arginfo = -9;
    } else if (nthreads < 2) {
        arginfo = -10;
    } else if (batchCount < 0) {
        arginfo = -11;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t ppairs   = magma_roundup(n,2) / 2;
    nthreads = max(nthreads, 2);

    magma_zheevj_batched_small_sm_driver(
        jobz, uplo, n,
        dA_array, NULL, ldda, 0,
        dW_array, NULL, 0,
        info_array, batch_mask, num_sweeps,
        heevj_tol, sort_flag, max_sweeps,
        nthreads, ppairs,
        batchCount, queue );

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZHEEVJ computes all eigenvalues and, optionally, eigenvectors of a
    Hermitian matrix A based on the Jacobi method

    NOTES:

    ** This is the batch version of the routine, which accepts a base pointer and a
       constant stride. It solves the eigenvalue problem on a batch of matrices having
       the same dimensions.

    ** This is an experimental routine, currently supporting matrices that
       fit in the shared memory of the GPU. If the eigenvectors are required,
       additional shared memory workspace is required.

    ** This is an expert interface, through which the user can control several
       aspects of the eigensolver

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA       DOUBLE PRECISION base pointer for the matrices in the batch,
             such that:
                dAi = dA + i * strideA, i = 0, 1, 2, ..., batchCount-1

             Each dAi is a COMPLEX_16 array, dimension (LDA, N)
             On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
             leading N-by-N upper triangular part of A contains the
             upper triangular part of the matrix A.  If UPLO = MagmaLower,
             the leading N-by-N lower triangular part of A contains
             the lower triangular part of the matrix A.
             On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
             orthonormal eigenvectors of the matrix A.
             If JOBZ = MagmaNoVec, then A is unchanged upon completion

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[in]
    strideA INTEGER
            Specifies the distance between two consecutive matrices in the batch.

    @param[out]
    dW       DOUBLE PRECISION base pointer for the eigenvalues in the batch,
             such that:
                dWi = dW + i * strideW, i = 0, 1, 2, ..., batchCount-1
             Each dWi is a DOUBLE PRECISION array, dimension (N)
             The eigenvalues of each matrix A,
             If sort_flag = 1, eigenvalues are sorted so that S(i) >= S(i+1).
             If sort_flag = 0, eigenvalues are unsorted.

    @param[in]
    strideW INTEGER
            Specifies the distance between two consecutive eigenvalue arrays
            in the batch.

    @param[out]
    info_array  INTEGER array, dimension(batchCount)
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  the algorithm failed to converge;

    @param[in]
    heevj_tol   DOUBLE PRECISION
                A scalar controlling the tolerance upon which Jacobi rotations
                are generated for an off-diagonal element ajk. During a Jacobi sweep,
                Jacobi rotations are generated for an off-diagonal element ajk if the
                condition below is true:
                |ajk| > heevj_tol * eps * sqrt( |ajj| * |akk| ),
                where eps is the machine epsilon.

    @param[in]
    sort_flag   INTEGER
                = 0: Do not sort the eigenvalues/eigenvectors
                = 1: Sort eigenvalues/eigenvectors in ascending order

    @param[in]
    nthreads    INTEGER
                Number of threads assigned for each off-diagonal element during a parallel
                Jacobi sweep. Minimum is 2 threads.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_heevd
*******************************************************************************/
extern "C" magma_int_t
magma_zheevj_batched_strided_expert_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n, magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    double *dW, magma_int_t strideW,
    magma_int_t* info_array, int* batch_mask, int *num_sweeps,
    double heevj_tol, magma_int_t sort_flag, magma_int_t max_sweeps,
    magma_int_t nthreads, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    // Test the input arguments
    if (! (jobz == MagmaNoVec || jobz == MagmaVec) ) {
        arginfo = -1;
    } else if (! (uplo == MagmaLower || uplo == MagmaUpper) ) {
        arginfo = -2;
    } else if (n < 0 ) {
        arginfo = -3;
    } else if (ldda < max(1,n)) {
        arginfo = -5;
    } else if (strideA < ldda*n) {
        arginfo = -6;
    } else if (strideW < n) {
        arginfo = -8;
    } else if (heevj_tol < 0) {
        arginfo = -10;
    } else if (sort_flag != 1 && sort_flag != 0) {
        arginfo = -11;
    } else if (nthreads < 2) {
        arginfo = -12;
    } else if (batchCount < 0) {
        arginfo = -13;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t ppairs   = magma_roundup(n,2) / 2;
    nthreads = max(nthreads, 2);

    magma_zheevj_batched_small_sm_driver(
        jobz, uplo, n,
        NULL, dA, ldda, strideA,
        NULL, dW, strideW,
        info_array, batch_mask, num_sweeps,
        heevj_tol, sort_flag, max_sweeps,
        nthreads, ppairs,
        batchCount, queue );

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZHEEVJ computes all eigenvalues and, optionally, eigenvectors of a
    Hermitian matrix A based on the Jacobi method

    NOTES:

    ** This is the batch version of the routine, which solves the eigenvalue
       problem on a batch of matrices having the same dimensions.

    ** This is an experimental routine, currently supporting matrices that
       fit in the shared memory of the GPU. If the eigenvectors are required,
       additional shared memory workspace is required.

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    dA_array Array of pointers, length (batchCount).
             Each is a COMPLEX_16 array, dimension (LDA, N)
             On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
             leading N-by-N upper triangular part of A contains the
             upper triangular part of the matrix A.  If UPLO = MagmaLower,
             the leading N-by-N lower triangular part of A contains
             the lower triangular part of the matrix A.
             On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
             orthonormal eigenvectors of the matrix A.
             If JOBZ = MagmaNoVec, then A is unchanged upon completion

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[out]
    dW_array Array of pointers, length (batchCount)
             Each is a DOUBLE PRECISION array, dimension (N)
             The eigenvalues of each matrix A, sorted so that S(i) >= S(i+1).

    @param[out]
    info_array  INTEGER array, dimension(batchCount)
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  the algorithm failed to converge;

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_heevd
*******************************************************************************/
extern "C" magma_int_t
magma_zheevj_batched_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dW_array,
    magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    // Test the input arguments
    if (! (jobz == MagmaNoVec || jobz == MagmaVec) ) {
        arginfo = -1;
    } else if (! (uplo == MagmaLower || uplo == MagmaUpper) ) {
        arginfo = -2;
    } else if (n < 0 ) {
        arginfo = -3;
    } else if (ldda < max(1,n)) {
        arginfo = -5;
    } else if (batchCount < 0) {
        arginfo = -8;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t nthreads   = magma_get_zheevj_batched_small_nthreads(n);

    // default params
    magma_int_t max_sweeps = 100;
    magma_int_t sort_flag  = 1;
    double heevj_tol       = 30;

    magma_zheevj_batched_expert_small_sm(
        jobz, uplo, n, dA_array, ldda,
        dW_array, info_array, NULL, NULL,
        heevj_tol, sort_flag, max_sweeps,
        nthreads, batchCount, queue );

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZHEEVJ computes all eigenvalues and, optionally, eigenvectors of a
    Hermitian matrix A based on the Jacobi method

    NOTES:

    ** This is the batch version of the routine, which accepts a base pointer and a
       constant stride. It solves the eigenvalue problem on a batch of matrices having
       the same dimensions.

    ** This is an experimental routine, currently supporting matrices that
       fit in the shared memory of the GPU. If the eigenvectors are required,
       additional shared memory workspace is required.

    ** This is an expert interface, through which the user can control several
       aspects of the eigensolver

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA       DOUBLE PRECISION base pointer for the matrices in the batch,
             such that:
                dAi = dA + i * strideA, i = 0, 1, 2, ..., batchCount-1

             Each dAi is a COMPLEX_16 array, dimension (LDA, N)
             On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
             leading N-by-N upper triangular part of A contains the
             upper triangular part of the matrix A.  If UPLO = MagmaLower,
             the leading N-by-N lower triangular part of A contains
             the lower triangular part of the matrix A.
             On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
             orthonormal eigenvectors of the matrix A.
             If JOBZ = MagmaNoVec, then A is unchanged upon completion

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[in]
    strideA INTEGER
            Specifies the distance between two consecutive matrices in the batch.

    @param[out]
    dW       DOUBLE PRECISION base pointer for the eigenvalues in the batch,
             such that:
                dWi = dW + i * strideW, i = 0, 1, 2, ..., batchCount-1
             Each dWi is a DOUBLE PRECISION array, dimension (N)
             The eigenvalues of each matrix A,
             If sort_flag = 1, eigenvalues are sorted so that S(i) >= S(i+1).
             If sort_flag = 0, eigenvalues are unsorted.

    @param[in]
    strideW INTEGER
            Specifies the distance between two consecutive eigenvalue arrays
            in the batch.

    @param[out]
    info_array  INTEGER array, dimension(batchCount)
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  the algorithm failed to converge;

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_heevd
*******************************************************************************/
extern "C" magma_int_t
magma_zheevj_batched_strided_small_sm(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    double *dW, magma_int_t strideW,
    magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    // Test the input arguments
    if (! (jobz == MagmaNoVec || jobz == MagmaVec) ) {
        arginfo = -1;
    } else if (! (uplo == MagmaLower || uplo == MagmaUpper) ) {
        arginfo = -2;
    } else if (n < 0 ) {
        arginfo = -3;
    } else if (ldda < max(1,n)) {
        arginfo = -5;
    } else if (strideA < ldda * n) {
        arginfo = -6;
    } else if (strideW < n) {
        arginfo = -8;
    }else if (batchCount < 0) {
        arginfo = -10;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t nthreads   = magma_get_zheevj_batched_small_nthreads(n);

    // default params
    magma_int_t max_sweeps = 100;
    magma_int_t sort_flag  = 1;
    double heevj_tol       = 30;


    magma_zheevj_batched_strided_expert_small_sm(
        jobz, uplo, n, dA, ldda, strideA,
        dW, strideW,
        info_array, NULL, NULL,
        heevj_tol, sort_flag, max_sweeps,
        nthreads, batchCount, queue );

    return arginfo;
}
