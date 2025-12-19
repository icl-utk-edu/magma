/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "swap_scalar.cuh"
#include "batched_gesvj_param.h"

#define COMPLEX
#define PRECISION_z

#define SLDA(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )
#define sA(i,j)    (sA[(j)*slda+(i)])
#define sV(i,j)    (sV[(j)*sldv+(i)])

////////////////////////////////////////////////////////////////////////////////
static __device__ __inline__ int
get_next_column(int n, int col)
{
    int newcol = 0;
    newcol = (col % 2 == 0          ) ? -2 : 2;
    newcol = (col == n-1 || col == 2) ? -1 : newcol;
    newcol = (col == 0              ) ?  0 : newcol;
    return (col + newcol);
}

////////////////////////////////////////////////////////////////////////////////
__device__ static __noinline__ void
magmablas_svd_compute_norms_n(int n, int i, double* xjj, double *xkk, magmaDoubleComplex *xjk)
{
    #pragma unroll
    for(int step = 1024; step > 0; step >>= 1) {
        if(n > step) {
            if ( i < step && i + step < n ) { xjj[i] += xjj[i+step]; xkk[i] += xkk[i+step]; xjk[i] += xjk[i+step]; }
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////
__device__ static __noinline__ void
magmablas_dmax_key_device(const int n, const int i, double* x, int* ind)
{
    #pragma unroll
    for(int step = 1024; step > 0; step >>= 1) {
        if ( n > step ) {
            if ( i < step && i + step < n ) {
                if ( x[i] < x[i+step] ) {
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
magmablas_zsort_svd(const int n, const int tx, double* dS, int* sindex)
{
    #pragma unroll
    for(int iN = 0; iN < n-1; iN++) {
        magmablas_dmax_key_device(n-iN, tx, dS+iN, sindex+iN);
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void
zgesvj_batched_small_sm_kernel(
    int jobu, int jobv,
    int morg, int norg,
    magmaDoubleComplex **dA_array, int ldda, double** dS_array,
    magmaDoubleComplex **dU_array, int lddu,
    magmaDoubleComplex **dV_array, int lddv,
    double tol, int max_sweeps, magma_int_t *info_array, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];

    // this kernel supports only (morg >= norg), so if otherwise,
    // we solve the conj-transpose of A
    const int m    = ( morg >= norg ) ? morg : norg;
    const int n    = ( morg >= norg ) ? norg : morg;
    //const int jobu = ( morg >= norg ) ? jobu_org : jobv_org;
    //const int jobv = ( morg >= norg ) ? jobv_org : jobu_org;

    const int  tx = threadIdx.x;
    const int  ty = threadIdx.y;
    const int ntx = blockDim.x;
    const int nty = blockDim.y;
    const int gtx = (ty * ntx) + tx;
    const int total_threads = nty * ntx;
    const int batchid = blockIdx.x;
    const int n2      = 2 * ( (n+1)/2 );
    double s = MAGMA_D_ZERO;

    magmaDoubleComplex* dA = dA_array[batchid];
    double* dS = dS_array[batchid];
    magma_int_t* info = &info_array[batchid];
    const int slda = SLDA(m);
    const int sldv = SLDA(n2);

    magmaDoubleComplex *sA      = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex *sV      = sA + slda*n2;
    magmaDoubleComplex *sxjk    = (jobv == 1) ? (sV + sldv * n2):
                                                (sA + slda * n2);
    magmaDoubleComplex *sJo     = sxjk + (nty * ntx);            // j10, j01
    double             *sxjj    = (double*)(sJo + nty * 2);
    double             *sxkk    = sxjj + (nty * ntx) ;
    double             *sJd     = sxkk + (nty * ntx); // j00, j11
    double             *dsSigma = sJd +  (nty * 2);
    int                *sorder  = (int*)(dsSigma + n2);
    int                *noconv  = sorder + n2;

    // advance pointers w.r.t ty
    sxjk   += ty * ntx;
    sxjj   += ty * ntx;
    sxkk   += ty * ntx;
    sJd    += ty * 2;
    sJo    += ty * 2;

    // init sorder and sV
    if( gtx < n2 ) {
        sorder[gtx] = gtx;
        if(jobv == 1) {
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
    if(morg >= norg) {
        for(int im = gtx; im < morg; im+=total_threads) {
            for(int in = 0; in < norg; in++){
                sA(im,in) = dA[in * ldda + im];
            }
        }
    }
    else {
        for(int im = gtx; im < morg; im+=total_threads) {
            for(int in = 0; in < norg; in++){
                sA(in,im) = MAGMA_Z_CONJ( dA[in * ldda + im] );
            }
        }
    }

    // if n is odd, init the last column in sA to zero
    if( !(n == n2) ) {
        for(int im = gtx; im < m; im+=total_threads) {
            sA(im,n2-1) = MAGMA_Z_ZERO;
        }
    }
    __syncthreads();

    int sweep    = 0;
    int converge = 0;
    int j, k;
    while ( sweep < max_sweeps && converge == 0 ) {
        j = 2*ty;
        k = j + 1;

        converge = 1;
        sweep++;

        for(int ipair = 0; ipair < n2-1; ++ipair) {
            // compute norms
            double ajj = MAGMA_D_ZERO;
            double akk = MAGMA_D_ZERO;
            magmaDoubleComplex ajk = MAGMA_Z_ZERO;

            // compute the norms
            for(int im = tx; im < m; im+=ntx) {
                magmaDoubleComplex ej = sA(im,j);
                magmaDoubleComplex ek = sA(im,k);
                ajj += MAGMA_Z_REAL(ej) * MAGMA_Z_REAL(ej) + MAGMA_Z_IMAG(ej) * MAGMA_Z_IMAG(ej);
                akk += MAGMA_Z_REAL(ek) * MAGMA_Z_REAL(ek) + MAGMA_Z_IMAG(ek) * MAGMA_Z_IMAG(ek);
                ajk += MAGMA_Z_CONJ(ej) * ek;
            }
            sxjj[tx] = ajj;
            sxkk[tx] = akk;
            sxjk[tx] = ajk;
            __syncthreads();
            magmablas_svd_compute_norms_n(ntx, tx, sxjj, sxkk, sxjk);
            // no need to sync -- the function already syncs at the end

            if(tx == 0) {
                noconv[ty] = 0;
                ajj        = sxjj[0];
                akk        = sxkk[0];
                ajk        = sxjk[0];

                double threshold   = tol * sqrt(ajj * akk);
                double abs_ajk     = MAGMA_Z_ABS( ajk );
                double abs_ajj_akk = MAGMA_D_ABS(ajj - akk);

                if( abs_ajk > threshold) {
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
                    sJo[0] = -sn_theta * cs_alpha;
                    sJo[1] =  sn_theta * cs_alpha;
                    #endif
                }
            }
            __syncthreads();

            if( noconv[ty] == 1 ) {
                double j00 = sJd[0];
                double j11 = sJd[1];
                magmaDoubleComplex j01 = sJo[0];
                magmaDoubleComplex j10 = sJo[1];

                // update vectors
                for(int im = tx; im < m; im+=ntx) {
                    magmaDoubleComplex aj_new = j00 * sA(im,j) + j10 * sA(im,k);
                    magmaDoubleComplex ak_new = j01 * sA(im,j) + j11 * sA(im,k);
                    sA(im,j) = aj_new;
                    sA(im,k) = ak_new;
                }

                // update rV if needed
                if(jobv == 1) {
                    for(int in = tx; in < n2; in+=ntx) {
                        ajk      = j00 * sV(in,j) + j10 * sV(in,k);
                        sV(in,k) = j01 * sV(in,j) + j11 * sV(in,k);
                        sV(in,j) = ajk;
                    }
                }
            }
            converge &= !noconv[ty];
            __syncthreads();

            // get the next j, k
            j = get_next_column(n2, j);
            k = get_next_column(n2, k);
        }

        // test for convergence
        if(tx == 0) noconv[ty] = !converge;
         __syncthreads();
        converge = 1;
        for(int iconv = 0; iconv < nty; ++iconv) {
            converge &= !noconv[iconv];
        }
    }
    __syncthreads();

    // singular values are now the vec-norm of sA
    s = MAGMA_D_ZERO;
    if(gtx < n2) {
        for(int i = 0; i < m; i++) {
            s += MAGMA_Z_REAL( sA(i,gtx) ) * MAGMA_Z_REAL( sA(i,gtx) ) +
                 MAGMA_Z_IMAG( sA(i,gtx) ) * MAGMA_Z_IMAG( sA(i,gtx) );
        }
        s = sqrt(s);
        dsSigma[gtx] = s;
    }
    __syncthreads();
    magmablas_zsort_svd(n2, gtx, dsSigma, sorder);
    __syncthreads();

    if(gtx < n) {
        dS[gtx] = dsSigma[gtx];
    }

    if(gtx == 0) {
        *info = (converge == 1) ? 0 : sweep;
    }

    int ii;
    if(jobu == 1) {
        magmaDoubleComplex* dU  = (morg >= norg) ? dU_array[batchid] : dV_array[batchid];
        int lddu_ = (morg >= norg) ? lddu : lddv;
        if( gtx < n ) {
            s = (s == MAGMA_D_ZERO) ? MAGMA_D_ONE : MAGMA_D_DIV(1., s);
            #pragma unroll
            for(int i = 0; i < m; i++) {
                sA(i,gtx) *= s;
            }
        }
        __syncthreads();

        for(int im = gtx; im < m; im+=total_threads) {
            for(int in = 0; in < n; in++){
                ii = sorder[in];
                dU[ in * lddu_ + im ] = sA(im, ii);
            }
        }
    }

    if(jobv == 1) {
        magmaDoubleComplex* dV = (morg >= norg) ? dV_array[batchid] : dU_array[batchid];
        int lddv_ = (morg >= norg) ? lddv : lddu;
        if( gtx < n) {
            #pragma unroll
            for(int i = 0; i < n; i++){
                ii = sorder[i];
                dV[i * lddv_ + gtx] = sV(gtx,ii);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// @return the shared memory requirements (in bytes) for the batch fused gesvj routine
static magma_int_t
magma_zgesvj_batched_small_sm_size(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg, magma_int_t nthreads)
{
    // this kernel supports only (morg >= norg), so if otherwise,
    // we solve the conj-transpose of A
    magma_int_t m = ( morg >= norg ) ? morg : norg;
    magma_int_t n = ( morg >= norg ) ? norg : morg;
    magma_int_t n2     = magma_roundup(n, 2);
    magma_int_t ppairs = n2/2;

    magma_int_t slda  = SLDA(m);
    magma_int_t sldv  = SLDA(n2);
    magma_int_t shmem  = (slda  * n2 * sizeof(magmaDoubleComplex) ); // the input matrix
                shmem += (jobv == MagmaNoVec) ? 0 : (sldv  * n2 * sizeof(magmaDoubleComplex)); // right singular vectors if required
                shmem += ppairs * (1 * nthreads * sizeof(magmaDoubleComplex)); // norm of ajk
                shmem += ppairs * (2 * sizeof(magmaDoubleComplex));            // j01, j10
                shmem += ppairs * (2 * nthreads * sizeof(double));             // norms of ajj, akk
                shmem += ppairs * (2 * sizeof(double));                        // j00, j11
                shmem += (n2 * sizeof(double));      // singular values
                shmem += (n2)  * sizeof(int);        // used internally, no need to make it magma_int_t
                shmem += ppairs * 1 * sizeof(int);   // used internally, no need to make it magma_int_t

    return shmem;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgesvj_batched_small_sm_driver(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t morg, magma_int_t norg,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* info_array,
    magma_int_t batchCount,
    magma_int_t ntx, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    // this kernel supports only (morg >= norg), so if otherwise,
    // we solve the conj-transpose of A
    int m    = ( morg >= norg ) ? morg : norg;
    int n    = ( morg >= norg ) ? norg : morg;

    magma_int_t ijobu = (jobu == MagmaNoVec) ? 0 : 1;
    magma_int_t ijobv = (jobv == MagmaNoVec) ? 0 : 1;
    if( morg < norg ) {
        // swap jobu & jobv
        magma_int_t tmp = ijobu;
        ijobu = ijobv;
        ijobv = tmp;
    }

    // TODO: choose tolerance
    double tol = 30*lapackf77_dlamch("E");
    magma_int_t max_sweeps = 100;
    if(m == 0 || n == 0 || batchCount == 0) return 0;


    const int n2          = 2 * ( (n+1)/2 );
    const int nthreads    = ntx; //min(m, 32); //max(min(m, max_threads), n);
    const int ppairs      = n2/2;

    dim3 threads(nthreads, ppairs, 1);
    dim3 grid(batchCount, 1, 1);

    assert( (nthreads * ppairs) >= n2 );
    magma_int_t shmem = magma_zgesvj_batched_small_sm_size(jobu, jobv, morg, norg, nthreads);

    magma_int_t nthreads_max, shmem_max;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgesvj_batched_small_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000
    if ( nthreads > nthreads_max || shmem > shmem_max ) {
        //printf("error: function %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;;
    }

    void *kernel_args[] = {&ijobu, &ijobv, &morg, &norg, &dA_array, &ldda, &dS_array, &dU_array, &lddu, &dV_array, &lddv, &tol, &max_sweeps, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgesvj_batched_small_sm_kernel, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
// helper function to get the number of threads
static magma_int_t magma_zgesvj_batched_get_nthreads(
                magma_vec_t jobu, magma_vec_t jobv,
                magma_int_t morg, magma_int_t norg )
{
    // tuning is for m >=  only
    magma_int_t m = (morg >= norg) ? morg : norg;
    //magma_int_t n = (morg >= norg) ? norg : morg;

    magma_int_t nthreads = 4; // a default value
    magma_int_t *nthread_table = NULL;

    magma_int_t arch = magma_getdevice_arch();

    #ifdef MAGMA_HAVE_CUDA
    magma_int_t nthreads_ss_sm80[] = {zgesvj_ss_nthreads_sm80};
    magma_int_t nthreads_sn_sm80[] = {zgesvj_sn_nthreads_sm80};
    magma_int_t nthreads_ss_sm70[] = {zgesvj_ss_nthreads_sm70};
    magma_int_t nthreads_sn_sm70[] = {zgesvj_sn_nthreads_sm70};
    if(arch >= 800) {
        nthread_table = (jobu == MagmaSomeVec && jobv == MagmaSomeVec) ?
                        nthreads_ss_sm80 : nthreads_sn_sm80;
    }
    else {
        nthread_table = (jobu == MagmaSomeVec && jobv == MagmaSomeVec) ?
                        nthreads_ss_sm70 : nthreads_sn_sm70;
    }

    #elif defined(MAGMA_HAVE_HIP)
    magma_int_t nthreads_ss_gfx908[] = {zgesvj_ss_nthreads_gfx908};
    magma_int_t nthreads_sn_gfx908[] = {zgesvj_sn_nthreads_gfx908};
    nthread_table = (jobu == MagmaSomeVec && jobv == MagmaSomeVec) ?
                        nthreads_ss_gfx908 : nthreads_sn_gfx908;
    #endif

    if( nthread_table != NULL) {
        magma_int_t m_index = min(m-1, 64-1);
        nthreads = nthread_table[m_index];
        nthreads = min(nthreads, m); // nthreads should not exceed #rows
    }

    return nthreads;
}

////////////////////////////////////////////////////////////////////////////////
/// @return a recommendation whether the fused batch gesvj batched kernel should be used
bool magma_zgesvj_batched_use_fused( magma_vec_t jobu, magma_vec_t jobv, magma_int_t m, magma_int_t n )
{
    // Decision is based on shared memory requirement
    // NVIDIA GPUs usually have more memory than AMD GPUs
    // Empirically chosen values for occupancy (resident svd's per SM/CU)
    #ifdef MAGMA_HAVE_CUDA
    magma_int_t min_resident_svds = 16;
    #else
    magma_int_t min_resident_svds =  8;
    #endif

    magma_int_t max_shmem = magma_getdevice_shmem_block_optin();
    magma_int_t nthreads  = magma_zgesvj_batched_get_nthreads( jobu, jobv, m, n );
    magma_int_t shmem     = magma_zgesvj_batched_small_sm_size( jobu, jobv, m, n, nthreads );

    magma_int_t estimated_occupancy = max_shmem / shmem;

    return ( estimated_occupancy >=  min_resident_svds ) ? true : false;
}

/***************************************************************************//**
    Purpose
    -------
    ZGESVJ computes the singular value decomposition (SVD) of an M-by-N
    matrix A , optionally computing the left and/or right singular vectors.
    The SVD is written as:

         A = U * SIGMA * conjugate-transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
    V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    NOTES:

    ** This routines computes only the economy size SVD based on the one-sided
       Jacobi algorithm

    ** This is the batch version of the routine, which performs the SVD
       on a batch of matrices having the same dimensions.

    ** This is an internal routine.
       Each individual matrix should fit in the shared memory of the
       GPU. If the right singular vectors are required, additional
       shared memory workspace is required.

    Arguments
    ---------
    @param[in]
    jobu    magma_vec_t
            Specifies options for computing all or part of the matrix U:
      -     = MagmaSomeVec: the first min(m,n) columns of U (the left singular
                            vectors) are computed.
      -     = MagmaNoVec:   no columns of U (no left singular vectors) are
                            computed.

    @param[in]
    jobv    magma_vec_t
            Specifies options for computing the matrix V:
      -     = MagmaSomeVec: the first min(m,n) rows of V (the right singular
                            vectors) are returned in the array V;
      -     = MagmaNoVec:   no rows of V (no right singular vectors) are
                            computed.
    @param[in]
    m       INTEGER
            The number of rows of each input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each input matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, length (batchCount).
             Each is a COMPLEX_16 array, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit,
      -      if JOBU = MagmaSomeVec, and dA_array is the same as dU_array,
             A is overwritten with the first min(m,n) columns of U
             (the left singular vectors, stored columnwise);
      -      if JOBU = MagmaNoVec, or dA_array is different from dU_array, then
             A is unchanged on exit

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDA >= max(1,M).

    @param[out]
    dS_array Array of pointers, length (batchCount)
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The singular values of each matrix A, sorted so that S(i) >= S(i+1).

    @param[out]
    dU_array Array of pointers, length (batchCount)
             Each is a COMPLEX_16 array, dimension (LDDU,N)
      -      if JOBU = MagmaSomeVec, U contains the first min(m,n) columns of U
             (the left singular vectors, stored columnwise);
      -      if JOBU = MagmaNoVec, U is not referenced.

    @param[in]
    lddu    INTEGER
            The leading dimension of each array U.  lddu >= max(1,M);

    @param[out]
    dV_array Array of pointers, length (batchCount)
             Each is a COMPLEX_16 array, dimension (LDDV,N)
      -      if JOBV = MagmaSomeVec, V contains the first n columns of V
             (the right singular vectors, stored columnwise);
      -      if JOBV = MagmaNoVec, V is not referenced.

    @param[in]
    lddv    INTEGER
            The leading dimension of each array V.  lddv >= max(1,N);

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gesvd
*******************************************************************************/

extern "C"
magma_int_t
magma_zgesvj_batched_small_sm(
    magma_vec_t jobu, magma_vec_t jobv,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t ldda, double **dS_array,
    magmaDoubleComplex** dU_array, magma_int_t lddu,
    magmaDoubleComplex** dV_array, magma_int_t lddv,
    magma_int_t* info_array, magma_int_t batchCount,
    magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const bool want_us  = (jobu == MagmaSomeVec);
    const bool want_un  = (jobu == MagmaNoVec);

    const bool want_vs  = (jobv == MagmaSomeVec);
    const bool want_vn  = (jobv == MagmaNoVec);

    // Test the input arguments
    if (! (want_us || want_un) ) {
        arginfo = -1;
    } else if (! (want_vs || want_vn) ) {
        arginfo = -2;
    } else if (m < 0 ) {
        arginfo = -3;
    } else if (n < 0) {
        arginfo = -4;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    } else if ((lddv < 1) || (want_vs && (lddv < n)) ) {
        arginfo = -9;
    } else if (batchCount < 0) {
        arginfo = -11;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t nthreads = magma_zgesvj_batched_get_nthreads(jobu, jobv, m, n);

    arginfo = magma_zgesvj_batched_small_sm_driver(
                jobu, jobv, m, n,
                dA_array, ldda, dS_array,
                dU_array, lddu,
                dV_array, lddv,
                info_array, batchCount,
                nthreads, queue );

    return arginfo;
}
