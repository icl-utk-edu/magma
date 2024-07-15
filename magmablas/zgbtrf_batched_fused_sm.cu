/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Stan Tomov

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"
#include "zgbtf2_devicefunc.cuh"

// use this so magmasubs will replace with relevant precision, so we can comment out
// the switch case that causes compilation failure
#define PRECISION_z

#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)        (max(1,64/(M)))
#else
#define NTCOL(M)        (max(1,64/(M)))
#endif

#define SLDAB(MBAND)    ((MBAND)+1)

////////////////////////////////////////////////////////////////////////////////
__global__ void
zgbtrf_batched_kernel_fused_sm(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, int lddab,
    magma_int_t** ipiv_array, magma_int_t *info_array,
    int batchCount)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    extern __shared__ magmaDoubleComplex zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    const int minmn   = min(m,n);
    const int kv      = kl + ku;
    const int mband   = (kl + 1 + kv);
    const int sldab   = SLDAB(mband);
    const int sldab_1 = sldab-1;

    magmaDoubleComplex* dAB = dAB_array[batchid];
    int linfo = 0;

    // shared memory pointers
    magmaDoubleComplex *sAB = (magmaDoubleComplex*)(zdata);
    double* dsx             = (double*)(sAB + blockDim.y * n * sldab);
    int* sipiv              = (int*)(dsx + blockDim.y * (kl+1));
    sAB   += ty * n * sldab;
    dsx   += ty * (kl+1);
    sipiv += ty * minmn;

    // init sAB
    for(int i = tx; i < n*sldab; i+=ntx) {
        sAB[i] = MAGMA_Z_ZERO;
    }
    __syncthreads();

    // read
    read_sAB(mband, n, kl, ku, dAB, lddab, sAB, sldab, ntx, tx);
    __syncthreads();

    int ju = 0;
    for(int j = 0; j < minmn; j++) {
        // izamax
        int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
        if(tx < km) {
            dsx[ tx ] = fabs(MAGMA_Z_REAL( sAB(kv+tx,j) )) + fabs(MAGMA_Z_IMAG( sAB(kv+tx,j) ));
        }
        __syncthreads();

        double rx_abs_max = dsx[0];
        int    jp       = 0;
        for(int i = 1; i < km; i++) {
            if( dsx[i] > rx_abs_max ) {
                rx_abs_max = dsx[i];
                jp         = i;
            }
        }

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (j+1) : linfo;

        if(tx == 0) {
            sipiv[j] = jp + j + 1;    // +1 for fortran indexing
        }

        ju = max(ju, min(j+ku+jp, n-1));
        int swap_len = ju - j + 1;

        // swap
        if( !(jp == 0) ) {
            magmaDoubleComplex tmp;
            magmaDoubleComplex *sR1 = &sAB(kv   ,j);
            magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
            for(int i = tx; i < swap_len; i+=ntx) {
                tmp              = sR1[i * sldab_1];
                sR1[i * sldab_1] = sR2[i * sldab_1];
                sR2[i * sldab_1] = tmp;
            }
        }
        __syncthreads();

        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        __syncthreads();

        // ger
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        magmaDoubleComplex *sU = &sAB(kv,j);
        magmaDoubleComplex *sV = &sAB(kv+1,j);
        if( tx < (km-1) ) {
            for(int jj = 1; jj < swap_len; jj++) {
                sV[jj * sldab_1 + tx] -= sV[tx] * sU[jj * sldab_1 + 0] * reg;
            }
        }
        __syncthreads();
    }

    // write info
    if(tx == 0) info_array[batchid] = linfo;

    // write pivot
    magma_int_t* ipiv = ipiv_array[batchid];
    for(int i = tx; i < minmn; i+=ntx) {
        ipiv[i] = (magma_int_t)sipiv[i];
    }

    write_sAB(mband, n, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

#undef sAB
#undef dAB
}

/***************************************************************************//**
    Purpose
    -------
    ZGBTRF computes an LU factorization of a COMPLEX m-by-n band matrix A
    using partial pivoting with row interchanges.

    This is the batched version of the algorithm, which performs the factorization
    on a batch of matrices with the same size and lower/upper bandwidths.

    This routine has shared memory requirements that may exceed the capacity of
    the GPU. In such a case, the routine exits immediately, returning a negative
    error code.

    Arguments
    ---------
    @param[in]
    M     INTEGER
          The number of rows of the matrix A.  M >= 0.

    @param[in]
    N     INTEGER
          The number of columns of the matrix A.  N >= 0.

    @param[in]
    KL    INTEGER
          The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    KU    INTEGER
          The number of superdiagonals within the band of A.  KU >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
          Each is a COMPLEX_16 array, dimension (LDDAB,N)
          On entry, the matrix AB in band storage, in rows KL+1 to
          2*KL+KU+1; rows 1 to KL of the array need not be set.
          The j-th column of A is stored in the j-th column of the
          array AB as follows:
          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)

          On exit, details of the factorization: U is stored as an
          upper triangular band matrix with KL+KU superdiagonals in
          rows 1 to KL+KU+1, and the multipliers used during the
          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
          See below for further details.

    @param[in]
    LDDAB INTEGER
          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.

    @param[out]
    dIPIV_array    Array of pointers, dimension (batchCount).
          Each is an INTEGER array, dimension (min(M,N))
          The pivot indices; for 1 <= i <= min(M,N), row i of the
          matrix was interchanged with row IPIV(i).

    @param[out]
    dINFO_array    INTEGER array, dimension (batchCount)
          Each is the INFO output for a given matrix
          = 0: successful exit
          < 0: if INFO = -i, the i-th argument had an illegal value
          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
               has been completed, but the factor U is exactly
               singular, and division by zero will occur if it is used
               to solve a system of equations.

    @param[in]
    nthreads    INTEGER
                The number of threads assigned to a single matrix.
                nthreads >= (KL+1)

    @param[in]
    ntcol       INTEGER
                The number of concurrent factorizations in a thread-block
                ntcol >= 1

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------

    The band storage scheme is illustrated by the following example, when
    M = N = 6, KL = 2, KU = 1:

    On entry:                       On exit:

      *    *    *    +    +    +       *    *    *   u14  u25  u36
      *    *    +    +    +    +       *    *   u13  u24  u35  u46
      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *

    Array elements marked * are not used by the routine, but may be set
    to zero after completion. Elements marked
    + need not be set on entry, but are required by the routine to store
    elements of U because of fill-in resulting from the row interchanges.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrf_batched_fused_sm(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    magma_int_t kv      = kl + ku;
    magma_int_t mband   = kv + 1 + kl;
    magma_int_t sldab   = SLDAB(mband);

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 ) return 0;

    nthreads = max( nthreads, (kl + 1) );
    ntcol    = max(1, ntcol);

    magma_int_t shmem  = 0;
    shmem += sldab * n * sizeof(magmaDoubleComplex); // sAB
    shmem += (kl + 1)  * sizeof(double);        // dsx
    shmem += min(m,n)  * sizeof(magma_int_t);   // pivot
    shmem *= ntcol;


    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 threads(nthreads, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgbtrf_batched_kernel_fused_sm, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &n, &kl, &ku, &dAB_array, &lddab, &ipiv_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgbtrf_batched_kernel_fused_sm, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}
