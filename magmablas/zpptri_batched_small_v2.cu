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

#define dA(i_, j_)   dA[ (j_) * ldda  + (i_)]
#define dAP(i_, j_) dAP[N*(j_) - (j_)*((j_)+1)/2 + (i_)]

#define dW11(i_, j_) dW11[N1*(j_) - (j_)*((j_)+1)/2 + (i_)]
#define dW22(i_, j_) dW22[N1*(j_) - (j_)*((j_)+1)/2 + (i_)]
#define dW21(i_, j_) dW21[(j_) * N2 + (i_)]

#define  sA(i_, j_) sA[(j_)*slda + (i_)]
#define  sB(i_, j_) sB[(j_)*sldb + (i_)]

#define SLDA(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )
#define SLDB(n)    ( ((n+1)%4) == 0 ? (n) : (n+1) )

#define BLK_X 32
#define BLK_Y  4

////////////////////////////////////////////////////////////////////////////////
// Subdivides the lower-packed matrix L into [L11 0; L21 L22]
// L11 is N1xN1, L22 is N2xN2
// N1 = ceil(N/2), N2 = N-N1, so N1 >= N2
// inverts L11 and L22 into W11 and W22 (workspace)
// computes the diagonal blocks of B = inv(L):
// B11 = W11' * W11 (overwrites L11)
// B22 = W22' * W22 (overwrites L22)
template<int N, int N1, int N2>
__global__
__launch_bounds__(N)
void
zpptri_lower_1of3_small_batched_kernel(
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex*  dW,
        int batchCount,
        magma_int_t *info_array )
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeW           = (N1+1)*N1/2;
    constexpr int sizeW_aligned   = ( (sizeW + alignment - 1) / alignment) * alignment;
    constexpr int sizeW_N1        = ( sizeW / N1 ) * N1;

    const int batchid  = blockIdx.x;
    const int batchgrp = batchid / 2;
    bool inverting_n2  = ( batchid % 2 == 1 ) ? true : false;

    // set workspace ptrs
    magmaDoubleComplex *dW = dW + batchid * sizeW;

    const int sldb = SLDB(N);
    const int tx  = threadIdx.x;
    const int Ai  = inverting_n2 ? N1 : 0;
    const int myN = inverting_n2 ? N2 : N1;

    magmaDoubleComplex* dAP = dAP_array[batchgrp];
    magmaDoubleComplex *sA  = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB  = sA + sizeW_aligned;

    // init B to identity
    #pragma unroll
    for(int i = 0; i < N1; i++) {
        sB(tx, i) = MAGMA_Z_ZERO;
    }
    sB(tx, tx) = MAGMA_Z_ONE;

    // read A
    #pragma unroll
    for(int j = 0; j < N2; j++){
        if(tx >= j && tx < myN) {
            sA(tx, j) = ( tx < myN ) ? dAP(Ai+tx, Ai+j) : MAGMA_Z_ZERO;
        }
    }
    if( tx == N1-1 ) sA(tx, N-1) = inverting_n2 ? MAGMA_Z_ONE : dAP(Ai+tx, Ai+N-1)
    __syncthreads();

    // compute 1 / diagonal(sA)
    sA(tx, tx) = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(tx,tx));
    __syncthreads();

    print_memory( "sA", sizeW, 1, sA, sizeW, 0, 0, 0, 0, 0, 0);
    print_memory( "sB",    N1, 1, sB, sldb,  0, 0, 0, 0, 0, 0);

    // Solving L L^T x = b
    // First, solve L y = b for y
    #pragma unroll
    for(int i = 0; i < N1; i++) {
        sB(i,tx) *= sA(i,i);
        #pragma unroll
        for(int j = i+1; j < N1; j++) {
            sB(j,tx) -= sB(i,tx) * sA(j,i);
        }
    }

    // Second, solve L^T x = y for x
    #pragma unroll
    for(int i = N1-1; i >= 0; i--) {
        sB(i,tx) *= MAGMA_Z_CONJ(sA(i,i));
        #pragma unroll
        for(int j = i-1; j >= 0; j--) {
            sB(j,tx) -= sB(i, tx) * MAGMA_Z_CONJ( sA(i,j) );
        }
    }
    __syncthreads();

    print_memory( "sB",    N1, 1, sB, sldb,  0, 0, 0, 0, 0, 0);

    // convert the inverse in sB from col-major to packed in sA
    #pragma unroll
    for(int j = 0; j < N1; j++) {
        if(tx >= j) {
            sA(tx,j) = sB(tx,j);
        }
    }
    __syncthreads();

    // write the inverse to global memory (contiguously in workspace dW)
    #pragma unroll
    for(int i = 0; i < sizeW_N1; i+=N1){
        dW[ i + tx ] = sA[ i + tx ];
    }

    if(tx < (sizeA - sizeA_N)) {
        dW[sizeA_N + tx] = sA[sizeA_N + tx];
    }

    __syncthreads();

    // now compute sB' * sB
    int offset = 0;
    #pragma unroll
    for(int j = 0; j < N1; j++) {
        magmaDoubleComplex rTmp = MAGMA_Z_ZERO;
        #pragma unroll
        for(int i = 0; i < N1; i++) {
            rTmp += MAGMA_Z_CONJ( sB(i,tx) ) * sB(i,j);
        }
        if( tx >= j && tx < myN ) sA[offset + tx] = rTmp;
        offset += N1-j;
    }
    __syncthreads();

    // overwrite dAP with sA
    // write the inverse to global memory (lower packed)
    #pragma unroll
    for(int j = 0; j < N2; j++){
        if(tx >= j && tx < myN) {
            dAP(Ai+tx, Ai+j) = sA(tx, j);
        }
    }
    if( inverting_n2 == false && tx == N1-1 ) dAP(Ai+tx, Ai+N-1) = sA(tx, N-1);
}

////////////////////////////////////////////////////////////////////////////////
// computes the following product
// W21 = W22 * (-L21 * W11);
// B21 = B22 * (-L21 * W11);
// where
// * W11, W22, and B22 are computed by 'zpptri_lower_1of3_small_batched_kernel'
// * L21 is the off-diagonal block of AP (packed)
template<int N, int BX, int BY, int TX, int TY>
__global__
__launch_bounds__(TX*TY)
void
zpptri_lower_2of3_small_batched_kernel(
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex*  dW,
        int batchCount,
        magma_int_t *info_array )
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int N1 = (N + 1) / 2;
    constexpr int N2 = N - N1;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int batchid  = blockIdx.x;

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeW           = (N1+1)*N1/2;
    //constexpr int sizeW_aligned   = ( (sizeW + alignment - 1) / alignment) * alignment;
    constexpr int sizeW_N1        = ( sizeW / N1 ) * N1;

    // set workspace ptrs
    magmaDoubleComplex *dW11 = dW   + batchid * 2 * sizeW;
    magmaDoubleComplex *dW22 = dW11 + sizeW;
    magmaDoubleComplex *dW21 = dW11; // W21 overwrites W11 and W22

    const int slda = SLDA(N);
    const int sldb = SLDB(N);

    magmaDoubleComplex* dAP = dAP_array[batchid];
    magmaDoubleComplex *sA  = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex *sB  = sA + (BX * BY);

    magmaDoubleComplex rC[BY/TY][BX/TX] = {MAGMA_Z_ZERO};

    // init sA/sB to zero
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            sA(i+tx,j+ty) = MAGMA_Z_ZERO;
            sB(i+tx,j+ty) = MAGMA_Z_ZERO;
        }

    // read L21 into sA and W11 into sB
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            sA(cx, cy) = ( cx < N2 && cy < N1            ) ? dAP(N1+cx,cy) : MAGMA_Z_ZERO;
            sB(cx, cy) = ( cx < N1 && cy < N1 && cy < cx ) ? dW11(cx, cy)  : MAGMA_Z_ZERO;
        }
    __syncthreads();

    // compute rC = -L21 * W11
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX)
            #pragma unroll
            for(int k = 0; k < N1; k++)
                rC[j/TY][i/TX] -= sA(i+tx,k) * sB(k,j+ty);
    __syncthreads();

    // write rC into sB
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX)
            sB(i+tx,j+ty) = rC[j/TY][i/TX];
    __syncthreads();

    // read W22 into sA
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            sA(cx, cy) = ( cx < N1 && cy < N1 && cy < cx ) ? dW22(cx, cy)  : MAGMA_Z_ZERO;
        }
    __syncthreads();

    // rC = W22 * sB
    // compute rC = -L21 * W11
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            rC[j/TY][i/TX] = MAGMA_Z_ZERO;
            #pragma unroll
            for(int k = 0; k < N1; k++)
                rC[j/TY][i/TX] += sA(i+tx,k) * sB(k,j+ty);
        }
    __syncthreads();

    // write rC into dW21 -- column major in a workspace
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            if( cx < N2 && cy < N1 )
                dW21(cx, cy) = rC[j/TY][i/TX];
        }

    // read B22 into sA
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            sA(cx, cy) = ( cx < N2 && cy < N2 && cy < cx ) ? dAP(N1+cx, N1+cy)  : MAGMA_Z_ZERO;
        }
    __syncthreads();

    // rC = B22 * sB
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            rC[j/TY][i/TX] = MAGMA_Z_ZERO;
            #pragma unroll
            for(int k = 0; k < N1; k++)
                rC[j/TY][i/TX] += sA(i+tx,k) * sB(k,j+ty);
        }
    __syncthreads();

    // write rC into dB21 -- off diagonal block in packed format
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            if( cx < N2 && cy < N1 )
                dAP(N1+cx, cy) = rC[j/TY][i/TX];
        }
}

////////////////////////////////////////////////////////////////////////////////
// Compute the following product:
// B11 = B11 + W21'* W21;
// B11 is lower packed  (N1 * N1)
// W21 is column major in workspace (N2 * N1)
template<int N, int BX, int BY, int TX, int TY>
__global__
__launch_bounds__(TX*TY)
void
zpptri_lower_3of3_small_batched_kernel(
        magmaDoubleComplex** dAP_array,
        magmaDoubleComplex*  dW,
        int batchCount,
        magma_int_t *info_array )
{
    extern __shared__ magmaDoubleComplex zdata[];

    constexpr int N1 = (N + 1) / 2;
    constexpr int N2 = N - N1;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int batchid  = blockIdx.x;

    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);
    constexpr int sizeW           = (N1+1)*N1/2;

    // set workspace ptrs
    magmaDoubleComplex *dW21 = dW   + batchid * 2 * sizeW;

    const int slda = SLDA(N);

    magmaDoubleComplex* dAP = dAP_array[batchid];
    magmaDoubleComplex *sA  = (magmaDoubleComplex*)zdata;

    magmaDoubleComplex rC[BY/TY][BX/TX] = {MAGMA_Z_ZERO};

    // read B11 into rC
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            rC[j/TY][i/TX] = ( cx < N1 && cy < N1 && cy < cx ) ? dAP(cx, cy)  : MAGMA_Z_ZERO;
        }

    // read W21 into sA
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            sA(cx, cy) = ( cx < N2 && cy < N1 && ) ? dW21(N1+cx,cy) : MAGMA_Z_ZERO;
        }
    __syncthreads();

    // compute W21'*W21 and accumulate into rC
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX)
            #pragma unroll
            for(int k = 0; k < N1; k++)
                rC[j/TY][i/TX] += MAGMA_Z_CONJ( sB(k,i+tx) ) * sB(k,j+ty);
    __syncthreads();

    // write rC into B11 -- diagonal block in packed format
    #pragma unroll
    for(int j = 0; j < BY; j+= TY)
        #pragma unroll
        for(int i = 0; i < BX; i+=TX) {
            int cx = i + tx;
            int cy = j + ty;
            if( cx < N1 && cy < N1 && cy < cx )
                dAP(cx, cy) = rC[j/TY][i/TX];
        }
}

////////////////////////////////////////////////////////////////////////////////
template<int N>
magma_int_t
zpptri_lower_batched_small_v2_kernel_driver(
    magmaDoubleComplex** dAP_array, magmaDoubleComplex* dW,
    int batchCount, magma_int_t *info_array, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    if ( batchCount < 0 )
        arginfo = -2;

    if (arginfo != 0) {
        return arginfo;
    }

    if( N == 0 || batchCount == 0 ) return 0;

    // constants
    constexpr int N1              = (N + 1) / 2;
    constexpr int N2              = N - N1;
    constexpr int alignment_bytes = 128;
    constexpr int alignment       = alignment_bytes / sizeof(magmaDoubleComplex);

    constexpr int sizeA11         = (N1+1)*N1/2;
    constexpr int sizeA21         = N2 * N1;
    constexpr int sizeA11_aligned = ( (sizeA11 + alignment - 1) / alignment) * alignment;
    constexpr int sizeA21_aligned = ( (sizeA21 + alignment - 1) / alignment) * alignment;

    // configure shared memory
    magma_int_t shmem_1 = 0;
    shmem_1 += 2 * sizeA11_aligned * sizeof(magmaDoubleComplex);

    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zpptri_lower_1of3_small_batched_kernel<N, N1, N2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_1);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        arginfo = -100;;
    }
    else {
        dim3 threads(N, 1, 1);
        dim3 grid(2*batchCount, 1, 1);
        void *kernel_args[] = {&dAP_array, &dW, &batchCount, &info_array};
        cudaError_t e = cudaLaunchKernel((void*)zpptri_lower_1of3_small_batched_kernel<N, N1, N2>, grid, threads, kernel_args, shmem, queue->cuda_stream());
        if( e != cudaSuccess ) {
            //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
                arginfo = -100;
        }
    }
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    PPTRI computes the inverse of a Hermitian positive definite matrix A using
    the Cholesky factorization A = U**H*U or A = L*L**H computed by PPTRF.

    This is a batched version that factors batchCount N-by-N matrices in parallel.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The size of each matrix A.  0 < N <= 64.

    @param[in,out]
    dAP_array  Array of pointers, dimension (batchCount).
             Each is COMPLEX*16 array, dimension (n*(n+1)/2)
             On entry, the upper or lower triangle of the Hermitian matrix
             A, packed columnwise in a linear array. The j-th column of AP
             is stored in the array AP as follows:
               - if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j)      for 1<=i<=j;
               - if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if INFO = i, the (i,i) element of the factor U or L is
                  zero, and the inverse could not be computed.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_potrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpptri_batched_small_v2(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex** dAP_array,
    magmaDoubleComplex* dW,
    magma_int_t batchCount, magma_int_t *info_array,
    magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if( uplo != MagmaLower ) {
        printf("%s only supports uplo = MagmaLower\n", __func__);
        arginfo = -1;
    }
    else if(n < 0 || n > 64)
        arginfo = -1;
    else if ( batchCount < 0 )
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || batchCount == 0 ) return 0;

    switch(n){
        case  1: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 1>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  2: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 2>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  3: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 3>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  4: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 4>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  5: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 5>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  6: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 6>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  7: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 7>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  8: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 8>( dAP_array, dW, batchCount, info_array, queue ); break;
        case  9: arginfo = zpptri_lower_batched_small_v2_kernel_driver< 9>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 10: arginfo = zpptri_lower_batched_small_v2_kernel_driver<10>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 11: arginfo = zpptri_lower_batched_small_v2_kernel_driver<11>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 12: arginfo = zpptri_lower_batched_small_v2_kernel_driver<12>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 13: arginfo = zpptri_lower_batched_small_v2_kernel_driver<13>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 14: arginfo = zpptri_lower_batched_small_v2_kernel_driver<14>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 15: arginfo = zpptri_lower_batched_small_v2_kernel_driver<15>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 16: arginfo = zpptri_lower_batched_small_v2_kernel_driver<16>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 17: arginfo = zpptri_lower_batched_small_v2_kernel_driver<17>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 18: arginfo = zpptri_lower_batched_small_v2_kernel_driver<18>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 19: arginfo = zpptri_lower_batched_small_v2_kernel_driver<19>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 20: arginfo = zpptri_lower_batched_small_v2_kernel_driver<20>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 21: arginfo = zpptri_lower_batched_small_v2_kernel_driver<21>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 22: arginfo = zpptri_lower_batched_small_v2_kernel_driver<22>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 23: arginfo = zpptri_lower_batched_small_v2_kernel_driver<23>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 24: arginfo = zpptri_lower_batched_small_v2_kernel_driver<24>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 25: arginfo = zpptri_lower_batched_small_v2_kernel_driver<25>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 26: arginfo = zpptri_lower_batched_small_v2_kernel_driver<26>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 27: arginfo = zpptri_lower_batched_small_v2_kernel_driver<27>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 28: arginfo = zpptri_lower_batched_small_v2_kernel_driver<28>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 29: arginfo = zpptri_lower_batched_small_v2_kernel_driver<29>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 30: arginfo = zpptri_lower_batched_small_v2_kernel_driver<30>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 31: arginfo = zpptri_lower_batched_small_v2_kernel_driver<31>( dAP_array, dW, batchCount, info_array, queue ); break;
        case 32: arginfo = zpptri_lower_batched_small_v2_kernel_driver<32>( dAP_array, dW, batchCount, info_array, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}


