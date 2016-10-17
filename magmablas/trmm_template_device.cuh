/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_DEVICE_CUH
#define TRMM_TEMPLATE_DEVICE_CUH

///////////////////////////////////////////////////////////////////////////////////////////////////
// op<trans>( x ) returns x or conj(x).
template<typename T, const int CONJA>
__device__ static inline T OP( T& x )
{
    if(CONJA == 1) return conj(x);
    else return x;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trmm modes
lNL: left  - NoTrans - Lower 
lNU: left  - NoTrans - Upper
lTL: left  - Trans   - Lower 
lTU: left  - Trans   - Upper
rNL: right - NoTrans - Lower 
rNU: right - NoTrans - Upper
rTL: right - Trans   - Lower 
rTU: right - Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// lNL, lNU
template<typename T, const int NB>
static __device__ 
void trmm_small_template_device_lNx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int bx = blockIdx.x; 
    
    const int nblocks = magma_ceildiv(n, NB);
    const int nn = (bx < nblocks-1) ? NB : n - (nblocks-1)*NB;
    B += bx * NB * lddb;
    
    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];
    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    
    // load A and B
    if(ty < m  && tx < m) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < nn && tx < m) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    __syncthreads(); 
    
    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sA[i * NB + tx] * sB[ty * NB + i];
    rb *= alpha;
    // write B
    if(ty < nn && tx < m) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// lTL, lTU, lCL, lCU
template<typename T, const int NB, const int CONJA>
static __device__ 
void trmm_small_template_device_lTx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int bx = blockIdx.x; 
    
    const int nblocks = magma_ceildiv(n, NB);
    const int nn = (bx < nblocks-1) ? NB : n - (nblocks-1)*NB;
    B += bx * NB * lddb;
    
    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];
    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    __syncthreads(); // needed because sA will be stored as transposed   
    
    // load A and B
    if(ty < m  && tx < m) sA[tx * NB + ty] = OP<T, CONJA>( A[ty * ldda + tx] );
    if(ty < nn && tx < m) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    __syncthreads(); 
    if(uplo == MagmaLower){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    __syncthreads(); 
    
    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sA[i * NB + tx] * sB[ty * NB + i];
    rb *= alpha;

    // write B
    if(ty < nn && tx < m) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rNL, rNU
template<typename T, const int NB>
static __device__ 
void trmm_small_template_device_rNx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int bx = blockIdx.x; 
    
    const int nblocks = magma_ceildiv(m, NB);
    const int mm = (bx < nblocks-1) ? NB : m - (nblocks-1)*NB;
    B += bx * NB;
    
    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];
    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    
    // load A and B
    if(ty < n && tx <  n) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < n && tx < mm) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    __syncthreads(); 
    
    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sB[i * NB + tx] * sA[ty * NB + i];
    rb *= alpha;
    // write B
    if(ty < n && tx < mm) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rTL, rTU, rCL, rCU
template<typename T, const int NB, const int CONJA>
static __device__ 
void trmm_small_template_device_rTx(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n,    // m must be <= NB
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int bx = blockIdx.x; 
    
    const int nblocks = magma_ceildiv(m, NB);
    const int mm = (bx < nblocks-1) ? NB : m - (nblocks-1)*NB;
    B += bx * NB;
    
    __shared__ T sA[NB * NB];
    __shared__ T sB[NB * NB];
    T rb = make_FloatingPoint(0.0, 0.0);
    
    // init sA and sB to zero
    sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    sB[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    
    // load A and B
    if(ty < n && tx < n ) sA[ty * NB + tx] = A[ty * ldda + tx];
    if(ty < n && tx < mm) sB[ty * NB + tx] = B[ty * lddb + tx];
    
    // handle diag
    if(diag == MagmaUnit){
        if(ty == tx) sA[ty * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }
    
    // handle uplo
    if(uplo == MagmaUpper){
        if(tx > ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }else{
        if(tx < ty) sA[ty * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    __syncthreads(); 
    
    // multiply
    #pragma unroll
    for(int i = 0; i < NB; i++)
        rb += sB[i * NB + tx] * OP<T, CONJA>( sA[i * NB + ty] );
    rb *= alpha;
    // write B
    if(ty < n && tx < mm) B[ty * lddb + tx] = rb;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
#endif //TRMM_TEMPLATE_DEVICE_CUH
