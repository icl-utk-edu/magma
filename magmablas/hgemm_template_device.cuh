/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
*/

#ifndef HGEMM_TEMPLATE_DEVICE_CUH
#define HGEMM_TEMPLATE_DEVICE_CUH

#include <cuda.h>  // for CUDA_VERSION
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700   
#include <mma.h>
using namespace nvcuda;
#endif

#include "sync.cuh"

#define WARP_SIZE 32
#define NWRPS (DIM_X*DIM_Y/WARP_SIZE)
#define fetch_half(Am, An, A, LDA, i, j)  (((i) < Am && (j) < An)? A[(j) * LDA + (i)] : make_FloatingPoint(0.0, 0.0)) 
#define ceildiv(a,b)    ((a+b-1)/b)

/******************************************************************************/
template<const int nthreads>
__device__ __inline__ void sync()
{
    __syncthreads();
}

template<>
__device__ __inline__ void sync<WARP_SIZE>()
{
    magmablas_syncwarp();
}

/******************************************************************************/
// read a block of A or B from global memory into registers
// It is not necessary that DIM_X/DIM_Y fully divide BLK_R/BLK_C
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_R, const int BLK_C>
static __device__ __inline__ void
read_global2reg( const int blk_m, const int blk_n, const T* __restrict__ A, int LDA, 
                 T reg[ceildiv(BLK_C,DIM_Y)][ceildiv(BLK_R,DIM_X)], 
                 const int tx, const int ty)
{
    int m, n;
    #pragma unroll
    for(n = 0; n < BLK_C-DIM_Y; n+=DIM_Y) {
        #pragma unroll
        for(m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            reg[n/DIM_Y][m/DIM_X] = fetch_half(blk_m, blk_n, A, LDA, m+tx, n+ty);
        }

        if(tx < BLK_R-m) {
            reg[n/DIM_Y][m/DIM_X] = fetch_half(blk_m, blk_n, A, LDA, m+tx, n+ty);
        }
    }
    
    if(ty < BLK_C-n) {
        #pragma unroll
        for(m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            reg[n/DIM_Y][m/DIM_X] = fetch_half(blk_m, blk_n, A, LDA, m+tx, n+ty);
        }

        if(tx < BLK_R-m) {
            reg[n/DIM_Y][m/DIM_X] = fetch_half(blk_m, blk_n, A, LDA, m+tx, n+ty);
        }
    }    
}

/******************************************************************************/
// read a block of A from global memory into registers
// DIM_X/DIM_Y must fully divide BLK_R/BLK_C
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_R, const int BLK_C>
__device__ __inline__ void
read_global(const int blk_m, const int blk_n, const T* __restrict__ A, int LDA, T reg[BLK_C/DIM_Y][BLK_R/DIM_X], const int tx, const int ty)
{
    #pragma unroll
    for (int n = 0; n < BLK_C/DIM_Y; n++)
        #pragma unroll
        for (int m = 0; m < BLK_R/DIM_X; m++)
            reg[n][m] = fetch_half(blk_m, blk_n, A, LDA, m*DIM_X+tx, n*DIM_Y+ty);        
}

/******************************************************************************/
// store half BLK_R x BLK_C reg. block --> BLK_R x BLK_C shared memory block (notrans)
template<const int DIM_X, const int DIM_Y, const int BLK_R, const int BLK_C>
static __device__ __inline__ void 
store_half_reg2smem_notrans(magmaHalf rA[ceildiv(BLK_C,DIM_Y)][ceildiv(BLK_R,DIM_X)], magmaHalf* sA, const int tx, const int ty)
{
    int m, n;
    #pragma unroll
    for (n = 0; n < BLK_C-DIM_Y; n+=DIM_Y) {
        #pragma unroll
        for (m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            sA[(n+ty) * BLK_R + (m+tx)] = rA[n/DIM_Y][m/DIM_X];
        }

        if(tx < BLK_R-m) {
            sA[(n+ty) * BLK_R + (m+tx)] = rA[n/DIM_Y][m/DIM_X];
        }
    }

    if(ty < BLK_C-n) {
        #pragma unroll
        for (m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            sA[(n+ty) * BLK_R + (m+tx)] = rA[n/DIM_Y][m/DIM_X];
        }

        if(tx < BLK_R-m) {
            sA[(n+ty) * BLK_R + (m+tx)] = rA[n/DIM_Y][m/DIM_X];
        }
    }
}

/******************************************************************************/
// store half BLK_R x BLK_C reg. block --> BLK_C x BLK_R shared memory block (trans)
template<const int DIM_X, const int DIM_Y, const int BLK_R, const int BLK_C>
static __device__ __inline__ void 
store_half_reg2smem_trans(magmaHalf rA[ceildiv(BLK_C,DIM_Y)][ceildiv(BLK_R,DIM_X)], magmaHalf* sA, const int tx, const int ty)
{
    int m, n;
    #pragma unroll
    for (n = 0; n < BLK_C-DIM_Y; n+=DIM_Y) {
        #pragma unroll
        for (m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            sA[(m+tx) * BLK_C + (n+ty)] = rA[n/DIM_Y][m/DIM_X];
        }

        if(tx < BLK_R-m) {
            sA[(m+tx) * BLK_C + (n+ty)] = rA[n/DIM_Y][m/DIM_X];
        }
    }

    if(ty < BLK_C-n) {
        #pragma unroll
        for (m = 0; m < BLK_R-DIM_X; m+=DIM_X) {
            sA[(m+tx) * BLK_C + (n+ty)] = rA[n/DIM_Y][m/DIM_X];
        }

        if(tx < BLK_R-m) {
            sA[(m+tx) * BLK_C + (n+ty)] = rA[n/DIM_Y][m/DIM_X];
        }
    }    
}


/******************************************************************************/
// Multiply blocks in shared memory using tensor cores 
// BLK_M/N/K must be divisible by TC_M/N/K, respectively
// sA is BLK_M x BLK_K
// sB is BLK_K x BLK_N
// sC is BLK_M x BLK_N
// multiple warps do the multiplication
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700
template<typename T, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K, const int NWARPS, const int NFRAG>
__device__ __inline__ void 
tc_multiply(T* sA, T* sB, wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, T> fC[NFRAG], 
            const int tid, const int wid, const int nbx, const int nby, const int nblks)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a,    TC_M, TC_N, TC_K, T, wmma::col_major> fA; 
    wmma::fragment<wmma::matrix_b,    TC_M, TC_N, TC_K, T, wmma::col_major> fB; 

    int b = 0;
    #pragma unroll
    for(b = 0; b < nblks - NWARPS; b += NWARPS){
        const int blkid = b + wid;
        const int i     = (blkid % nbx) * TC_M;
        const int j     = (blkid / nbx) * TC_N;
        const int fid   = b / NWARPS; 
        #pragma unroll
        for(int k = 0; k < BLK_K; k+=TC_K){
            T* ptrA = sA + k * BLK_M + i;
            T* ptrB = sB + j * BLK_K + k;
            wmma::load_matrix_sync(fA, ptrA, BLK_M);
            wmma::load_matrix_sync(fB, ptrB, BLK_K);
            wmma::mma_sync(fC[fid], fA, fB, fC[fid]);
        }
    }
    
    if(wid < nblks - b){
        const int blkid = b + wid;
        const int i = (blkid % nbx) * TC_M;
        const int j = (blkid / nbx) * TC_N;
        const int fid   = b / NWARPS;         
        #pragma unroll
        for(int k = 0; k < BLK_K; k+=TC_K){
            T* ptrA = sA + k * BLK_M + i;
            T* ptrB = sB + j * BLK_K + k;
            wmma::load_matrix_sync(fA, ptrA, BLK_M);
            wmma::load_matrix_sync(fB, ptrB, BLK_K);
            wmma::mma_sync(fC[fid], fA, fB, fC[fid]);
        }
    }
}
#endif    //CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __device__ 
void hgemm_template_device_nn(
    int M, int N, int K,
    const T* A, int LDA,
    const T* B, int LDB,
    T*       C, int LDC,
    T alpha, T beta, 
    T* sA, T* sB, T* sC)
{
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700
    // thread indexing 
    const int tid    = threadIdx.x;       // assume a 1D config with DIM_X * DIM_Y threads 
    const int tx     = tid % DIM_X;       
    const int ty     = tid / DIM_X;       
    const int wid    = tid / WARP_SIZE;
    const int nbx    = (BLK_M/TC_M);
    const int nby    = (BLK_N/TC_N);
    const int nblks  = nbx * nby;
    const int nfrag  = ceildiv(nblks, NWRPS);

    // block indexing
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // fC
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, T> fC[nfrag];
        
    // register 
    T rA[ceildiv(BLK_K,DIM_Y)][ceildiv(BLK_M,DIM_X)];
    T rB[ceildiv(BLK_N,DIM_Y)][ceildiv(BLK_K,DIM_X)];
    T rC[ceildiv(BLK_N,DIM_Y)][ceildiv(BLK_M,DIM_X)] = {make_FloatingPoint(0.0, 0.0)};

    // advance pointers of A, B, and C
    A += bx * BLK_M;
    B += by * BLK_N * LDB;
    
    const int ic = bx * BLK_M + tx;
    const int jc = by * BLK_N + ty;
    int m, n, kk;

    if( !(beta == make_FloatingPoint(0.0, 0.0)) ){
        T* lC = C + by * BLK_N * LDC + bx * BLK_M;
        const int cm = min(BLK_M, M-bx*BLK_M);
        const int cn = min(BLK_N, N-by*BLK_N);
        #pragma unroll
        for(n = 0; n < BLK_N/DIM_Y; n++)
            #pragma unroll
            for(m = 0; m < BLK_M/DIM_X; m++)
                rC[n][m] = beta * fetch_half(cm, cn, lC, LDC, m*DIM_X+tx, n*DIM_Y+ty); 
    }

    // Zero fC
    #pragma unroll
    for(int i = 0; i < nfrag; i++){
        wmma::fill_fragment(fC[i], make_FloatingPoint(0.0, 0.0));
    }

    for (kk = 0; kk < K; kk += BLK_K) {
        const int am = min(BLK_M, M-(bx * BLK_M));
        const int an = min(BLK_K, K-kk);
        const int bm = min(BLK_K, K-kk);
        const int bn = min(BLK_N, N-(by * BLK_N));

        read_global2reg<T, DIM_X, DIM_Y, BLK_M, BLK_K>(am, an, A, LDA, rA, tx, ty);
        read_global2reg<T, DIM_X, DIM_Y, BLK_K, BLK_N>(bm, bn, B, LDB, rB, tx, ty);


        store_half_reg2smem_notrans<DIM_X, DIM_Y, BLK_M, BLK_K>(rA, sA, tx, ty);
        store_half_reg2smem_notrans<DIM_X, DIM_Y, BLK_K, BLK_N>(rB, sB, tx, ty);

        sync<DIM_X * DIM_Y>();
        tc_multiply<T, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K, NWRPS, nfrag>(sA, sB, fC, tid, wid, nbx, nby, nblks);
        sync<DIM_X * DIM_Y>();

        A  += BLK_K * LDA;
        B  += BLK_K;
    }

    // store from fC to sC
    int b = 0;
    #pragma unroll
    for(b = 0; b < nblks - NWRPS; b += NWRPS){
        const int blkid = b + wid;
        const int i     = (blkid % nbx) * TC_M;
        const int j     = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS; 
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }

    if(wid < nblks - b){
        const int blkid = b + wid;
        const int i = (blkid % nbx) * TC_M;
        const int j = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS;         
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }
    sync<DIM_X * DIM_Y>();

    #pragma unroll
    for (n = 0; n < BLK_N/DIM_Y; n++){
        #pragma unroll
        for (m = 0; m < BLK_M/DIM_X; m++){
            const int ic_ = ic + m * DIM_X;
            const int jc_ = jc + n * DIM_Y;
            if(ic_ < M && jc_ < N){
                C[jc_ * LDC + ic_] = rC[n][m] + alpha * sC[(n*DIM_Y+ty) * BLK_M + (m*DIM_X+tx)];
            }
        }
    }
#endif /* #if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700    */
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __device__ 
void hgemm_template_device_nt(
    int M, int N, int K,
    const T* A, int LDA,
    const T* B, int LDB,
    T*       C, int LDC,
    T alpha, T beta, 
    T* sA, T* sB, T* sC )
{
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700
    // thread indexing 
    const int tid    = threadIdx.x;    // assume a 1D config with DIM_X * DIM_Y threads 
    const int tx     = tid % DIM_X;
    const int ty     = tid / DIM_X;
    const int wid    = tid / WARP_SIZE;
    const int nbx    = (BLK_M/TC_M);
    const int nby    = (BLK_N/TC_N);
    const int nblks  = nbx * nby;
    const int nfrag  = ceildiv(nblks, NWRPS);

    // block indexing
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // fC
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, T> fC[nfrag];
        
    // register 
    T rA[ceildiv(BLK_K,DIM_Y)][ceildiv(BLK_M,DIM_X)];
    T rB[ceildiv(BLK_K,DIM_Y)][ceildiv(BLK_N,DIM_X)];
    T rC[ceildiv(BLK_N,DIM_Y)][ceildiv(BLK_M,DIM_X)] = {make_FloatingPoint(0.0, 0.0)};

    // advance pointers of A, B, and C
    A += bx * BLK_M;
    B += by * BLK_N;
    
    const int ic = bx * BLK_M + tx;
    const int jc = by * BLK_N + ty;
    int m, n, kk;

    if( !(beta == make_FloatingPoint(0.0, 0.0)) ){
        T* lC = C + by * BLK_N * LDC + bx * BLK_M;
        const int cm = min(BLK_M, M-bx*BLK_M);
        const int cn = min(BLK_N, N-by*BLK_N);
        #pragma unroll
        for(n = 0; n < BLK_N/DIM_Y; n++)
            #pragma unroll
            for(m = 0; m < BLK_M/DIM_X; m++)
                rC[n][m] = beta * fetch_half(cm, cn, lC, LDC, m*DIM_X+tx, n*DIM_Y+ty); 
    }

    // Zero fC
    #pragma unroll
    for(int i = 0; i < nfrag; i++){
        wmma::fill_fragment(fC[i], make_FloatingPoint(0.0, 0.0));
    }

    for (kk = 0; kk < K; kk += BLK_K) {
        const int am = min(BLK_M, M-(bx * BLK_M));
        const int an = min(BLK_K, K-kk);
        const int bm = min(BLK_N, N-(by * BLK_N));
        const int bn = min(BLK_K, K-kk);

        read_global2reg<T, DIM_X, DIM_Y, BLK_M, BLK_K>(am, an, A, LDA, rA, tx, ty);
        read_global2reg<T, DIM_X, DIM_Y, BLK_N, BLK_K>(bm, bn, B, LDB, rB, tx, ty);

        store_half_reg2smem_notrans<DIM_X, DIM_Y, BLK_M, BLK_K>(rA, sA, tx, ty);
        store_half_reg2smem_trans<DIM_X, DIM_Y, BLK_N, BLK_K>(rB, sB, tx, ty);

        sync<DIM_X * DIM_Y>();
        tc_multiply<T, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K, NWRPS, nfrag>(sA, sB, fC, tid, wid, nbx, nby, nblks);
        sync<DIM_X * DIM_Y>();

        A  += BLK_K * LDA;
        B  += BLK_K * LDB;
    }

    // store from fC to sC
    int b = 0;
    #pragma unroll
    for(b = 0; b < nblks - NWRPS; b += NWRPS){
        const int blkid = b + wid;
        const int i     = (blkid % nbx) * TC_M;
        const int j     = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS; 
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }

    if(wid < nblks - b){
        const int blkid = b + wid;
        const int i = (blkid % nbx) * TC_M;
        const int j = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS;         
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }
    sync<DIM_X * DIM_Y>();

    #pragma unroll
    for (n = 0; n < BLK_N/DIM_Y; n++){
        #pragma unroll
        for (m = 0; m < BLK_M/DIM_X; m++){
            const int ic_ = ic + m * DIM_X;
            const int jc_ = jc + n * DIM_Y;
            if(ic_ < M && jc_ < N){
                C[jc_ * LDC + ic_] = rC[n][m] + alpha * sC[(n*DIM_Y+ty) * BLK_M + (m*DIM_X+tx)];
            }
        }
    }
#endif /* #if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700    */
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __device__ 
void hgemm_template_device_tn(
    int M, int N, int K,
    const T* A, int LDA,
    const T* B, int LDB,
    T*       C, int LDC,
    T alpha, T beta, 
    T* sA, T* sB, T* sC )
{
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700
    // thread indexing 
    const int tid    = threadIdx.x;    // assume a 1D config with DIM_X * DIM_Y threads 
    const int tx     = tid % DIM_X;
    const int ty     = tid / DIM_X;
    const int wid    = tid / WARP_SIZE;
    const int nbx    = (BLK_M/TC_M);
    const int nby    = (BLK_N/TC_N);
    const int nblks  = nbx * nby;
    const int nfrag  = ceildiv(nblks, NWRPS);

    // block indexing
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // fC
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, T> fC[nfrag];

    // register 
    T rA[BLK_M/DIM_Y][BLK_K/DIM_X];
    T rB[BLK_N/DIM_Y][BLK_K/DIM_X];
    T rC[BLK_N/DIM_Y][BLK_M/DIM_X] = {make_FloatingPoint(0.0, 0.0)};

    // advance pointers of A, B, and C
    A += bx * BLK_M * LDA;
    B += by * BLK_N * LDB;

    const int ic = bx * BLK_M + tx;
    const int jc = by * BLK_N + ty;
    int m, n, kk;

    if( !(beta == make_FloatingPoint(0.0, 0.0)) ){
        T* lC = C + by * BLK_N * LDC + bx * BLK_M;
        const int cm = min(BLK_M, M-bx*BLK_M);
        const int cn = min(BLK_N, N-by*BLK_N);
        #pragma unroll
        for(n = 0; n < BLK_N/DIM_Y; n++)
            #pragma unroll
            for(m = 0; m < BLK_M/DIM_X; m++)
                rC[n][m] = beta * fetch_half(cm, cn, lC, LDC, m*DIM_X+tx, n*DIM_Y+ty); 
    }

    // Zero fC
    #pragma unroll
    for(int i = 0; i < nfrag; i++){
        wmma::fill_fragment(fC[i], make_FloatingPoint(0.0, 0.0));
    }

    for (kk = 0; kk < K; kk += BLK_K) {
        const int am = min(BLK_K, K-kk);
        const int an = min(BLK_M, M-(bx * BLK_M));
        const int bm = min(BLK_K, K-kk);
        const int bn = min(BLK_N, N-(by * BLK_N));

        read_global<T, DIM_X, DIM_Y, BLK_K, BLK_M>(am, an, A, LDA, rA, tx, ty);
        read_global<T, DIM_X, DIM_Y, BLK_K, BLK_N>(bm, bn, B, LDB, rB, tx, ty);

        store_half_reg2smem_trans<DIM_X, DIM_Y, BLK_K, BLK_M>(rA, sA, tx, ty);
        store_half_reg2smem_notrans<DIM_X, DIM_Y, BLK_K, BLK_N>(rB, sB, tx, ty);

        sync<DIM_X * DIM_Y>();
        tc_multiply<T, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K, NWRPS, nfrag>(sA, sB, fC, tid, wid, nbx, nby, nblks);
        sync<DIM_X * DIM_Y>();

        A  += BLK_K;
        B  += BLK_K;
    }

    // store from fC to sC
    int b = 0;
    #pragma unroll
    for(b = 0; b < nblks - NWRPS; b += NWRPS){
        const int blkid = b + wid;
        const int i     = (blkid % nbx) * TC_M;
        const int j     = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS; 
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }

    if(wid < nblks - b){
        const int blkid = b + wid;
        const int i = (blkid % nbx) * TC_M;
        const int j = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS;         
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }
    sync<DIM_X * DIM_Y>();


    #pragma unroll
    for (n = 0; n < BLK_N/DIM_Y; n++){
        #pragma unroll
        for (m = 0; m < BLK_M/DIM_X; m++){
            const int ic_ = ic + m * DIM_X;
            const int jc_ = jc + n * DIM_Y;
            if(ic_ < M && jc_ < N){
                C[jc_ * LDC + ic_] = rC[n][m] + alpha * sC[(n*DIM_Y+ty) * BLK_M + (m*DIM_X+tx)];
            }
        }
    }
#endif /* #if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700    */
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int TC_M, const int TC_N, const int TC_K>
static __device__ 
void hgemm_template_device_tt(
    int M, int N, int K,
    const T* A, int LDA,
    const T* B, int LDB,
    T*       C, int LDC,
    T alpha, T beta, 
    T* sA, T* sB, T* sC )
{
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700
    // thread indexing 
    const int tid    = threadIdx.x;    // assume a 1D config with DIM_X * DIM_Y threads 
    const int tx     = tid % DIM_X;
    const int ty     = tid / DIM_X;
    const int wid    = tid / WARP_SIZE;
    const int nbx    = (BLK_M/TC_M);
    const int nby    = (BLK_N/TC_N);
    const int nblks  = nbx * nby;
    const int nfrag  = ceildiv(nblks, NWRPS);

    // block indexing
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // fC
    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, T> fC[nfrag];
        
    // register 
    T rA[BLK_M/DIM_Y][BLK_K/DIM_X];
    T rB[BLK_K/DIM_Y][BLK_N/DIM_X];
    T rC[BLK_N/DIM_Y][BLK_M/DIM_X] = {make_FloatingPoint(0.0, 0.0)};

    // advance pointers of A, B, and C
    A += bx * BLK_M * LDA;
    B += by * BLK_N;
    
    const int ic = bx * BLK_M + tx;
    const int jc = by * BLK_N + ty;
    int m, n, kk;

    if( !(beta == make_FloatingPoint(0.0, 0.0)) ){
        T* lC = C + by * BLK_N * LDC + bx * BLK_M;
        const int cm = min(BLK_M, M-bx*BLK_M);
        const int cn = min(BLK_N, N-by*BLK_N);
        #pragma unroll
        for(n = 0; n < BLK_N/DIM_Y; n++)
            #pragma unroll
            for(m = 0; m < BLK_M/DIM_X; m++)
                rC[n][m] = beta * fetch_half(cm, cn, lC, LDC, m*DIM_X+tx, n*DIM_Y+ty); 
    }

    // Zero fC
    #pragma unroll
    for(int i = 0; i < nfrag; i++){
        wmma::fill_fragment(fC[i], make_FloatingPoint(0.0, 0.0));
    }

    for (kk = 0; kk < K; kk += BLK_K) {
        const int am = min(BLK_K, K-kk);
        const int an = min(BLK_M, M-(bx * BLK_M));
        const int bm = min(BLK_N, N-(by * BLK_N));
        const int bn = min(BLK_K, K-kk);

        read_global<T, DIM_X, DIM_Y, BLK_K, BLK_M>(am, an, A, LDA, rA, tx, ty);
        read_global<T, DIM_X, DIM_Y, BLK_N, BLK_K>(bm, bn, B, LDB, rB, tx, ty);

        store_half_reg2smem_trans<DIM_X, DIM_Y, BLK_K, BLK_M>(rA, sA, tx, ty);
        store_half_reg2smem_trans<DIM_X, DIM_Y, BLK_N, BLK_K>(rB, sB, tx, ty);

        sync<DIM_X * DIM_Y>();
        tc_multiply<T, BLK_M, BLK_N, BLK_K, TC_M, TC_N, TC_K, NWRPS, nfrag>(sA, sB, fC, tid, wid, nbx, nby, nblks);
        sync<DIM_X * DIM_Y>();

        A  += BLK_K;
        B  += BLK_K * LDB;
    }

    // store from fC to sC
    int b = 0;
    #pragma unroll
    for(b = 0; b < nblks - NWRPS; b += NWRPS){
        const int blkid = b + wid;
        const int i     = (blkid % nbx) * TC_M;
        const int j     = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS; 
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }

    if(wid < nblks - b){
        const int blkid = b + wid;
        const int i = (blkid % nbx) * TC_M;
        const int j = (blkid / nbx) * TC_N;
        const int fid   = b / NWRPS;         
        T* ptrC = sC + j * BLK_M + i;
        wmma::store_matrix_sync(ptrC, fC[fid], BLK_M, wmma::mem_col_major);
    }
    sync<DIM_X * DIM_Y>();

    #pragma unroll
    for (n = 0; n < BLK_N/DIM_Y; n++){
        #pragma unroll
        for (m = 0; m < BLK_M/DIM_X; m++){
            const int ic_ = ic + m * DIM_X;
            const int jc_ = jc + n * DIM_Y;
            if(ic_ < M && jc_ < N){
                C[jc_ * LDC + ic_] = rC[n][m] + alpha * sC[(n*DIM_Y+ty) * BLK_M + (m*DIM_X+tx)];
            }
        }
    }
#endif /* #if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 700    */
}

#endif //HGEMM_TEMPLATE_DEVICE_CUH
