/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Ahmad Abdelfattah
       @author Azzam Haidar

*/

#ifndef GEMM_TEMPLATE_DEVICE_DEFS_H
#define GEMM_TEMPLATE_DEVICE_DEFS_H

// =============================================================================
// conjugation -- double complex
template<const int conjugate>
__device__ inline
magmaDoubleComplex conj(const magmaDoubleComplex &x) {return MAGMA_Z_CONJ(x);}

template<>
__device__ inline
magmaDoubleComplex conj<0>(const magmaDoubleComplex &x) {return x;}

// conjugation -- single complex
template<const int conjugate>
__device__ inline
magmaFloatComplex conj(const magmaFloatComplex &x) {return MAGMA_C_CONJ(x);}

template<>
__device__ inline
magmaFloatComplex conj<0>(const magmaFloatComplex &x) {return x;}

// conjugation -- real single, double, half
template<const int conjugate>
__device__ static inline
double conj(const double &x) {return x;}

template<const int conjugate>
__device__ static inline
float conj(const float &x) {return x;}

template<int conjugate>
__device__ static inline
magmaHalf conj(const magmaHalf &x) {return x;}

/******************************************************************************/
// op<trans>( x ) returns x or conj(x).
template<typename T, int conjugate>
static __device__ inline
T op(const T& x) {return conj<conjugate>(x);}

/******************************************************************************/
// shared memory accesses
#define sA(i,j)    sA[(j)*slda + (i)]
#define sB(i,j)    sB[(j)*sldb + (i)]
#define sC(i,j)    sC[(j)*sldc + (i)]

/******************************************************************************/
// shared memory leading dimensions
#define SLDA(N)    ((N)+1)
#define SLDB(N)    ((N)+1)
#define SLDC(N)    ((N)+1)

// =============================================================================
#define fetch(A, m, n, bound)         A[min(n*LD##A+m, bound)]

// =============================================================================
#if defined(PRECISION_z)
    #define add(A, B)        MAGMA_Z_ADD(A, B)
    #define mul(A, B)        MAGMA_Z_MUL(A, B)
    #define div(A, B)        MAGMA_Z_DIV(A, B)
    #define fma(A, B, C) C = magmaCfma(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_Z_MAKE(x, y)
#elif defined(PRECISION_c)
    #define add(A, B)        MAGMA_C_ADD(A, B)
    #define mul(A, B)        MAGMA_C_MUL(A, B)
    #define div(A, B)        MAGMA_C_DIV(A, B)
    #define fma(A, B, C) C = magmaCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_C_MAKE(x, y)
#elif defined(PRECISION_h)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define make_FloatingPoint(x, y) ((magmaHalf)x)
    #if defined(MAGMA_HAVE_CUDA)
        #define fma(A, B, C)      C = __hfma (A, B, C)
    #elif defined(MAGMA_HAVE_HIP)
        #define fma(A, B, C) C += (A*B)
    #endif
#else
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

#if defined(PRECISION_z)
    #define magmablas_atomic_add magmablas_zatomic_add
#elif defined(PRECISION_c)
    #define magmablas_atomic_add magmablas_catomic_add
#elif defined(PRECISION_d)
    #define magmablas_atomic_add magmablas_datomic_add
#else
    #define magmablas_atomic_add magmablas_satomic_add
#endif

/******************************************************************************/
// zero reg 2D
// For a block: BLK_M x BLK_N and thread config: DIM_X x DIM_Y,
// THR_M = BLK_M / DIM_X and THR_N = BLK_N / DIM_Y
template<typename T, int THR_M, int THR_N>
static __device__ __inline__ void
zero_rgArray2D(T rg[THR_N][THR_M])
{
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rg[n][m] = make_FloatingPoint(0.0, 0.0);
}

/******************************************************************************/
// zero reg 3D
// This routine is for future use
template<typename T, int THR_M, int THR_N, int THR_RC>
static __device__ __inline__ void
zero_rgArray3D(T rg[THR_N][THR_M][THR_RC])
{
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            #pragma unroll
            for(int k = 0; k < THR_RC; k++)
                rg[n][m][k] = make_FloatingPoint(0.0, 0.0);
}

/******************************************************************************/
// read a block from global memory to shared memory -- non-transposed
// Block dimensions are mb x nb, assumed be default as BLK_ROW x BLK_COL
// Assuming 2D thread-config: DIM_X x DIM_Y
// boundA is the last element that can be read without going out-of-bound
// Partial blocks are padded with zeros
template<typename T, int BLK_ROW, int BLK_COL, int DIM_X, int DIM_Y>
static __device__ __inline__ void
read_gm2sm_notrans(
    const T* __restrict__  A, int &LDA, ptrdiff_t &boundA,
          T*              sA, int &slda,
    const int &tx, const int &ty,
    const int mb = BLK_ROW, const int nb = BLK_COL )
{
    if(mb == BLK_ROW && nb == BLK_COL) {
        #pragma unroll
        for (int n = 0; n < BLK_COL; n += DIM_Y)
            #pragma unroll
            for (int m = 0; m < BLK_ROW; m += DIM_X) {
                sA(m+tx,n+ty) = fetch(A, m, n, boundA);
            }
    }
    else {
        int mtx, nty;
        #pragma unroll
        for (int n = 0; n < BLK_COL; n += DIM_Y) {
            nty = n + ty;
            #pragma unroll
            for (int m = 0; m < BLK_ROW; m += DIM_X) {
                mtx = m + tx;
                sA(mtx,nty) = (mtx >= mb || nty >= nb) ? make_FloatingPoint(0, 0) : fetch(A, m, n, boundA);
            }
        }
    }
}

/******************************************************************************/
// read a block from global memory to shared memory -- transpose it in shared memory
// Block dimensions are mb x nb, assumed be default as BLK_ROW x BLK_COL
// Assuming 2D thread-config: DIM_X x DIM_Y
// boundA is the last element that can be read without going out-of-bound
// Partial blocks are padded with zeros
// if CONJA = 1, the block is conjugate-transposed
template<typename T, int BLK_ROW, int BLK_COL, int DIM_X, int DIM_Y, int CONJA>
static __device__ __inline__ void
read_gm2sm_trans(
    const T* __restrict__  A, int &LDA, ptrdiff_t &boundA,
          T*              sA, int &slda,
    const int &tx, const int &ty,
    const int mb = BLK_ROW, const int nb = BLK_COL )
{
    if(mb == BLK_ROW && nb == BLK_COL) {
        #pragma unroll
        for (int n = 0; n < BLK_COL; n += DIM_Y)
            #pragma unroll
            for (int m = 0; m < BLK_ROW; m += DIM_X)
                sA(n+ty,m+tx) = op<T, CONJA>( fetch(A, m, n, boundA) );
    }
    else {
        int mtx, nty;
        #pragma unroll
        for (int n = 0; n < BLK_COL; n += DIM_Y) {
            nty = n+ty;
            #pragma unroll
            for (int m = 0; m < BLK_ROW; m += DIM_X) {
                mtx = m + tx;
                sA(nty,mtx) = (mtx >= mb || nty >= nb) ? make_FloatingPoint(0, 0) : op<T, CONJA>( fetch(A, m, n, boundA) );
            }
        }
    }
}

/******************************************************************************/
// read a block from global memory to registers -- non-transposed
// Block dimensions are BLK_ROW x BLK_COL
// Assuming 2D thread-config: DIM_X x DIM_Y
// Partial blocks are padded with the element pointed to by boundA
template<typename T, int BLK_ROW, int BLK_COL, int DIM_X, int DIM_Y>
static __device__ __inline__ void
read_gm2rg_notrans(
    const T* __restrict__  A, int &LDA, ptrdiff_t &boundA,
          T  rg[BLK_COL/DIM_Y][BLK_ROW/DIM_X])
{
    #pragma unroll
    for (int n = 0; n < BLK_COL/DIM_Y; n++)
        #pragma unroll
        for (int m = 0; m < BLK_ROW/DIM_X; m++)
            rg[n][m] = fetch(A, m*DIM_X, n*DIM_Y, boundA);
}


/******************************************************************************/
// store a block from registers to shared memory -- non-transposed
// Block dimensions are mb x nb, assumed be default as BLK_ROW x BLK_COL
// Assuming 2D thread-config: DIM_X x DIM_Y
// Partial blocks are padded with zeros
template<typename T, int BLK_ROW, int BLK_COL, int DIM_X, int DIM_Y>
static __device__ __inline__ void
write_rg2sm_notrans(
    T  rg[BLK_COL/DIM_Y][BLK_ROW/DIM_X],
    T* sA, int &slda, const int &tx, const int &ty,
    const int mb = BLK_ROW, const int nb = BLK_COL)
{
    if(mb == BLK_ROW && nb == BLK_COL) {
       #pragma unroll
        for (int n = 0; n < BLK_COL/DIM_Y; n++)
            #pragma unroll
            for (int m = 0; m < BLK_ROW/DIM_X; m++)
                sA(m*DIM_X+tx,n*DIM_Y+ty) = rg[n][m];
    }
    else {
        int mtx, nty;
        #pragma unroll
        for (int n = 0; n < BLK_COL/DIM_Y; n++) {
            nty = n*DIM_Y + ty;
            #pragma unroll
            for (int m = 0; m < BLK_ROW/DIM_X; m++) {
                mtx = m*DIM_X + tx;
                sA(mtx,nty) = (mtx >= mb || nty >= nb) ? make_FloatingPoint(0, 0) : rg[n][m];
            }
        }
    }
}

/******************************************************************************/
// store a block from registers to shared memory -- transpose it in sm
// Block dimensions are mb x nb, assumed be default as BLK_ROW x BLK_COL
// Assuming 2D thread-config: DIM_X x DIM_Y
// Partial blocks are padded with zeros
// if CONJA = 1, the block is conjugate-transposed
template<typename T, int BLK_ROW, int BLK_COL, int DIM_X, int DIM_Y, int CONJA>
static __device__ __inline__ void
write_rg2sm_trans(
    T  rg[BLK_COL/DIM_Y][BLK_ROW/DIM_X],
    T* sA, int &slda, const int &tx, const int &ty,
    const int mb = BLK_ROW, const int nb = BLK_COL )
{
    if(mb == BLK_ROW && nb == BLK_COL) {
        #pragma unroll
        for (int n = 0; n < BLK_COL/DIM_Y; n++)
            #pragma unroll
            for (int m = 0; m < BLK_ROW/DIM_X; m++)
            sA(n*DIM_Y+ty,m*DIM_X+tx) = op<T, CONJA>( rg[n][m] );
    }
    else {
        int mtx, nty;
        #pragma unroll
        for (int n = 0; n < BLK_COL/DIM_Y; n++) {
            nty = n*DIM_Y+ty;
            #pragma unroll
            for (int m = 0; m < BLK_ROW/DIM_X; m++) {
                mtx = m*DIM_X+tx;
                sA(nty,mtx) = (mtx >= mb || nty >= nb) ? make_FloatingPoint(0, 0) : op<T, CONJA>( rg[n][m] );
            }
        }
    }
}

/******************************************************************************/
// multiply a full block (k = BLK_K)
// 2D thread-config: DIM_X x DIM_Y
// sA is BLK_M x BLK_K
// sB is BLK_K x BLK_N
// THR_M = BLK_M / DIM_X
// THR_N = BLK_N / DIM_Y
template<typename T, int BLK_K, int THR_M, int THR_N, int DIM_X, int DIM_Y>
static __device__ __inline__ void
multiply_full_block(
    T* sA, int &slda,
    T* sB, int &sldb,
    T rC[THR_N][THR_M],
    const int &tx, const int &ty )
{
    T rA[THR_M];
    T rB[THR_N];

    #pragma unroll
    for (int k = 0; k < BLK_K; k++) {
        // Load A shmem->regs
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rA[m] = sA(m*DIM_X+tx,k);

        // Load B shmem->regs
        #pragma unroll
        for (int n = 0; n < THR_N; n++)
            rB[n] = sB(k,n*DIM_Y+ty);

        // Compute
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma(rA[m], rB[n], rC[n][m]);
            }
        }
    }
}

/******************************************************************************/
// multiply a partial block (k != BLK_K)
// 2D thread-config: DIM_X x DIM_Y
// sA is BLK_M x BLK_K
// sB is BLK_K x BLK_N
// THR_M = BLK_M / DIM_X
// THR_N = BLK_N / DIM_Y
template<typename T, int THR_M, int THR_N, int DIM_X, int DIM_Y>
static __device__ __inline__ void
multiply_partial_block(
    int &kk,
    T* sA, int &slda,
    T* sB, int &sldb,
    T rC[THR_N][THR_M],
    const int &tx, const int &ty )
{
    T rA[THR_M];
    T rB[THR_N];

    #pragma unroll
    for (int k = 0; k < kk; k++) {
        // Load A shmem->regs
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rA[m] = sA(m*DIM_X+tx,k);

        // Load B shmem->regs
        #pragma unroll
        for (int n = 0; n < THR_N; n++)
            rB[n] = sB(k,n*DIM_Y+ty);

        // Compute
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma(rA[m], rB[n], rC[n][m]);
            }
        }
    }
}

/******************************************************************************/
// store a BLK_M x BLK_N block in global memory
// 2D thread-config: DIM_X x DIM_Y
// THR_M = BLK_M / DIM_X
// THR_N = BLK_N / DIM_Y
// M and N are the dimensions of the C matrix
template<typename T, int BLK_M, int BLK_N, int THR_M, int THR_N, int DIM_X, int DIM_Y>
static __device__ __inline__ void
write_rg2gm(int &M, int &N, T rC[THR_N][THR_M], T* C, int &LDC, T &alpha, T &beta, int &bx, int &by, int &tx, int &ty)
{
    if( beta == make_FloatingPoint(0.0,0.0) ) {
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            int coord_dCn = by*BLK_N + n*DIM_Y + ty;
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                int coord_dCm = bx*BLK_M + m*DIM_X + tx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = (ptrdiff_t)coord_dCn*(ptrdiff_t)LDC + (ptrdiff_t)coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = alpha*regC;
                }
            }
        }
    } else {
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            int coord_dCn = by*BLK_N + n*DIM_Y + ty;
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                int coord_dCm = bx*BLK_M + m*DIM_X + tx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = (ptrdiff_t)coord_dCn*(ptrdiff_t)LDC + (ptrdiff_t)coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = alpha*regC + beta*memC;
                }
            }
        }
    }
}

#endif // GEMM_TEMPLATE_DEVICE_DEFS_H
