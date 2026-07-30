/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       @author Natalie Beams

       See gemm_template_device_defs.cuh for aux. device routines
*/

#ifndef GEMM_PACKED_TEMPLATE_DEVICE_CUH
#define GEMM_PACKED_TEMPLATE_DEVICE_CUH

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static __device__
void gemm_packed_template_device_nn(
    int M, int N, int K,
    const T* __restrict__ A, int coffA, int roffA, int LDA,
    const T* __restrict__ B, int coffB, int roffB, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta,
    T* sA, int slda,
    T* sB, int sldb,
    T* sC, int sldc )
{
    int tx = threadIdx.x;  // thread's m dimension
    int ty = threadIdx.y;  // thread's n dimension

    int tid = DIM_X * ty + tx;    // thread's global number

    int txA = tid % DIM_XA;    // tx within A
    int tyA = tid / DIM_XA;    // ty within A

    int txB = tid % DIM_XB;    // tx within B
    int tyB = tid / DIM_XB;    // ty within B

    int bx = blockIdx.x;   // block's m dimension
    int by = blockIdx.y;   // block's n dimension

    // Registers for the innermost loop
    T rC[THR_N][THR_M];

    // Registers for the dev->shmem copy
    T rA[BLK_K/DIM_YA][BLK_M/DIM_XA];
    T rB[BLK_N/DIM_YB][BLK_K/DIM_XB];

    ptrdiff_t  boundA = (LDA*(LDA+1)/2) - 1;

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dB = B + by*BLK_N*LDB + tyB*LDB + txB + coffB * LDB + roffB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( by*BLK_N*LDB + tyB*LDB + txB ) -1;

    int kk;

    // Zero C
    zero_rgArray2D<T, THR_M, THR_N>(rC);
    if(K > 0) {
        // read A/B gm2sm -- N/N
        int col = tyA + coffA;
        int row = bx*BLK_M + txA + roffA;
        read_packed_gm2sm_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, sA, slda, txA, tyA);

	read_gm2sm_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, sB, sldb, txB, tyB);
    }
    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K) {
        offs_dB += BLK_K;
        boundB  -= BLK_K;

        // prefetch A/B gm2reg -- ignore transposition for now
        int col = tyA + coffA + BLK_K + kk;
        int row = bx*BLK_M + txA + roffA;
        read_packed_gm2rg_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, rA);

	read_gm2rg_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, rB);

	// Multiply
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y>
        (sA, slda, sB, sldb, rC, tx, ty);

        __syncthreads();

        // mv prefetched blocks into sm with propoer transposition
        write_rg2sm_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>( rA, sA, slda, txA, tyA );
        write_rg2sm_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>( rB, sB, sldb, txB, tyB );
        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>
    (kk,sA, slda, sB, sldb, rC, tx, ty);

    // Store C regs->dev
    write_rg2gm<T, BLK_M, BLK_N, THR_M, THR_N, DIM_X, DIM_Y>
    (M, N, rC, C, LDC, alpha, beta, bx, by, tx, ty);
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static __device__
void gemm_packed_template_device_nt (
    int M, int N, int K,
    const T* __restrict__ A, int coffA, int roffA, int LDA,
    const T* __restrict__ B, int coffB, int roffB, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta,
    T* sA, int slda,
    T* sB, int sldb,
    T* sC, int sldc )
{
    int tx = threadIdx.x;  // thread's m dimension
    int ty = threadIdx.y;  // thread's n dimension

    int tid = DIM_X * ty + tx;    // thread's global number

    int txA = tid % DIM_XA;    // tx within A
    int tyA = tid / DIM_XA;    // ty within A

    int txB = tid % DIM_XB;    // tx within B
    int tyB = tid / DIM_XB;    // ty within B

    int bx = blockIdx.x;   // block's m dimension
    int by = blockIdx.y;   // block's n dimension

    // Registers for the innermost loop
    T rC[THR_N][THR_M];

    // Registers for the dev->shmem copy
    T rA[BLK_K/DIM_YA][BLK_M/DIM_XA];
    T rB[BLK_K/DIM_YB][BLK_N/DIM_XB];

    ptrdiff_t boundA = LDA*(LDA+1)/2 - 1;

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dB = B + by*BLK_N     + tyB*LDB + txB + coffB * LDB + roffB;
    ptrdiff_t boundB = (LDB*(K-1) + N) - ( by*BLK_N     + tyB*LDB + txB ) -1;

    int kk;

    // Zero C
    zero_rgArray2D<T, THR_M, THR_N>(rC);
    if(K > 0) {
        // read A/B gm2sm -- N/T
        int col = tyA + coffA;
	int row = bx*BLK_M + txA + roffA;
        read_packed_gm2sm_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, sA, slda, txA, tyA);

	read_gm2sm_trans<T, BLK_N, BLK_K, DIM_XB, DIM_YB, CONJB>(offs_dB, LDB, boundB, sB, sldb, txB, tyB );
    }
    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K) {
        offs_dB += BLK_K*LDB;
        boundB  -= BLK_K*LDB;

	// prefetch A/B gm2reg -- ignore transposition for now
        int col = tyA + coffA + BLK_K + kk;
	int row = bx*BLK_M + txA + roffA;
        read_packed_gm2rg_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, rA);

	read_gm2rg_notrans<T, BLK_N, BLK_K, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, rB);

	// Multiply
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y>
        (sA, slda, sB, sldb, rC, tx, ty);
        __syncthreads();

        // mv prefetched blocks into sm with propoer transposition
        write_rg2sm_notrans<T, BLK_M, BLK_K, DIM_XA, DIM_YA>( rA, sA, slda, txA, tyA );
        write_rg2sm_trans<T, BLK_N, BLK_K, DIM_XB, DIM_YB, CONJB>( rB, sB, sldb, txB, tyB );

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>
    (kk,sA, slda, sB, sldb, rC, tx, ty);

    // Store C regs->dev
    write_rg2gm<T, BLK_M, BLK_N, THR_M, THR_N, DIM_X, DIM_Y>
    (M, N, rC, C, LDC, alpha, beta, bx, by, tx, ty);
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static __device__
void gemm_packed_template_device_tn (
    int M, int N, int K,
    const T* __restrict__ A, int coffA, int roffA, int LDA,
    const T* __restrict__ B, int coffB, int roffB, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta,
    T* sA, int slda,
    T* sB, int sldb,
    T* sC, int sldc )
{
    int tx = threadIdx.x;  // thread's m dimension
    int ty = threadIdx.y;  // thread's n dimension

    int tid = DIM_X * ty + tx;    // thread's global number

    int txA = tid % DIM_XA;    // tx within A
    int tyA = tid / DIM_XA;    // ty within A

    int txB = tid % DIM_XB;    // tx within B
    int tyB = tid / DIM_XB;    // ty within B

    int bx = blockIdx.x;   // block's m dimension
    int by = blockIdx.y;   // block's n dimension

    // Registers for the innermost loop
    T rC[THR_N][THR_M];

    // Registers for the dev->shmem copy
    T rA[BLK_M/DIM_YA][BLK_K/DIM_XA];
    T rB[BLK_N/DIM_YB][BLK_K/DIM_XB];

    ptrdiff_t boundA = (LDA*(LDA+1)/2) - 1;

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dB = B + by*BLK_N*LDB + tyB*LDB + txB + coffB * LDB + roffB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( by*BLK_N*LDB + tyB*LDB + txB ) -1;

    int kk;


    // Zero C
    zero_rgArray2D<T, THR_M, THR_N>(rC);
    if(K > 0) {
        // read A/B gm2sm -- T/N
	int col = bx*BLK_M + tyA + coffA;
	int row = txA + roffA;
        read_packed_gm2sm_trans<T, BLK_K, BLK_M, DIM_XA, DIM_YA, CONJA>(A, col, row, LDA, boundA, sA, slda, txA, tyA );

	read_gm2sm_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, sB, sldb, txB, tyB );
    }
    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K) {
          offs_dB += BLK_K;
          boundB  -= BLK_K;

        // prefetch A/B gm2reg -- ignore transposition for now
	int col = bx*BLK_M + tyA + coffA;
	int row = txA + roffA + BLK_K + kk;
        read_packed_gm2rg_notrans<T, BLK_K, BLK_M, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, rA);

	read_gm2rg_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, rB);

        // Multiply
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y>
        (sA, slda, sB, sldb, rC, tx, ty);
        __syncthreads();

        // mv prefetched blocks into sm with propoer transposition
        write_rg2sm_trans<T, BLK_K, BLK_M, DIM_XA, DIM_YA, CONJA>( rA, sA, slda, txA, tyA );
        write_rg2sm_notrans<T, BLK_K, BLK_N, DIM_XB, DIM_YB>( rB, sB, sldb, txB, tyB );

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>
    (kk,sA, slda, sB, sldb, rC, tx, ty);

    // Store C regs->dev
    write_rg2gm<T, BLK_M, BLK_N, THR_M, THR_N, DIM_X, DIM_Y>
    (M, N, rC, C, LDC, alpha, beta, bx, by, tx, ty);
}

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static __device__
void gemm_packed_template_device_tt (
    int M, int N, int K,
    const T* __restrict__ A, int coffA, int roffA, int LDA,
    const T* __restrict__ B, int coffB, int roffB, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta,
    T* sA, int slda,
    T* sB, int sldb,
    T* sC, int sldc )
{
    int tx = threadIdx.x;  // thread's m dimension
    int ty = threadIdx.y;  // thread's n dimension

    int tid = DIM_X * ty + tx;    // thread's global number

    int txA = tid % DIM_XA;    // tx within A
    int tyA = tid / DIM_XA;    // ty within A

    int txB = tid % DIM_XB;    // tx within B
    int tyB = tid / DIM_XB;    // ty within B

    int bx = blockIdx.x;   // block's m dimension
    int by = blockIdx.y;   // block's n dimension

    // Registers for the innermost loop
    T rC[THR_N][THR_M];

    // Registers for the dev->shmem copy
    T rA[BLK_M/DIM_YA][BLK_K/DIM_XA];
    T rB[BLK_K/DIM_YB][BLK_N/DIM_XB];

    ptrdiff_t boundA = (LDA*(LDA+1)/2) - 1;

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dB = B + by*BLK_N     + tyB*LDB + txB + coffB * LDB + roffB;
    ptrdiff_t boundB = (LDB*(K-1) + N) - ( by*BLK_N     + tyB*LDB + txB ) -1;

    int kk;

    // Zero C
    zero_rgArray2D<T, THR_M, THR_N>(rC);
    if(K > 0) {
        // read A/B gm2sm -- T/T
	int col = bx*BLK_M + tyA + coffA;
	int row = txA + roffA;
        read_packed_gm2sm_trans<T, BLK_K, BLK_M, DIM_XA, DIM_YA, CONJA>(A, col, row, LDA, boundA, sA, slda, txA, tyA );

	read_gm2sm_trans<T, BLK_N, BLK_K, DIM_XB, DIM_YB, CONJB>(offs_dB, LDB, boundB, sB, sldb, txB, tyB );
    }
    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K) {
         offs_dB += BLK_K*LDB;
         boundB  -= BLK_K*LDB;

        // prefetch A/B gm2reg -- ignore transposition for now
        int col = bx*BLK_M + tyA + coffA;
        int row = txA + roffA + BLK_K + kk;
        read_packed_gm2rg_notrans<T, BLK_K, BLK_M, DIM_XA, DIM_YA>(A, col, row, LDA, boundA, rA);

	read_gm2rg_notrans<T, BLK_N, BLK_K, DIM_XB, DIM_YB>(offs_dB, LDB, boundB, rB);

        // Multiply
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y>
        (sA, slda, sB, sldb, rC, tx, ty);

        __syncthreads();

        // mv prefetched blocks into sm with propoer transposition
        write_rg2sm_trans<T, BLK_K, BLK_M, DIM_XA, DIM_YA, CONJA>( rA, sA, slda, txA, tyA );
        write_rg2sm_trans<T, BLK_N, BLK_K, DIM_XB, DIM_YB, CONJB>( rB, sB, sldb, txB, tyB );

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>
    (kk,sA, slda, sB, sldb, rC, tx, ty);

    // Store C regs->dev
    write_rg2gm<T, BLK_M, BLK_N, THR_M, THR_N, DIM_X, DIM_Y>
    (M, N, rC, C, LDC, alpha, beta, bx, by, tx, ty);
}

#endif //GEMM_PACKED_TEMPLATE_DEVICE_CUH
