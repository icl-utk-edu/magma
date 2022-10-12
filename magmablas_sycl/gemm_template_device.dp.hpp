#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

       See [zcds]gemm_fermi.cu for description of related files.
*/

#ifndef GEMM_TEMPLATE_DEVICE_CUH
#define GEMM_TEMPLATE_DEVICE_CUH

/******************************************************************************/
// op<trans>( x ) returns x or conj(x).
template< const int conjugate, typename T >
static inline
T op( T& x )
{
    if (conjugate == 1) {
        return conj(x);
    } else {
        return x;
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static 
void gemm_template_device_nn(
    int M, int N, int K,
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
#if (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)
    int idx = item_ct1.get_local_id(2); // thread's m dimension
    int idy = item_ct1.get_local_id(1); // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = item_ct1.get_group(2); // block's m dimension
    int bly = item_ct1.get_group(1); // block's n dimension

          // +1 only required if A is transposed
          // +1 always required

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    T ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    T rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    const T *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;

    const T *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;


    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            /*
            DPCT1064:292: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

#pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA)
            sA[n+idyA][m+idxA] = fetch(A, m, n, boundA);

    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB)
            sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);

    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;

        offs_dB += BLK_K;
        boundB  -= BLK_K;

        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
                }
            }
        }

        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    /*
    DPCT1064:293: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (beta == make_FloatingPoint(0.0, 0.0)) {
#pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = mul(alpha, regC);
                }
            }
        }
    } else {
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = add(mul(alpha, regC), mul(beta, memC));
                }
            }
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static 
void gemm_template_device_nt(
    int M, int N, int K,
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
#if (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)
    int idx = item_ct1.get_local_id(2); // thread's m dimension
    int idy = item_ct1.get_local_id(1); // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = item_ct1.get_group(2); // block's m dimension
    int bly = item_ct1.get_group(1); // block's n dimension

          // +1 only required if A is transposed
          // +1 always required

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    T ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    T rb[BLK_K/DIM_YB][BLK_N/DIM_XB];

    const T *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;

    const T *offs_dB = B + bly*BLK_N     + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(K-1) + N) - ( bly*BLK_N     + idyB*LDB + idxB ) -1;

    int m, n, k, kk;


    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            /*
            DPCT1064:294: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

#pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA)
            sA[n+idyA][m+idxA] = fetch(A, m, n, boundA);

    // Load B dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_N; m += DIM_XB)
            sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

    /*
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;

        offs_dB += BLK_K*LDB;
        boundB  -= BLK_K*LDB;

        // Load A dev->regs
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

        // Load B dev->regs
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_N/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
                }
            }
        }

        /*
        DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Load A regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_M/DIM_XA; m++)
                sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

        // Load B regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_N/DIM_XB; m++)
                sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[n][m];
        /*
        DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    /*
    DPCT1064:295: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (beta == make_FloatingPoint(0.0, 0.0)) {
#pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = mul(alpha, regC);
                }
            }
        }
    }else{
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = add(mul(alpha, regC), mul(beta, memC));
                }
            }
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static 
void gemm_template_device_tn(
    int M, int N, int K,
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
#if (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)
    int idx = item_ct1.get_local_id(2); // thread's m dimension
    int idy = item_ct1.get_local_id(1); // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = item_ct1.get_group(2); // block's m dimension
    int bly = item_ct1.get_group(1); // block's n dimension

          // +1 only required if A is transposed
          // +1 always required

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    // Registers for the dev->shmem copy
    T ra[BLK_M/DIM_YA][BLK_K/DIM_XA];
    T rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dA = A + blx*BLK_M*LDA + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(M-1) + K) - ( blx*BLK_M*LDA + idyA*LDA + idxA ) -1;

    const T *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            /*
            DPCT1064:296: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

    // Load A dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_M; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XA)
            sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB)
            sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);

    /*
    DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K;
        boundA  -= BLK_K;

        offs_dB += BLK_K;
        boundB  -= BLK_K;

        // Load A dev->regs
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

        // Load B dev->regs
        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
                }
            }
        }

        /*
        DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Load A regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++)
                sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[n][m];

        // Load B regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        /*
        DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    /*
    DPCT1064:297: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (beta == make_FloatingPoint(0.0, 0.0)) {
#pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = mul(alpha, regC);
                }
            }
        }
    }else{
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = add(mul(alpha, regC), mul(beta, memC));
                }
            }
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N, const int CONJA, const int CONJB>
static 
void gemm_template_device_tt(
    int M, int N, int K,
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta , sycl::nd_item<3> item_ct1,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<T, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{
#if (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)
    int idx = item_ct1.get_local_id(2); // thread's m dimension
    int idy = item_ct1.get_local_id(1); // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = item_ct1.get_group(2); // block's m dimension
    int bly = item_ct1.get_group(1); // block's n dimension

          // +1 only required if A is transposed
          // +1 always required

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    // Registers for the dev->shmem copy
    T ra[BLK_M/DIM_YA][BLK_K/DIM_XA];
    T rb[BLK_K/DIM_YB][BLK_N/DIM_XB];

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dA = A + blx*BLK_M*LDA + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(M-1) + K) - ( blx*BLK_M*LDA + idyA*LDA + idxA ) -1;

    const T *offs_dB = B + bly*BLK_N     + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(K-1) + N) - ( bly*BLK_N     + idyB*LDB + idxB ) -1;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            /*
            DPCT1064:298: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

    // Load A dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_M; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XA)
            sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);

    // Load B dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_N; m += DIM_XB)
            sB[m+idxB][n+idyB] = fetch(B, m, n, boundB);

    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K;
        boundA  -= BLK_K;

        offs_dB += BLK_K*LDB;
        boundB  -= BLK_K*LDB;

        // Load A dev->regs
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++)
                ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);

        // Load B dev->regs
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_N/DIM_XB; m++)
                rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
                }
            }
        }

        /*
        DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Load A regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++)
                sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[n][m];

        // Load B regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_K/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_N/DIM_XB; m++)
                sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[n][m];

        /*
        DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                fma(op<CONJA>(rA[m]), op<CONJB>(rB[n]), rC[n][m]);
            }
        }
    }

    // Store C regs->dev
    /*
    DPCT1064:299: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (beta == make_FloatingPoint(0.0, 0.0)) {
#pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = coord_dCn*(ptrdiff_t)LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = mul(alpha, regC);
                }
            }
        }
    } else {
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    ptrdiff_t offsC = (ptrdiff_t)coord_dCn*(ptrdiff_t)LDC + (ptrdiff_t)coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = add(mul(alpha, regC), mul(beta, memC));
                }
            }
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}

#endif //GEMM_TEMPLATE_DEVICE_CUH
