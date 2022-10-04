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

       See [zcds]gemm_fermi.cu for description of related files.
*/

// =============================================================================
// reset variables from previous includes of this file.
#undef TRANS_A
#undef TRANS_B
#undef CONJ_A
#undef CONJ_B

#undef BLK_M
#undef BLK_N

#undef THR_M
#undef THR_N

#undef batched_herk_kernel_name_
#undef batched_herk_kernel_name
#undef batched_gemm_kernel_name_
#undef batched_gemm_kernel_name
#undef gemm_kernel_name_
#undef gemm_kernel_name

#undef devfunc_name_
#undef devfunc_name

// =============================================================================

#if   (version == trans_nn)
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nn_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nn
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_nn
  #define BLK_M BLK_M_nn
  #define BLK_N BLK_N_nn

#elif (version == trans_nt)
  #define TRANS_B
  #define batched_herk_kernel_name_(p)  magmablas_ ## p ## _herk_kernel_fermi_nt_batched
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nt_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nt
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_nt
  #define BLK_M BLK_M_nt
  #define BLK_N BLK_N_nt

#elif (version == trans_nc)
  #define TRANS_B
  #define CONJ_B
  #define batched_herk_kernel_name_(p)  magmablas_ ## p ## _herk_kernel_fermi_nc_batched
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nc_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_nc
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_nc
  #define BLK_M BLK_M_nc
  #define BLK_N BLK_N_nc

#elif (version == trans_tn)
  #define TRANS_A
  #define batched_herk_kernel_name_(p)  magmablas_ ## p ## _herk_kernel_fermi_tn_batched
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tn_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tn
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_tn
  #define BLK_M BLK_M_tn
  #define BLK_N BLK_N_tn

#elif (version == trans_tt)
  #define TRANS_A
  #define TRANS_B
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tt_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tt
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_tt
  #define BLK_M BLK_M_tt
  #define BLK_N BLK_N_tt

#elif (version == trans_tc)
  #define TRANS_A
  #define TRANS_B
  #define CONJ_B
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tc_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_tc
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_tc
  #define BLK_M BLK_M_tc
  #define BLK_N BLK_N_tc

#elif (version == trans_cn)
  #define TRANS_A
  #define CONJ_A
  #define batched_herk_kernel_name_(p)  magmablas_ ## p ## _herk_kernel_fermi_cn_batched
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_cn_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_cn
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_cn
  #define BLK_M BLK_M_cn
  #define BLK_N BLK_N_cn

#elif (version == trans_ct)
  #define TRANS_A
  #define CONJ_A
  #define TRANS_B
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_ct_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_ct
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_ct
  #define BLK_M BLK_M_ct
  #define BLK_N BLK_N_ct

#elif (version == trans_cc)
  #define TRANS_A
  #define CONJ_A
  #define TRANS_B
  #define CONJ_B
  #define batched_gemm_kernel_name_(p)  p ## gemm_kernel_fermi_cc_batched
  #define gemm_kernel_name_(p)  p ## gemm_kernel_fermi_cc
  #define devfunc_name_(p) p ## gemm_devfunc_fermi_cc
  #define BLK_M BLK_M_cc
  #define BLK_N BLK_N_cc

#endif

// need a second macro in order to expand precision;
// see http://gcc.gnu.org/onlinedocs/cpp/Argument-Prescan.html
#define batched_herk_kernel_name(p) batched_herk_kernel_name_(p)
#define batched_gemm_kernel_name(p) batched_gemm_kernel_name_(p)
#define gemm_kernel_name(p) gemm_kernel_name_(p)
#define devfunc_name(p) devfunc_name_(p)

// =============================================================================

// size of work for a thread
#define THR_M ( BLK_M / DIM_X )
#define THR_N ( BLK_N / DIM_Y )

/******************************************************************************/

extern "C" {

static 
void devfunc_name(precision) (
    int M, int N, int K,
    const FloatingPoint_t* __restrict__ A, int LDA,
    const FloatingPoint_t* __restrict__ B, int LDB,
    FloatingPoint_t*       __restrict__ C, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB , sycl::nd_item<3> item_ct1,
    sycl::accessor<FloatingPoint_t, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    sycl::accessor<FloatingPoint_t, 2, sycl::access_mode::read_write, sycl::access::target::local> sB,
    dpct::image_accessor_ext<sycl::int4, 1> tex_ref_Amagma_z,
    dpct::image_accessor_ext<sycl::int4, 1> tex_ref_Bmagma_z,
    dpct::image_accessor_ext<sycl::float2, 1> tex_ref_Amagma_c,
    dpct::image_accessor_ext<sycl::float2, 1> tex_ref_Bmagma_c,
    dpct::image_accessor_ext<sycl::int2, 1> tex_ref_Amagma_d,
    dpct::image_accessor_ext<sycl::int2, 1> tex_ref_Bmagma_d,
    dpct::image_accessor_ext<float, 1> tex_ref_Amagma_s,
    dpct::image_accessor_ext<float, 1> tex_ref_Bmagma_s)
{
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
    FloatingPoint_t rC[THR_N][THR_M];
    FloatingPoint_t rA[THR_M];
    FloatingPoint_t rB[THR_N];

    // Registers for the dev->shmem copy
    #ifdef TRANS_A
        FloatingPoint_t ra[BLK_M/DIM_YA][BLK_K/DIM_XA];
    #else
        FloatingPoint_t ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    #endif
    #ifdef TRANS_B
        FloatingPoint_t rb[BLK_K/DIM_YB][BLK_N/DIM_XB];
    #else
        FloatingPoint_t rb[BLK_N/DIM_YB][BLK_K/DIM_XB];
    #endif

    #ifdef TEXTURE_1D
        #ifdef TRANS_A
            int coord_A = offsetA + blx*BLK_M*LDA + idyA*LDA + idxA;
        #else
            int coord_A = offsetA + blx*BLK_M     + idyA*LDA + idxA;
        #endif
        #ifdef TRANS_B
            int coord_B = offsetB + bly*BLK_N     + idyB*LDB + idxB;
        #else
            int coord_B = offsetB + bly*BLK_N*LDB + idyB*LDB + idxB;
        #endif
    #else
        // bound is the correction to offs_d in order to not get out of memory bound
        // so bound could be negative value since offs_d could be out of bound
        #ifdef TRANS_A
            const FloatingPoint_t *offs_dA = A + blx*BLK_M*LDA + idyA*LDA + idxA;
            ptrdiff_t boundA = (LDA*(M-1) + K) - ( blx*BLK_M*LDA + idyA*LDA + idxA ) -1;
        #else
            const FloatingPoint_t *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
            ptrdiff_t boundA = (LDA*(K-1) + M) - ( blx*BLK_M  + idyA*LDA + idxA ) -1;
        #endif
        #ifdef TRANS_B
            const FloatingPoint_t *offs_dB = B + bly*BLK_N     + idyB*LDB + idxB;
            ptrdiff_t boundB = (LDB*(K-1) + N) - ( bly*BLK_N     + idyB*LDB + idxB ) -1;
        #else
            const FloatingPoint_t *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
            ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;
        #endif
    #endif

    int m, n, k, kk;


    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);

    // Load A dev->shmem
    #ifdef TRANS_A
        #pragma unroll
        for (n = 0; n < BLK_M; n += DIM_YA)
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XA)
                /*
                DPCT1084:16: The function call has multiple migration results in
                different template instantiations that could not be unified. You
                may need to adjust the code.
                */
                sA[m + idxA][n + idyA] = fetch(A, m, n, boundA);
#else
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA)
            #pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
                /*
                DPCT1084:10: The function call has multiple migration results in
                different template instantiations that could not be unified. You
                may need to adjust the code.
                */
                sA[n + idyA][m + idxA] = fetch(A, m, n, boundA);
    #endif

    // Load B dev->shmem
    #ifdef TRANS_B
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YB)
            #pragma unroll
            for (m = 0; m < BLK_N; m += DIM_XB)
                /*
                DPCT1084:14: The function call has multiple migration results in
                different template instantiations that could not be unified. You
                may need to adjust the code.
                */
                sB[m + idxB][n + idyB] = fetch(B, m, n, boundB);
#else
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                /*
                DPCT1084:11: The function call has multiple migration results in
                different template instantiations that could not be unified. You
                may need to adjust the code.
                */
                sB[n + idyB][m + idxB] = fetch(B, m, n, boundB);
    #endif

    /*
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        #ifdef TEXTURE_1D
            #ifdef TRANS_A
                coord_A += BLK_K;
            #else
                coord_A += BLK_K*LDA;
            #endif
            #ifdef TRANS_B
                coord_B += BLK_K*LDB;
            #else
                coord_B += BLK_K;
            #endif
        #else
            #ifdef TRANS_A
                offs_dA += BLK_K;
                boundA  -= BLK_K;
            #else
                offs_dA += BLK_K*LDA;
                boundA  -= BLK_K*LDA;
            #endif
            #ifdef TRANS_B
                offs_dB += BLK_K*LDB;
                boundB  -= BLK_K*LDB;
            #else
                offs_dB += BLK_K;
                boundB  -= BLK_K;
            #endif
        #endif

        // Load A dev->regs
        #ifdef TRANS_A
            #pragma unroll
            for (n = 0; n < BLK_M/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XA; m++)
                    /*
                    DPCT1084:17: The function call has multiple migration
                    results in different template instantiations that could not
                    be unified. You may need to adjust the code.
                    */
                    ra[n][m] = fetch(A, m * DIM_XA, n * DIM_YA, boundA);
#else
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    /*
                    DPCT1084:12: The function call has multiple migration
                    results in different template instantiations that could not
                    be unified. You may need to adjust the code.
                    */
                    ra[n][m] = fetch(A, m * DIM_XA, n * DIM_YA, boundA);
        #endif

        // Load B dev->regs
        #ifdef TRANS_B
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_N/DIM_XB; m++)
                    /*
                    DPCT1084:15: The function call has multiple migration
                    results in different template instantiations that could not
                    be unified. You may need to adjust the code.
                    */
                    rb[n][m] = fetch(B, m * DIM_XB, n * DIM_YB, boundB);
#else
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    /*
                    DPCT1084:13: The function call has multiple migration
                    results in different template instantiations that could not
                    be unified. You may need to adjust the code.
                    */
                    rb[n][m] = fetch(B, m * DIM_XB, n * DIM_YB, boundB);
        #endif

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
                    #ifdef CONJ_A
                      #ifdef CONJ_B
                        fma(conj(rA[m]), conj(rB[n]), rC[n][m]);
                      #else
                        fma(conj(rA[m]), rB[n], rC[n][m]);
                      #endif
                    #else
                      #ifdef CONJ_B
                        fma(rA[m], conj(rB[n]), rC[n][m]);
                      #else
                        fma(rA[m], rB[n], rC[n][m]);
                      #endif
                    #endif
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
        #ifdef TRANS_A
            #pragma unroll
            for (n = 0; n < BLK_M/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XA; m++)
                    sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[n][m];
        #else
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];
            #endif

        // Load B regs->shmem
        #ifdef TRANS_B
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_N/DIM_XB; m++)
                    sB[m*DIM_XB+idxB][n*DIM_YB+idyB] = rb[n][m];
        #else
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];
        #endif

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
                #ifdef CONJ_A
                  #ifdef CONJ_B
                    fma(conj(rA[m]), conj(rB[n]), rC[n][m]);
                  #else
                    fma(conj(rA[m]), rB[n], rC[n][m]);
                  #endif
                #else
                  #ifdef CONJ_B
                    fma(rA[m], conj(rB[n]), rC[n][m]);
                  #else
                    fma(rA[m], rB[n], rC[n][m]);
                  #endif
                #endif
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx*BLK_M + m*DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                ptrdiff_t offsC = (ptrdiff_t)coord_dCn*(ptrdiff_t)LDC + (ptrdiff_t)coord_dCm;

                FloatingPoint_t &regC = rC[n][m];
                FloatingPoint_t &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

} /* extern "C" { */
