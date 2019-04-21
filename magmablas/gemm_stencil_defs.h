/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       [zcds]gemm_fermi.cu        defines the CPU driver.
       [zcds]gemm_fermi_kernels.h defines the block sizes for each precision.
       [zcds]gemm_fermi_kernels_batched.h defines the block sizes for each precision.
       gemm_stencil_defs.h        defines types and functions for precision-independent code.
       gemm_stencil.cu            defines the GPU kernel. It gets included
                                  multiple times, once for each transpose version.
*/

// =============================================================================

#ifdef COMPLEX
  #ifdef DOUBLE
    typedef magmaDoubleComplex FloatingPoint_t;
    #define precision z
  #else
    typedef magmaFloatComplex FloatingPoint_t;
    #define precision c
  #endif
#else
  #ifdef DOUBLE
    typedef double FloatingPoint_t;
    #define precision d
  #else
    typedef float FloatingPoint_t;
    #define precision s
  #endif
#endif

// =============================================================================

  #ifdef TEXTURE_1D

    #ifdef COMPLEX
      #ifdef DOUBLE
        static __device__
        FloatingPoint_t tex_fetch(texture<int4> tex_ref, int coord)
        {
            #if (__CUDA_ARCH__ >= 200)
            int4 v = tex1Dfetch(tex_ref, coord);
            return make_cuDoubleComplex(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
            #else
            return make_cuDoubleComplex( 0., 0. );  // dummy code for 1.x compile
            #endif
        }
      #else
        static __device__
        FloatingPoint_t tex_fetch(texture<float2> tex_ref, int coord)
        {
            return tex1Dfetch(tex_ref, coord);
        }
      #endif
    #else
      #ifdef DOUBLE
        static __device__
        FloatingPoint_t tex_fetch(texture<int2> tex_ref, int coord)
        {
            #if (__CUDA_ARCH__ >= 200)
            int2 v = tex1Dfetch(tex_ref, coord);
            return __hiloint2double(v.y, v.x);
            #else
            return 0.;  // dummy code for 1.x compile
            #endif
        }
      #else
        static __device__
        FloatingPoint_t tex_fetch(texture<float> tex_ref, int coord)
        {
            return tex1Dfetch(tex_ref, coord);
        }
      #endif
    #endif

  #endif

// =============================================================================
  #ifdef TEXTURE_1D
    #define fetch(A, m, n, bound) tex_fetch(Mjoin1(tex_ref_##A##magma_,precision), coord_##A + n*LD##A+m)
    #define Mjoin1(Mname,Mp) Mjoin(Mname,Mp)
    #define Mjoin(Mname,Mp) Mname##Mp
  #else
    #define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]
  #endif

#ifdef COMPLEX
  #ifdef DOUBLE
    #define conj(A)          cuConj(A)
    #define add(A, B)        cuCadd(A, B)
    #define mul(A, B)        cuCmul(A, B)
    #define fma(A, B, C) C = cuCfma(A, B, C)
    #define make_FloatingPoint(x, y) make_cuDoubleComplex(x, y);
  #else
    #define conj(A)          cuConjf(A)
    #define add(A, B)        cuCaddf(A, B)
    #define mul(A, B)        cuCmulf(A, B)
    #define fma(A, B, C) C = cuCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) make_cuFloatComplex(x, y);
  #endif
#else
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

// =============================================================================

#define trans_nn 1
#define trans_nt 2
#define trans_nc 3
        
#define trans_tn 4
#define trans_tt 5
#define trans_tc 6
        
#define trans_cn 7
#define trans_ct 8
#define trans_cc 9

// =============================================================================

  #ifdef TEXTURE_1D

    #ifdef COMPLEX
      #ifdef DOUBLE
        texture<int4, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_z;
        texture<int4, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_z;
      #else
        texture<float2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_c;
        texture<float2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_c;
      #endif
    #else
      #ifdef DOUBLE
        texture<int2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_d;
        texture<int2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_d;
      #else
        texture<float, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_s;
        texture<float, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_s;
      #endif
    #endif

  #endif
