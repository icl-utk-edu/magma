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
#ifdef TEXTURE_1D

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

static __device__
FloatingPoint_t tex_fetch(texture<float2> tex_ref, int coord)
{
    return tex1Dfetch(tex_ref, coord);
}

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

static __device__
FloatingPoint_t tex_fetch(texture<float> tex_ref, int coord)
{
    return tex1Dfetch(tex_ref, coord);
}
#endif


// =============================================================================
#ifdef TEXTURE_1D
    #define fetch(A, m, n, bound) tex_fetch(Mjoin1(tex_ref_##A##magma_,precision), coord_##A + n*LD##A+m)
    #define Mjoin1(Mname,Mp) Mjoin(Mname,Mp)
    #define Mjoin(Mname,Mp) Mname##Mp
#else
    #define fetch(A, m, n, bound) offs_d##A[min(n*LD##A+m, bound)]
#endif


#if defined(PRECISION_z)
    #define conj(A)          cuConj(A)
    #define add(A, B)        cuCadd(A, B)
    #define mul(A, B)        cuCmul(A, B)
    #define div(A, B)        cuCdiv(A, B)
    #define fma(A, B, C) C = cuCfma(A, B, C)
    #define make_FloatingPoint(x, y) make_cuDoubleComplex(x, y)
#elif defined(PRECISION_c)
    #define conj(A)          cuConjf(A)
    #define add(A, B)        cuCaddf(A, B)
    #define mul(A, B)        cuCmulf(A, B)
    #define div(A, B)        cuCdivf(A, B)
    #define fma(A, B, C) C = cuCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) make_cuFloatComplex(x, y)
#elif defined(PRECISION_h)
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) ((magmaHalf)x)
#else
    #define conj(A)           (A)
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

// =============================================================================
#ifdef TEXTURE_1D

    #if defined(PRECISION_z)
        texture<int4, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_z;
        texture<int4, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_z;
    #elif defined(PRECISION_c)
        texture<float2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_c;
        texture<float2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_c;
    #elif defined(PRECISION_d)
        texture<int2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_d;
        texture<int2, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_d;
    #elif defined(PRECISION_s)
        texture<float, cudaTextureType1D, cudaReadModeElementType> tex_ref_Amagma_s;
        texture<float, cudaTextureType1D, cudaReadModeElementType> tex_ref_Bmagma_s;
    #endif
    
#endif

#endif // GEMM_TEMPLATE_DEVICE_DEFS_H
