/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       

*/
#include <cuda.h>    // for CUDA_VERSION
#include <cuda_runtime.h>

#if CUDA_VERSION >= 7500
#include <cuda_fp16.h>

#if CUDA_VERSION < 9020
// conversion float to half are not defined for host in CUDA version <9.2
// thus uses the conversion below when CUDA VERSION is < 9.2.
#include <string.h>
//
// Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This code modified from the public domain code here: 
// https://gist.github.com/rygorous/2156668
// The URL above includes more robust conversion routines
// that handle Inf and NaN correctly. 
// 
// It is recommended to use the more robust versions in production code.

typedef unsigned uint;

union FP32
{
    uint u;
    float f;
    struct
    {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
};

union FP16
{
    unsigned short u;
    struct
    {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
};

// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
static half approx_float_to_half(float fl)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16max = { (127 + 16) << 23 };
    FP32 magic = { 15 << 23 };
    FP32 expinf = { (255 ^ 31) << 23 };
    uint sign_mask = 0x80000000u;
    FP16 o = { 0 };

    FP32 f = *((FP32*)&fl);

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    if (!(f.f < f32infty.u)) // Inf or NaN
        o.u = f.u ^ expinf.u;
    else
    {
        if (f.f > f16max.f) f.f = f16max.f;
        f.f *= magic.f;
    }

    o.u = f.u >> 13; // Take the mantissa bits
    o.u |= sign >> 16;
    half tmp;
    memcpy(&tmp, &o, sizeof(half));
    //return *((half*)&o);
    return tmp;
}

// from half->float code - just for verification.
static float half_to_float(half hf)
{
    FP16 h;
    memcpy(&h, &hf, sizeof(half));

    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}
#endif
#endif // CUDA_VERSION >= 7500

#include "magma_internal.h"
//#include "nvToolsExt.h"

//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)

/***************************************************************************//**
    Purpose
    -------
    XHSGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. It uses mixed precision 
    FP32/FP16-w/o TensorCores factorization techniques.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    enable_tc  MAGMA_MP_TYPE_T
               internal and expert API uses. enable/disable tensor cores

    @param[in]
    mp_algo_type  MAGMA_MP_TYPE_T
               internal and expert API uses.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_xhsgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info,
    magma_mp_type_t enable_tc,
    magma_mp_type_t mp_algo_type )
{
#if CUDA_VERSION >= 7500
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_) dA,  (dA_offset  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) dAT, (dAT_offset + (i_)*lddat + (j_))
    #define dAP(i_, j_) dAP, (             (i_)          + (j_)*maxm)
    #else
    #define  dA(i_, j_) (dA  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) (dAT + (i_)*lddat + (j_))
    #define dAT_hp(i_, j_) (dAT_hp + (i_)*lddat + (j_))
    #define dAP(i_, j_) (dAP + (i_)       + (j_)*maxm)
    #endif

    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    #if 1
    #if CUDA_VERSION >= 9020
    const magmaHalf h_one     = (magmaHalf) 1.0;
    const magmaHalf h_neg_one = (magmaHalf)-1.0;
    #else
    const magmaHalf h_one = approx_float_to_half(1.0);
    const magmaHalf h_neg_one = approx_float_to_half(-1.0);
    #endif
    #else
    FP32 float_one    = *((FP32*)&c_one);
    FP16 half_one     = float_to_half_full(float_one);
    magmaHalf h_one;
    memcpy(&h_one, &half_one, sizeof(half));

    FP32 float_negone = *((FP32*)&c_neg_one);
    FP16 half_negone  = float_to_half_full(float_negone);
    magmaHalf h_neg_one;
    memcpy(&h_neg_one, &half_negone, sizeof(half));
    #endif

    magma_int_t iinfo, nb, jb, nextj, nextjb;
    magma_int_t maxm, maxn, minmn, maxnb;
    magma_int_t i, j, rows, lddat, ldwork;
    magmaFloat_ptr dAT=NULL, dAP=NULL, work=NULL;
    magmaHalf *dAT_hp = NULL;
        
    cublasMath_t mode;
    cublasStatus_t cuerr;
    cublasGemmAlgo_t ALGO = CUBLAS_GEMM_DFALT;


//#define CHECKFOR_NAN_INF
#ifdef CHECKFOR_NAN_INF
    magma_int_t c_gpu_nan=-1, c_gpu_inf=-1;
#endif
    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    minmn = min( m, n );
    maxnb = 512;
    nb    = magma_get_xgetrf_nb( m, n, n, enable_tc, mp_algo_type );

    magma_queue_t queues[2] = { NULL };
    magma_event_t event[2] = { NULL };
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create( &event[0] );
    magma_event_create( &event[1] );

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        if ( MAGMA_SUCCESS != magma_smalloc_cpu( &work, m*n )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }

#if 0
        magmaHalf *dA_hp=NULL;
        magma_malloc( (void**)&dA_hp,  ldda*n *sizeof(magmaHalf) );
        magmablas_convert_sp2hp(m, n, dA, ldda, dA_hp, ldda, queues[0]);
        
        magmablas_convert_hp2sp(m, n, dA_hp, ldda, dA, ldda, queues[0]);      
        magma_sgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
        lapackf77_sgetrf( &m, &n, work, &m, ipiv, info );
        magma_ssetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
        
        magma_free_cpu( work );  work=NULL;

#else
        magma_sgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
        lapackf77_sgetrf( &m, &n, work, &m, ipiv, info );
        magma_ssetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
        magma_free_cpu( work );  work=NULL;
#endif
    }
    else {
        /* Use hybrid blocked code. */
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        if (MAGMA_SUCCESS != magma_smalloc( &dAP, maxnb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if ( m == n ) {
            dAT = dA;
            lddat = ldda;
            magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[0] );
        }
        else {
            lddat = maxn;  // N-by-M
            if (MAGMA_SUCCESS != magma_smalloc( &dAT, lddat*maxm )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
            magmablas_stranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
        }
        magma_queue_sync( queues[0] );  // finish transpose

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, ldwork*maxnb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }

        magma_malloc( (void**)&dAT_hp,  lddat*maxm *sizeof(magmaHalf) );
        magmablas_convert_sp2hp(n, m, dAT, lddat, dAT_hp, lddat, queues[0]);
        magma_queue_sync( queues[0] );  // finish convert whole matrix

        for( j=0; j < minmn; j += jb )
        {
            jb = min(nb, minmn-j);
            rows = m - j;
            if(j==0)
            {
                magmablas_convert_hp2sp(jb, m-j, dAT_hp(j,j), lddat, dAT(j,j), lddat, queues[1]);
                magmablas_stranspose( jb, m-j, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
                magma_queue_sync( queues[1] );
                magma_sgetmatrix_async( m-j, jb, dAP(0,0), maxm, work, ldwork, queues[0] );
            }

            // do the cpu part
            magma_queue_sync( queues[0] );  // wait to get work
            lapackf77_sgetrf( &rows, &jb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 ){
                *info = iinfo + j;
                printf("error sgetrf inside xhsgetrf voici info %d\n",(int)*info);
                goto cleanup;
            }

            magma_ssetmatrix_async( m-j, jb, work, ldwork, dAP, maxm, queues[0] );

            for( i=j; i < j + jb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_hlaswp( n, dAT_hp(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );

            magma_queue_sync( queues[0] );
            magmablas_stranspose( m-j, jb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );
            magma_event_record( event[0], queues[1] );

            nextj  = j+jb;
            nb     = magma_get_xgetrf_nb( minmn-nextj, minmn-nextj, jb, enable_tc, mp_algo_type );
            nextjb = min(nb, minmn-nextj);

            magmablas_convert_hp2sp(nextjb, jb, 
                    dAT_hp(j, nextj), lddat, 
                    dAT(j, nextj), lddat, queues[1]);
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    nextjb, jb,
                    c_one, dAT(j, j    ), lddat,
                    dAT(j, nextj), lddat, queues[1] );
            magmablas_convert_sp2hp(nextjb, jb, 
                    dAT(j, nextj), lddat, 
                    dAT_hp(j, nextj), lddat, queues[1]);
            magma_queue_wait_event(queues[0], event[0]);
            magmablas_convert_sp2hp(jb, m-j, dAT(j,j), lddat, dAT_hp(j,j), lddat, queues[0]);
            magma_queue_sync( queues[0] );
            
            if( nextjb > 0 &&  (m-nextj) > 0 )
            {
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT_TENSOR_OP;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
                }
                if( mp_algo_type == Magma_MP_GEMEX_I16_O16_C32 )
                {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(nextjb), int(m-nextj), int(jb),
                            &c_neg_one, dAT_hp(j,     nextj), CUDA_R_16F, int(lddat),
                            dAT_hp(nextj, j    ), CUDA_R_16F, int(lddat),
                            &c_one,     dAT_hp(nextj, nextj), CUDA_R_16F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_GEMEX_I16_O16_C16 ) {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(nextjb), int(m-nextj), int(jb),
                            &h_neg_one, dAT_hp(j,     nextj), CUDA_R_16F, int(lddat),
                            dAT_hp(nextj, j    ), CUDA_R_16F, int(lddat),
                            &h_one,     dAT_hp(nextj, nextj), CUDA_R_16F, int(lddat),
                            CUDA_R_16F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_HGEMM ) {
                    cublasHgemm( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(nextjb), int(m-nextj), int(jb),
                            &h_neg_one, dAT_hp(j,     nextj), int(lddat),
                            dAT_hp(nextj, j    ), int(lddat),
                            &h_one,     dAT_hp(nextj, nextj), int(lddat) );
                }
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_DEFAULT_MATH);
                }
            }
            magmablas_convert_hp2sp(nextjb, m-nextj, dAT_hp(nextj, nextj), lddat, dAT(nextj, nextj), lddat, queues[1]);
            magmablas_stranspose( nextjb, m-nextj, dAT(nextj, nextj), lddat, dAP(0,0), maxm, queues[1] );
            magma_queue_sync( queues[1] );
            magma_sgetmatrix_async( m-nextj, nextjb, dAP(0,0), maxm, work, ldwork, queues[0] );

            magmablas_convert_hp2sp(n-(nextj+nextjb), jb, 
                    dAT_hp(j, nextj+nextjb), lddat, 
                    dAT(j, nextj+nextjb), lddat, queues[1]);
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    n-(nextj+nextjb), jb,
                    c_one, dAT(j, j           ), lddat,
                    dAT(j, nextj+nextjb), lddat, queues[1] );
            magmablas_convert_sp2hp(n-(nextj+nextjb), jb, 
                    dAT(j, nextj+nextjb), lddat, 
                    dAT_hp(j, nextj+nextjb), lddat, queues[1]);

            if( (n-(nextj+nextjb)) > 0 &&  (m-nextj) > 0 )
            {
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT_TENSOR_OP;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
                }
                if( mp_algo_type == Magma_MP_GEMEX_I16_O16_C32 )
                {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(maxn-(nextj+nextjb)), int(m-nextj), int(jb),
                            &c_neg_one, dAT_hp(j,     nextj+nextjb), CUDA_R_16F, int(lddat),
                            dAT_hp(nextj, j           ), CUDA_R_16F, int(lddat),
                            &c_one,     dAT_hp(nextj, nextj+nextjb), CUDA_R_16F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_GEMEX_I16_O16_C16 ) {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(maxn-(nextj+nextjb)), int(m-nextj), int(jb),
                            &h_neg_one, dAT_hp(j,     nextj+nextjb), CUDA_R_16F, int(lddat),
                            dAT_hp(nextj, j           ), CUDA_R_16F, int(lddat),
                            &h_one,     dAT_hp(nextj, nextj+nextjb), CUDA_R_16F, int(lddat), 
                            CUDA_R_16F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_HGEMM ) {
                    cublasHgemm( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(maxn-(nextj+nextjb)), int(m-nextj), int(jb),
                            &h_neg_one, dAT_hp(j,     nextj+nextjb), int(lddat),
                            dAT_hp(nextj, j           ), int(lddat),
                            &h_one,     dAT_hp(nextj, nextj+nextjb), int(lddat) );
                }
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_DEFAULT_MATH);
                }
            }
        }
        magmablas_convert_hp2sp(n, m, dAT_hp, lddat, dAT, lddat, queues[1]);

        // undo transpose
        if ( m == n ) {
            magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[1] );
        }
        else {
            magmablas_stranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[1] );
        }
    }
    
#ifdef CHECKFOR_NAN_INF
    magma_snan_inf_gpu( MagmaFull, m, n, dA, ldda, &c_gpu_nan, &c_gpu_inf, queues[1] );
    printf("from inside xhsgetrf here is c_gpu_nan %d c_gpu_inf %d\n",(int)c_gpu_nan, (int)c_gpu_inf);
#endif    
cleanup:
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_event_destroy( event[0] );
    magma_event_destroy( event[1] );

    magma_free( dAP );
    if (m != n) {
        magma_free( dAT );
    }
    magma_free( dAT_hp );
    magma_free_pinned( work );
    
    MAGMA_UNUSED( cuerr );
    MAGMA_UNUSED( mode  );
    return *info;
#else
    return MAGMA_ERR_NOT_SUPPORTED;
#endif
} /* magma_xhsgetrf_gpu */

/***************************************************************************//**
    Purpose
    -------
    HGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. It uses mixed precision 
    FP32/FP16 techniques.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    More details can be found in 
    Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham. 2018. 
    Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision 
    iterative refinement solvers. In Proceedings of the International Conference for 
    High Performance Computing, Networking, Storage, and Analysis (SC '18). 
    IEEE Press, Piscataway, NJ, USA, Article 47, 11 pages.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_hgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_xhsgetrf_gpu(m, n, dA, ldda, ipiv, info, 
            Magma_MP_ENABLE_DFLT_MATH, Magma_MP_HGEMM);
    return *info; 
}

