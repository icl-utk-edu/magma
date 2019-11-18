/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgemm
*/
int main( int argc, char** argv)
{

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time;
    float          dev_error, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn, Wm, Wn, info;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc, lddw;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    float *hA, *hB, *hC, *hCdev;
    float *dW;
    magmaHalf_ptr dA, dB, dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    #if CUDA_VERSION >= 9020
    const magmaHalf h_alpha = (magmaHalf) alpha;
    const magmaHalf h_beta  = (magmaHalf) beta;
    #else
    const magmaHalf h_alpha = approx_float_to_half(alpha);
    const magmaHalf h_beta  = approx_float_to_half(beta);
    #endif
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // Allow 3*eps; real needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    // For half precision, there is no lapackf77_hlamch, please visit: 
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float eps = (float)(0.00097656);
    float tol = 3*eps;

    
    printf("%% If running with option --lapack (-l) or with checking (-c), GPU error is computed\n"
            "%% relative to CPU BLAS result in single precision.\n\n");
    printf("%% transA = %s, transB = %s\n",
            lapack_trans_const(opts.transA),
            lapack_trans_const(opts.transB) );
    printf("%%   M     N     K   GPU Gflop/s (ms)      GPU error\n");
    printf("%%========================================================================================================\n");

    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;
            
            // workspace for conversion between single anf half precisions
            Wm    = max(Am, Bm);
            Wn    = max(An, Bn);

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            lddw = magma_roundup(  Wm, opts.align );  // multiple of 32 by default
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_smalloc_cpu( &hA,       lda*An ));
            TESTING_CHECK( magma_smalloc_cpu( &hB,       ldb*Bn ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,       ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCdev,    ldc*N  ));
            
            TESTING_CHECK( magma_smalloc( &dW, lddw*Wn  ));
            TESTING_CHECK( magma_malloc( (void**)&dA, ldda*An*sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**)&dB, lddb*Bn*sizeof(magmaHalf) ));
            TESTING_CHECK( magma_malloc( (void**)&dC, lddc*N *sizeof(magmaHalf)  ));

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );

            /* Convert the matrices to half precision */
            // A
            magma_ssetmatrix( Am, An, hA, lda,  dW, lddw, opts.queue );
            magmablas_slag2h( Am, An, dW, lddw, dA, ldda, &info, opts.queue);
            if(info != 0) {
                printf("error in magmablas_slag2h( dA )\n");
            }
            
            // B
            magma_ssetmatrix( Bm, Bn, hB, ldb,  dW, lddw, opts.queue );
            magmablas_slag2h( Bm, Bn, dW, lddw, dB, lddb, &info, opts.queue);
            if(info != 0) {
                printf("error in magmablas_slag2h( dB )\n");
            }

            // C
            magma_ssetmatrix( M, N, hC, ldc, dW, lddw, opts.queue );
            magmablas_slag2h( M, N, dW, lddw, dC, lddc, &info, opts.queue);
            if(info != 0) {
                printf("error in magmablas_slag2h( dC )\n");
            }

            // for error checks
            float Anorm = lapackf77_slange( "F", &Am, &An, hA, &lda, work );
            float Bnorm = lapackf77_slange( "F", &Bm, &Bn, hB, &ldb, work );
            float Cnorm = lapackf77_slange( "F", &M,  &N,  hC, &ldc, work );
            
            /* =====================================================================
               Performs operation using GPU
               =================================================================== */
            #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
                magma_flush_cache( opts.cache );
                dev_time = magma_sync_wtime( opts.queue );

                magma_hgemm( opts.transA, opts.transB, M, N, K,
                             h_alpha, dA, ldda,
                                      dB, lddb,
                             h_beta,  dC, lddc,
                             opts.queue );

                dev_time = magma_sync_wtime( opts.queue ) - dev_time;
                dev_perf = gflops / dev_time;
                
                magmablas_hlag2s( M, N, dC, lddc, dW, lddw, opts.queue );
                magma_sgetmatrix( M, N, dW, lddw, hCdev, ldc, opts.queue );
            #else
                dev_time = 0.0;
                dev_perf = 0.0;
            #endif
            
            
            /* =====================================================================
               Check the result
               =================================================================== */
            #if defined(HAVE_CUBLAS) || defined(HAVE_HIP)
            if ( opts.lapack || opts.check ) {
                /* =====================================================================
                   Performs operation using CPU BLAS
                   =================================================================== */

                blasf77_sgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                
                // Compute forward error bound (see Higham, 2002, sec. 3.5),
                // modified to include alpha, beta, and input C.
                // ||R_magma - R_ref||_p / (gamma_{K+2} |alpha| ||A||_p ||B||_p + 2 |beta| ||C||_p ) < eps/2.
                // This should work with p = 1, inf, fro, but numerical tests
                // show p = 1, inf are very spiky and sometimes exceed eps.
                // We use gamma_n = sqrt(n)*u instead of n*u/(1-n*u), since the
                // former accurately represents statistical average rounding.
                // We allow a slightly looser tolerance.
                
                // use LAPACK for R_ref
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = lapackf77_slange( "F", &M, &N, hCdev, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                
                bool okay = (dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld %5lld   %7.2f (%7.2f)    %8.2e   %s\n",
                        (long long) M, (long long) N, (long long) K,
                        dev_perf,    1000.*dev_time,
                        dev_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                    printf("%5lld %5lld %5lld   %7.2f (%7.2f)       ---\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time );
            }
            #else
                printf("%5lld %5lld %5lld   %7.2f (%7.2f)       ---\n",
                           (long long) M, (long long) N, (long long) K,
                           dev_perf,    1000.*dev_time );
            #endif
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCdev    );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            magma_free( dW );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
