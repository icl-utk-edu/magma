/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Yaohung Mike Tsai
*/

#include "magma_internal.h"
#include "shuffle.cuh"
//#define FULL_MASK 0xffffffff
__global__
void magma_znormalize_kernel( magma_int_t n, magma_int_t m, magmaDoubleComplex *dX, magma_int_t ldx) {
    int idx = threadIdx.x;
    int blkx = blockIdx.x;

    dX += blkx * ldx;

    double dot = 0.0;

    for( int i=idx; i<n; i+=warpSize ) {
        dot += MAGMA_Z_REAL(dX[i] * MAGMA_Z_CONJ(dX[i]));
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        //dot += __shfl_xor_sync(FULL_MASK, dot, offset);
        dot += magmablas_dshfl_xor(dot, offset);
    }

    dot = sqrt(1.0/dot);

    for( int i=idx; i<n; i+=warpSize ) {
        dX[i] *= dot;
    }
}

extern "C"
void magma_znormalize( magma_int_t n, magma_int_t m, magmaDoubleComplex *dX, magma_int_t ldx, magma_queue_t queue) {
    dim3 threads( 32 ); // Hardcode the warp size
    dim3 grid( m );
    magma_znormalize_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(n, m, dX, ldx);
}
__global__
void magma_dnormalize_kernel( magma_int_t n, magma_int_t m, double *dX, magma_int_t ldx) {
    int idx = threadIdx.x;
    int blkx = blockIdx.x;

    dX += blkx * ldx;

    double dot = 0.0;

    for( int i=idx; i<n; i+=warpSize ) {
        dot += dX[i] * dX[i];
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        //dot += __shfl_xor_sync(FULL_MASK, dot, offset);
        dot += magmablas_dshfl_xor(dot, offset);
    }

    dot = sqrt(1.0/dot);

    for( int i=idx; i<n; i+=warpSize ) {
        dX[i] *= dot;
    }
}

extern "C"
void magma_dnormalize( magma_int_t n, magma_int_t m, double *dX, magma_int_t ldx, magma_queue_t queue) {
    dim3 threads( 32 ); // Hardcode the warp size
    dim3 grid( m );
    magma_dnormalize_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(n, m, dX, ldx);
}

__global__
void magma_zdiag_scale_kernel( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                              double *x, magma_int_t incx,
                              magmaDoubleComplex *C, magma_int_t ldc)
{
    int idx = threadIdx.x;
    int blkx = blockIdx.x;

    A += blkx * lda;
    C += blkx * ldc;
    magmaDoubleComplex w = MAGMA_Z_MAKE( x[incx*blkx], 0.0 );

    for( int i=idx; i<m; i+=blockDim.x ) {
        C[i] = A[i] * w;
    }
}

extern "C"
void magma_zdiag_scale( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                        double *x, magma_int_t incx, 
                        magmaDoubleComplex *C, magma_int_t ldc, magma_queue_t queue )
{
    dim3 threads( 32 );
    dim3 grid( n );
    magma_zdiag_scale_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(m, n, A, lda, x, incx, C, ldc);
    // C = A * diag(X)
}

extern "C"
void magma_ddiag_scale( magma_int_t m, magma_int_t n, double *A, magma_int_t lda,
                        double *x, magma_int_t incx,
                        double *C, magma_int_t ldc, magma_queue_t queue )
{
    // C = A * diag(X)
    cublasDdgmm( queue->cublas_handle(), CUBLAS_SIDE_RIGHT,
                 (int)m, (int)n,
                  A, (int)lda,
                  x, (int)incx,
                  C, (int)ldc);
}

__global__
void magma_zsolve_sicesm_kernel( magma_int_t n, magma_int_t m, magmaDoubleComplex *dR, magma_int_t ldr,
                          magmaDoubleComplex *dC, double *dd, double* de, magmaDoubleComplex *df, double *dw,
                          magmaDoubleComplex *dwork, double *drwork )
{
    int idx = threadIdx.x;
    int blkx = blockIdx.x;
    double w = dw[blkx];

    dR += blkx*ldr;
    dC += blkx*ldr;

    drwork += blkx * ldr * 6;
    double *dD = drwork;
    double *dD2 = dD + ldr;
    double *dL = dD2 + ldr;
    double *dL2 = dL + ldr;
    double *dU = dL2 + ldr;
    double *dU2 = dU + ldr;

    dwork += blkx * ldr * 2;
    magmaDoubleComplex *dR2 = dwork;
    magmaDoubleComplex *dC2 = dR2+ldr;

    magmaDoubleComplex rdot=MAGMA_Z_ZERO;
    magmaDoubleComplex cdot=MAGMA_Z_ZERO;

    // Setup tridiag systems from dd and de.
    for( int i=idx; i<n; i+=warpSize ) {
        //Diagonal shift
        dD[i]  = dd[i]-w;
        if( dD[i]==0.0 ) {
            dD[i] = 2.22e-16;
        }
        dD2[i] = dD[i];
        if( i==0 ) {
            dL[i] = 0.0;
        } else {
            dL[i] = de[i-1];
        }
        dL2[i] = dL[i];
        if( i==n-1 ) {
            dU[i] = 0.0;
        } else {
            dU[i] = de[i];
        }
        dU2[i] = dU[i];
        dR2[i] = dR[i];
        dC2[i] = dC[i];
    }

    for(int offset=1; offset<n; offset*=2 ) {
        for( int i=idx; i<n; i+=warpSize ) {
            if( offset!=1 ) {
                dD[i] = dD2[i];
                dL[i] = dL2[i];
                dU[i] = dU2[i];
                dC[i] = dC2[i];
                dR[i] = dR2[i];
            }
        }
        __syncwarp();
        for( int i=idx; i<n; i+=warpSize ) {
            if( i-offset >=0 ) {
                double k = dL[i] / dD[i-offset];
                if( !isfinite(k) ) {
                    k = dL[i] * 4.5036e+15;
                }
                dD2[i] -=  dU[i-offset] * k;
                dL2[i]  = -dL[i-offset] * k;
                dC2[i] -=  dC[i-offset] * k;
                dR2[i] -=  dR[i-offset] * k;
            }
            if( i+offset < n ) {
                double k = dU[i] / dD[i+offset];
                if( !isfinite(k) ) {
                    k = dU[i] * 4.5036e+15;
                }
                dD2[i] -=  dL[i+offset] * k;
                dU2[i]  = -dU[i+offset] * k;
                dC2[i] -=  dC[i+offset] * k;
                dR2[i] -=  dR[i+offset] * k;
            }
            //if( blkx==0 ) printf("offset %d dD[%d]=%12e dL[%d]=%12e dU[%d]=%12e dC[%d]=%12e dRHS[%d]=%12e\n", offset, i, dD_work[i], i, dL_work[i], i, dU_work[i], i, dC_work[i], i, dRHS_work[i]);
        }
        __syncwarp();
    }

    for( int i=idx; i<n; i+=warpSize ) {
        if( dD2[i]==0.0 ) {
            dD2[i] = 2.22e-16;
        }
        dC2[i]  /= dD2[i];
        cdot += dC2[i] * df[i];
        dR2[i] /= dD2[i];
        rdot += dR2[i] * df[i];
        //if( blkx==0 ) printf("dD[%d]=%12e dC[%d]=%12e dRHS[%d]=%12e\n", i, dD2[i], i, MAGMA_Z_REAL(dC2[i]), i, MAGMA_Z_REAL(dR2[i]));
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        //rdot += __shfl_xor_sync(FULL_MASK, rdot, offset);
        //cdot += __shfl_xor_sync(FULL_MASK, cdot, offset);
        rdot += magmablas_zshfl_xor(rdot, offset);
        cdot += magmablas_zshfl_xor(cdot, offset);
    }

    magmaDoubleComplex scalar = MAGMA_Z_DIV(rdot, MAGMA_Z_ONE+cdot); //rdot / ( 1.0 + cdot )
    //if(threadIdx.x==0) printf("scalar = %12e + %12ei\n", MAGMA_Z_REAL(scalar), MAGMA_Z_IMAG(scalar));

    for( int i=idx; i<n; i+=warpSize ) {
        dR[i] = dR2[i] - scalar * dC2[i];
    }

}

extern "C"
void magma_zsolve_sicesm( magma_int_t n, magma_int_t m, magmaDoubleComplex *dR, magma_int_t ldr,
                          magmaDoubleComplex *dC, double *dd, double* de, magmaDoubleComplex *df, double *dw,
                          magmaDoubleComplex *dwork, double *drwork, magma_queue_t queue )
{
    dim3 threads( 32 ); // Hardcode the warp size
    dim3 grid( m );

    magma_zsolve_sicesm_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(n, m, dR, ldr, dC, dd, de, df,
                                                                           dw, dwork, drwork);
}
    
    
__global__
void magma_dsolve_sicesm_kernel( magma_int_t n, magma_int_t m, double *dR, magma_int_t ldr,
                          double *dC, double *dd, double* de, double *df, double *dw,
                          double *dwork, double *drwork )
{
    int idx = threadIdx.x;
    int blkx = blockIdx.x;
    double w = dw[blkx];

    dR += blkx * ldr;
    dC += blkx * ldr;

    drwork += blkx * ldr * 6;
    double *dD = drwork;
    double *dD2 = dD + ldr;
    double *dL = dD2 + ldr;
    double *dL2 = dL + ldr;
    double *dU = dL2 + ldr;
    double *dU2 = dU + ldr;

    dwork += blkx * ldr * 2;
    double *dR2 = dwork;
    double *dC2 = dR2+ldr;

    double rdot=0.0;
    double cdot=0.0;

    // Setup tridiag systems from dd and de.
    for( int i=idx; i<n; i+=warpSize ) {
        //Diagonal shift
        dD[i]  = dd[i]-w;
        if( dD[i]==0.0 ) {
            dD[i] = 2.22e-16;
        }
        dD2[i] = dD[i];
        if( i==0 ) {
            dL[i] = 0.0;
        } else {
            dL[i] = de[i-1];
        }
        dL2[i] = dL[i];
        if( i==n-1 ) {
            dU[i] = 0.0;
        } else {
            dU[i] = de[i];
        }
        dU2[i] = dU[i];
        dR2[i] = dR[i];
        dC2[i] = dC[i];
    }

    for(int offset=1; offset<n; offset*=2 ) {
        for( int i=idx; i<n; i+=warpSize ) {
            if( offset!=1 ) {
                dD[i] = dD2[i];
                dL[i] = dL2[i];
                dU[i] = dU2[i];
                dC[i] = dC2[i];
                dR[i] = dR2[i];
            }
        }
        __syncwarp();
        for( int i=idx; i<n; i+=warpSize ) {
            if( i-offset >=0 ) {
                double k = dL[i] / dD[i-offset];
                if( !isfinite(k) ) {
                    k = dL[i] * 4.5036e+15;
                }
                dD2[i] -=  dU[i-offset] * k;
                dL2[i]  = -dL[i-offset] * k;
                dC2[i] -=  dC[i-offset] * k;
                dR2[i] -=  dR[i-offset] * k;
            }
            if( i+offset < n ) {
                double k = dU[i] / dD[i+offset];
                if( !isfinite(k) ) {
                    k = dU[i] * 4.5036e+15;
                }
                dD2[i] -=  dL[i+offset] * k;
                dU2[i]  = -dU[i+offset] * k;
                dC2[i] -=  dC[i+offset] * k;
                dR2[i] -=  dR[i+offset] * k;
            }
            //if( blkx==0 ) printf("offset %d dD[%d]=%12e dL[%d]=%12e dU[%d]=%12e dC[%d]=%12e dRHS[%d]=%12e\n", offset, i, dD_work[i], i, dL_work[i], i, dU_work[i], i, dC_work[i], i, dRHS_work[i]);
        }
        __syncwarp();
    }

    for( int i=idx; i<n; i+=warpSize ) {
        if( dD2[i]==0.0 ) {
            dD2[i] = 2.22e-16;
        }
        dC2[i]   /= dD2[i];
        cdot += dC2[i] * df[i];
        dR2[i] /= dD2[i];
        rdot += dR2[i] * df[i];
        //if( blkx==0 ) printf("dD[%d]=%12e dC[%d]=%12e dRHS[%d]=%12e\n", i, dD_work[i], i, dC_work[i], i, dRHS_work[i]);
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        //rdot += __shfl_xor_sync(FULL_MASK, rdot, offset);
        //cdot += __shfl_xor_sync(FULL_MASK, cdot, offset);
        rdot += magmablas_dshfl_xor(rdot, offset);
        cdot += magmablas_dshfl_xor(cdot, offset);
    }

    double scalar = rdot / ( 1.0 + cdot );

    for( int i=idx; i<n; i+=warpSize ) {
        dR[i] = dR2[i] - scalar * dC2[i];
    }

}

extern "C"
void magma_dsolve_sicesm( magma_int_t n, magma_int_t m, double *dR, magma_int_t ldr,
                          double *dC, double *dd, double* de, double *df, double *dw,
                          double *dwork, double *drwork, magma_queue_t queue )
{
    dim3 threads( 32 ); // Hardcode the warp size
    dim3 grid( m );

    magma_dsolve_sicesm_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(n, m, dR, ldr, dC, dd, de, df,
                                                                           dw, dwork, drwork);
}

__global__
void magma_zupdate_eigenvalues_kernel( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx, double *dw )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > m) return;
    dw[idx] += MAGMA_Z_REAL( dX[idx*incx] );
    dX[idx*incx] = MAGMA_Z_ZERO; 
}

extern "C"
void magma_zupdate_eigenvalues( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx,
                                double *dw, magma_queue_t queue )
{
    dim3 threads( 32 );
    dim3 grid( magma_ceildiv(m, 32) );

    magma_zupdate_eigenvalues_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(m, dX, incx, dw);
}

__global__
void magma_zadd_eigenvalues_kernel( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx, double *dw )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > m) return;
    dX[idx*incx] += MAGMA_Z_MAKE( dw[idx], 0.0 );
}

extern "C"
void magma_zadd_eigenvalues( magma_int_t m, magmaDoubleComplex *dX, magma_int_t incx,
                                        double *dw, magma_queue_t queue )
{
    dim3 threads( 32 );
    dim3 grid( magma_ceildiv(m, 32) );

    magma_zadd_eigenvalues_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(m, dX, incx, dw);
}
