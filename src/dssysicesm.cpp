/*
    -- MAGMA (version 2.5.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Yaohung Mike Tsai

*/
#include "magma_internal.h"
#include "magma_timer.h"

#define REAL

extern "C" magma_int_t
magma_dssysicesm(
    magma_int_t n, magma_int_t m,
    double *dA, magma_int_t lda,
    float *dfloatQ, magma_int_t ldq,
    double *dX, magma_int_t ldx,
    double *w, double *d, double *e,
    double *dwork, magma_int_t ldwork,
    magma_queue_t queue)
{
    double *dR, *dC, *dRHS, *df, *dY, *dtemp, *dwork_solve;
    float *dfloattemp1, *dfloattemp2;
    double *dw, *dd, *de, *drwork_solve;

    double z_one     = MAGMA_D_ONE;
    double z_neg_one = MAGMA_D_NEG_ONE;
    float c_one      = MAGMA_S_ONE;
    float c_zero     = MAGMA_S_ZERO;

    magma_int_t info;

    // Use ldr as the leading dimensions for workspace matrices.
    magma_int_t ldr = magma_roundup( n, 32 );
    if( ldwork < 11 * m * ldr + 4 * ldr) {
        return -12;
    }

    double residual = 1.0;
    double last_residual = 1.0;

    magma_timer_t time;

    dR          = dwork;
    dC          = dR + m*ldr;
    df          = dC + m*ldr;
    dwork_solve = df + ldr;
    drwork_solve = (double*) (dwork_solve + 2*m*ldr);
    dw = drwork_solve + 6*m*ldr;
    dd = dw + ldr;
    de = dd + ldr;

    //One temp matrix in high precision or two temp matrices in low precisions.
    dtemp      = (double*)(de + ldr);
    dfloattemp1 = (float*) dtemp;
    dfloattemp2 = dfloattemp1 + m*ldr;

    magma_dsetvector( m, w, 1, dw, 1, queue );
    magma_dsetvector( n, d, 1, dd, 1, queue );
    magma_dsetvector( n, e, 1, de, 1, queue );

    // Apply the back transformations to eigenvectors.
    // X <- Q * X
    magmablas_dlag2s(n, m, dX, ldx, dfloattemp1, ldr, queue, &info);
    magma_sgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
    magmablas_slag2d(n, m, dfloattemp2, ldr, dX, ldx, queue, &info);

    //magma_dprint_gpu( n, n, dA, lda, queue );
    //magma_dprint_gpu( n, n, dX, ldx, queue );
    //magma_sprint_gpu( n, n, dfloatQ, ldq, queue );
    
    // X <- X + 1/2 X * ( I - X^T * X ) 
    magmablas_dlaset( MagmaFull, m, m, MAGMA_D_ZERO, MAGMA_D_ONE, dtemp, m, queue );
    magmablas_dgemm( MagmaConjTrans, MagmaNoTrans, m, m, n, MAGMA_D_NEG_ONE, dX, ldx, dX, ldx, MAGMA_D_ONE, dtemp, m, queue );
    magmablas_dlacpy( MagmaFull, n, m, dX, ldx, dR, ldr, queue );
    magmablas_dgemm( MagmaNoTrans, MagmaNoTrans, n, m, m, MAGMA_D_HALF, dR, ldr, dtemp, m, MAGMA_D_ONE, dX, ldx, queue );

    // Refinement iterations
    for(int it=0; it < 15; it++) {
        int s = it + 1;

        // R <- X * diag(w) - A * X
        timer_start( time );
        magma_ddiag_scale( n, m, dX, ldx, dw, 1, dR, ldr, queue); 
        //magma_dsymm( MagmaLeft, uplo, n, m, z_neg_one, dA, lda, dX, ldx, z_one, dR, ldr, queue );
        magma_dgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, z_neg_one, dA, lda, dX, ldx, z_one, dR, ldr, queue );
        residual = magmablas_dlange(MagmaInfNorm, n, m, dR, ldr, drwork_solve, 6*m*ldr, queue );
        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time computing residual (X*diag(w) - A*X) = %10.6f\n", time );
        timer_start( time );
        printf("|X*diag(w) - A*X|_inf = %.12e at iteration %d\n", residual, s);

        // Form C
        magmablas_dlacpy( MagmaFull, n, m, dX, ldx, dC, ldr, queue );
        magmablas_dlascl( MagmaFull, n, n, 1.0, -1.0, n, m, dC, ldr, queue, &info );
        for( int i=0; i<m; i++ ) {
            magma_daxpy( n, z_neg_one, dA+s*lda, 1, dC+i*ldr, 1, queue );
        }
        magma_daxpy( m, 1.0, dw, 1, dC+s, ldr, queue );
        // C <- Q^H * C
        magmablas_dlag2s(n, m, dC, ldr, dfloattemp1, ldr, queue, &info);
        magma_sgemm( MagmaConjTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_slag2d(n, m, dfloattemp2, ldr, dC, ldr, queue, &info);

        // R <- Q^H * R;
        magmablas_dlag2s(n, m, dR, ldr, dfloattemp1, ldr, queue, &info);
        magma_sgemm( MagmaConjTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_slag2d(n, m, dfloattemp2, ldr, dR, ldr, queue, &info);

        // f = Q(s,:)
        magma_scopy( n, dfloatQ+s, ldq, dfloattemp1, 1, queue );
        magmablas_slag2d(n, 1, dfloattemp1, ldr, df, ldr, queue, &info);

        /*printf("dC=");
        magma_dprint_gpu( n, m, dC, ldr, queue );
        printf("dR=");
        magma_dprint_gpu( n, m, dR, ldr, queue );
        printf("df=");
        magma_dprint_gpu( 1, n, df, 1, queue );
        printf("dd=");
        magma_dprint_gpu( 1, n, dd, 1, queue );
        printf("de=");
        magma_dprint_gpu( 1, n, de, 1, queue );
        printf("dw=");
        magma_dprint_gpu( 1, m, dw, 1, queue );*/
        // Solver
        magma_dsolve_sicesm( n, m, dR, ldr, dC, dd, de, df, dw, dwork_solve, drwork_solve, queue);
        //printf("dR=");
        //magma_dprint_gpu( n, m, dR, ldr, queue );

        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time tridiag solve = %10.6f\n", time );
        timer_start( time );

        // R <- Q * R (correction vectors)
        magmablas_dlag2s(n, m, dR, ldr, dfloattemp1, ldr, queue, &info);
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_slag2d(n, m, dfloattemp2, ldr, dR, ldr, queue, &info);

        //magma_dprint_gpu( 10, 10, dR, ldr, queue );

        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time GEMM after solve = %10.6f\n", time );
        timer_start( time );

        //printf("correction=");
        //magma_dprint_gpu( n, *m, dtemp, lda, queue );
        //magma_dprint_gpu( 5, 5, dtemp+122*lda, lda, queue );
        magma_daxpy( m, 1.0, dR+s, ldr, dw, 1, queue );
        //printf("dR=");
        //magma_dprint_gpu( 10, 10, dR, ldr, queue );
        if( it!=0 ) {
            magmablas_dgeadd( n, m, MAGMA_D_ONE, dR, ldr, dX, ldx, queue );
            magma_dnormalize( n, m, dX, ldx, queue );
        }

        if( residual < 1e-12 && residual > last_residual ) {
            break;
        }
        last_residual = residual;
        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time update eigenvalues and eigenvectors = %10.6f\n", time ); 
    }

    // X <- X + 1/2 X * ( I - X^T * X )
    magmablas_dlaset( MagmaFull, m, m, MAGMA_D_ZERO, MAGMA_D_ONE, dtemp, m, queue );
    magmablas_dgemm( MagmaConjTrans, MagmaNoTrans, m, m, n, MAGMA_D_NEG_ONE, dX, ldx, dX, ldx, MAGMA_D_ONE, dtemp, m, queue );
    magmablas_dlacpy( MagmaFull, n, m, dX, ldx, dR, ldr, queue );
    magmablas_dgemm( MagmaNoTrans, MagmaNoTrans, n, m, m, MAGMA_D_HALF, dR, ldr, dtemp, m, MAGMA_D_ONE, dX, ldx, queue );
    magma_dnormalize( n, m, dX, ldx, queue );

    magma_dgetvector(m, dw, 1, w, 1, queue);
    return 0;
}
