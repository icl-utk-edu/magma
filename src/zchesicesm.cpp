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

#define COMPLEX

extern "C" magma_int_t
magma_zchesicesm(
    magma_int_t n, magma_int_t m,
    magmaDoubleComplex *dA, magma_int_t lda,
    magmaFloatComplex *dfloatQ, magma_int_t ldq,
    magmaDoubleComplex *dX, magma_int_t ldx,
    double *w, double *d, double *e,
    magmaDoubleComplex *dwork, magma_int_t ldwork,
    magma_queue_t queue)
{
    magmaDoubleComplex *dR, *dC, *dRHS, *df, *dY, *dtemp, *dwork_solve;
    magmaFloatComplex *dfloattemp1, *dfloattemp2;
    double *dw, *dd, *de, *drwork_solve;

    magmaDoubleComplex z_one     = MAGMA_Z_ONE;
    magmaDoubleComplex z_neg_one = MAGMA_Z_NEG_ONE;
    magmaFloatComplex c_one      = MAGMA_C_ONE;
    magmaFloatComplex c_zero     = MAGMA_C_ZERO;

    magma_int_t info;

    // Use ldr as the leading dimensions for workspace matrices.
    magma_int_t ldr = magma_roundup( n, 32 );
    if( ldwork < 8 * m * ldr + 3 * ldr) {
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
    dtemp      = (magmaDoubleComplex*)(de + ldr);
    dfloattemp1 = (magmaFloatComplex*) dtemp;
    dfloattemp2 = dfloattemp1 + m*ldr;

    magma_dsetvector( m, w, 1, dw, 1, queue );
    magma_dsetvector( n, d, 1, dd, 1, queue );
    magma_dsetvector( n, e, 1, de, 1, queue );

    // Apply the back transformations to eigenvectors.
    // X <- Q * X
    magmablas_zlag2c(n, m, dX, ldx, dfloattemp1, ldr, queue, &info);
    magma_cgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
    magmablas_clag2z(n, m, dfloattemp2, ldr, dX, ldx, queue, &info);

    //magma_zprint_gpu( n, n, dA, lda, queue );
    //magma_zprint_gpu( n, n, dX, ldx, queue );
    //magma_cprint_gpu( n, n, dfloatQ, ldq, queue );
    
    // X <- X + 1/2 X * ( I - X^T * X ) 
    magmablas_zlaset( MagmaFull, m, m, MAGMA_Z_ZERO, MAGMA_Z_ONE, dtemp, m, queue );
    magmablas_zgemm( MagmaConjTrans, MagmaNoTrans, m, m, n, MAGMA_Z_NEG_ONE, dX, ldx, dX, ldx, MAGMA_Z_ONE, dtemp, m, queue );
    magmablas_zlacpy( MagmaFull, n, m, dX, ldx, dR, ldr, queue );
    magmablas_zgemm( MagmaNoTrans, MagmaNoTrans, n, m, m, MAGMA_Z_HALF, dR, ldr, dtemp, m, MAGMA_Z_ONE, dX, ldx, queue );

    // Refinement iterations
    for(int it=0; it < 15; it++) {
        int s = it + 1;

        // R <- X * diag(w) - A * X
        timer_start( time );
        magma_zdiag_scale( n, m, dX, ldx, dw, 1, dR, ldr, queue); 
        //magma_zhemm( MagmaLeft, uplo, n, m, z_neg_one, dA, lda, dX, ldx, z_one, dR, ldr, queue );
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, z_neg_one, dA, lda, dX, ldx, z_one, dR, ldr, queue );
        residual = magmablas_zlange(MagmaInfNorm, n, m, dR, ldr, drwork_solve, 6*m*ldr, queue );
        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time computing residual (X*diag(w) - A*X) = %10.6f\n", time );
        timer_start( time );
        printf("|X*diag(w) - A*X|_inf = %.12e at iteration %d\n", residual, s);

        // Form C
        magmablas_zlacpy( MagmaFull, n, m, dX, ldx, dC, ldr, queue );
        magmablas_zlascl( MagmaFull, n, n, 1.0, -1.0, n, m, dC, ldr, queue, &info );
        for( int i=0; i<m; i++ ) {
            magma_zaxpy( n, z_neg_one, dA+s*lda, 1, dC+i*ldr, 1, queue );
        }
        magma_zadd_eigenvalues( m, dC+s, ldr, dw, queue);
        // C <- Q^H * C
        magmablas_zlag2c(n, m, dC, ldr, dfloattemp1, ldr, queue, &info);
        magma_cgemm( MagmaConjTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_clag2z(n, m, dfloattemp2, ldr, dC, ldr, queue, &info);

        // R <- Q^H * R;
        magmablas_zlag2c(n, m, dR, ldr, dfloattemp1, ldr, queue, &info);
        magma_cgemm( MagmaConjTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_clag2z(n, m, dfloattemp2, ldr, dR, ldr, queue, &info);

        // f = Q(s,:)
        magma_ccopy( n, dfloatQ+s, ldq, dfloattemp1, 1, queue );
        magmablas_clag2z(n, 1, dfloattemp1, ldr, df, ldr, queue, &info);

        /*printf("dC=");
        magma_zprint_gpu( n, m, dC, ldr, queue );
        printf("dR=");
        magma_zprint_gpu( n, m, dR, ldr, queue );
        printf("df=");
        magma_zprint_gpu( 1, n, df, 1, queue );
        printf("dd=");
        magma_dprint_gpu( 1, n, dd, 1, queue );
        printf("de=");
        magma_dprint_gpu( 1, n, de, 1, queue );
        printf("dw=");
        magma_dprint_gpu( 1, m, dw, 1, queue );*/
        // Solver
        magma_zsolve_sicesm( n, m, dR, ldr, dC, dd, de, df, dw, dwork_solve, drwork_solve, queue);
        //printf("dR=");
        //magma_zprint_gpu( n, m, dR, ldr, queue );

        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time tridiag solve = %10.6f\n", time );
        timer_start( time );

        // R <- Q * R (correction vectors)
        magmablas_zlag2c(n, m, dR, ldr, dfloattemp1, ldr, queue, &info);
        magma_cgemm( MagmaNoTrans, MagmaNoTrans, n, m, n, c_one, dfloatQ, ldq, dfloattemp1, ldr, c_zero, dfloattemp2, ldr, queue );
        magmablas_clag2z(n, m, dfloattemp2, ldr, dR, ldr, queue, &info);

        //magma_zprint_gpu( 10, 10, dR, ldr, queue );

        magma_queue_sync( queue );
        timer_stop( time );
        timer_printf( "time GEMM after solve = %10.6f\n", time );
        timer_start( time );

        //printf("correction=");
        //magma_dprint_gpu( n, *m, dtemp, lda, queue );
        //magma_dprint_gpu( 5, 5, dtemp+122*lda, lda, queue );

        magma_zupdate_eigenvalues( m, dR+s, ldr, dw, queue);
        //printf("dR=");
        //magma_zprint_gpu( 10, 10, dR, ldr, queue );
        if( it!=0 ) {
            magmablas_zgeadd( n, m, MAGMA_Z_ONE, dR, ldr, dX, ldx, queue );
            magma_znormalize( n, m, dX, ldx, queue );
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
    magmablas_zlaset( MagmaFull, m, m, MAGMA_Z_ZERO, MAGMA_Z_ONE, dtemp, m, queue );
    magmablas_zgemm( MagmaConjTrans, MagmaNoTrans, m, m, n, MAGMA_Z_NEG_ONE, dX, ldx, dX, ldx, MAGMA_Z_ONE, dtemp, m, queue );
    magmablas_zlacpy( MagmaFull, n, m, dX, ldx, dR, ldr, queue );
    magmablas_zgemm( MagmaNoTrans, MagmaNoTrans, n, m, m, MAGMA_Z_HALF, dR, ldr, dtemp, m, MAGMA_Z_ONE, dX, ldx, queue );
    magma_znormalize( n, m, dX, ldx, queue );

    magma_dgetvector(m, dw, 1, w, 1, queue);
    return 0;
}
