/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names

#define COMPLEX

// These MAGMA v1 routines are all deprecated.
// See corresponding v2 functions for documentation.

/******************************************************************************/
extern "C" void
magmablas_zaxpycp_v1(
    magma_int_t m,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b)
{
    magmablas_zaxpycp( m, r, x, b, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgeadd_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zgeadd( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgeadd2_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zgeadd2( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgemm_v1(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_zgemm( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgemv_conj_v1(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zgemv_conj(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_zgemm_reduce(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT, magma_int_t ldda,
    magmaDoubleComplex          *hA,  magma_int_t lda,
    magmaDoubleComplex_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_zgetmatrix_transpose( m, n, nb, dAT, ldda, hA, lda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" magma_int_t
magmablas_zhemv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_zhemv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" magma_int_t
magmablas_zsymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_zsymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_zprbt_v1(
    magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *du, magmaDoubleComplex *dv)
{
    magmablas_zprbt(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zprbt_mv_v1(
    magma_int_t n,
    magmaDoubleComplex *dv, magmaDoubleComplex *db)
{
    magmablas_zprbt_mv(n, dv, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zprbt_mtv_v1(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex *db)
{
    magmablas_zprbt_mtv(n, du, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlacpy_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlacpy_conj_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1,
    magmaDoubleComplex_ptr dA2, magma_int_t lda2)
{
    magmablas_zlacpy_conj( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy_sym_in( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy_sym_out( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magmablas_zlange_v1(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_zlange( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magmablas_zlanhe_v1(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_zlanhe( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zlarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr tau,
    magmaDoubleComplex_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDoubleComplex_ptr dT, magma_int_t iter,
    magmaDoubleComplex_ptr work )
{
    magma_zlarfx_gpu(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/******************************************************************************/
extern "C" void
magma_zlarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork)
{
    magma_zlarfbx_gpu( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlarfg_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dalpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dtau )
{
    magmablas_zlarfg( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zlarfg_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dAkk )
{
    magma_zlarfg_gpu( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zlarfgx_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter)
{
    magma_zlarfgx_gpu( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zlarfgtx_gpu_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork )
{
    magma_zlarfgtx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_zlascl2_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_zlascl2( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_zlascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaDoubleComplex_const_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl_2x2( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_zlascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl_diag( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_zlaset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_zlaset( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlaset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda)
{
    magmablas_zlaset_band(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlaswp_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_zlaswp( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlaswpx_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_zlaswpx( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlaswp2_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_zlaswp2( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zlaswp_sym_v1( magma_int_t n, magmaDoubleComplex *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_zlaswp_sym( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dznrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc ) 
{
    magmablas_dznrm2_check( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dznrm2_adjust_v1(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDoubleComplex_ptr dc )
{
    magmablas_dznrm2_adjust( k, dxnorm, dc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dznrm2_row_check_adjust_v1(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc )
{
    magmablas_dznrm2_row_check_adjust( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dznrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm ) 
{
    magmablas_dznrm2_cols( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zsetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex     *hA, magma_int_t lda,
    magmaDoubleComplex_ptr       dAT, magma_int_t ldda,
    magmaDoubleComplex_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_zsetmatrix_transpose( m, n, nb, hA, lda, dAT, ldda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" void
magmablas_zswap_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zswapblk_v1(
    magma_order_t order, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_zswapblk(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_zswapdblk( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zsymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_zsymmetrize( uplo, m, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_zsymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_zsymmetrize_tiles( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ztranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" void
magmablas_ztranspose_conj_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ztranspose_conj( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztranspose_conj_inplace_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ztranspose_conj_inplace( n, dA, ldda, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_ztranspose_inplace_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ztranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_ztrsm( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztrsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ztrsm_outofplace( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztrsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ztrsm_work( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ztrtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr d_dinvA)
{
    magmablas_ztrtri_diag( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_zgetmatrix_1D_row_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_zgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_zgetmatrix_1D_col_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_zsetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex    *hA, magma_int_t lda,
    magmaDoubleComplex_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_zsetmatrix_1D_row_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_zsetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA, magma_int_t lda,
    magmaDoubleComplex_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_zsetmatrix_1D_col_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/zlarfb_gpu.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_zlarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV,    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT,    magma_int_t lddt,
    magmaDoubleComplex_ptr dC,          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,       magma_int_t ldwork )
{
    return magma_zlarfb_gpu( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/zlarfb_gpu_gemm.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_zlarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV,    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT,    magma_int_t lddt,
    magmaDoubleComplex_ptr dC,          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_zlarfb_gpu_gemm( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
