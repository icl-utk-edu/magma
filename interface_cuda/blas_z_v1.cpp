/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#define COMPLEX

#ifdef HAVE_CUBLAS

// These MAGMA v1 routines are all deprecated.
// See blas_z_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx )
{
    return magma_izamax( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" magma_int_t
magma_izamin_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx )
{
    return magma_izamin( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magma_dzasum_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx )
{
    return magma_dzasum( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zaxpy_v1(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    magma_zaxpy( n, alpha, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zcopy_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    magma_zcopy( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C"
magmaDoubleComplex magma_zdotc_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy )
{
    return magma_zdotc( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
magmaDoubleComplex magma_zdotu_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy )
{
    return magma_zdotu( n, dx, incx, dy, incy, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" double
magma_dznrm2_v1(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx )
{
    return magma_dznrm2( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zrot_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double c, magmaDoubleComplex s )
{
    magma_zrot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zdrot_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double c, double s )
{
    magma_zdrot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_zrotm_v1(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    const double *param )
{
    magma_zrotm( n, dx, incx, dy, incy, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_zrotmg_v1(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    magma_zrotmg( d1, d2, x1, y1, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
magma_zscal_v1(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx )
{
    magma_zscal( n, alpha, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zdscal_v1(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx )
{
    magma_zdscal( n, alpha, dx, incx, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_zswap_v1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy )
{
    magma_zswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
magma_zgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    magma_zgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zgerc_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda )
{
    magma_zgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zgeru_v1(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda )
{
    magma_zgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_zhemv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    magma_zhemv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zher_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda )
{
    magma_zher(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zher2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda )
{
    magma_zher2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ztrmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx )
{
    magma_ztrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ztrsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx )
{
    magma_ztrsv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


// =============================================================================
// Level 3 BLAS

/******************************************************************************/
extern "C" void
magma_zgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zsymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zsymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zsyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zsyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_zsyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zsyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zhemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zhemm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zherk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_zher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magma_zher2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_ztrmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magma_ztrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ztrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magma_ztrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}

#endif // HAVE_CUBLAS

#undef COMPLEX

#endif // MAGMA_NO_V1
