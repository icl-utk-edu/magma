/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef MAGMA_LAPACK_H
#define MAGMA_LAPACK_H

#include "magma_mangling.h"

#include "magma_zlapack.h"
#include "magma_clapack.h"
#include "magma_dlapack.h"
#include "magma_slapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define lapackf77_ieeeck FORTRAN_NAME( ieeeck, IEEECK )
#define lapackf77_lsame  FORTRAN_NAME( lsame,  LSAME  )

#define lapackf77_slamch FORTRAN_NAME( slamch, SLAMCH )
#define lapackf77_dlamch FORTRAN_NAME( dlamch, DLAMCH )
#define lapackf77_slabad FORTRAN_NAME( slabad, SLABAD )
#define lapackf77_dlabad FORTRAN_NAME( dlabad, DLABAD )
#define lapackf77_zcgesv FORTRAN_NAME( zcgesv, ZCGESV )
#define lapackf77_dsgesv FORTRAN_NAME( dsgesv, DSGESV )

#define lapackf77_dsterf FORTRAN_NAME( dsterf, DSTERF )
#define lapackf77_ssterf FORTRAN_NAME( ssterf, SSTERF )

#define lapackf77_zlag2c FORTRAN_NAME( zlag2c, ZLAG2C )
#define lapackf77_clag2z FORTRAN_NAME( clag2z, CLAG2Z )
#define lapackf77_dlag2s FORTRAN_NAME( dlag2s, DLAG2S )
#define lapackf77_slag2d FORTRAN_NAME( slag2d, SLAG2D )

#define lapackf77_zlat2c FORTRAN_NAME( zlat2c, ZLAT2C )
#define lapackf77_dlat2s FORTRAN_NAME( dlat2s, DLAT2S )
//#define lapackf77_clat2z FORTRAN_NAME( clat2z, CLAT2Z )
//#define lapackf77_slat2d FORTRAN_NAME( slat2d, SLAT2D )

#define lapackf77_dlapy2 FORTRAN_NAME( dlapy2, DLAPY2 )
#define lapackf77_slapy2 FORTRAN_NAME( slapy2, SLAPY2 )

magma_int_t lapackf77_ieeeck( const magma_int_t *ispec, const float *zero, const float *one );

long   lapackf77_lsame(  const char *ca, const char *cb );

float  lapackf77_slamch( const char *cmach );
double lapackf77_dlamch( const char *cmach );

// "small" (lowercase) defined as char on Windows (reported by MathWorks)
void   lapackf77_slabad( float  *Small, float  *large );
void   lapackf77_dlabad( double *Small, double *large );

void   lapackf77_zcgesv( const magma_int_t *n, const magma_int_t *nrhs,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         const magmaDoubleComplex *B, const magma_int_t *ldb,
                               magmaDoubleComplex *X, const magma_int_t *ldx,
                         magmaDoubleComplex *work, magmaFloatComplex *swork, double *rwork,
                         magma_int_t *iter,
                         magma_int_t *info );

void   lapackf77_dsgesv( const magma_int_t *n, const magma_int_t *nrhs,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         const double *B, const magma_int_t *ldb,
                               double *X, const magma_int_t *ldx,
                         double *work, float *swork,
                         magma_int_t *iter,
                         magma_int_t *info );

void   lapackf77_dsterf( const magma_int_t *n,
                         double *d, double *e,
                         magma_int_t *info );

void   lapackf77_ssterf( const magma_int_t *n,
                         float *d, float *e,
                         magma_int_t *info );

// precision conversion, general matrix
void   lapackf77_zlag2c( const magma_int_t *m, const magma_int_t *n,
                         const magmaDoubleComplex *A,  const magma_int_t *lda,
                               magmaFloatComplex  *SA, const magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_clag2z( const magma_int_t *m, const magma_int_t *n,
                         const magmaFloatComplex  *SA, const magma_int_t *ldsa,
                               magmaDoubleComplex *A,  const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dlag2s( const magma_int_t *m, const magma_int_t *n,
                         const double *A,  const magma_int_t *lda,
                               float  *SA, const magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_slag2d( const magma_int_t *m, const magma_int_t *n,
                         const float  *SA, const magma_int_t *ldsa,
                               double *A,  const magma_int_t *lda,
                         magma_int_t *info );

// precision conversion, triangular (or symmetric) matrix
void   lapackf77_zlat2c( const char *uplo, const magma_int_t *n,
                         const magmaDoubleComplex *A,  const magma_int_t *lda,
                               magmaFloatComplex  *SA, const magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_dlat2s( const char *uplo, const magma_int_t *n,
                         const double *A,  const magma_int_t *lda,
                               float  *SA, const magma_int_t *ldsa,
                         magma_int_t *info );

// not implemented in LAPACK
//void lapackf77_clat2z(const char *uplo, const magma_int_t *n,
//                       const magmaFloatComplex  *SA, const magma_int_t *ldsa,
//                             magmaDoubleComplex *A,  const magma_int_t *lda,
//                       magma_int_t *info );
//
//void lapackf77_slat2d( const char *uplo, const magma_int_t *n,
//                       const float  *SA, const magma_int_t *ldsa,
//                             double *A,  const magma_int_t *lda,
//                       magma_int_t *info );

double lapackf77_dlapy2( const double *x, const double *y );
float  lapackf77_slapy2( const float  *x, const float  *y );

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_LAPACK_H */
