/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c
       
       This is simply a copy of part of magma_zlapack.h,
       with the { printf(...); } function body added to each function.
*/
#include <stdio.h>

#include "magma_v2.h"
#include "magma_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

static const char* format = "Cannot check results: %s unavailable, since there was no Fortran compiler.\n";

/*
 * Testing functions
 */
void   lapackf77_zbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magmaDoubleComplex *Q, const magma_int_t *ldq,
                         double *d, double *e,
                         magmaDoubleComplex *Pt, const magma_int_t *ldpt,
                         magmaDoubleComplex *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *resid )
                         { printf( format, __func__ ); }

void   lapackf77_zget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magmaDoubleComplex *E, const magma_int_t *lde,
                         #ifdef COMPLEX
                         magmaDoubleComplex *w,
                         #else
                         magmaDoubleComplex *wr,
                         magmaDoubleComplex *wi,
                         #endif
                         magmaDoubleComplex *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *result )
                         { printf( format, __func__ ); }

void   lapackf77_zhet21( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kband,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         double *d, double *e,
                         magmaDoubleComplex *U, const magma_int_t *ldu,
                         magmaDoubleComplex *V, const magma_int_t *ldv,
                         magmaDoubleComplex *tau,
                         magmaDoubleComplex *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *result )
                         { printf( format, __func__ ); }

void   lapackf77_zhst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magmaDoubleComplex *H, const magma_int_t *ldh,
                         magmaDoubleComplex *Q, const magma_int_t *ldq,
                         magmaDoubleComplex *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *result )
                         { printf( format, __func__ ); }

void   lapackf77_zstt21( const magma_int_t *n, const magma_int_t *kband,
                         double *AD,
                         double *AE,
                         double *SD,
                         double *SE,
                         magmaDoubleComplex *U, const magma_int_t *ldu,
                         magmaDoubleComplex *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *result )
                         { printf( format, __func__ ); }

void   lapackf77_zunt01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         magmaDoubleComplex *U, const magma_int_t *ldu,
                         magmaDoubleComplex *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         double *resid )
                         { printf( format, __func__ ); }

// testing/eig
void   lapackf77_zlarfy( const char *uplo, const magma_int_t *n,
                         magmaDoubleComplex *V, const magma_int_t *incv,
                         magmaDoubleComplex *tau,
                         magmaDoubleComplex *C, const magma_int_t *ldc,
                         magmaDoubleComplex *work )
                         { printf( format, __func__ ); }

double lapackf77_zqpt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         magmaDoubleComplex *A,
                         magmaDoubleComplex *Af, const magma_int_t *lda,
                         magmaDoubleComplex *tau, magma_int_t *jpvt,
                         magmaDoubleComplex *work, const magma_int_t *lwork )
                         { printf( format, __func__ ); return -1; }

void   lapackf77_zqrt02( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         magmaDoubleComplex *A,
                         magmaDoubleComplex *AF,
                         magmaDoubleComplex *Q,
                         magmaDoubleComplex *R, const magma_int_t *lda,
                         magmaDoubleComplex *tau,
                         magmaDoubleComplex *work, const magma_int_t *lwork,
                         double *rwork,
                         double *result )
                         { printf( format, __func__ ); }

// testing/matgen
void   lapackf77_zlatms( const magma_int_t *m, const magma_int_t *n,
                         const char *dist, magma_int_t *iseed, const char *sym,
                         double *d,
                         const magma_int_t *mode, const double *cond,
                         const double *dmax,
                         const magma_int_t *kl, const magma_int_t *ku,
                         const char *pack,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magmaDoubleComplex *work,
                         magma_int_t *info )
                         { printf( format, __func__ ); }

#ifdef __cplusplus
}
#endif
