/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov

       @precisions d -> s
*/
#include "magma_internal.h"

#ifdef LAPACK_RETURN_DOUBLE
typedef double RETURN_FLOAT;
#else
typedef float  RETURN_FLOAT;
#endif

// -------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#define blasf77_ddot     FORTRAN_NAME( ddot,   DDOT   )

RETURN_FLOAT
blasf77_sdot(   const magma_int_t *n,
                const float *x, const magma_int_t *incx,
                const float *y, const magma_int_t *incy );

double
blasf77_ddot(   const magma_int_t *n,
                const double *x, const magma_int_t *incx,
                const double *y, const magma_int_t *incy );

#ifdef __cplusplus
}
#endif
// ------------------------------------------------------------

// To do: add complex version for determinant & inverse, if needed (no intertia)  

/***************************************************************************//**
    Purpose
    -------
    dsidi computes the determinant, inertia and inverse
    of a double precision symmetric matrix using the factors from
    dsytrf.

    Arguments
    ---------
    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the output from dsytrf.
            On exit, the upper triangle of the inverse of
            the original matrix. The strict lower triangle
            is never referenced.  

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N). 

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    ipiv    INTEGER array, dimension (N) 
            The pivot vector from dsytrf.

    @param[out]
    det     DOUBLE PRECISION array, dimension (2)
            Determinant of the original matrix:
            Determinant = det(0) * 10^det(1) 
            with 1<=fabs(det(0))<10 or det(0) = 0.

    @param[out]
    inert   INTEGER array, dimension (3)
            The inertia of the original matrix:
            inert(0)  =  number of positive eigenvalues.
            inert(1)  =  number of negative eigenvalues.
            inert(2)  =  number of zero eigenvalues.

    @param[in]
    work    (workspace) DOUBLE PRECISION array, dimension (N)
            Work vector.  Contents destroyed.

    @param[in]
    job     INTEGER 
            JOB has the decimal expansion  abc  where
            if  c != 0, the inverse is computed,
            if  b != 0, the determinant is computed,
            if  a != 0, the inertia is computed.

            for example, job = 111  gives all three.

    Further Details 
    ===============
    Error condition: a division by zero may occur if the inverse is requested
    and dsytrf has set info != 0.

    This routine is based on the LINPACK dsidi routine 
    (see also LINPACK USERS' guide, page 5.15).

    @ingroup magma_hetrf
*******************************************************************************/
extern "C" magma_int_t
magma_dsidi(
    double *A, magma_int_t lda, magma_int_t n, magma_int_t *ipiv, 
    double *det, magma_int_t *inert, 
    double *work, magma_int_t job,
    magma_int_t *info)
{
    #define  A(i_, j_) ( A + (i_) + (j_)*lda )

    /* .. Local variables .. */
    double akkp1, temp;
    double ten, d, t, ak, akp1;
    magma_int_t j, jb, k, ks, kstep, ione = 1;
    bool noinv, nodet, noert;

    /* Test the input parameters. */
    *info = 0;
    if ( lda < 0 ) {
        *info = -2; 
    } else if ( n < 0 ) {
        *info = -3;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }   

    noinv = ( (job  %10)      == 0);
    nodet = (((job %100)/ 10) == 0);
    noert = (((job%1000)/100) == 0);

    if (!nodet || !noert) {
        if (!noert) {
            inert[0] = inert[1] = inert[2] = 0;
        }
        if (!nodet) {
            det[0] = 1.0;
            det[1] = 0.0;
            ten = 10.0;
        }

        t = 0.0;
        for(k = 0; k<n; k++) {
            d = *A(k,k);
            
            /* check if 1 by 1 */
            if (ipiv[k]-1 < 0) {
                /* 2 by 2 block
                   use det (d  s)  =  (d/t * c - t) * t  ,  t = fabs(s)
                           (s  c)
                   to avoid underflow/overflow troubles.
                   take two passes through scaling. Use t for flag. */
                if (t != 0.) {
                    d = t;
                    t = 0.;
                } else {
                    t = fabs(*A(k,k+1));
                    d = (d/t)*(*A(k+1,k+1)) - t;
                }
            }

            if (!noert) {
                if (d > 0.) inert[0]++;
                if (d < 0.) inert[1]++;
                if (d ==0.) inert[2]++;
            }
            
            if (!nodet) {
                det[0] *= d;
                if (det[0] != 0.) {
                    while (fabs(det[0]) < 1.0) {
                        det[0] *= ten;
                        det[1] -= 1.0;
                    }
                    while (fabs(det[0]) >= ten) {
                        det[0] /= ten;
                        det[1] += 1.0;
                    }
                }
            }
        }
    }
    
    /* compute inverse(a) */
    if (!noinv) {
        k = 0;
        while (k < n) {
            //fprintf(stderr, "k = %4d ipiv = %4d\n", k, ipiv[k]);
            if (ipiv[k]-1 >= 0) {
                /* 1 by 1 */
                *A(k,k) = 1. / *A(k,k);
                if (k > 0) {
                    blasf77_dcopy(&k, A(0, k), &ione, work, &ione);
                    for(j = 0; j<k; j++) {
                        const magma_int_t jp1 = j+1;
                        *A(j,k) = blasf77_ddot(&jp1, A(0,j),&ione, work,&ione);
                        blasf77_daxpy(&j, &work[j], A(0,j),&ione, A(0,k),&ione);
                    }
                    *A(k,k) += blasf77_ddot(&k, work,&ione, A(0,k),&ione);
                }
                kstep = 1;
            } else {
                /* 2 by 2 */
                t = fabs(*A(k,k+1));
                ak = *A(k,k)/t;
                akp1 = *A(k+1,k+1)/t;
                akkp1 = *A(k,k+1)/t;
                d = t*(ak*akp1 - 1.);
                *A(k,k) = akp1/d;
                *A(k+1,k+1) = ak/d;
                *A(k,k+1) = -akkp1/d;
                if (k > 0) {
                    blasf77_dcopy(&k, A(0,k+1),&ione, work,&ione);
                    for(j = 0; j<k; j++) {
                        const magma_int_t jp1 = j+1;
                        *A(j,k+1) = blasf77_ddot(&jp1, A(0,j),&ione, work,&ione);
                        blasf77_daxpy(&j, &work[j], A(0,j),&ione, A(0,k+1),&ione);
                    }
                    *A(k+1,k+1) += blasf77_ddot(&k, work,&ione, A(0,k+1),&ione);
                    *A(k  ,k+1) += blasf77_ddot(&k, A(0,k),&ione, A(0,k+1),&ione);
                    blasf77_dcopy(&k, A(0,k),&ione, work,&ione);
                    
                    for(j = 0; j<k; j++) {
                        const magma_int_t jp1 = j+1;
                        *A(j,k) = blasf77_ddot(&jp1, A(0,j),&ione, work,&ione);
                        blasf77_daxpy(&j, &work[j], A(0,j),&ione, A(0,k),&ione);
                    }
                    *A(k,k) += blasf77_ddot(&k, work,&ione, A(0,k),&ione);
                }
                kstep = 2;
            }
            /* swap */
            ks = abs(ipiv[k])-1;
            if (ks != k) {
                const magma_int_t ksp1 = ks+1;
                blasf77_dswap(&ksp1, A(0,ks),&ione, A(0,k),&ione);
                for(jb = ks; jb<=k; jb++) {
                    j = k + ks - jb;
                    temp = *A(j,k);
                    *A(j,k) = *A(ks,j);
                    *A(ks,j) = temp;
                }
                if (kstep != 1) {
                    temp = *A(ks,k+1);
                    *A(ks,k+1) = *A(k,k+1);
                    *A(k,k+1) = temp;
                }
            }
            k = k + kstep;
        }
    }

    return *info;
}  /* End of DSIDI */
