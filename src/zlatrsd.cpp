/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> c
       Making s,d precisions requires fixing dot call.
*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZLATRSD solves one of the triangular systems with modified diagonal
       (A - lambda*I)    * x = s*b,
       (A - lambda*I)**T * x = s*b,  or
       (A - lambda*I)**H * x = s*b,
    with scaling to prevent overflow.  Here A is an upper or lower
    triangular matrix, A**T denotes the transpose of A, A**H denotes the
    conjugate transpose of A, x and b are n-element vectors, and s is a
    scaling factor, usually less than or equal to 1, chosen so that the
    components of x will be less than the overflow threshold.  If the
    unscaled problem will not cause overflow, the Level 2 BLAS routine
    ZTRSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
    then s is set to 0 and a non-trivial solution to A*x = 0 is returned.

    This version subtracts lambda from the diagonal, for use in ztrevc
    to compute eigenvectors. It does not modify A during the computation.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the matrix A is upper or lower triangular.
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    trans   magma_trans_t
            Specifies the operation applied to A.
      -     = MagmaNoTrans:    Solve (A - lambda*I)    * x = s*b  (No transpose)
      -     = MagmaTrans:      Solve (A - lambda*I)**T * x = s*b  (Transpose)
      -     = MagmaConjTrans:  Solve (A - lambda*I)**H * x = s*b  (Conjugate transpose)

    @param[in]
    diag    magma_diag_t
            Specifies whether or not the matrix A is unit triangular.
      -     = MagmaNonUnit:  Non-unit triangular
      -     = MagmaUnit:     Unit triangular

    @param[in]
    normin  magma_bool_t
            Specifies whether CNORM has been set or not.
      -     = MagmaTrue:   CNORM contains the column norms on entry
      -     = MagmaFalse:  CNORM is not set on entry.  On exit, the norms will
                           be computed and stored in CNORM.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    A       COMPLEX_16 array, dimension (LDA,N)
            The triangular matrix A.  If UPLO = MagmaUpper, the leading n by n
            upper triangular part of the array A contains the upper
            triangular matrix, and the strictly lower triangular part of
            A is not referenced.  If UPLO = MagmaLower, the leading n by n lower
            triangular part of the array A contains the lower triangular
            matrix, and the strictly upper triangular part of A is not
            referenced.  If DIAG = MagmaUnit, the diagonal elements of A are
            also not referenced and are assumed to be 1.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max (1,N).

    @param[in]
    lambda  COMPLEX_16
            Eigenvalue to subtract from diagonal of A.

    @param[in,out]
    x       COMPLEX_16 array, dimension (N)
            On entry, the right hand side b of the triangular system.
            On exit, X is overwritten by the solution vector x.

    @param[out]
    scale   DOUBLE PRECISION
            The scaling factor s for the triangular system
               A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b.
            If SCALE = 0, the matrix A is singular or badly scaled, and
            the vector x is an exact or approximate solution to A*x = 0.
    
    @param[in,out]
    cnorm   (input or output) DOUBLE PRECISION array, dimension (N)
      -     If NORMIN = MagmaTrue, CNORM is an input argument and CNORM(j)
            contains the norm of the off-diagonal part of the j-th column
            of A.  If TRANS = MagmaNoTrans, CNORM(j) must be greater than or equal
            to the infinity-norm, and if TRANS = MagmaTrans or MagmaConjTrans, CNORM(j)
            must be greater than or equal to the 1-norm.
      -     If NORMIN = MagmaFalse, CNORM is an output argument and CNORM(j)
            returns the 1-norm of the offdiagonal part of the j-th column
            of A.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -k, the k-th argument had an illegal value

    Further Details
    ---------------
    A rough bound on x is computed; if that is less than overflow, ZTRSV
    is called, otherwise, specific code is used which checks for possible
    overflow or divide-by-zero at every operation.

    A columnwise scheme is used for solving A*x = b.  The basic algorithm
    if A is lower triangular is

         x[1:n] := b[1:n]
         for j = 1, ..., n
              x(j) := x(j) / A(j,j)
              x[j+1:n] := x[j+1:n] - x(j) * A[j+1:n,j]
         end

    Define bounds on the components of x after j iterations of the loop:
       M(j) = bound on x[1:j]
       G(j) = bound on x[j+1:n]
    Initially, let M(0) = 0 and G(0) = max{x(i), i=1,...,n}.

    Then for iteration j+1 we have
       M(j+1) <= G(j) / | A(j+1,j+1) |
       G(j+1) <= G(j) + M(j+1) * | A[j+2:n,j+1] |
              <= G(j) ( 1 + CNORM(j+1) / | A(j+1,j+1) | )

    where CNORM(j+1) is greater than or equal to the infinity-norm of
    column j+1 of A, not counting the diagonal.  Hence

       G(j) <= G(0) product     ( 1 + CNORM(i) / | A(i,i) | )
                    1 <= i <= j
    and

       |x(j)| <= ( G(0) / |A(j,j)| ) product     ( 1 + CNORM(i) / |A(i,i)| )
                                     1 <= i < j

    Since |x(j)| <= M(j), we use the Level 2 BLAS routine ZTRSV if the
    reciprocal of the largest M(j), j=1,..,n, is larger than
    max(underflow, 1/overflow).

    The bound on x(j) is also used to determine when a step in the
    columnwise method can be performed without fear of overflow.  If
    the computed bound is greater than a large constant, x is scaled to
    prevent overflow, but if the bound overflows, x is set to 0, x(j) to
    1, and scale to 0, and a non-trivial solution to A*x = 0 is found.

    Similarly, a row-wise scheme is used to solve A**T *x = b  or
    A**H *x = b.  The basic algorithm for upper triangular A is:

         for j = 1, ..., n
              x(j) := ( b(j) - A[1:j-1,j]' * x[1:j-1] ) / A(j,j)
         end

    We simultaneously compute two bounds
         G(j) = bound on ( b(i) - A[1:i-1,i]' * x[1:i-1] ), 1 <= i <= j
         M(j) = bound on x(i), 1 <= i <= j

    The initial values are G(0) = 0, M(0) = max{b(i), i=1,..,n}, and we
    add the constraint G(j) >= G(j-1) and M(j) >= M(j-1) for j >= 1.
    Then the bound on x(j) is

         M(j) <= M(j-1) * ( 1 + CNORM(j) ) / | A(j,j) |

              <= M(0) * product      ( ( 1 + CNORM(i) ) / |A(i,i)| )
                        1 <= i <= j

    and we can safely call ZTRSV if 1/M(n) and 1/G(n) are both greater
    than max(underflow, 1/overflow).

    @ingroup magma_latrsd
*******************************************************************************/
extern "C"
magma_int_t magma_zlatrsd(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_bool_t normin,
    magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex lambda,
    magmaDoubleComplex *x,
    double *scale, double *cnorm,
    magma_int_t *info)
{
#define A(i,j) (A + (i) + (j)*lda)

    /* constants */
    const magma_int_t ione = 1;
    const double d_half = 0.5;
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    /* System generated locals */
    magma_int_t len;
    magmaDoubleComplex ztmp;

    /* Local variables */
    magma_int_t i, j;
    double xj, rec, tjj;
    magma_int_t jinc;
    double xbnd;
    magma_int_t imax;
    double tmax;
    magmaDoubleComplex tjjs;
    double xmax, grow;

    double tscal;
    magmaDoubleComplex uscal;
    magma_int_t jlast;
    magmaDoubleComplex csumj;

    double bignum;
    magma_int_t jfirst;
    double smlnum;

    /* Function Body */
    *info = 0;
    magma_int_t upper  = (uplo  == MagmaUpper);
    magma_int_t notran = (trans == MagmaNoTrans);
    magma_int_t nounit = (diag  == MagmaNonUnit);

    /* Test the input parameters. */
    if ( ! upper && uplo != MagmaLower ) {
        *info = -1;
    }
    else if (! notran &&
             trans != MagmaTrans &&
             trans != MagmaConjTrans) {
        *info = -2;
    }
    else if ( ! nounit && diag != MagmaUnit ) {
        *info = -3;
    }
    else if ( ! (normin == MagmaTrue) &&
              ! (normin == MagmaFalse) ) {
        *info = -4;
    }
    else if ( n < 0 ) {
        *info = -5;
    }
    else if ( lda < max(1,n) ) {
        *info = -7;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( n == 0 ) {
        return *info;
    }

    /* Determine machine dependent parameters to control overflow. */
    smlnum = lapackf77_dlamch( "Safe minimum" );
    bignum = 1. / smlnum;
    lapackf77_dlabad( &smlnum, &bignum );
    smlnum /= lapackf77_dlamch( "Precision" );
    bignum = 1. / smlnum;
    *scale = 1.;

    if ( normin == MagmaFalse ) {
        /* Compute the 1-norm of each column, not including the diagonal. */
        if ( upper ) {
            /* A is upper triangular. */
            cnorm[0] = 0.;
            for( j = 1; j < n; ++j ) {
                cnorm[j] = magma_cblas_dzasum( j, A(0,j), ione );
            }
        }
        else {
            /* A is lower triangular. */
            for( j = 0; j < n-1; ++j ) {
                cnorm[j] = magma_cblas_dzasum( n-(j+1), A(j+1,j), ione );
            }
            cnorm[n-1] = 0.;
        }
    }

    /* Scale the column norms by TSCAL if the maximum element in CNORM is */
    /* greater than BIGNUM/2. */
    imax = blasf77_idamax( &n, &cnorm[0], &ione ) - 1;
    tmax = cnorm[imax];
    if ( tmax <= bignum * 0.5 ) {
        tscal = 1.;
    }
    else {
        tscal = 0.5 / (smlnum * tmax);
        blasf77_dscal( &n, &tscal, &cnorm[0], &ione );
    }

    /* ================================================================= */
    /* Compute a bound on the computed solution vector to see if the */
    /* Level 2 BLAS routine ZTRSV can be used. */
    xmax = 0.;
    for( j = 0; j < n; ++j ) {
        xmax = max( xmax, 0.5*MAGMA_Z_ABS1( x[j] ));
    }
    xbnd = xmax;

    if ( notran ) {
        /* ---------------------------------------- */
        /* Compute the growth in A * x = b. */
        if ( upper ) {
            jfirst = n-1;
            jlast  = 0;
            jinc   = -1;
        }
        else {
            jfirst = 0;
            jlast  = n;
            jinc   = 1;
        }

        if ( tscal != 1. ) {
            grow = 0.;
            goto L60;
        }

        /* A is non-unit triangular. */
        /* Compute GROW = 1/G(j) and XBND = 1/M(j). */
        /* Initially, G(0) = max{x(i), i=1,...,n}. */
        grow = 0.5 / max( xbnd, smlnum );
        xbnd = grow;
        for( j = jfirst; (jinc < 0 ? j >= jlast : j < jlast); j += jinc ) {
            /* Exit the loop if the growth factor is too small. */
            if ( grow <= smlnum ) {
                goto L60;
            }

            if ( nounit ) {
                tjjs = *A(j,j) - lambda;
            }
            else {
                tjjs = c_one - lambda;
            }
            tjj = MAGMA_Z_ABS1( tjjs );

            if ( tjj >= smlnum ) {
                /* M(j) = G(j-1) / abs(A(j,j)) */
                xbnd = min( xbnd, min(1.,tjj)*grow );
            }
            else {
                /* M(j) could overflow, set XBND to 0. */
                xbnd = 0.;
            }

            if ( tjj + cnorm[j] >= smlnum ) {
                /* G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) ) */
                grow *= (tjj / (tjj + cnorm[j]));
            }
            else {
                /* G(j) could overflow, set GROW to 0. */
                grow = 0.;
            }
        }
        grow = xbnd;
L60:
        ;
    }
    else {
        /* ---------------------------------------- */
        /* Compute the growth in A**T * x = b  or  A**H * x = b. */
        if ( upper ) {
            jfirst = 0;
            jlast  = n;
            jinc   = 1;
        }
        else {
            jfirst = n-1;
            jlast  = 0;
            jinc   = -1;
        }

        if ( tscal != 1. ) {
            grow = 0.;
            goto L90;
        }

        /* A is non-unit triangular. */
        /* Compute GROW = 1/G(j) and XBND = 1/M(j). */
        /* Initially, M(0) = max{x(i), i=1,...,n}. */
        grow = 0.5 / max( xbnd, smlnum );
        xbnd = grow;
        for( j = jfirst; (jinc < 0 ? j >= jlast : j < jlast); j += jinc ) {
            /* Exit the loop if the growth factor is too small. */
            if ( grow <= smlnum ) {
                goto L90;
            }

            /* G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) ) */
            xj = 1. + cnorm[j];
            grow = min( grow, xbnd / xj );

            if ( nounit ) {
                tjjs = *A(j,j) - lambda;
            }
            else {
                tjjs = c_one - lambda;
            }
            tjj = MAGMA_Z_ABS1( tjjs );

            if ( tjj >= smlnum ) {
                /* M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j)) */
                if ( xj > tjj ) {
                    xbnd *= (tjj / xj);
                }
            }
            else {
                /* M(j) could overflow, set XBND to 0. */
                xbnd = 0.;
            }
        }
        grow = min( grow, xbnd );
L90:
        ;
    }
        
    /* ================================================================= */
    /* Due to modified diagonal, we can't use regular BLAS ztrsv. */
    
    /* Use a Level 1 BLAS solve, scaling intermediate results. */
    if ( xmax > bignum * 0.5 ) {
        /* Scale X so that its components are less than or equal to */
        /* BIGNUM in absolute value. */
        *scale = (bignum * 0.5) / xmax;
        blasf77_zdscal( &n, scale, &x[0], &ione );
        xmax = bignum;
    }
    else {
        xmax *= 2.;
    }

    if ( notran ) {
        /* ---------------------------------------- */
        /* Solve A * x = b */
        for( j = jfirst; (jinc < 0 ? j >= jlast : j < jlast); j += jinc ) {
            /* Compute x(j) = b(j) / A(j,j), scaling x if necessary. */
            xj = MAGMA_Z_ABS1( x[j] );
            if ( nounit ) {
                tjjs = (*A(j,j) - lambda ) * tscal;
            }
            else {
                tjjs = (c_one - lambda) * tscal;
                if ( tscal == 1. ) {
                    goto L110;
                }
            }
            tjj = MAGMA_Z_ABS1( tjjs );
            if ( tjj > smlnum ) {
                /* abs(A(j,j)) > SMLNUM: */
                if ( tjj < 1. ) {
                    if ( xj > tjj * bignum ) {
                        /* Scale x by 1/b(j). */
                        rec = 1. / xj;
                        blasf77_zdscal( &n, &rec, &x[0], &ione );
                        *scale *= rec;
                        xmax *= rec;
                    }
                }
                x[j] = x[j] / tjjs;
                xj   = MAGMA_Z_ABS1( x[j] );
            }
            else if ( tjj > 0. ) {
                /* 0 < abs(A(j,j)) <= SMLNUM: */
                if ( xj > tjj * bignum ) {
                    /* Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM */
                    /* to avoid overflow when dividing by A(j,j). */
                    rec = (tjj * bignum) / xj;
                    if ( cnorm[j] > 1. ) {
                        /* Scale by 1/CNORM(j) to avoid overflow when */
                        /* multiplying x(j) times column j. */
                        rec /= cnorm[j];
                    }
                    blasf77_zdscal( &n, &rec, &x[0], &ione );
                    *scale *= rec;
                    xmax *= rec;
                }
                x[j] = x[j] / tjjs;
                xj   = MAGMA_Z_ABS1( x[j] );
            }
            else {
                /* A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
                /* scale = 0, and compute a solution to A*x = 0. */
                for( i = 0; i < n; ++i ) {
                    x[i] = c_zero;
                }
                x[j]   = c_one;
                xj     = 1.;
                *scale = 0.;
                xmax   = 0.;
            }
L110:

            /* Scale x if necessary to avoid overflow when adding a */
            /* multiple of column j of A. */
            if ( xj > 1. ) {
                rec = 1. / xj;
                if ( cnorm[j] > (bignum - xmax) * rec ) {
                    /* Scale x by 1/(2*abs(x(j))). */
                    rec *= 0.5;
                    blasf77_zdscal( &n, &rec, &x[0], &ione );
                    *scale *= rec;
                }
            }
            else if ( xj * cnorm[j] > bignum - xmax ) {
                /* Scale x by 1/2. */
                blasf77_zdscal( &n, &d_half, &x[0], &ione );
                *scale *= 0.5;
            }

            if ( upper ) {
                if ( j > 0 ) {
                    /* Compute the update */
                    /* x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j) */
                    len = j;
                    ztmp = -tscal * x[j];
                    blasf77_zaxpy( &len, &ztmp, A(0,j), &ione, &x[0], &ione );
                    i = blasf77_izamax( &len, &x[0], &ione ) - 1;
                    xmax = MAGMA_Z_ABS1( x[i] );
                }
            }
            else {
                if ( j < n-1 ) {
                    /* Compute the update */
                    /* x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j) */
                    len = n - (j+1);
                    ztmp = -tscal * x[j];
                    blasf77_zaxpy( &len, &ztmp, A(j+1,j), &ione, &x[j + 1], &ione );
                    i = j + blasf77_izamax( &len, &x[j + 1], &ione );
                    xmax = MAGMA_Z_ABS1( x[i] );
                }
            }
        }
    }
    else if ( trans == MagmaTrans ) {
        /* ---------------------------------------- */
        /* Solve A**T * x = b */
        for( j = jfirst; (jinc < 0 ? j >= jlast : j < jlast); j += jinc ) {
            /* Compute x(j) = b(j) - sum A(k,j)*x(k). */
            /*                       k<>j             */
            xj = MAGMA_Z_ABS1( x[j] );
            uscal = MAGMA_Z_MAKE( tscal, 0. );
            rec = 1. / max( xmax, 1. );
            if ( cnorm[j] > (bignum - xj) * rec ) {
                /* If x(j) could overflow, scale x by 1/(2*XMAX). */
                rec *= 0.5;
                if ( nounit ) {
                    tjjs = (*A(j,j) - lambda) * tscal;
                }
                else {
                    tjjs = (c_one - lambda) * tscal;
                }
                tjj = MAGMA_Z_ABS1( tjjs );
                if ( tjj > 1. ) {
                    /* Divide by A(j,j) when scaling x if A(j,j) > 1. */
                    rec = min( 1., rec * tjj );
                    uscal = uscal / tjjs;
                }
                if ( rec < 1. ) {
                    blasf77_zdscal( &n, &rec, &x[0], &ione );
                    *scale *= rec;
                    xmax *= rec;
                }
            }

            csumj = c_zero;
            if ( uscal == c_one ) {
                /* If the scaling needed for A in the dot product is 1, */
                /* call ZDOTU to perform the dot product. */
                if ( upper ) {
                    csumj = magma_cblas_zdotu( j, A(0,j), ione, &x[0], ione );
                }
                else if ( j < n-1 ) {
                    csumj = magma_cblas_zdotu( n-(j+1), A(j+1,j), ione, &x[j+1], ione );
                }
            }
            else {
                /* Otherwise, use in-line code for the dot product. */
                if ( upper ) {
                    for( i = 0; i < j; ++i ) {
                        csumj = csumj + (*A(i,j) * uscal) * x[i];
                    }
                }
                else if ( j < n-1 ) {
                    for( i = j+1; i < n; ++i ) {
                        csumj = csumj + (*A(i,j) * uscal) * x[i];
                    }
                }
            }

            if ( uscal == MAGMA_Z_MAKE( tscal, 0. )) {
                /* Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
                /* was not used to scale the dotproduct. */
                x[j] = x[j] - csumj;
                xj = MAGMA_Z_ABS1( x[j] );
                if ( nounit ) {
                    tjjs = (*A(j,j) - lambda) * tscal;
                }
                else {
                    tjjs = (c_one - lambda) * tscal;
                    if ( tscal == 1. ) {
                        goto L160;
                    }
                }

                /* Compute x(j) = x(j) / A(j,j), scaling if necessary. */
                tjj = MAGMA_Z_ABS1( tjjs );
                if ( tjj > smlnum ) {
                    /* abs(A(j,j)) > SMLNUM: */
                    if ( tjj < 1. ) {
                        if ( xj > tjj * bignum ) {
                            /* Scale X by 1/abs(x(j)). */
                            rec = 1. / xj;
                            blasf77_zdscal( &n, &rec, &x[0], &ione );
                            *scale *= rec;
                            xmax   *= rec;
                        }
                    }
                    x[j] = x[j] / tjjs;
                }
                else if ( tjj > 0. ) {
                    /* 0 < abs(A(j,j)) <= SMLNUM: */
                    if ( xj > tjj * bignum ) {
                        /* Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */
                        rec = (tjj * bignum) / xj;
                        blasf77_zdscal( &n, &rec, &x[0], &ione );
                        *scale *= rec;
                        xmax   *= rec;
                    }
                    x[j] = x[j] / tjjs;
                }
                else {
                    /* A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
                    /* scale = 0 and compute a solution to A**T *x = 0. */
                    for( i = 0; i < n; ++i ) {
                        x[i] = c_zero;
                    }
                    x[j]   = c_one;
                    *scale = 0.;
                    xmax   = 0.;
                }
L160:
                ;
            }
            else {
                /* Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
                /* product has already been divided by 1/A(j,j). */
                x[j] = (x[j] / tjjs) - csumj;
            }
            xmax = max( xmax, MAGMA_Z_ABS1( x[j] ));
        }
    }
    else {
        /* ---------------------------------------- */
        /* Solve A**H * x = b */
        for( j = jfirst; (jinc < 0 ? j >= jlast : j < jlast); j += jinc ) {
            /* Compute x(j) = b(j) - sum A(k,j)*x(k). */
            /*                       k<>j             */
            xj = MAGMA_Z_ABS1( x[j] );
            uscal = MAGMA_Z_MAKE( tscal, 0. );
            rec = 1. / max(xmax, 1.);
            if ( cnorm[j] > (bignum - xj) * rec ) {
                /* If x(j) could overflow, scale x by 1/(2*XMAX). */
                rec *= 0.5;
                if ( nounit ) {
                    tjjs = MAGMA_Z_CONJ( *A(j,j) - lambda ) * tscal;
                }
                else {
                    tjjs = (c_one - lambda) * tscal;
                }
                tjj = MAGMA_Z_ABS1( tjjs );
                if ( tjj > 1. ) {
                    /* Divide by A(j,j) when scaling x if A(j,j) > 1. */
                    rec = min( 1., rec * tjj );
                    uscal = uscal / tjjs;
                }
                if ( rec < 1. ) {
                    blasf77_zdscal( &n, &rec, &x[0], &ione );
                    *scale *= rec;
                    xmax   *= rec;
                }
            }

            csumj = c_zero;
            if ( uscal == c_one ) {
                /* If the scaling needed for A in the dot product is 1, */
                /* call ZDOTC to perform the dot product. */
                if ( upper ) {
                    csumj = magma_cblas_zdotc( j, A(0,j), ione, &x[0], ione );
                }
                else if ( j < n-1 ) {
                    csumj = magma_cblas_zdotc( n-(j+1), A(j+1,j), ione, &x[j+1], ione );
                }
            }
            else {
                /* Otherwise, use in-line code for the dot product. */
                if ( upper ) {
                    for( i = 0; i < j; ++i ) {
                        csumj = csumj + (MAGMA_Z_CONJ( *A(i,j) ) * uscal) * x[i];
                    }
                }
                else if ( j < n-1 ) {
                    for( i = j + 1; i < n; ++i ) {
                        csumj = csumj + (MAGMA_Z_CONJ( *A(i,j) ) * uscal) * x[i];
                    }
                }
            }

            if ( uscal == tscal ) {
                /* Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
                /* was not used to scale the dotproduct. */
                x[j] = x[j] - csumj;
                xj = MAGMA_Z_ABS1( x[j] );
                if ( nounit ) {
                    tjjs = MAGMA_Z_CONJ( *A(j,j) - lambda ) * tscal;
                }
                else {
                    tjjs = (c_one - lambda) * tscal;
                    if ( tscal == 1. ) {
                        goto L210;
                    }
                }

                /* Compute x(j) = x(j) / A(j,j), scaling if necessary. */
                tjj = MAGMA_Z_ABS1( tjjs );
                if ( tjj > smlnum ) {
                    /* abs(A(j,j)) > SMLNUM: */
                    if ( tjj < 1. ) {
                        if ( xj > tjj * bignum ) {
                            /* Scale X by 1/abs(x(j)). */
                            rec = 1. / xj;
                            blasf77_zdscal( &n, &rec, &x[0], &ione );
                            *scale *= rec;
                            xmax   *= rec;
                        }
                    }
                    x[j] = x[j] / tjjs;
                }
                else if ( tjj > 0. ) {
                    /* 0 < abs(A(j,j)) <= SMLNUM: */
                    if ( xj > tjj * bignum ) {
                        /* Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */
                        rec = (tjj * bignum) / xj;
                        blasf77_zdscal( &n, &rec, &x[0], &ione );
                        *scale *= rec;
                        xmax   *= rec;
                    }
                    x[j] = x[j] / tjjs;
                }
                else {
                    /* A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
                    /* scale = 0 and compute a solution to A**H *x = 0. */
                    for( i = 0; i < n; ++i ) {
                        x[i] = c_zero;
                    }
                    x[j] = c_one;
                    *scale = 0.;
                    xmax   = 0.;
                }
L210:
                ;
            }
            else {
                /* Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
                /* product has already been divided by 1/A(j,j). */
                x[j] = (x[j] / tjjs) - csumj;
            }
            xmax = max( xmax, MAGMA_Z_ABS1( x[j] ));
        }
    }
    *scale /= tscal;
    
    /* Scale the column norms by 1/TSCAL for return. */
    if ( tscal != 1. ) {
        double d = 1. / tscal;
        blasf77_dscal( &n, &d, &cnorm[0], &ione );
    }

    return *info;
} /* end zlatrsd */
