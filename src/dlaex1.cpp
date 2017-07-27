/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    DLAEX1 computes the updated eigensystem of a diagonal
    matrix after modification by a rank-one symmetric matrix.

      T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out)

    where Z = Q'u, u is a vector of length N with ones in the
    CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.

    The eigenvectors of the original matrix are stored in Q, and the
    eigenvalues are in D.  The algorithm consists of three stages:

    The first stage consists of deflating the size of the problem
    when there are multiple eigenvalues or if there is a zero in
    the Z vector.  For each such occurence the dimension of the
    secular equation problem is reduced by one.  This stage is
    performed by the routine DLAED2.

    The second stage consists of calculating the updated
    eigenvalues. This is done by finding the roots of the secular
    equation via the routine DLAED4 (as called by DLAED3).
    This routine also calculates the eigenvectors of the current
    problem.

    The final stage consists of computing the updated eigenvectors
    directly using the updated eigenvalues.  The eigenvectors for
    the current problem are multiplied with the eigenvectors from
    the overall problem.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.
            
    @param[in,out]
    d       DOUBLE PRECISION array, dimension (N)
            On entry, the eigenvalues of the rank-1-perturbed matrix.
            On exit, the eigenvalues of the repaired matrix.
            
    @param[in,out]
    Q       DOUBLE PRECISION array, dimension (LDQ,N)
            On entry, the eigenvectors of the rank-1-perturbed matrix.
            On exit, the eigenvectors of the repaired tridiagonal matrix.
            
    @param[in]
    ldq     INTEGER
            The leading dimension of the array Q.  LDQ >= max(1,N).
            
    @param[in,out]
    indxq   INTEGER array, dimension (N)
            On entry, the permutation which separately sorts the two
            subproblems in D into ascending order.
            On exit, the permutation which will reintegrate the
            subproblems back into sorted order,
            i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.
            
    @param[in]
    rho     DOUBLE PRECISION
            The subdiagonal entry used to create the rank-1 modification.
            
    @param[in]
    cutpnt  INTEGER
            The location of the last eigenvalue in the leading sub-matrix.
            min(1,N) <= CUTPNT <= N/2.
            
    @param
    work    (workspace) DOUBLE PRECISION array, dimension (4*N + N**2)
            
    @param
    iwork   (workspace) INTEGER array, dimension (4*N)
            
    @param
    dwork   (workspace) DOUBLE PRECISION array, dimension (3*N*N/2+3*N)
            
    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.
            
    @param[in]
    vl      DOUBLE PRECISION
    @param[in]
    vu      DOUBLE PRECISION
            if RANGE=MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.
            
    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            if RANGE=MagmaRangeI, the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeV.
            
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ---------------
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    @ingroup magma_laex1
*******************************************************************************/
extern "C" magma_int_t
magma_dlaex1(
    magma_int_t n,
    double *d,
    double *Q, magma_int_t ldq,
    magma_int_t *indxq, double rho, magma_int_t cutpnt,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info)
{
#define Q(i_,j_) (Q + (i_) + (j_)*ldq)

    magma_int_t coltyp, i, idlmda;
    magma_int_t indx, indxc, indxp;
    magma_int_t iq2, is, iw, iz, k, tmp;
    magma_int_t ione = 1;

    //  Test the input parameters.
    *info = 0;

    if ( n < 0 )
        *info = -1;
    else if ( ldq < max(1, n) )
        *info = -4;
    else if ( min( 1, n/2 ) > cutpnt || n/2 < cutpnt )
        *info = -7;
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    //  Quick return if possible
    if ( n == 0 )
        return *info;

    //  The following values are integer pointers which indicate
    //  the portion of the workspace
    //  used by a particular array in DLAED2 and DLAED3.
    iz = 0;
    idlmda = iz + n;
    iw = idlmda + n;
    iq2 = iw + n;

    indx = 0;
    indxc = indx + n;
    coltyp = indxc + n;
    indxp = coltyp + n;

    //  Form the z-vector which consists of the last row of Q_1 and the
    //  first row of Q_2.
    blasf77_dcopy( &cutpnt, Q(cutpnt-1, 0), &ldq, &work[iz], &ione);
    tmp = n-cutpnt;
    blasf77_dcopy( &tmp, Q(cutpnt, cutpnt), &ldq, &work[iz+cutpnt], &ione);

    //  Deflate eigenvalues.
    lapackf77_dlaed2(&k, &n, &cutpnt, d, Q, &ldq, indxq, &rho, &work[iz],
                     &work[idlmda], &work[iw], &work[iq2],
                     &iwork[indx], &iwork[indxc], &iwork[indxp],
                     &iwork[coltyp], info);

    if ( *info != 0 )
        return *info;

    //  Solve Secular Equation.
    if ( k != 0 ) {
        is = (iwork[coltyp]+iwork[coltyp+1])*cutpnt + (iwork[coltyp+1]+iwork[coltyp+2])*(n-cutpnt) + iq2;
        magma_dlaex3(k, n, cutpnt, d, Q, ldq, rho,
                     &work[idlmda], &work[iq2], &iwork[indxc],
                     &iwork[coltyp], &work[iw], &work[is],
                     indxq, dwork, range, vl, vu, il, iu, info );
        if ( *info != 0 )
            return *info;
    }
    else {
        for (i = 0; i < n; ++i)
            indxq[i] = i+1;
    }

    return *info;
} /* magma_dlaex1 */
