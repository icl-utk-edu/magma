/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "magma_internal.h"
#include "magma_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t magma_get_dlaed3_k() { return 512; }

void magma_dvrange(
    magma_int_t k, double *d, magma_int_t *il, magma_int_t *iu, double vl, double vu)
{
    magma_int_t i;

    *il = 1;
    *iu = k;
    for (i = 0; i < k; ++i) {
        if (d[i] > vu) {
            *iu = i;
            break;
        }
        else if (d[i] < vl) {
            ++*il;
        }
    }
    return;
}

void magma_dirange(
    magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu)
{
    magma_int_t i;

    *iil = 1;
    *iiu = 0;
    for (i = il; i <= iu; ++i) {
        if (indxq[i-1] <= k) {
            *iil = indxq[i-1];
            break;
        }
    }
    for (i = iu; i >= il; --i) {
        if (indxq[i-1] <= k) {
            *iiu = indxq[i-1];
            break;
        }
    }
    return;
}

#ifdef __cplusplus
}  // end extern "C"
#endif


/***************************************************************************//**
    Purpose
    -------
    DLAEX3 finds the roots of the secular equation, as defined by the
    values in D, W, and RHO, between 1 and K.  It makes the
    appropriate calls to DLAED4 and then updates the eigenvectors by
    multiplying the matrix of eigenvectors of the pair of eigensystems
    being combined by the matrix of eigenvectors of the K-by-K system
    which is solved here.

    It is used in the last step when only a part of the eigenvectors
    is required. It computes only the required portion of the eigenvectors
    and the rest is not used.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    k       INTEGER
            The number of terms in the rational function to be solved by
            DLAED4.  K >= 0.

    @param[in]
    n       INTEGER
            The number of rows and columns in the Q matrix.
            N >= K (deflation may result in N > K).

    @param[in]
    n1      INTEGER
            The location of the last eigenvalue in the leading submatrix.
            min(1,N) <= N1 <= N/2.

    @param[out]
    d       DOUBLE PRECISION array, dimension (N)
            D(I) contains the updated eigenvalues for
            1 <= I <= K.

    @param[out]
    Q       DOUBLE PRECISION array, dimension (LDQ,N)
            Initially the first K columns are used as workspace.
            On output the columns ??? to ??? contain
            the updated eigenvectors.

    @param[in]
    ldq     INTEGER
            The leading dimension of the array Q.  LDQ >= max(1,N).

    @param[in]
    rho     DOUBLE PRECISION
            The value of the parameter in the rank one update equation.
            RHO >= 0 required.

    @param[in,out]
    dlamda  DOUBLE PRECISION array, dimension (K)
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation. May be changed on output by
            having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
            Cray-2, or Cray C-90, as described above.

    @param[in]
    Q2      DOUBLE PRECISION array, dimension (LDQ2, N)
            The first K columns of this matrix contain the non-deflated
            eigenvectors for the split problem.
            TODO what is LDQ2?

    @param[in]
    indx    INTEGER array, dimension (N)
            The permutation used to arrange the columns of the deflated
            Q matrix into three groups (see DLAED2).
            The rows of the eigenvectors found by DLAED4 must be likewise
            permuted before the matrix multiply can take place.

    @param[in]
    ctot    INTEGER array, dimension (4)
            A count of the total number of the various types of columns
            in Q, as described in INDX.  The fourth column type is any
            column which has been deflated.

    @param[in,out]
    w       DOUBLE PRECISION array, dimension (K)
            The first K elements of this array contain the components
            of the deflation-adjusted updating vector. Destroyed on
            output.

    @param
    s       (workspace) DOUBLE PRECISION array, dimension (N1 + 1)*K
            Will contain the eigenvectors of the repaired matrix which
            will be multiplied by the previously accumulated eigenvectors
            to update the system.

    @param[out]
    indxq   INTEGER array, dimension (N)
            On exit, the permutation which will reintegrate the
            subproblems back into sorted order,
            i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.

    @param
    dwork   (workspace) DOUBLE PRECISION array, dimension (3*N*N/2 + 3*N)

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.
            TODO verify range, vl, vu, il, iu -- copied from dlaex1.

    @param[in]
    vl      DOUBLE PRECISION
    @param[in]
    vu      DOUBLE PRECISION
            if RANGE = MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.

    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            if RANGE = MagmaRangeI, the indices (in ascending order) of the
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

    @ingroup magma_laex3
*******************************************************************************/
extern "C" magma_int_t
magma_dlaex3(
    magma_int_t k, magma_int_t n, magma_int_t n1,
    double *d,
    double *Q, magma_int_t ldq, double rho,
    double *dlamda, double *Q2, magma_int_t *indx,
    magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info )
{
    #define   Q(i_,j_) (Q   + (i_) + (j_)*ldq)
    #define  dQ(i_,j_) (dQ  + (i_) + (j_)*lddq)
    #define dQ2(i_,j_) (dQ2 + (i_) + (j_)*lddq)
    #define  dS(i_,j_) (dS  + (i_) + (j_)*lddq)

    double d_one  = 1.;
    double d_zero = 0.;
    magma_int_t ione = 1;
    magma_int_t ineg_one = -1;

    magma_int_t iil, iiu, rk;

    magma_int_t lddq = n/2 + 1;
    magmaDouble_ptr dQ2 = dwork;
    magmaDouble_ptr dS  = dQ2  + n*lddq;
    magmaDouble_ptr dQ  = dS   + n*lddq;

    magma_int_t i, iq2, j, n12, n2, n23, tmp, lq2;
    double temp;
    magma_int_t alleig, valeig, indeig;

    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);

    *info = 0;

    if (k < 0)
        *info = -1;
    else if (n < k)
        *info = -2;
    else if (ldq < max(1,n))
        *info = -6;
    else if (! (alleig || valeig || indeig))
        *info = -15;
    else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -17;
        }
        else if (indeig) {
            if (il < 1 || il > max(1,n))
                *info = -18;
            else if (iu < min(n,il) || iu > n)
                *info = -19;
        }
    }


    if (*info != 0) {
        magma_xerbla(__func__, -(*info));
        return *info;
    }

    // Quick return if possible
    if (k == 0)
        return *info;
    /*
     Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
     be computed with high relative accuracy (barring over/underflow).
     This is a problem on machines without a guard digit in
     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
     The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
     which on any of these machines zeros out the bottommost
     bit of DLAMDA(I) if it is 1; this makes the subsequent
     subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
     occurs. On binary machines with a guard digit (almost all
     machines) it does not change DLAMDA(I) at all. On hexadecimal
     and decimal machines with a guard digit, it slightly
     changes the bottommost bits of DLAMDA(I). It does not account
     for hexadecimal or decimal machines without guard digits
     (we know of none). We use a subroutine call to compute
     2*DLAMBDA(I) to prevent optimizing compilers from eliminating
     this code.*/

    n2 = n - n1;

    n12 = ctot[0] + ctot[1];
    n23 = ctot[1] + ctot[2];

    iq2 = n1 * n12;
    lq2 = iq2 + n2 * n23;
    
    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    magma_dsetvector_async( lq2, Q2, 1, dQ2(0,0), 1, queue );

#ifdef _OPENMP
    // -------------------------------------------------------------------------
    // openmp implementation
    // -------------------------------------------------------------------------
    //magma_timer_t time = 0;
    //timer_start( time );

    #pragma omp parallel private(i, j, tmp, temp)
    {
        magma_int_t tid     = omp_get_thread_num();
        magma_int_t nthread = omp_get_num_threads();

        magma_int_t ibegin = ( tid    * k) / nthread; // start index of local loop
        magma_int_t iend   = ((tid+1) * k) / nthread; // end   index of local loop
        magma_int_t ik     = iend - ibegin;           // number of local indices

        for (i = ibegin; i < iend; ++i)
            dlamda[i] = lapackf77_dlamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

        for (j = ibegin; j < iend; ++j) {
            magma_int_t tmpp = j+1;
            magma_int_t iinfo = 0;
            lapackf77_dlaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
            // If the zero finder fails, the computation is terminated.
            if (iinfo != 0) {
                #pragma omp critical (magma_dlaex3)
                *info = iinfo;
                break;
            }
        }

        #pragma omp barrier

        if (*info == 0) {
            #pragma omp single
            {
                // Prepare the INDXQ sorting permutation.
                magma_int_t nk = n - k;
                lapackf77_dlamrg( &k, &nk, d, &ione, &ineg_one, indxq);

                // compute the lower and upper bound of the non-deflated eigenvectors
                if (valeig) {
                    magma_dvrange(k, d, &iil, &iiu, vl, vu);
                }
                else if (indeig) {
                    magma_dirange(k, indxq, &iil, &iiu, il, iu);
                }
                else {
                    iil = 1;
                    iiu = k;
                }
                rk = iiu - iil + 1;
            }

            if (k == 2) {
                #pragma omp single
                {
                    for (j = 0; j < k; ++j) {
                        w[0] = *Q(0,j);
                        w[1] = *Q(1,j);

                        i = indx[0] - 1;
                        *Q(0,j) = w[i];
                        i = indx[1] - 1;
                        *Q(1,j) = w[i];
                    }
                }
            }
            else if (k != 1) {
                // Compute updated W.
                blasf77_dcopy( &ik, &w[ibegin], &ione, &s[ibegin], &ione);

                // Initialize W(I) = Q(I,I)
                tmp = ldq + 1;
                blasf77_dcopy( &ik, Q(ibegin,ibegin), &tmp, &w[ibegin], &ione);

                for (j = 0; j < k; ++j) {
                    magma_int_t i_tmp = min(j, iend);
                    for (i = ibegin; i < i_tmp; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                    i_tmp = max(j+1, ibegin);
                    for (i = i_tmp; i < iend; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                }

                for (i = ibegin; i < iend; ++i)
                    w[i] = copysign( sqrt( -w[i] ), s[i]);

                #pragma omp barrier

                // reduce the number of threads used to have enough S workspace
                nthread = min(n1, omp_get_num_threads());

                if (tid < nthread) {
                    ibegin = ( tid    * rk) / nthread + iil - 1;
                    iend   = ((tid+1) * rk) / nthread + iil - 1;
                    ik     = iend - ibegin;
                }
                else {
                    ibegin = -1;
                    iend   = -1;
                    ik     = -1;
                }

                // Compute eigenvectors of the modified rank-1 modification.
                for (j = ibegin; j < iend; ++j) {
                    for (i = 0; i < k; ++i)
                        s[tid*k + i] = w[i] / *Q(i,j);
                    temp = magma_cblas_dnrm2( k, s + tid*k, 1 );
                    for (i = 0; i < k; ++i) {
                        magma_int_t iii = indx[i] - 1;
                        *Q(i,j) = s[tid*k + iii] / temp;
                    }
                }
            }
        }
    }  // end omp parallel
    if (*info != 0)
        return *info;

    //timer_stop( time );
    //timer_printf( "eigenvalues/vector D+zzT = %6.2f\n", time );

#else
    // -------------------------------------------------------------------------
    // Non openmp implementation
    // -------------------------------------------------------------------------
    // magma_timer_t time = 0;
    // timer_start( time );

    for (i = 0; i < k; ++i)
        dlamda[i] = lapackf77_dlamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

    for (j = 0; j < k; ++j) {
        magma_int_t tmpp = j+1;
        magma_int_t iinfo = 0;
        lapackf77_dlaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
        // If the zero finder fails, the computation is terminated.
        if (iinfo != 0)
            *info = iinfo;
    }
    if (*info != 0)
        return *info;

    // Prepare the INDXQ sorting permutation.
    magma_int_t nk = n - k;
    lapackf77_dlamrg( &k, &nk, d, &ione, &ineg_one, indxq);

    // compute the lower and upper bound of the non-deflated eigenvectors
    if (valeig) {
        magma_dvrange(k, d, &iil, &iiu, vl, vu);
    }
    else if (indeig) {
        magma_dirange(k, indxq, &iil, &iiu, il, iu);
    }
    else {
        iil = 1;
        iiu = k;
    }
    rk = iiu - iil + 1;

    if (k == 2) {
        for (j = 0; j < k; ++j) {
            w[0] = *Q(0,j);
            w[1] = *Q(1,j);

            i = indx[0] - 1;
            *Q(0,j) = w[i];
            i = indx[1] - 1;
            *Q(1,j) = w[i];
        }
    }
    else if (k != 1) {
        // Compute updated W.
        blasf77_dcopy( &k, w, &ione, s, &ione);

        // Initialize W(I) = Q(I,I)
        tmp = ldq + 1;
        blasf77_dcopy( &k, Q, &tmp, w, &ione);

        for (j = 0; j < k; ++j) {
            for (i = 0; i < j; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
            for (i = j+1; i < k; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
        }

        for (i = 0; i < k; ++i)
            w[i] = copysign( sqrt( -w[i] ), s[i]);

        // Compute eigenvectors of the modified rank-1 modification.
        for (j = iil-1; j < iiu; ++j) {
            for (i = 0; i < k; ++i)
                s[i] = w[i] / *Q(i,j);
            temp = magma_cblas_dnrm2( k, s, 1 );
            for (i = 0; i < k; ++i) {
                magma_int_t iii = indx[i] - 1;
                *Q(i,j) = s[iii] / temp;
            }
        }
    }

    //timer_stop( time );
    //timer_printf( "eigenvalues/vector D+zzT = %6.2f\n", time );

#endif // _OPENMP
    // Compute the updated eigenvectors.

    //timer_start( time );
    //magma_queue_sync( queue );  // previously, needed to setvector finished. Now all on same queue, so not needed?

    if (rk != 0) {
        if ( n23 != 0 ) {
            if (rk < magma_get_dlaed3_k()) {
                lapackf77_dlacpy("A", &n23, &rk, Q(ctot[0],iil-1), &ldq, s, &n23);
                blasf77_dgemm("N", "N", &n2, &rk, &n23, &d_one, &Q2[iq2], &n2,
                              s, &n23, &d_zero, Q(n1,iil-1), &ldq );
            }
            else {
                magma_dsetmatrix( n23, rk, Q(ctot[0],iil-1), ldq, dS(0,0), n23, queue );
                magma_dgemm( MagmaNoTrans, MagmaNoTrans, n2, rk, n23,
                             d_one,  dQ2(iq2,0), n2,
                                     dS(0,0), n23,
                             d_zero, dQ(0,0), lddq, queue );
                magma_dgetmatrix( n2, rk, dQ(0,0), lddq, Q(n1,iil-1), ldq, queue );
            }
        }
        else {
            lapackf77_dlaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);
        }

        if ( n12 != 0 ) {
            if (rk < magma_get_dlaed3_k()) {
                lapackf77_dlacpy("A", &n12, &rk, Q(0,iil-1), &ldq, s, &n12);
                blasf77_dgemm("N", "N", &n1, &rk, &n12, &d_one, Q2, &n1,
                              s, &n12, &d_zero, Q(0,iil-1), &ldq);
            }
            else {
                magma_dsetmatrix( n12, rk, Q(0,iil-1), ldq, dS(0,0), n12, queue );
                magma_dgemm( MagmaNoTrans, MagmaNoTrans, n1, rk, n12,
                             d_one,  dQ2(0,0), n1,
                                     dS(0,0), n12,
                             d_zero, dQ(0,0), lddq, queue );
                magma_dgetmatrix( n1, rk, dQ(0,0), lddq, Q(0,iil-1), ldq, queue );
            }
        }
        else {
            lapackf77_dlaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
        }
    }
    //timer_stop( time );
    //timer_printf( "gemms = %6.2f\n", time );

    magma_queue_destroy( queue );

    return *info;
} /* magma_dlaex3 */
