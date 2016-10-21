/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar

       @precisions normal z -> s d c

*/
#include "magma_internal.h"


#define A(m,n)   (A + lda * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup magma_hbtype1cb
 *
 *  magma_zhbtype1cb is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel eliminate a column by an column-wise
 *  annihiliation, then it apply a left+right update on the Hermitian triangle.
 *  Note that the column to be eliminated is located at st-1.
 *
 *  All detail are available on technical report or SC11 paper.
 *  Azzam Haidar, Hatem Ltaief, and Jack Dongarra. 2011.
 *  Parallel reduction to condensed forms for symmetric eigenvalue problems
 *  using aggregated fine-grained and memory-aware kernels. In Proceedings
 *  of 2011 International Conference for High Performance Computing,
 *  Networking, Storage and Analysis (SC '11). ACM, New York, NY, USA,
 *  Article 8, 11 pages.
 *  http://doi.acm.org/10.1145/2063384.2063394
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in, out] A
 *          A pointer to the matrix A of size (2*nb+1)-by-n.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1,2*nb+1)
 *
 * @param[out] V
 *          magmaDoubleComplex array, dimension 2*n if eigenvalue only
 *          requested or (ldv*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[in] ldv
 *          The leading dimension of the matrix V. ldv >= TODO.
 *
 * @param[out] TAU
 *          magmaDoubleComplex array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
 *
 * @param[in] st
 *          A pointer to the start index where this kernel will operate.
 *
 * @param[in] ed
 *          A pointer to the end index where this kernel will operate.
 *
 * @param[in] sweep
 *          The sweep number that is eliminated. it serve to calculate the
 *          pointer to the position where to store the Vs and Ts.
 *
 * @param[in] Vblksiz
 *          constant which correspond to the blocking used when applying the Vs.
 *          it serve to calculate the pointer to the position where to store the
 *          Vs and Ts.
 *
 * @param[in] wantz
 *          constant which indicate if Eigenvalue are requested or both
 *          Eigenvalue/Eigenvectors.
 *
 * @param[in] work
 *          Workspace of size nb.
 *
 *******************************************************************************
 *
 * @return
 *          \retval MAGMA_SUCCESS successful exit
 *          \retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/

// -----------------------------------------------------------------------------
// TYPE 1-BAND Lower-columnwise-Householder

extern "C" void
magma_zhbtype1cb(magma_int_t n, magma_int_t nb,
                magmaDoubleComplex *A, magma_int_t lda,
                magmaDoubleComplex *V, magma_int_t ldv, 
                magmaDoubleComplex *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                magmaDoubleComplex *work)
{
    magma_int_t len;
    magma_int_t vpos, taupos;
    //magma_int_t blkid, tpos;
    
    magma_int_t ione = 1;
    magmaDoubleComplex c_one    =  MAGMA_Z_ONE;

    /* find the pointer to the Vs and Ts as stored by the bulgechasing
     * note that in case no eigenvector required V and T are stored
     * on a vector of size n
     * */
    if (wantz == 0) {
        vpos   = (sweep%2)*n + st;
        taupos = (sweep%2)*n + st;
    }
    else {
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, st, ldv, &vpos, &taupos);
        //findVTpos(n, nb, Vblksiz, sweep, st, &vpos, &taupos, &tpos, &blkid);
    }

    len = ed-st+1;
    *(V(vpos)) = c_one;

    //magma_int_t len2 = len-1;
    //blasf77_zcopy( &len2, A(st+1, st-1), &ione, V(vpos+1), &ione );
    memcpy( V(vpos+1), A(st+1, st-1), (len-1)*sizeof(magmaDoubleComplex) );
    memset( A(st+1, st-1), 0, (len-1)*sizeof(magmaDoubleComplex) );

    /* Eliminate the col  at st-1 */
    lapackf77_zlarfg( &len, A(st, st-1), V(vpos+1), &ione, TAU(taupos) );

    /* Apply left and right on A(st:ed,st:ed) */
    magma_zlarfy(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);
}

#undef A
#undef V
#undef TAU
