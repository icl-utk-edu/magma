/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

// === Define what BLAS to use ============================================
    //#undef  magma_ztrsm
    //#define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use =======================================

/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_rectile_native(
    magma_uplo_t uplo, magma_int_t n, magma_int_t recnb,    
    magmaDoubleComplex* dA,    magma_int_t ldda, magma_int_t gbstep, 
    magma_int_t *dinfo,  magma_int_t *info, magma_queue_t queue)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Constants */
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;

    *info = 0;
    // check arguments
    if ( uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if ( n == 0 ) {
        return *info;
    }

    if (n <= recnb) {
        // if (DEBUG == 1) printf("calling bottom panel recursive with n=%lld\n", (long long) n);
        //  panel factorization
        //magma_zpotf2_lpout(MagmaLower, n, dA, ldda, gbstep, dinfo, queue );
        magma_zpotf2_lpin(MagmaLower, n, dA, ldda, gbstep, dinfo, queue );
        //magma_zpotf2_native(MagmaLower, n, dA, ldda, gbstep, dinfo, queue );
    }
    else {
        // split A over two [A11 A12;  A21 A22]
        // panel on tile A11, 
        // trsm on A21, using A11
        // update on A22 then panel on A22.  
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;
        magma_int_t p1 = 0;
        magma_int_t p2 = n1;

        // panel on A11
        //if (DEBUG == 1) printf("calling recursive panel on A11=A(%d,%d) with n=%d recnb %d\n", p1, p1, n1, recnb);
        magma_zpotrf_rectile_native( uplo, n1, recnb, 
                                     dA(p1, p1), ldda, 
                                     gbstep+p1, dinfo, info, queue );

        // TRSM on A21
        //if (DEBUG == 1) printf("calling trsm on A21=A(%d,%d) using A11 == A(%d,%d) with m=%d k=%d\n",p2,p1,p1,p1,n2,n1);
       magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                    n2, n1,
                    c_one, dA(p1, p1), ldda,
                           dA(p2, p1), ldda, queue );

        // update A22
        //if (DEBUG == 1) printf("calling update A22=A(%d,%d) using A21 == A(%d,%d) with m=%d n=%d k=%d\n",p2,p2,p2,p1,n2,n2,n1);
       magma_zherk( MagmaLower, MagmaNoTrans, n2, n1,
                    d_neg_one, dA(p2, p1), ldda,
                    d_one,     dA(p2, p2), ldda, queue );

        // panel on A22
        //if (DEBUG == 1) printf("calling recursive panel on A22=A(%d,%d) with n=%d recnb %d\n",p2,p2,n2,recnb);
        magma_zpotrf_rectile_native( uplo, n2, recnb, 
                                     dA(p2, p2), ldda, 
                                     gbstep+n1, dinfo, info, queue );
    }

    return *info;
    #undef dA
}
