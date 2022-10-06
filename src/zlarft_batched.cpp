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

#define  max_shared_bsiz 32

/******************************************************************************/
extern "C" void
magma_zlarft_sm32x32_batched(
        magma_int_t n, magma_int_t k,
        magmaDoubleComplex **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        magmaDoubleComplex **tau_array, magma_int_t taui,
        magmaDoubleComplex **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        magma_int_t batchCount, magma_queue_t queue)
{
    if ( k <= 0) return;

    magma_zgemm_batched_core(
            MagmaConjTrans, MagmaNoTrans,
            k, k, n,
            MAGMA_Z_ONE,
            v_array, vi, vj, ldv,
            v_array, vi, vj, ldv,
            MAGMA_Z_ZERO,
            T_array, Ti, Tj, ldt,
            batchCount, queue );

    magmablas_zlaset_internal_batched(
            MagmaLower, k, k,
            MAGMA_Z_ZERO, MAGMA_Z_ZERO,
            T_array, Ti, Tj, ldt,
            batchCount, queue );

    // TRMV
    // T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    magmablas_zlarft_ztrmv_sm32x32_batched(
        k, k, tau_array, taui,
        T_array, Ti, Tj, ldt,
        T_array, Ti, Tj, ldt,
        batchCount, queue);
}


/******************************************************************************/
extern "C" magma_int_t
magma_zlarft_internal_batched(
        magma_int_t n, magma_int_t k, magma_int_t stair_T,
        magmaDoubleComplex **v_array,   magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        magmaDoubleComplex **tau_array, magma_int_t taui,
        magmaDoubleComplex **T_array,   magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        magmaDoubleComplex **work_array, magma_int_t lwork,
        magma_int_t batchCount, magma_queue_t queue)
{
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;

    if ( k <= 0) return 0;
    if ( stair_T > 0 && k <= stair_T) return 0;

    magma_int_t maxnb = max_shared_bsiz;

    magma_int_t info = 0;
    if (stair_T > 0 && stair_T > maxnb) {
        info = -3;
    }
    else if (lwork < k*ldt) {
        info = -10;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_int_t DEBUG=0;
    magma_int_t nb = stair_T == 0 ? min(k,maxnb) : stair_T;

    magma_int_t i, j, prev_n, mycol, rows;

    magmaDoubleComplex **dTstep_array  = NULL;

    magma_int_t Tstepi, Tstepj;
    if (k > nb) {
        dTstep_array = work_array;
        Tstepi = 0;
        Tstepj = 0;
    }
    else {
        dTstep_array = T_array;
        Tstepi = Ti;
        Tstepj = Tj;
    }

    magma_int_t ldtstep = ldt; //a enlever
    // stair_T = 0 meaning all T
    // stair_T > 0 meaning the triangular portion of T has been computed.
    //                    the value of stair_T is the nb of these triangulars

    magma_zgemm_batched_core( MagmaConjTrans, MagmaNoTrans,
                              k, k, n,
                              c_one,  v_array, vi, vj, ldv,
                                      v_array, vi, vj, ldv,
                              c_zero, dTstep_array, Tstepi, Tstepj, ldtstep,
                              batchCount, queue );

    magmablas_zlaset_internal_batched(
            MagmaLower, k, k, MAGMA_Z_ZERO, MAGMA_Z_ZERO,
            dTstep_array, 0, 0, ldtstep, batchCount, queue );

    //TRMV
    //T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    // TRMV is split over block of column of size nb
    // the update should be done from top to bottom so:
    // 1- a gemm using the previous computed columns
    //    of T to update rectangular upper protion above
    //    the triangle of my columns
    // 2- the columns need to be updated by a serial
    //    loop over of gemv over itself. since we limit the
    //    shared memory to nb, this nb column
    //    are split vertically by chunk of nb rows

    // This causes errors when building with dpcpp and doesn't appear to be used anywhere else in the file?
    // Can we remove?  - N Beams 
//    dim3 grid(1, 1, batchCount);

    for (j=0; j < k; j += nb) {
        prev_n =  j;
        mycol  =  min(nb, k-j);
        // note that myrow = prev_n + mycol;
        if (prev_n > 0 && mycol > 0) {
            if (DEBUG == 3) {
                printf("doing gemm on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) prev_n, (long long) mycol, (long long) 0, (long long) j );
            }

            magma_zgemm_batched_core( MagmaNoTrans, MagmaNoTrans,
                                 prev_n, mycol, prev_n,
                                 c_one,  T_array,            Ti,       Tj, ldt,
                                         dTstep_array, Tstepi+0, Tstepj+j, ldtstep,
                                 c_zero, T_array,            Ti,     Tj+j, ldt,
                                 batchCount, queue );

            // update my rectangular portion (prev_n,mycol) using sequence of gemv
            for (i=0; i < prev_n; i += nb) {
                rows = min(nb,prev_n-i);
                if (DEBUG == 3) {
                    printf("        doing recztrmv on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                            (long long) rows, (long long) mycol, (long long) i, (long long) j );
                }

                if (rows > 0 && mycol > 0) {
                    magmablas_zlarft_recztrmv_sm32x32_batched(
                            rows, mycol,
                            tau_array,      taui+j,
                            T_array,          Ti+i,     Tj+j, ldt,
                            dTstep_array, Tstepi+j, Tstepj+j, ldtstep,
                            batchCount, queue);
                }
            }
        }

        // the upper rectangular protion is updated, now if needed update the triangular portion
        if (stair_T == 0) {
            if (DEBUG == 3) {
                printf("doing ztrmv on the triangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) mycol, (long long) mycol, (long long) j, (long long) j );
            }

            if (mycol > 0) {
                magmablas_zlarft_ztrmv_sm32x32_batched(
                        mycol, mycol,
                        tau_array,      taui+j,
                        dTstep_array, Tstepi+j, Tstepj+j, ldtstep,
                        T_array,          Ti+j,     Tj+j, ldt,
                        batchCount, queue);
            }
        }
    } // end of j

    return 0;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zlarft_batched(magma_int_t n, magma_int_t k, magma_int_t stair_T,
                magmaDoubleComplex **v_array, magma_int_t ldv,
                magmaDoubleComplex **tau_array, magmaDoubleComplex **T_array, magma_int_t ldt,
                magmaDoubleComplex **work_array, magma_int_t lwork,
                magma_int_t batchCount, magma_queue_t queue)
{
    magma_zlarft_internal_batched(
        n, k, stair_T,
        v_array,   0, 0, ldv,
        tau_array, 0,
        T_array,   0, 0, ldt,
        work_array, lwork,
        batchCount, queue);

    return 0;
}
