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

#define RFT_MAG_GEM


/******************************************************************************/
extern "C" void
magma_zlarft_sm32x32_batched(magma_int_t n, magma_int_t k, 
                    magmaDoubleComplex **v_array, magma_int_t ldv,
                    magmaDoubleComplex **tau_array, 
                    magmaDoubleComplex **T_array, magma_int_t ldt, 
                    magma_int_t batchCount, magma_queue_t queue)
{
    if ( k <= 0) return;

     //==================================
     //          GEMV
     //==================================
#define USE_GEMV2
#define use_gemm_larft_sm32

    #if defined(use_gemm_larft_sm32)
    magma_zgemm_batched( MagmaConjTrans, MagmaNoTrans, 
                         k, k, n, 
                         MAGMA_Z_ONE, v_array, ldv, 
                         v_array, ldv, 
                         MAGMA_Z_ZERO, T_array, ldt, 
                         batchCount, queue );
    magmablas_zlaset_batched( MagmaLower, k, k, 
            MAGMA_Z_ZERO, MAGMA_Z_ZERO, 
            T_array, ldt, batchCount, queue );
    #else
    #if 1
    for (magma_int_t i=0; i < k; i++)
    {
        //W(1:i-1) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
        //T( i, i ) = tau( i ) 
        //custom implementation.
        #ifdef USE_GEMV2
        magmablas_zlarft_gemvrowwise_batched( n-i, i, 
                            tau_array,
                            v_array, ldv, 
                            T_array, ldt,
                            batchCount, queue);
                            
        #else       
        magmablas_zlarft_gemvcolwise_batched( n-i, i, v_array, ldv, T_array, ldt, tau_array, batchCount, queue);
        #endif
    }
    #else
        //seems to be very slow when k=32 while the one by one loop above is faster
        zlarft_gemv_loop_inside_kernel_batched(n, k, tau_array, v_array, ldv, T_array, ldt, batchCount, queue); 
    #endif
    #endif
    //==================================
    //          TRMV
    //==================================
    //T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    magmablas_zlarft_ztrmv_sm32x32_batched(k, k, tau_array, T_array, ldt, T_array, ldt, batchCount, queue);
}


/******************************************************************************/
extern "C" magma_int_t
magma_zlarft_batched(magma_int_t n, magma_int_t k, magma_int_t stair_T, 
                magmaDoubleComplex **v_array, magma_int_t ldv,
                magmaDoubleComplex **tau_array, magmaDoubleComplex **T_array, magma_int_t ldt, 
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

    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dTstep_array  = NULL;

    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dTstep_array,  batchCount * sizeof(*dTstep_array));

    //magmaDoubleComplex *Tstep =  k > nb ? work : T;
    if (k > nb)
    {
        magma_zdisplace_pointers(dTstep_array, work_array, lwork, 0, 0, batchCount, queue);
    }
    else
    {
        magma_zdisplace_pointers(dTstep_array, T_array, ldt, 0, 0, batchCount, queue);
    }

    //magma_int_t ldtstep = k > nb ? k : ldt;
    magma_int_t ldtstep = ldt; //a enlever
    // stair_T = 0 meaning all T
    // stair_T > 0 meaning the triangular portion of T has been computed. 
    //                    the value of stair_T is the nb of these triangulars
   

    //GEMV compute the whole triangular upper portion of T (phase 1)
    // TODO addcublas to check perf

    magma_zgemm_batched( MagmaConjTrans, MagmaNoTrans, 
                         k, k, n, 
                         c_one,  v_array, ldv, 
                                 v_array, ldv, 
                         c_zero, dTstep_array, ldtstep, 
                         batchCount, queue );

    magmablas_zlaset_batched( MagmaLower, k, k, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dTstep_array, ldtstep, batchCount, queue );
    // no need for it as T is expected to be lower zero
    //if (k > nb) magmablas_zlaset_batched( MagmaLower, k, k, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dTstep_array, ldtstep, batchCount, queue );
    

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

    dim3 grid(1, 1, batchCount);

    for (j=0; j < k; j += nb)
    {
        prev_n =  j;
        mycol  =  min(nb, k-j);
        // note that myrow = prev_n + mycol;
        if (prev_n > 0 && mycol > 0) {
            if (DEBUG == 3) {
                printf("doing gemm on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) prev_n, (long long) mycol, (long long) 0, (long long) j );
            }

            magma_zdisplace_pointers(dW1_displ, dTstep_array, ldtstep, 0, j, batchCount, queue);
            magma_zdisplace_pointers(dW2_displ, T_array,     ldt, 0, j, batchCount, queue);
            magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, 
                                 prev_n, mycol, prev_n, 
                                 c_one,  T_array, ldt, 
                                         dW1_displ, ldtstep, 
                                 c_zero, dW2_displ, ldt, 
                                 batchCount, queue );

            // update my rectangular portion (prev_n,mycol) using sequence of gemv 
            magma_zdisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount, queue);
            magma_zdisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount, queue);

            for (i=0; i < prev_n; i += nb)
            {
                rows = min(nb,prev_n-i);
                if (DEBUG == 3) {
                    printf("        doing recztrmv on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                            (long long) rows, (long long) mycol, (long long) i, (long long) j );
                }

                if (rows > 0 && mycol > 0)
                {
                    magma_zdisplace_pointers(dW2_displ, T_array,     ldt, i, j, batchCount, queue);
                    magmablas_zlarft_recztrmv_sm32x32_batched(rows, mycol, dW3_displ, dW2_displ, ldt, dW1_displ, ldtstep, batchCount, queue);
                }
            }
        }

        // the upper rectangular protion is updated, now if needed update the triangular portion
        if (stair_T == 0) {
            if (DEBUG == 3) {
                printf("doing ztrmv on the triangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) mycol, (long long) mycol, (long long) j, (long long) j );
            }

            if (mycol > 0)
            {
                magma_zdisplace_pointers(dW1_displ, dTstep_array, ldtstep, j, j, batchCount, queue);
                magma_zdisplace_pointers(dW3_displ, tau_array,  1, j, 0, batchCount, queue);
                magma_zdisplace_pointers(dW2_displ, T_array,     ldt, j, j, batchCount, queue);
                magmablas_zlarft_ztrmv_sm32x32_batched(mycol, mycol, dW3_displ, dW1_displ, ldtstep, dW2_displ, ldt, batchCount, queue);
            }
        }
    }// end of j

    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dTstep_array);

    return 0;
}
