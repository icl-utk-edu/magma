/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#define PRECISION_z

#include "magma_internal.h"
#include "magma_templates.h"
#include "zunm2r_batched.cuh"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZUNM2R overwrites the general complex m-by-n matrix C with

       Q * C    if SIDE = MagmaLeft and TRANS = 'N', or
       Q**H* C  if SIDE = MagmaLeft and TRANS = 'C', or

       C * Q    if SIDE = MagmaRight and TRANS = 'N', or
       C * Q**H if SIDE = MagmaRight and TRANS = 'C',

   where Q is a complex unitary matrix defined as the product of k
   elementary reflectors

       Q = H(1) H(2) . . . H(k)

   as returned by ZGEQRF. Q is of order m if SIDE = MagmaLeft and of order n
   if SIDE = MagmaRight.

  - This is an internal batch implementation of ZUNM2R
  - The implementation uses register blocking
  - Only SIDE = MagmaLeft is currently supported

    @ingroup magma_unmqr
*******************************************************************************/
extern "C" magma_int_t
magma_zunm2r_reg_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t k,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m32 = magma_roundup(m, 32);

    if (side != MagmaLeft) {
        printf("%s currently supports side = MagmaLeft only\n", __func__);
        arginfo = -1;
    } else if (trans != MagmaNoTrans && trans != Magma_ConjTrans) {
        arginfo = -2;
    } else if (m < 0 || (m32 > 0 && m32 < nb)) {
        arginfo = -3;
    } else if (n < 0) {
        arginfo = -4;
    } else if (nb <= 0) {
        arginfo = -5;
    } else if (k < 0) {
        arginfo = -6;
    } else if (ldda < max(1,m)) {
        arginfo = -10;
    } else if (lddv < max(1,m)) {
        arginfo = -14;
    } else if (batchCount < 0) {
        arginfo = -18;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0 || batchCount == 0)
        return arginfo;

    magma_int_t istart, istop, istep;
    if( (side == MagmaLeft  && trans != MagmaNoTrans) ||
        (side == MagmaRight && trans == MagmaNoTrans) ) {
        istart = 0;
        istop  = k;
        istep  = nb;
    }
    else {
        istart = magma_roundup(k,nb); //max(nb, magma_roundup(k-1,nb));
        istop  = 0;
        istep  = -nb;
    }

    magma_int_t locinfo = 0;
    for(magma_int_t i = istart; (istep < 0 ? i >= istop : i < istop); i+=istep) {
        magma_int_t ib   = min(nb, k-i);
        magma_int_t mi32 = magma_roundup(m-i,32);

        if(m-i <= 0 || ib <= 0) continue;

        if( mi32 >= 768 ) {
            locinfo = magma_zunm2r_reg_tall_batched(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue);
        }
        else if (mi32 >= 384 ) {
            locinfo = magma_zunm2r_reg_medium_batched(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue);
        }
        else {
            switch( magma_ceildiv(m-i,32) ) {
                case  1: locinfo = magma_zunm2r_reg_NB_batched< 32>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  2: locinfo = magma_zunm2r_reg_NB_batched< 64>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  3: locinfo = magma_zunm2r_reg_NB_batched< 96>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  4: locinfo = magma_zunm2r_reg_NB_batched<128>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  5: locinfo = magma_zunm2r_reg_NB_batched<160>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  6: locinfo = magma_zunm2r_reg_NB_batched<192>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  7: locinfo = magma_zunm2r_reg_NB_batched<224>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  8: locinfo = magma_zunm2r_reg_NB_batched<256>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case  9: locinfo = magma_zunm2r_reg_NB_batched<288>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case 10: locinfo = magma_zunm2r_reg_NB_batched<320>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                case 11: locinfo = magma_zunm2r_reg_NB_batched<352>(side, trans, m-i, n, nb, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, check_launch_only, batchCount, queue ); break;
                default: locinfo = -3; // unsupported m
            }
        }

        // accumulate locinfo (useful in checking launch only)
        // if arginfo = 0 at the end, it means that check launch was successful at every iteration
        arginfo += locinfo;
    }

    return arginfo;
}
