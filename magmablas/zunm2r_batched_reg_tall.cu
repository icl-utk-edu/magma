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

#include <cuda.h>    // for CUDA_VERSION
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
magma_zunm2r_reg_tall_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
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
    } else if (ib < 0 || ib > nb) {
        printf("Error in %s: ib %d must be <= nb %d\n", __func__, ib, nb);
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
    if (m == 0 || n == 0 || ib == 0 || batchCount == 0)
        return arginfo;

    switch( magma_ceildiv(m,32) ) {
        case 24: arginfo = magma_zunm2r_reg_NB_batched< 768>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 25: arginfo = magma_zunm2r_reg_NB_batched< 800>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 26: arginfo = magma_zunm2r_reg_NB_batched< 832>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 27: arginfo = magma_zunm2r_reg_NB_batched< 864>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 28: arginfo = magma_zunm2r_reg_NB_batched< 896>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 29: arginfo = magma_zunm2r_reg_NB_batched< 928>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 30: arginfo = magma_zunm2r_reg_NB_batched< 960>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 31: arginfo = magma_zunm2r_reg_NB_batched< 992>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 32: arginfo = magma_zunm2r_reg_NB_batched<1024>(side, trans, m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        default: arginfo = -3; // unsupported m
    }
    return arginfo;
}
