/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "magma_internal.h"

/******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_panel_fused_update_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex** tau_array, magma_int_t taui,
        magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        magma_int_t *info_array, magma_int_t separate_R_V,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    magma_int_t minmn = min(m,n);
    if( m < nb ) return -1;

#ifndef MAGMA_HAVE_SYCL
    // check for square sizes <= 32
    if( m == n && m <= 32 ){
        info = magma_zgeqrf_batched_smallsq(
                    m, dA_array, Ai, Aj, ldda,
                    tau_array, taui,
                    info_array, batchCount, queue );
    }
    else{
#endif
        for(magma_int_t j = 0; j < minmn; j+=nb) {
            magma_int_t ib = min(minmn-j,nb);

            // if there is no trailing matrix, then we should not
            // check the update kernels
            magma_int_t update_needed = (n-(j+ib)) > 0 ? 1 : 0;

            // check launch first of both register and sm versions
            // checks for shmem requirement and #threads
            // when check_launch is 1, use batchCount = 1
            magma_int_t nthreads        = magma_get_zgeqr2_fused_sm_batched_nthreads(m-j, ib);
            magma_int_t zgeqr2_reg_info = magma_zgeqr2_fused_reg_batched(m-j, ib, dA_array, Ai+j, Aj+j, ldda, tau_array, taui+j, info_array, 1, 1, queue );
            magma_int_t zgeqr2_sm_info  = magma_zgeqr2_fused_sm_batched(m-j, ib, dA_array, Ai+j, Aj+j, ldda, tau_array, taui+j, info_array, nthreads, 1, 1, queue );
            magma_int_t zlarf_reg_info  = magma_zlarf_fused_reg_batched(m-j, n-(j+ib), nb, ib, dA_array, Ai+j, Aj+j+ib, ldda, dA_array, Ai+j, Aj+j, ldda, tau_array, taui+j, 1, 1, queue );
            magma_int_t zlarf_sm_info   = magma_zlarf_fused_sm_batched(m-j, n-(j+ib), nb, ib, dA_array, Ai+j, Aj+j+ib, ldda, dA_array, Ai+j, Aj+j, ldda, tau_array, taui+j, nthreads, 1, 1, queue );

            magma_int_t zgeqr2_reg_ok = (zgeqr2_reg_info == 0) ? 1 : 0;
            magma_int_t zgeqr2_sm_ok  = (zgeqr2_sm_info  == 0) ? 1 : 0;
            magma_int_t zlarf_reg_ok  = (zlarf_reg_info  == 0) ? 1 : 0;
            magma_int_t zlarf_sm_ok   = (zlarf_sm_info   == 0) ? 1 : 0;

            if( (zgeqr2_reg_ok == 0 && zgeqr2_sm_ok == 0) ||
                (update_needed == 1 && zlarf_reg_ok  == 0 && zlarf_sm_ok  == 0) ) {
                return -1;
            }

            // panel
            if( zgeqr2_reg_ok == 1 ) {
                info = magma_zgeqr2_fused_reg_batched(
                            m-j, ib,
                            dA_array, Ai+j, Aj+j, ldda,
                            tau_array, taui+j,
                            info_array, 0, batchCount, queue );
            }
            else {
                info = magma_zgeqr2_fused_sm_batched(
                            m-j, ib,
                            dA_array, Ai+j, Aj+j, ldda,
                            tau_array, taui+j,
                            info_array, nthreads, 0, batchCount, queue );
            }

            // update -- try reg first
            info = magma_zlarf_fused_reg_batched(
                            m-j, n-(j+ib), nb, ib,
                            dA_array, Ai+j, Aj+j+ib, ldda,
                            dA_array, Ai+j, Aj+j,    ldda,
                            tau_array, taui+j, 0, batchCount, queue );


            // if update on reg failed to launch, use sm version
            if( info != 0 ) {
                info = magma_zlarf_fused_sm_batched(
                            m-j, n-(j+ib), nb, ib,
                            dA_array, Ai+j, Aj+j+ib, ldda,
                            dA_array, Ai+j, Aj+j,    ldda,
                            tau_array, taui+j, nthreads, 0,
                            batchCount, queue );
            }
        }
#ifndef MAGMA_HAVE_SYCL
    }
#endif
    if( info == 0 && separate_R_V == 1 ) {
        // copy to dR
        magmablas_zlacpy_internal_batched(
                MagmaUpper, minmn, minmn,
                dA_array, Ai+0, Aj+0, ldda,
                dR_array, Ri+0, Rj+0, lddr,
                batchCount, queue );


        // set the upper nxn portion of dA to 1/0s
        magmablas_zlaset_internal_batched(
                MagmaUpper, minmn, minmn,
                MAGMA_Z_ZERO, MAGMA_Z_ONE,
                dA_array, Ai+0, Aj+0, ldda,
                batchCount, queue );

    }

    return info;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_panel_internal_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex** tau_array, magma_int_t taui,
        magmaDoubleComplex** dT_array, magma_int_t Ti, magma_int_t Tj, magma_int_t lddt,
        magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr,
        magmaDoubleComplex** dwork_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0, j, jb;
    magma_int_t ldw = nb;
    magma_int_t minmn = min(m,n);

    // try optimizations for small sizes first
    // try different nb values
    for(int inb = 16; inb >= 2; inb/=2) {
        info = magma_zgeqrf_panel_fused_update_batched(
                    m, n, inb,
                    dA_array, Ai, Aj, ldda,
                    tau_array, taui,
                    dR_array, Ri, Rj, lddr,
                    info_array, 1,
                    batchCount, queue);

        if( info == 0) return MAGMA_SUCCESS;
    }

    // otherwise, fall back to lapack-style panel
    for( j=0; j < minmn; j += nb) {
        jb = min(nb, minmn-j);

        //sub-panel factorization
        magma_zgeqr2_batched(
                m-j, jb,
                dA_array, Ai+j, Aj+j, ldda,
                tau_array, taui+j,
                info_array,
                batchCount,
                queue);

        // copy th whole rectangular n,jb from of dA to dR (it's lower portion (which is V's)
        // will be set to zero if needed at the end)
        magmablas_zlacpy_internal_batched(
                MagmaFull, minmn, jb,
                dA_array, Ai+0, Aj+j, ldda,
                dR_array, Ri+0, Rj+j, lddr,
                batchCount, queue );

        // set the upper jbxjb portion of V dA(j,j) to 1/0s (note that the rectangular on
        // the top of this triangular of V still non zero but has been copied to dR).
        magmablas_zlaset_internal_batched(
                MagmaUpper, jb, jb,
                MAGMA_Z_ZERO, MAGMA_Z_ONE,
                dA_array, Ai+j, Aj+j, ldda,
                batchCount, queue );

        if ( (n-j-jb) > 0) {
            //update the trailing matrix inside the panel
            magma_zlarft_sm32x32_batched(m-j, jb,
                    dA_array,  Ai+j, Aj+j, ldda,
                    tau_array, taui+j,
                    dT_array,  Ti+0, Tj+0, lddt,
                    batchCount, queue);

            magma_zlarfb_gemm_internal_batched(
                    MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                    m-j, n-j-jb, jb,
                    (const magmaDoubleComplex**)dA_array, Ai+j, Aj+j, ldda,
                    (const magmaDoubleComplex**)dT_array, Ti+0, Tj+0, lddt,
                    dA_array, Ai+j, Aj+j+jb, ldda,
                    dwork_array,              ldw,
                    dwork_array + batchCount, ldw,
                    batchCount, queue );
        }
    }

    // copy the remaining portion of dR from dA in case m < n
    if ( m < n ) {
        magmablas_zlacpy_internal_batched(
            MagmaFull, minmn, n-minmn,
            dA_array, Ai+0, Aj+minmn, ldda,
            dR_array, Ri+0, Rj+minmn, lddr,
            batchCount, queue );
    }

    // to be consistent set the whole upper nbxnb of V to 0/1s,
    // in this case no need to set it inside zgeqrf_batched
    magmablas_zlaset_internal_batched(
            MagmaUpper, minmn, n,
            MAGMA_Z_ZERO, MAGMA_Z_ONE,
            dA_array, Ai+0, Aj+0, ldda,
            batchCount, queue );

    return MAGMA_SUCCESS;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_panel_batched(
        magma_int_t m, magma_int_t n, magma_int_t nb,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** tau_array,
        magmaDoubleComplex** dT_array, magma_int_t lddt,
        magmaDoubleComplex** dR_array, magma_int_t lddr,
        magmaDoubleComplex** dwork_array,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_zgeqrf_panel_internal_batched(
        m, n, nb,
        dA_array, 0, 0, ldda,
        tau_array, 0,
        dT_array,  0,  0,  lddt,
        dR_array,  0,  0, lddr,
        dwork_array,
        info_array, batchCount, queue);

    return MAGMA_SUCCESS;
}
