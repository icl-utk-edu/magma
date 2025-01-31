/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah

*/
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_z
#include "trsv_template_kernel_batched.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsv_small_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue )
{
    if     ( n <=  2 )
        trsv_small_batched<magmaDoubleComplex,  2>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <=  4 )
        trsv_small_batched<magmaDoubleComplex,  4>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <=  8 )
        trsv_small_batched<magmaDoubleComplex,  8>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <= 16 )
        trsv_small_batched<magmaDoubleComplex, 16>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else if( n <= 32 )
        trsv_small_batched<magmaDoubleComplex, 32>(uplo, transA, diag, n, dA_array, ldda, dx_array, incx, Ai, Aj, xi, batchCount, queue );
    else
        printf("error in function %s: nrowA must be less than 32\n", __func__);
}

