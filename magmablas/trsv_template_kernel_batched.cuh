/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSV_TEMPLATE_KERNEL_BATCHED_CUH
#define TRSV_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB)
void trsv_template_batched_NL_kernel(
        magma_diag_t diag, int n,
        T** Aarray,  int ldda,
        T** xarray,  int incx,
        int roffA, int coffA, int offx)
{
    const int batchid = blockIdx.z;


    trsv_template_device_NL<T, NB>(
        diag, n,
        Aarray[batchid] + coffA * ldda + roffA, ldda,
        xarray[batchid] +  offx * incx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB)
void trsv_template_batched_NU_kernel(
        magma_diag_t diag, int n,
        T** Aarray,  int ldda,
        T** xarray,  int incx,
        int roffA, int coffA, int offx)
{
    int batchid = blockIdx.z;

    trsv_template_device_NU<T, NB>(
        diag, n,
        Aarray[batchid] + coffA * ldda + roffA, ldda,
        xarray[batchid] +  offx * incx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB)
void trsv_template_batched_TL_kernel(
        magma_diag_t diag, int n,
        T** Aarray, int ldda,
        T** xarray, int incx,
        int roffA, int coffA, int offx)
{
    int batchid = blockIdx.z;

    trsv_template_device_TL<T, NB, CONJA>(
        diag, n,
        Aarray[batchid] + coffA * ldda + roffA, ldda,
        xarray[batchid] +  offx * incx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB)
void trsv_template_batched_TU_kernel(
        magma_diag_t diag, int n,
        T** Aarray, int ldda,
        T** xarray, int incx,
        int roffA, int coffA, int offx)
{
    int batchid = blockIdx.z;

    trsv_template_device_TU<T, NB, CONJA>(
        diag, n,
        Aarray[batchid] + coffA * ldda + roffA, ldda,
        xarray[batchid] +  offx * incx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB>
void trsv_small_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        T **dA_array, magma_int_t ldda,
        T **dx_array, magma_int_t incx,
        magma_int_t roffA, magma_int_t coffA, magma_int_t offx,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t shape = 0;
    if      (uplo == MagmaLower  && transA == MagmaNoTrans   ) { shape = 0; } // NL
    else if (uplo == MagmaLower  && transA == MagmaTrans     ) { shape = 1; } // TL
    else if (uplo == MagmaLower  && transA == MagmaConjTrans ) { shape = 2; } // CL
    else if (uplo == MagmaUpper  && transA == MagmaNoTrans   ) { shape = 3; } // NU
    else if (uplo == MagmaUpper  && transA == MagmaTrans     ) { shape = 4; } // TU
    else if (uplo == MagmaUpper  && transA == MagmaConjTrans ) { shape = 5; } // CU


    dim3 threads(NB, 1, 1);
    magma_int_t max_batchCount = queue->get_maxBatch();

    switch(shape) {
        case 0: // NL
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_NL_kernel<T, NB>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        case 1: // TL
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_TL_kernel<T, NB, 0>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        case 2: // CL
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_TL_kernel<T, NB, 1>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        case 3: // NU
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_NU_kernel<T, NB>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        case 4: // TU
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_TU_kernel<T, NB, 0>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        case 5: // CU
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( 1, 1, ibatch );

                trsv_template_batched_TU_kernel<T, NB, 1>
                <<< grid, threads, 0, queue->cuda_stream() >>>
                (diag, n, dA_array+i, ldda, dx_array+i, incx, roffA, coffA, offx);
            }
            break;
        default:; // propose something
    }
}


#endif //TRSV_TEMPLATE_KERNEL_BATCHED_CUH
