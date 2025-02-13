/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSV_TEMPLATE_KERNEL_CUH
#define TRSV_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB)
void trsv_template_NL_kernel(
        magma_diag_t diag, int n,
        T* dA,  int ldda,
        T* dx,  int incx)
{
    trsv_template_device_NL<T, NB>
    (diag, n, dA, ldda, dx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB)
void trsv_template_NU_kernel(
        magma_diag_t diag, int n,
        T* dA,  int ldda,
        T* dx,  int incx )
{
    trsv_template_device_NU<T, NB>
    (diag, n, dA, ldda, dx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB)
void trsv_template_TL_kernel(
        magma_diag_t diag, int n,
        T* dA, int ldda,
        T* dx, int incx )
{
    trsv_template_device_TL<T, NB, CONJA>
    (diag, n, dA, ldda, dx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB)
void trsv_template_TU_kernel(
        magma_diag_t diag, int n,
        T* dA, int ldda,
        T* dx, int incx)
{
    trsv_template_device_TU<T, NB, CONJA>
    (diag, n, dA, ldda, dx, incx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrapper
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB>
void trsv_small(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        T *dA, magma_int_t ldda,
        T *dx, magma_int_t incx, magma_queue_t queue )
{
    magma_int_t shape = 0;
    if      (uplo == MagmaLower  && transA == MagmaNoTrans   ) { shape = 0; } // NL
    else if (uplo == MagmaLower  && transA == MagmaTrans     ) { shape = 1; } // TL
    else if (uplo == MagmaLower  && transA == MagmaConjTrans ) { shape = 2; } // CL
    else if (uplo == MagmaUpper  && transA == MagmaNoTrans   ) { shape = 3; } // NU
    else if (uplo == MagmaUpper  && transA == MagmaTrans     ) { shape = 4; } // TU
    else if (uplo == MagmaUpper  && transA == MagmaConjTrans ) { shape = 5; } // CU

    dim3 threads(NB, 1, 1);
    dim3 grid( 1, 1, 1 );

    switch(shape) {
        case 0: // NL
            trsv_template_NL_kernel<T, NB><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        case 1: // TL
            trsv_template_TL_kernel<T, NB, 0> <<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        case 2: // CL
            trsv_template_TL_kernel<T, NB, 1><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        case 3: // NU
            trsv_template_NU_kernel<T, NB><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        case 4: // TU
            trsv_template_TU_kernel<T, NB, 0><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        case 5: // CU
            trsv_template_TU_kernel<T, NB, 1><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, n, dA, ldda, dx, incx);
            break;
        default:; // propose something
    }
}

#endif //TRSV_TEMPLATE_KERNEL_CUH
