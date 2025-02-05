/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define PRECISION_z

#include "gemm_template_device_defs.cuh"
#include "trsv_template_device.cuh"
#include "trsv_template_kernel_batched.cuh"

#define NB 256  //NB is the 1st level blocking in recursive blocking, BLOCK_SIZE is the 2ed level, NB=256, BLOCK_SIZE=64 is optimal for batched

#define NUM_THREADS 128 //64 //128

#define BLOCK_SIZE_N 128
#define DIM_X_N 128
#define DIM_Y_N 1

#define BLOCK_SIZE_T 32
#define DIM_X_T 16
#define DIM_Y_T 8

#include "ztrsv_template_device.cuh"

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column


/******************************************************************************/
static void
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

/******************************************************************************/
static magma_int_t magma_get_ztrsv_batched_nb(magma_int_t n)
{
    if      ( n > 2048 ) return 2048;
    else if ( n > 1024 ) return 1024;
    else if ( n >  512 ) return 512;
    else if ( n >  256 ) return 256;
    else if ( n >  128 ) return 128;
    else if ( n >   64 ) return  64;
    else if ( n >   32 ) return  32;
    else if ( n >   16 ) return  16;
    else if ( n >    8 ) return   8;
    else if ( n >    4 ) return   4;
    else if ( n >    2 ) return   2;
    else return 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magmablas_ztrsv_recursive_batched(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dx_array, magma_int_t xi, magma_int_t incx,
        magma_int_t batchCount, magma_queue_t queue )
{
#define dA_array(i,j) dA_array, i, j
#define dx_array(i)   dx_array, i

    const magmaDoubleComplex c_one    = MAGMA_Z_ONE;
    const magmaDoubleComplex c_negone = MAGMA_Z_NEG_ONE;

    magma_int_t shape = -1;
    if      (transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // NL
    else if (transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // NU
    else if (transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // TL | CL
    else if (transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // TU | CU

    // stopping condition
    if(n <= 32){
        magmablas_ztrsv_small_batched(
            uplo, transA, diag, n,
            dA_array(Ai, Aj), ldda,
            dx_array(xi), incx, batchCount, queue );
        return;
    }

    const int n2 = magma_get_ztrsv_batched_nb(n);
    const int n1 = n - n2;


//#define DBG
#ifdef DBG
    printf("n1 = %d, n2 = %d\n", n1, n2);
    magmaDoubleComplex* tmpx = NULL;
    magma_getvector(1, sizeof(magmaDoubleComplex*), dx_array, 1, &tmpx, 1, queue);
#endif

    switch(shape) {
        case 0: // Nl
        {

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );

            #ifdef DBG
            magma_zprint_gpu(n, 1, tmpx, n, queue);
            #endif

            magmablas_zgemv_batched_core(
                transA, n2, n1,
                c_negone, dA_array(Ai+n1, Aj), ldda,
                          dx_array(xi       ), incx,
                c_one,    dx_array(xi+n1    ), incx,
                batchCount, queue );
            #ifdef DBG
            magma_zprint_gpu(n, 1, tmpx, n, queue);
            #endif

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );
            #ifdef DBG
            magma_zprint_gpu(n, 1, tmpx, n, queue);
            #endif
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 1: // NU
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n1, n2,
                c_negone, dA_array(Ai, Aj+n1), ldda,
                          dx_array(xi+n1    ), incx,
                c_one,    dx_array(xi       ), incx,
                batchCount, queue );

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 2: // TL || CL
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n2, n1,
                c_negone, dA_array(Ai+n1, Aj), ldda,
                          dx_array(xi+n1    ), incx,
                c_one,    dx_array(xi       ), incx,
                batchCount, queue );


            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        case 3: // TU | lCU
        {
            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n1,
                dA_array(Ai, Aj), ldda,
                dx_array(xi    ), incx,
                batchCount, queue );

            magmablas_zgemv_batched_core(
                transA, n1, n2,
                c_negone, dA_array(Ai, Aj+n1), ldda,
                          dx_array(xi       ), incx,
                c_one,    dx_array(xi+n1    ), incx,
                batchCount, queue );

            magmablas_ztrsv_recursive_batched(
                uplo, transA, diag, n2,
                dA_array(Ai+n1, Aj+n1), ldda,
                dx_array(xi+n1       ), incx,
                batchCount, queue );
        }
        break;
        ////////////////////////////////////////////////////////////////////////
        default:; // propose something
    }
#undef dA_array
#undef dx_array
}


/******************************************************************************/
template< const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
__global__ void
ztrsv_notrans_kernel_outplace_batched(
    int n,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **b_array, int incb,
    magmaDoubleComplex **x_array)
{
    int batchid = blockIdx.z;

    ztrsv_notrans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>(n, A_array[batchid], lda, b_array[batchid], incb, x_array[batchid]);
}


/******************************************************************************/
template<const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
__global__ void
ztrsv_trans_kernel_outplace_batched(
    int n,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **b_array, int incb,
    magmaDoubleComplex **x_array)
{
    int batchid = blockIdx.z;
    ztrsv_trans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>(n, A_array[batchid], lda, b_array[batchid], incb, x_array[batchid]);
}

/******************************************************************************/
// notrans driver
template< const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
void
ztrsv_notrans_outplace_batched(
    int n,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **b_array, int incb,
    magmaDoubleComplex **x_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    size_t shmem = n * sizeof(magmaDoubleComplex);
    dim3 threads( NUM_THREADS, 1, 1 );

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( 1, 1, ibatch );

        ztrsv_notrans_kernel_outplace_batched<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>
        <<<grid, threads, shmem, queue->cuda_stream()>>>
        (n, A_array+i, lda, b_array+i, incb, x_array+i);
    }
}

/******************************************************************************/
// trans driver
template<const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
void
ztrsv_trans_outplace_batched(
    int n,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **b_array, int incb,
    magmaDoubleComplex **x_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    size_t shmem = n * sizeof(magmaDoubleComplex);
    dim3 threads( NUM_THREADS, 1, 1 );

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( 1, 1, ibatch );

        ztrsv_trans_kernel_outplace_batched<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>
        <<<grid, threads, shmem, queue->cuda_stream()>>>
        (n, A_array+i, lda, b_array+i, incb, x_array+i);
    }
}


/******************************************************************************/
extern "C" void
magmablas_ztrsv_outofplace_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex ** A_array, magma_int_t lda,
    magmaDoubleComplex **b_array, magma_int_t incb,
    magmaDoubleComplex **x_array,
    magma_int_t batchCount, magma_queue_t queue,
    magma_int_t flag)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }


    // quick return if possible.
    if (n == 0)
        return;

    if (trans == MagmaNoTrans) {
        if (uplo == MagmaUpper) {
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaUpper, MagmaNoTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaUpper, MagmaNoTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaUpper, MagmaNoTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaUpper, MagmaNoTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
        else { //Lower
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaLower, MagmaNoTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaLower, MagmaNoTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaLower, MagmaNoTrans, MagmaUnit>
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_notrans_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaLower, MagmaNoTrans, MagmaUnit>
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
    }
    else if (trans == MagmaTrans) {
        if (uplo == MagmaUpper) {
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
        else {
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
    }
    else if (trans == MagmaConjTrans) {
        if (uplo == MagmaUpper) {
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaConjTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaConjTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaConjTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaConjTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
        else {
            if (diag == MagmaNonUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaConjTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaConjTrans, MagmaNonUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0) {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaConjTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
                else {
                    ztrsv_trans_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaConjTrans, MagmaUnit >
                    (n, A_array, lda, b_array, incb, x_array, batchCount, queue);
                }
            }
        }
    }
}


/******************************************************************************/
extern "C" void
magmablas_ztrsv_recursive_outofplace_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex **A_array, magma_int_t lda,
    magmaDoubleComplex **b_array, magma_int_t incb,
    magmaDoubleComplex **x_array,
    magma_int_t batchCount, magma_queue_t queue)
{
#define dW0_displ_ (const magmaDoubleComplex**) dW0_displ
#define dW1_displ_ (const magmaDoubleComplex**) dW1_displ

    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;


    //Init x_array with zero
    //magmablas_zlaset_batched(MagmaFull, n, incb, MAGMA_Z_ZERO, MAGMA_Z_ZERO, x_array, n, batchCount, queue);

    //memory allocation takes 0.32ms

    magmaDoubleComplex **dW0_displ  = NULL;
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;

    magma_int_t alloc = 0;

    alloc += magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    alloc += magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    alloc += magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));

    if (alloc != 0)
    {
        magma_free( dW0_displ );
        magma_free( dW1_displ );
        magma_free( dW2_displ );

        info = MAGMA_ERR_DEVICE_ALLOC;
        return;
    }

    magma_int_t col = n;

    if (trans == MagmaNoTrans)
    {
        for (magma_int_t i=0; i < n; i+= NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaUpper)
            {
                col -= jb;

                magma_zdisplace_pointers(dW0_displ, A_array, lda, col, col+jb, batchCount, queue);
                magma_zdisplace_pointers(dW1_displ, x_array, 1, col+jb, 0,     batchCount, queue);
                magma_zdisplace_pointers(dW2_displ, x_array, 1, col,    0,     batchCount, queue);
            }
            else
            {
                col = i;

                magma_zdisplace_pointers(dW0_displ, A_array, lda, col, 0, batchCount, queue);
                magma_zdisplace_pointers(dW1_displ, x_array, 1,   0,   0, batchCount, queue);
                magma_zdisplace_pointers(dW2_displ, x_array, 1,   col, 0, batchCount, queue);
            }

            //assume x_array contains zero elements
            magmablas_zgemv_batched(MagmaNoTrans, jb, i, MAGMA_Z_ONE, dW0_displ_, lda, dW1_displ_, 1, MAGMA_Z_ONE, dW2_displ, 1, batchCount, queue);

            magma_zdisplace_pointers(dW0_displ, A_array, lda,  col, col, batchCount, queue);
            magma_zdisplace_pointers(dW1_displ, b_array, 1, col*incb,   0, batchCount, queue);
            magma_zdisplace_pointers(dW2_displ, x_array, 1,    col,   0, batchCount, queue);

            magmablas_ztrsv_outofplace_batched(uplo, trans, diag,jb, dW0_displ, lda, dW1_displ, incb, dW2_displ, batchCount, queue, i);
        }
    }
    else
    {
        for (magma_int_t i=0; i < n; i += NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaLower)
            {
                col -= jb;

                magma_zdisplace_pointers(dW0_displ, A_array, lda, col+jb, col, batchCount, queue);
                magma_zdisplace_pointers(dW1_displ, x_array, 1, col+jb, 0,     batchCount, queue);
                magma_zdisplace_pointers(dW2_displ, x_array, 1, col,    0,     batchCount, queue);
            }
            else
            {
                col = i;

                magma_zdisplace_pointers(dW0_displ, A_array, lda, 0, col,  batchCount, queue);
                magma_zdisplace_pointers(dW1_displ, x_array, 1,   0,   0, batchCount, queue);
                magma_zdisplace_pointers(dW2_displ, x_array, 1,   col, 0, batchCount, queue);
            }


            //assume x_array contains zero elements

            magmablas_zgemv_batched(trans, i, jb, MAGMA_Z_ONE, dW0_displ_, lda, dW1_displ_, 1, MAGMA_Z_ONE, dW2_displ, 1, batchCount, queue);

            magma_zdisplace_pointers(dW0_displ, A_array, lda,  col, col, batchCount, queue);
            magma_zdisplace_pointers(dW1_displ, b_array, 1, col*incb,   0, batchCount, queue);
            magma_zdisplace_pointers(dW2_displ, x_array, 1,    col,   0, batchCount, queue);

            magmablas_ztrsv_outofplace_batched(uplo, trans, diag, jb, dW0_displ, lda, dW1_displ, incb, dW2_displ, batchCount, queue, i);
        }
    }

    magma_free(dW0_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
}


/******************************************************************************/
extern "C" void
magmablas_ztrsv_work_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex **A_array, magma_int_t lda,
    magmaDoubleComplex **b_array, magma_int_t incb,
    magmaDoubleComplex **x_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    //magmablas_zlaset_batched(MagmaFull, n, incb, MAGMA_Z_ZERO, MAGMA_Z_ZERO, x_array, n, batchCount, queue);

    //magmablas_ztrsv_recursive_outofplace_batched

    magmablas_ztrsv_recursive_outofplace_batched(uplo, trans, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue);

    magmablas_zlacpy_batched( MagmaFull, n, incb, x_array, n, b_array, n, batchCount, queue);
}


/***************************************************************************//**
    Purpose
    -------
    ztrsv solves one of the matrix equations on gpu

        op(A)*x = b,   or
        x*op(A) = b,

    where alpha is a scalar, X and B are vectors, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The vector x is overwritten on b.


    Arguments
    ----------

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    trans  magma_trans_t.
            On entry, trans specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n N specifies the order of the matrix A. n >= 0.

    @param[in]
    A_array       Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array A of dimension ( lda, n ),
            Before entry with uplo = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    b_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array of dimension  n
            On exit, b is overwritten with the solution vector X.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsv_batched
*******************************************************************************/
extern "C" void
magmablas_ztrsv_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex **A_array, magma_int_t lda,
    magmaDoubleComplex **b_array, magma_int_t incb,
    magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t size_x = n * incb;

    magmaDoubleComplex *x=NULL;
    magmaDoubleComplex **x_array = NULL;

    magma_zmalloc( &x, size_x * batchCount);
    magma_malloc((void**)&x_array,  batchCount * sizeof(*x_array));

    magma_zset_pointer( x_array, x, n, 0, 0, size_x, batchCount, queue );

    magmablas_ztrsv_work_batched(uplo, trans, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue);

    magma_free(x);
    magma_free(x_array);
}
