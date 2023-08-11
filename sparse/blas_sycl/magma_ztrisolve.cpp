/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel

       @precisions normal z -> s d c
*/
#include "magma_trisolve.h"

#define PRECISION_z

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasDoubleComplex hipDoubleComplex
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

#if defined(MAGMA_HAVE_CUDA)
magma_int_t magma_ztrisolve_analysis(magma_z_matrix M, magma_solve_info_t *solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_queue_t queue)
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle = NULL;
    cusparseFillMode_t fill_mode = upper_triangular ? CUSPARSE_FILL_MODE_UPPER
                                                    : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type = unit_diagonal ? CUSPARSE_DIAG_TYPE_UNIT
                                                 : CUSPARSE_DIAG_TYPE_NON_UNIT;
    magmaDoubleComplex one = MAGMA_Z_ONE;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t M_op = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                         : CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, queue->cuda_stream()));

#if CUDA_VERSION >= 11031
    cusparseSpMatDescr_t descr;
    cusparseDnMatDescr_t in;
    cusparseDnMatDescr_t out;
    {
        cusparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;
        cudaDataType data_type = CUDA_C_64F;
        CHECK_CUSPARSE(cusparseCreateCsr(&descr, M.num_rows, M.num_rows, M.nnz,
                                         M.drow, M.dcol, M.dval,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, data_type));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_FILL_MODE,
                                                 &fill_mode, sizeof(fill_mode)));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_DIAG_TYPE,
                                                 &diag_type, sizeof(diag_type)));
        // create dummy input and output vectors with distinct non-null pointers
        // otherwise cuSPARSE complains, even though it doesn't use the vectors
        CHECK_CUSPARSE(cusparseCreateDnMat(&in, M.num_rows, 1, M.num_rows,
                                           (void *)0xF0, data_type,
                                           CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(cusparseCreateDnMat(&out, M.num_rows, 1, M.num_rows,
                                           (void *)0xE0, data_type,
                                           CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(cusparseSpSM_createDescr(&solve_info->descr));
        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseSpSM_bufferSize(cusparseHandle, M_op, op, &one,
                                               descr, in, out, data_type, alg,
                                               solve_info->descr, &buffer_size));
        if (buffer_size > 0)
            magma_malloc(&solve_info->buffer, buffer_size);
        CHECK_CUSPARSE(cusparseSpSM_analysis(cusparseHandle, M_op, op, &one,
                                             descr, in, out, data_type, alg,
                                             solve_info->descr,
                                             solve_info->buffer));
    }

cleanup:
    cusparseDestroyDnMat(out);
    cusparseDestroyDnMat(in);
    cusparseDestroySpMat(descr);
#else
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr, diag_type));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr, fill_mode));
    {
        int algo = 0;
        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseCreateCsrsm2Info(&solve_info->descr));
        CHECK_CUSPARSE(cusparseZcsrsm2_bufferSizeExt(cusparseHandle, algo, M_op, op,
                                                     M.num_rows, 1, M.nnz, (const cuDoubleComplex*)&one,
                                                     descr, (cuDoubleComplex*)M.dval, M.drow, M.dcol,
                                                     NULL, M.num_rows,
                                                     solve_info->descr,
                                                     CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                     &buffer_size));
        if (buffer_size > 0)
            magma_malloc(&solve_info->buffer, buffer_size);
        CHECK_CUSPARSE(cusparseZcsrsm2_analysis(cusparseHandle, algo, M_op, op,
                                                M.num_rows, 1, M.nnz, (const cuDoubleComplex*)&one, descr,
                                                (cuDoubleComplex*)M.dval, M.drow, M.dcol, NULL,
                                                M.num_rows, solve_info->descr,
                                                CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                solve_info->buffer));
    }

cleanup:
    cusparseDestroyMatDescr(descr);
#endif
    cusparseDestroy(cusparseHandle);

    return info;
}

magma_int_t magma_ztrisolve(magma_z_matrix M, magma_solve_info_t solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_z_matrix b, magma_z_matrix x, magma_queue_t queue)
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle = NULL;
    cusparseFillMode_t fill_mode = upper_triangular ? CUSPARSE_FILL_MODE_UPPER
                                                    : CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type = unit_diagonal ? CUSPARSE_DIAG_TYPE_UNIT
                                                 : CUSPARSE_DIAG_TYPE_NON_UNIT;
    magmaDoubleComplex one = MAGMA_Z_ONE;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t M_op = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                         : CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, queue->cuda_stream()));

#if CUDA_VERSION >= 11031
    cusparseSpMatDescr_t descr;
    cusparseDnMatDescr_t in;
    cusparseDnMatDescr_t out;
    {
        cusparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;
        cudaDataType data_type = CUDA_C_64F;
        CHECK_CUSPARSE(cusparseCreateCsr(&descr, M.num_rows, M.num_rows, M.nnz,
                                         M.drow, M.dcol, M.dval,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, data_type));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_FILL_MODE,
                                                 &fill_mode, sizeof(fill_mode)));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_DIAG_TYPE,
                                                 &diag_type, sizeof(diag_type)));
        CHECK_CUSPARSE(cusparseCreateDnMat(&in, b.num_rows, b.num_cols, b.num_rows,
                                           b.dval, data_type, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(cusparseCreateDnMat(&out, x.num_rows, x.num_cols, x.num_rows,
                                           x.dval, data_type, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(cusparseSpSM_solve(cusparseHandle, M_op, op, &one, descr,
                                          in, out, data_type, alg,
                                          solve_info.descr));
    }

cleanup:
    cusparseDestroyDnMat(out);
    cusparseDestroyDnMat(in);
    cusparseDestroySpMat(descr);
#else
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr, diag_type));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr, fill_mode));
    {
        int algo = 0;
        magmablas_zlacpy(MagmaFull, M.num_rows, b.num_cols, b.dval, M.num_rows,
                         x.dval, M.num_rows, queue);
        CHECK_CUSPARSE(cusparseZcsrsm2_solve(cusparseHandle, algo, M_op, op,
                                             M.num_rows, b.num_cols, M.nnz, (const cuDoubleComplex*)&one,
                                             descr, (cuDoubleComplex*)M.dval, M.drow, M.dcol, (cuDoubleComplex*)x.dval,
                                             M.num_rows, solve_info.descr,
                                             CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             solve_info.buffer));
    }

cleanup:
    cusparseDestroyMatDescr(descr);
#endif
    cusparseDestroy(cusparseHandle);

    return info;
}
#endif
