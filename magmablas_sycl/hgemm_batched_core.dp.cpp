/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_h
#define version(v) NN_V_ ## v

extern "C" magma_int_t
magmablas_hgemm_batched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaHalf alpha,
    magmaHalf const * const * dAarray, magma_int_t ldda,
    magmaHalf const * const * dBarray, magma_int_t lddb,
    magmaHalf beta,
    magmaHalf **dCarray, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

  // For now, converted via dpct from modified hipblas code,
  // rather than from the specialized CUDA code.
  oneapi::mkl::transpose transpose_ct1 = syclblas_trans_const(transA);
  oneapi::mkl::transpose transpose_ct2 = syclblas_trans_const(transB);
  int64_t m_ct3 = int(m), n_ct4 = int(n), k_ct5 = int(k), lda_ct6 = int(ldda),
          ldb_ct7 = int(lddb), ldc_ct8 = int(lddc),
          group_size_ct9 = int(batchCount);
  oneapi::mkl::blas::column_major::gemm_batch(
      *queue->sycl_stream(), &transpose_ct1, &transpose_ct2, &m_ct3, &n_ct4,
      &k_ct5, (sycl::half*) &alpha, (const sycl::half **)dAarray, &lda_ct6,
      (const sycl::half **)dBarray, &ldb_ct7, (sycl::half*) &beta, (sycl::half **)dCarray,
      &ldc_ct8, 1, &group_size_ct9, {});

    return 0;
}
