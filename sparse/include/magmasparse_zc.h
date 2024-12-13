/*
-- MAGMA (version 2.0) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date

 @precisions mixed zc -> ds
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_ZC_H
#define MAGMASPARSE_ZC_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Matrix Descriptors
*/


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/
MAGMA_DEPRECATE("magma_vector_zlac2c is deprecated and will be removed in the next release")
magma_int_t
magma_vector_zlag2c(
    magma_z_matrix x,
    magma_c_matrix *y,
    magma_queue_t queue );

MAGMA_DEPRECATE("magma_sparse_matrix_zlac2c is deprecated and will be removed in the next release")
magma_int_t
magma_sparse_matrix_zlag2c(
    magma_z_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );


MAGMA_DEPRECATE("magma_vector_clag2z is deprecated and will be removed in the next release")
magma_int_t
magma_vector_clag2z(
    magma_c_matrix x,
    magma_z_matrix *y,
    magma_queue_t queue );

MAGMA_DEPRECATE("magma_sparse_matrix_clag2z is deprecated and will be removed in the next release")
magma_int_t
magma_sparse_matrix_clag2z(
    magma_c_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

MAGMA_DEPRECATE("magmablas_zlag2c_sparse is deprecated and will be removed in the next release")
void
magmablas_zlag2c_sparse(
    magma_int_t M, 
    magma_int_t N , 
    magmaDoubleComplex_const_ptr dA, 
    magma_int_t lda, 
    magmaFloatComplex_ptr dSA, 
    magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

MAGMA_DEPRECATE("magmablas_clag2z_sparse is deprecated and will be removed in the next release")
void
magmablas_clag2z_sparse(
    magma_int_t M, 
    magma_int_t N , 
    magmaFloatComplex_const_ptr dSA, 
    magma_int_t ldsa, 
    magmaDoubleComplex_ptr dA, 
    magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

MAGMA_DEPRECATE("magma_zlag2c_CSR_DENSE is deprecated and will be removed in the next release")
void 
magma_zlag2c_CSR_DENSE(
    magma_z_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );

MAGMA_DEPRECATE("magma_zlag2c_CSR_DENSE_alloc is deprecated and will be removed in the next release")
void 
magma_zlag2c_CSR_DENSE_alloc(
    magma_z_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );

MAGMA_DEPRECATE("magma_zlag2c_CSR_DENSE_convert is deprecated and will be removed in the next release")
void 
magma_zlag2c_CSR_DENSE_convert(
    magma_z_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on GPU
*/

MAGMA_DEPRECATE("magma_zlag2c_CSR_DENSE_convert is deprecated and will be removed in the next release")
magma_int_t
magma_zcgecsrmv_mixed_prec(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr ddiagval,
    magmaFloatComplex_ptr doffdiagval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/



#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_ZC_H */
