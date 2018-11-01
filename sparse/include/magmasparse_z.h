/*
 -- MAGMA (version 1.1) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date

 @precisions normal z -> s d c
 @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_Z_H
#define MAGMASPARSE_Z_H

#include "magma_types.h"
#include "magmasparse_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_z


#ifdef __cplusplus
extern "C" {
#endif


/* ////////////////////////////////////////////////////////////////////////////
 -- For backwards compatability, map old (1.6.1) to new (1.6.2) function names
*/

#define magma_z_mtranspose  magma_zmtranspose
#define magma_z_mtransfer   magma_zmtransfer
#define magma_z_vtransfer   magma_zmtransfer
#define magma_z_mconvert    magma_zmconvert
#define magma_z_vinit       magma_zvinit
#define magma_z_vvisu       magma_zprint_vector
#define magma_z_vread       magma_zvread
#define magma_z_vspread     magma_zvspread
#define magma_z_mvisu       magma_zprint_matrix
#define magma_z_mfree       magma_zmfree
#define magma_z_vfree       magma_zmfree
#define write_z_csr_mtx     magma_zwrite_csr_mtx
#define write_z_csrtomtx    magma_zwrite_csrtomtx
#define print_z_csr         magma_zprint_csr_mtx


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE Auxiliary functions
*/

magma_int_t
magma_zwrapper(
    magma_zopts *zopts,
    magma_z_matrix *A, 
    magma_z_matrix *x, 
    magma_z_matrix *b,
    magma_queue_t queue );

magma_int_t
magma_zparse_opts( 
    int argc, 
    char** argv, 
    magma_zopts *opts, 
    int *matrices, 
    magma_queue_t queue );

magma_int_t 
read_z_csr_from_binary( 
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    const char * filename,
    magma_queue_t queue );

magma_int_t 
read_z_csr_from_mtx( 
    magma_storage_t *type, 
    magma_location_t *location,
    magma_int_t* n_row, 
    magma_int_t* n_col, 
    magma_int_t* nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_z_csr_mtx( 
    magma_z_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zcsrset( 
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col, 
    magmaDoubleComplex *val,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_zcsrget( 
    magma_z_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    magmaDoubleComplex **val,
    magma_queue_t queue );


magma_int_t 
magma_zvset( 
    magma_int_t m, 
    magma_int_t n, 
    magmaDoubleComplex *val,
    magma_z_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_zvget( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex **val,
    magma_queue_t queue );

magma_int_t 
magma_zvcopy( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex *val,
    magma_queue_t queue );

magma_int_t 
magma_zvset_dev( 
    magma_int_t m, 
    magma_int_t n, 
    magmaDoubleComplex_ptr val,
    magma_z_matrix *v,
    magma_queue_t queue );

magma_int_t 
magma_zvget_dev( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex_ptr *val,
    magma_queue_t queue );

magma_int_t 
magma_zvcopy_dev( 
    magma_z_matrix v,
    magma_int_t *m, 
    magma_int_t *n, 
    magmaDoubleComplex *val,
    magma_queue_t queue );


magma_int_t 
magma_z_csr_mtxsymm( 
    magma_z_matrix *A, 
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_z_csr_compressor( 
    magmaDoubleComplex ** val, 
    magma_index_t ** row, 
    magma_index_t ** col, 
    magmaDoubleComplex ** valn, 
    magma_index_t ** rown, 
    magma_index_t ** coln, 
    magma_int_t *n,
    magma_queue_t queue );

magma_int_t
magma_zmcsrcompressor( 
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zmshrink(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zmcsrcompressor_gpu( 
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zvtranspose( 
    magma_z_matrix x,
    magma_z_matrix *y,
    magma_queue_t queue );

magma_int_t 
magma_z_cucsrtranspose( 
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
    magma_zmtransposeconjugate(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zmconjugate(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t 
z_transpose_csr( 
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz,
    magmaDoubleComplex *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    magmaDoubleComplex **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue );

magma_int_t
magma_zcsrsplit( 
    magma_int_t offset,
    magma_int_t bsize,
    magma_z_matrix A,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_zmscale( 
    magma_z_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t
magma_zmscale_matrix_rhs(
    magma_z_matrix *A,
    magma_z_matrix *b,
    magma_z_matrix *scaling_factors,
    magma_scale_t scaling,
    magma_queue_t queue );

magma_int_t
magma_zmscale_generate( 
      magma_int_t n, 
      magma_scale_t* scaling, 
      magma_side_t* side, 
      magma_z_matrix* A, 
      magma_z_matrix* scaling_factors,
      magma_queue_t queue  );

magma_int_t
magma_zmscale_apply( 
      magma_int_t n,  
      magma_side_t* side, 
      magma_z_matrix* scaling_factors, 
      magma_z_matrix* A,
    magma_queue_t queue );

magma_int_t
magma_zdimv( 
  magma_z_matrix* vecA, 
  magma_z_matrix* vecB,
  magma_queue_t queue );

magma_int_t
magma_zmslice(
    magma_int_t num_slices,
    magma_int_t slice,
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_z_matrix *ALOC,
    magma_z_matrix *ANLOC,
    magma_index_t *comm_i,
    magmaDoubleComplex *comm_v,
    magma_int_t *start,
    magma_int_t *end,
    magma_queue_t queue );

magma_int_t
magma_zmdiagdom(
    magma_z_matrix M,
    double *min_dd,
    double *max_dd,
    double *avg_dd,
    magma_queue_t queue );

magma_int_t
magma_zmbdiagdom(
    magma_z_matrix M,
    magma_z_matrix blocksizes,
    double *min_dd,
    double *max_dd,
    double *avg_dd,
    magma_queue_t queue );

magma_int_t 
magma_zmdiff( 
    magma_z_matrix A, 
    magma_z_matrix B, 
 real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_zmdiagadd( 
    magma_z_matrix *A, 
    magmaDoubleComplex add,
    magma_queue_t queue );

magma_int_t 
magma_zmsort(
    magmaDoubleComplex *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_zindexsort(
    magma_index_t *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_zsort(
    magmaDoubleComplex *x, 
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_zbitonic_sort(
    magma_int_t start, 
    magma_int_t length, 
    magmaDoubleComplex *seq, 
    magma_int_t flag,
    magma_queue_t queue );

magma_int_t
magma_zindexsortval(
    magma_index_t *x,
    magmaDoubleComplex *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue );

magma_int_t
magma_zorderstatistics(
    magmaDoubleComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue );

magma_int_t
magma_zorderstatistics_inc(
    magmaDoubleComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t inc,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue );

magma_int_t
magma_zmorderstatistics(
    magmaDoubleComplex *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue );

magma_int_t
magma_zpartition( 
    magmaDoubleComplex *a, 
    magma_int_t size, 
    magma_int_t pivot,
    magma_queue_t queue );

magma_int_t
magma_zmedian5( 
    magmaDoubleComplex *a, 
    magma_queue_t queue );

magma_int_t
magma_zselect( 
    magmaDoubleComplex *a, 
    magma_int_t size, 
    magma_int_t k,
    magma_queue_t queue );

magma_int_t
magma_zselectrandom( 
    magmaDoubleComplex *a, 
    magma_int_t size, 
    magma_int_t k,
    magma_queue_t queue );

magma_int_t
magma_zdomainoverlap(
    magma_index_t num_rows,
    magma_int_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue );

magma_int_t
magma_zsymbilu( 
    magma_z_matrix *A, 
    magma_int_t levels,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );


magma_int_t 
magma_zwrite_csr_mtx( 
    magma_z_matrix A,
    magma_order_t MajorType,
 const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zwrite_vector( 
    magma_z_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zwrite_csrtomtx( 
    magma_z_matrix A,
    const char *filename,
    magma_queue_t queue );

magma_int_t 
magma_zprint_csr( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col,
    magma_queue_t queue );

magma_int_t 
magma_zprint_csr_mtx( 
    magma_int_t n_row, 
    magma_int_t n_col, 
    magma_int_t nnz, 
    magmaDoubleComplex **val, 
    magma_index_t **row, 
    magma_index_t **col, 
    magma_order_t MajorType,
    magma_queue_t queue );


magma_int_t 
magma_zmtranspose(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmtranspose_cpu(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmtransposeabs_cpu(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmtransposestruct_cpu(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmtransposeconj_cpu(
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmtransfer(
    magma_z_matrix A, 
    magma_z_matrix *B, 
    magma_location_t src, 
    magma_location_t dst,
    magma_queue_t queue );

magma_int_t 
magma_zmconvert(
    magma_z_matrix A, 
    magma_z_matrix *B, 
    magma_storage_t old_format, 
    magma_storage_t new_format,
    magma_queue_t queue );


magma_int_t
magma_zvinit(
    magma_z_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    magmaDoubleComplex values,
    magma_queue_t queue );

magma_int_t
magma_zvinit_rand(
    magma_z_matrix *x, 
    magma_location_t memory_location,
    magma_int_t num_rows, 
    magma_int_t num_cols,
    magma_queue_t queue );

magma_int_t
magma_zprint_vector(
    magma_z_matrix x, 
    magma_int_t offset, 
    magma_int_t displaylength,
    magma_queue_t queue );

magma_int_t
magma_zvread(
    magma_z_matrix *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue );

magma_int_t
magma_zvspread(
    magma_z_matrix *x, 
    const char * filename,
    magma_queue_t queue );

magma_int_t
magma_zprint_matrix(
    magma_z_matrix A,
    magma_queue_t queue );

magma_int_t 
magma_zdiameter(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t 
magma_zrowentries(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zmfree(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zresidual(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix x, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zresidualvec(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix x,
    magma_z_matrix *r,
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zresidual_slice(
    magma_int_t start,
    magma_int_t end,
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix x,
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zmgenerator(
    magma_int_t n,
    magma_int_t offdiags,
    magma_index_t *diag_offset,
    magmaDoubleComplex *diag_vals,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zm_27stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zm_5stencil(
    magma_int_t n,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zsolverinfo(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zsolverinfo_init(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zeigensolverinfo_init(
    magma_z_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_zprecondfree(
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zsolverinfo_free(
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zKrylov_check( magma_solver_type solver );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE parallel incomplete factorizations (ParILU / ParILUT)
*/


magma_int_t
magma_zparilusetup( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparilu_gpu( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparilu_cpu( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparic_gpu( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparic_cpu( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparicsetup( 
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparicupdate( 
    magma_z_matrix A, 
    magma_z_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_zapplyiteric_l( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplyiteric_r( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zparilu_csr( 
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_queue_t queue );

magma_int_t
magma_zpariluupdate(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue );

magma_int_t
magma_zparic_csr( 
    magma_z_matrix A,
    magma_z_matrix A_CSR,
    magma_queue_t queue );

magma_int_t 
magma_zfrobenius( 
    magma_z_matrix A, 
    magma_z_matrix B, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t
magma_zmfrobenius(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix S,
    double *norm,
    magma_queue_t queue );

magma_int_t 
magma_znonlinres(   
    magma_z_matrix A, 
    magma_z_matrix L,
    magma_z_matrix U, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    magma_queue_t queue );

magma_int_t 
magma_zilures(   
    magma_z_matrix A, 
    magma_z_matrix L,
    magma_z_matrix U, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_zicres(       
    magma_z_matrix A, 
    magma_z_matrix C,
    magma_z_matrix CT, 
    magma_z_matrix *LU, 
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue );

magma_int_t 
magma_zinitguess( 
    magma_z_matrix A, 
    magma_z_matrix *L, 
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t 
magma_zinitrecursiveLU( 
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t 
magma_zmLdiagadd( 
    magma_z_matrix *L,
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative dynamic ILU
*/
// #ifdef _OPENMP

magma_int_t
magma_zmatrix_tril(
    magma_z_matrix A,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_triu(
    magma_z_matrix A,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_cup(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_cup_gpu(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue);

magma_int_t
magma_zmatrix_cap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_tril_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_triu_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_addrowindex(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_abssum(
    magma_z_matrix A,
    double *sum,
    magma_queue_t queue );

magma_int_t
magma_zparilut_thrsrm(
    magma_int_t order,
    magma_z_matrix *A,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_thrsrm_semilinked(
    magma_z_matrix *U,
    magma_z_matrix *UT,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_rmselected(
    magma_z_matrix R,
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zparilut_selectoneperrow(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zparilut_selecttwoperrow(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zparilut_selectoneperrowthrs_lower(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *A,
    double  rtol,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zparilut_selectoneperrowthrs_upper(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *A,
    double  rtol,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zparilut_selectonepercol(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_swap(
    magma_z_matrix *A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zcsrcoo_transpose(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zparilut_transpose_select_one(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zmatrix_createrowptr(
    magma_int_t n,
    magma_index_t *row,
    magma_queue_t queue );

magma_int_t
magma_zparilut_insert_LU(
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_index_t *rm_loc2,
    magma_z_matrix *LU_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_approx_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_thrs_randomselect(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_thrs_randomselect_approx(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_thrs_randomselect_factors(
    magma_int_t num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_exact_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_set_approx_thrs_inc(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_LU_approx_thrs(
    magma_int_t num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_reorder(
    magma_z_matrix *LU,
    magma_queue_t queue );

magma_int_t
magma_zparict_sweep(
    magma_z_matrix *A,
    magma_z_matrix *LU,
    magma_queue_t queue );

magma_int_t
magma_zparilut_zero(
    magma_z_matrix *A,
    magma_queue_t queue );

magma_int_t
magma_zparilu_sweep(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilu_sweep_sync(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparic_sweep(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_queue_t queue );

magma_int_t
magma_zparic_sweep_sync(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_queue_t queue );

magma_int_t
magma_zparict_sweep_sync(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_queue_t queue );

magma_int_t
magma_zparilut_sweep_sync(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilut_sweep_gpu( 
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals_gpu( 
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *R,
    magma_queue_t queue );

magma_int_t
magma_zthrsholdrm_gpu(
    magma_int_t order,
    magma_z_matrix* A,
    double* thrs,
    magma_queue_t queue);

magma_int_t
magma_zget_row_ptr(
    const magma_int_t num_rows,
    magma_int_t* nnz,
    const magma_index_t* rowidx,
    magma_index_t* rowptr,
    magma_queue_t queue);

magma_int_t
magma_zvalinit_gpu(
    magma_int_t num_el,
    magmaDoubleComplex_ptr dval,
    magma_queue_t queue);

magma_int_t
magma_zindexinit_gpu(
    magma_int_t num_el,
    magmaIndex_ptr dind,
    magma_queue_t queue);

magma_int_t
magma_zparilut_align_residuals(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *Lnew,
    magma_z_matrix *Unew,
    magma_queue_t queue );

magma_int_t
magma_zparilut_preselect_scale(
    magma_z_matrix *L,
    magma_z_matrix *oneL,
    magma_z_matrix *U,
    magma_z_matrix *oneU,
    magma_queue_t queue );

magma_int_t
magma_zparilut_thrsrm_U(
    magma_int_t order,
    magma_z_matrix L,
    magma_z_matrix *A,
    double *thrs,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals_transpose(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals_semilinked(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix US,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_sweep_semilinked(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *US,
    magma_queue_t queue );

magma_int_t
magma_zparilut_sweep_list(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals_list(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_sweep_linkedlist(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zparilut_residuals_linkedlist(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_colmajor(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue );

magma_int_t
magma_zparilut_colmajorup(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue );

magma_int_t
magma_zparict(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparict_cpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue);

magma_int_t
magma_zparilut(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparilut_cpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparilut_gpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zparilut_insert(
    magma_int_t *num_rmL,
    magma_int_t *num_rmU,
    magma_index_t *rm_locL,
    magma_index_t *rm_locU,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *UR,
    magma_queue_t queue );

magma_int_t
magma_zparilut_create_collinkedlist(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue );

magma_int_t
magma_zparilut_candidates(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_candidates_gpu(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue );

magma_int_t
magma_zparict_candidates(
    magma_z_matrix L0,
    magma_z_matrix L,
    magma_z_matrix LT,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_candidates_semilinked(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UT,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_candidates_linkedlist(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_rm_thrs(
    double *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    magma_queue_t queue );

magma_int_t
magma_zparilut_count(
    magma_z_matrix L,
    magma_int_t *num,
    magma_queue_t queue );

magma_int_t
magma_zparilut_randlist(
    magma_z_matrix *LU,
    magma_queue_t queue );

magma_int_t
magma_zparilut_select_candidates_L(
    magma_int_t *num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_select_candidates_U(
    magma_int_t *num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue );

magma_int_t
magma_zparilut_preselect(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zpreselect_gpu(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue );

magma_int_t
magma_zsampleselect(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue );

magma_int_t
magma_zsampleselect_approx(
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_ptr *tmp_ptr,
    magma_int_t *tmp_size,
    magma_queue_t queue );

// ISAI preconditioner

magma_int_t
magma_zmprepare_batched(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_zmtrisolve_batched(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_zmbackinsert_batched(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_zmprepare_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_zmtrisolve_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_zmbackinsert_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue );

magma_int_t
magma_ziluisaisetup_lower(
    magma_z_matrix L,
    magma_z_matrix S,
    magma_z_matrix *ISAIL,
    magma_queue_t queue );

magma_int_t
magma_ziluisaisetup_upper(
    magma_z_matrix U,
    magma_z_matrix S,
    magma_z_matrix *ISAIU,
    magma_queue_t queue );

magma_int_t
magma_zicisaisetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zisai_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zisai_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zisai_l_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zisai_r_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zmiluisai_sizecheck(
    magma_z_matrix A,
    magma_index_t batchsize,
    magma_index_t *maxsize,
    magma_queue_t queue );

magma_int_t
magma_zgeisai_maxblock(
    magma_z_matrix L,
    magma_z_matrix *MT,
    magma_queue_t queue );

magma_int_t
magma_zisai_generator_regs(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix *M,
    magma_queue_t queue );

magma_int_t
magma_zcsr_sort(
    magma_z_matrix *A,
    magma_queue_t queue);

magma_int_t
magma_zcsr_sort_gpu(
    magma_z_matrix *A,
    magma_queue_t queue);

// #endif
/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE supernodal and RCM reordering
*/
magma_int_t
magma_zmsupernodal(
    magma_int_t *max_bs,
    magma_z_matrix A,
    magma_z_matrix *S,
    magma_queue_t queue );

magma_int_t
magma_zmvarsizeblockstruct(
    magma_int_t n,
    magma_int_t *bs,
    magma_int_t bsl,
    magma_uplo_t uplotype,
    magma_z_matrix *A,
    magma_queue_t queue );



/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE iterative solvers (Data on GPU)
*/

magma_int_t 
magma_zcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_zcg_res(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t 
magma_zcg_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpcg_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zcgs(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zcgs_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpcgs(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpcgs_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zqmr_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpqmr_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_ztfqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_unrolled(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zptfqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zptfqmr_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x, 
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicg(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_merge2(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_merge3(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbpcg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpbicg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpbicgstab(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpbicgstab_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zfgmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbfgmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zidr(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zidr_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zidr_strms(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zpidr(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpidr_merge(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zpidr_strms(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbombard(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbombard_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobi(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobidomainoverlap(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zbaiter(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbaiter_overlap(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zftjacobicontractions(
    magma_z_matrix xkm2,
    magma_z_matrix xkm1, 
    magma_z_matrix xk, 
    magma_z_matrix *z,
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zftjacobiupdatecheck(
    double delta,
    magma_z_matrix *xold,
    magma_z_matrix *xnew, 
    magma_z_matrix *zprev, 
    magma_z_matrix c,
    magma_int_t *flag_t,
    magma_int_t *flag_fp,
    magma_queue_t queue );

magma_int_t
magma_ziterref(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlu(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );


magma_int_t
magma_zbcsrlutrf(
    magma_z_matrix A, 
    magma_z_matrix *M,
    magma_int_t *ipiv, 
    magma_int_t version,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlusv(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par, 
    magma_int_t *ipiv,
    magma_queue_t queue );



magma_int_t
magma_zilucg(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zilugmres(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue ); 


magma_int_t
magma_zlobpcg_shift(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magma_int_t shift,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue );

magma_int_t
magma_zlobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    double *evalues, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr R, 
    double *res,
    magma_queue_t queue );

magma_int_t
magma_zlobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr Y,
    magma_queue_t queue );


/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE eigensolvers (Data on GPU)
*/
magma_int_t
magma_zlobpcg(
    magma_z_matrix A, 
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par, 
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE LSQR (Data on GPU)
*/
magma_int_t
magma_zlsqr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE direct solvers (Data on GPU)
*/

magma_int_t 
magma_zpardiso(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue );

/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE preconditioners (Data on GPU)
*/

magma_int_t
magma_zjacobisetup(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *M, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_matrix(
    magma_z_matrix A, 
    magma_z_matrix *M, 
    magma_z_matrix *d,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_vector(
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter(
    magma_z_matrix M, 
    magma_z_matrix c, 
    magma_z_matrix *x, 
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter_precond( 
    magma_z_matrix M, 
    magma_z_matrix *x, 
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zjacobiiter_sys(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix t, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

magma_int_t
magma_zftjacobi(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue );

// custom preconditioner
magma_int_t
magma_zapplycustomprecond_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycustomprecond_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


// CUSPARSE preconditioner

magma_int_t
magma_zcuilusetup(
    magma_z_matrix A, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zilut_saad( 
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zilut_saad_apply( 
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zcumilusetup_transpose(
    magma_z_matrix A, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuilu_l(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuilu_r(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcuiccsetup(
    magma_z_matrix A, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuicc_l(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycuicc_r(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_zcumilusetup(
    magma_z_matrix A, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcustomilusetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcustomicsetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcumilugeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_l_transpose(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumilu_r_transpose(
    magma_z_matrix b, magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );



magma_int_t
magma_zcumiccsetup(
    magma_z_matrix A, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zcumicgeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumicc_l(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_zapplycumicc_r(
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue );


// block-asynchronous iteration

magma_int_t
magma_zbajac_csr(
    magma_int_t localiters,
    magma_z_matrix D,
    magma_z_matrix R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zbajac_csr_overlap(
    magma_int_t localiters,
    magma_int_t matrices,
    magma_int_t overlap,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_z_spmv(
    magmaDoubleComplex alpha, 
    magma_z_matrix A, 
    magma_z_matrix x, 
    magmaDoubleComplex beta, 
    magma_z_matrix y,
    magma_queue_t queue );

magma_int_t
magma_zcustomspmv(
    magma_int_t m,
    magma_int_t n,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex beta, 
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr y, 
    magma_queue_t queue );

magma_int_t
magma_z_spmv_shift(
    magmaDoubleComplex alpha, 
    magma_z_matrix A, 
    magmaDoubleComplex lambda,
    magma_z_matrix x, 
    magmaDoubleComplex beta, 
    magma_int_t offset, 
    magma_int_t blocksize,
    magmaIndex_ptr dadd_vecs, 
    magma_z_matrix y,
    magma_queue_t queue );

magma_int_t
magma_zcuspmm(
    magma_z_matrix A, 
    magma_z_matrix B, 
    magma_z_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_z_spmm(
    magmaDoubleComplex alpha, 
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *C,
    magma_queue_t queue );

magma_int_t
magma_zsymbilu( 
    magma_z_matrix *A, 
    magma_int_t levels, 
    magma_z_matrix *L, 
    magma_z_matrix *U,
    magma_queue_t queue );

magma_int_t
magma_zcuspaxpy(
    magmaDoubleComplex_ptr alpha, magma_z_matrix A, 
    magmaDoubleComplex_ptr beta, magma_z_matrix B, 
    magma_z_matrix *AB,
    magma_queue_t queue );

magma_int_t
magma_z_precond(
    magma_z_matrix A, 
    magma_z_matrix b, magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_solver(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_zopts *zopts,
    magma_queue_t queue );

magma_int_t
magma_z_precondsetup(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_solver_par *solver,
    magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_applyprecond(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_z_applyprecond_left(
    magma_trans_t trans,
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );


magma_int_t
magma_z_applyprecond_right(
    magma_trans_t trans,
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_preconditioner *precond,
    magma_queue_t queue );

magma_int_t
magma_z_initP2P(
    magma_int_t *bandwidth_benchmark,
    magma_int_t *num_gpus,
    magma_queue_t queue );

magma_int_t
magma_zcompact(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    double *dnorms, double tol, 
    magma_int_t *activeMask, magma_int_t *cBlockSize,
    magma_queue_t queue );

magma_int_t
magma_zcompactActive(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magma_int_t *active,
    magma_queue_t queue );

magma_int_t
magma_zmlumerge(    
    magma_z_matrix L, 
    magma_z_matrix U,
    magma_z_matrix *A, 
    magma_queue_t queue );

magma_int_t
magma_zdiagcheck(
    magma_z_matrix dA,
    magma_queue_t queue );

magma_int_t
magma_zdiagcheck_cpu(
    magma_z_matrix A,
    magma_queue_t queue );



/*/////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE wrappers to dense MAGMA
*/
magma_int_t
magma_zqr(
    magma_int_t m, 
    magma_int_t n, 
    magma_z_matrix A, 
    magma_int_t lda, 
    magma_z_matrix *Q, 
    magma_z_matrix *R, 
    magma_queue_t queue );


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE BLAS function definitions
*/

magma_int_t
magma_zgeaxpy(
    magmaDoubleComplex alpha,
    magma_z_matrix X,
    magmaDoubleComplex beta,
    magma_z_matrix *Y,
    magma_queue_t queue );

magma_int_t
magma_zgecsrreimsplit(
    magma_z_matrix A,
    magma_z_matrix *ReA,
    magma_z_matrix *ImA,
    magma_queue_t queue );

magma_int_t
magma_zgedensereimsplit(
    magma_z_matrix A,
    magma_z_matrix *ReA,
    magma_z_matrix *ImA,
    magma_queue_t queue );

magma_int_t 
magma_zgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgecsrmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zmgecsrmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr dadd_rows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );


magma_int_t 
magma_zmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t 
magma_zgeellrtmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowlength,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_int_t num_threads,
    magma_int_t threads_per_row,
    magma_queue_t queue );

magma_int_t 
magma_zgesellcmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zmgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zmgesellpmv_blocked(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

magma_int_t
magma_zgecsr5mv(
    magma_trans_t           transA,
    magma_int_t             m, 
    magma_int_t             n, 
    magma_int_t             p,
    magmaDoubleComplex      alpha,
    magma_int_t             sigma,
    magma_int_t             bit_y_offset,
    magma_int_t             bit_scansum_offset,
    magma_int_t             num_packet,
    magmaUIndex_ptr         dtile_ptr,
    magmaUIndex_ptr         dtile_desc,
    magmaIndex_ptr          dtile_desc_offset_ptr,
    magmaIndex_ptr          dtile_desc_offset,
    magmaDoubleComplex_ptr  dcalibrator,
    magma_int_t             tail_tile_start,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          drowptr,
    magmaIndex_ptr          dcolind,
    magmaDoubleComplex_ptr  dx,
    magmaDoubleComplex      beta,
    magmaDoubleComplex_ptr  dy,
    magma_queue_t           queue );

magma_int_t
magma_zgecscsyncfreetrsm_analysis(
    magma_int_t             m, 
    magma_int_t             nnz,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind, 
    magmaIndex_ptr          dgraphindegree, 
    magmaIndex_ptr          dgraphindegree_bak, 
    magma_queue_t           queue );

magma_int_t
magma_zgecscsyncfreetrsm_solve(
    magma_int_t             m, 
    magma_int_t             nnz,
    magmaDoubleComplex_ptr  dval,
    magmaIndex_ptr          dcolptr,
    magmaIndex_ptr          drowind,
    magmaIndex_ptr          dgraphindegree, 
    magmaIndex_ptr          dgraphindegree_bak, 
    magmaDoubleComplex_ptr  dx,
    magmaDoubleComplex_ptr  db,
    magma_int_t             substitution, 
    magma_int_t             rhs, 
    magma_queue_t           queue );

magma_int_t
magma_zmergedgs(
    magma_int_t n, 
    magma_int_t ldh,
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zcopyscale(    
    magma_int_t n, 
    magma_int_t k,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_dznrm2scale(    
    magma_int_t m, 
    magmaDoubleComplex_ptr dr,    
    magma_int_t lddr, 
    magmaDoubleComplex *drnorm,
    magma_queue_t queue );


magma_int_t
magma_zjacobisetup_vector_gpu(
    magma_int_t num_rows, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix c,
    magma_z_matrix *x,
    magma_queue_t queue );


magma_int_t
magma_zjacobi_diagscal(    
    magma_int_t num_rows, 
    magma_z_matrix d, 
    magma_z_matrix b, 
    magma_z_matrix *c,
    magma_queue_t queue );

magma_int_t
magma_zjacobiupdate(
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdate(
    magma_int_t maxiter,
    magma_z_matrix A, 
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdate_bw(
    magma_int_t maxiter,
    magma_z_matrix A, 
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_z_matrix A,
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix tmp, 
    magma_z_matrix *x,
    magma_queue_t queue );

magma_int_t
magma_zjacobisetup_diagscal(
    magma_z_matrix A, magma_z_matrix *d,
    magma_queue_t queue );


//##################   kernel fusion for Krylov methods

magma_int_t
magma_zmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr alpha, 
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge1(    
    magma_int_t n, 
    magmaDoubleComplex_ptr dskp,
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dp,
    magma_queue_t queue );


magma_int_t
magma_zbicgmerge2(
    magma_int_t n, 
    magmaDoubleComplex_ptr dskp, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr ds,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge3(
    magma_int_t n, 
    magmaDoubleComplex_ptr dskp, 
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dx, 
    magmaDoubleComplex_ptr dr,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge4(
    magma_int_t type, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zidr_smoothing_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr drs,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dt, 
    magma_queue_t queue );

magma_int_t
magma_zidr_smoothing_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dxs, 
    magma_queue_t queue );

magma_int_t
magma_zcgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr q, 
    magmaDoubleComplex_ptr u,
    magmaDoubleComplex_ptr p,
    magma_queue_t queue );

magma_int_t
magma_zcgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr u,
    magmaDoubleComplex_ptr p, 
    magma_queue_t queue );

magma_int_t
magma_zcgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr v_hat,
    magmaDoubleComplex_ptr u, 
    magmaDoubleComplex_ptr q,
    magmaDoubleComplex_ptr t, 
    magma_queue_t queue );

magma_int_t
magma_zcgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr u_hat,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_zqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue );

magma_int_t
magma_zqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex pde,
    magmaDoubleComplex rde,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr p, 
    magmaDoubleComplex_ptr q, 
    magma_queue_t queue );

magma_int_t
magma_zqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr y,
    magma_queue_t queue );

magma_int_t
magma_zqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr d, 
    magmaDoubleComplex_ptr s, 
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_zqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex pds,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr d, 
    magmaDoubleComplex_ptr s, 
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_zqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magmaDoubleComplex_ptr wt,
    magma_queue_t queue );

magma_int_t
magma_zqmr_7(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr pt,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr vt,
    magma_queue_t queue );

magma_int_t
magma_zqmr_8(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex rho,
    magmaDoubleComplex psi,
    magmaDoubleComplex_ptr vt,
    magmaDoubleComplex_ptr wt,
    magmaDoubleComplex_ptr y, 
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr r, 
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr p,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr s, 
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_zbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr Au,
    magmaDoubleComplex_ptr u_m,
    magmaDoubleComplex_ptr pu_m,
    magmaDoubleComplex_ptr u_mp1,
    magmaDoubleComplex_ptr w, 
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr w,
    magmaDoubleComplex_ptr u_m,
    magmaDoubleComplex_ptr u_mp1, 
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr Au_new,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr Au, 
    magma_queue_t queue );

magma_int_t
magma_ztfqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr Au,
    magmaDoubleComplex_ptr u_mp1,
    magmaDoubleComplex_ptr w, 
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magma_queue_t queue );

magma_int_t
magma_zcgmerge_spmv1( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zcgmerge_xrbeta( 
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_zpcgmerge_xrbeta1(
    magma_int_t n,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zpcgmerge_xrbeta2(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr dh,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dd, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zjcgmerge_xrbeta(
    magma_int_t n,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr diag,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd,
    magmaDoubleComplex_ptr dz,
    magmaDoubleComplex_ptr dh, 
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zgemvmdot_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_zgemvmdot(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr dv, 
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );


magma_int_t
magma_zmdotc1(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc2(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc3(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr v2, 
    magmaDoubleComplex_ptr w2,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zmdotc4(
    magma_int_t n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr v2, 
    magmaDoubleComplex_ptr w2,
    magmaDoubleComplex_ptr v3, 
    magmaDoubleComplex_ptr w3,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_spmv1( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_spmv2( 
    magma_z_matrix A,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbicgmerge_xrbeta( 
    magma_int_t n,
    magmaDoubleComplex_ptr dd1,
    magmaDoubleComplex_ptr dd2,
    magmaDoubleComplex_ptr drr,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dp,
    magmaDoubleComplex_ptr ds,
    magmaDoubleComplex_ptr dt,
    magmaDoubleComplex_ptr dx, 
    magmaDoubleComplex_ptr dskp,
    magma_queue_t queue );

magma_int_t
magma_zbcsrswp(
    magma_int_t n,
    magma_int_t size_b, 
    magma_int_t *ipiv,
    magmaDoubleComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_zbcsrtrsv(
    magma_uplo_t uplo,
    magma_int_t r_blocks,
    magma_int_t c_blocks,
    magma_int_t size_b, 
    magmaDoubleComplex_ptr dA,
    magma_index_t *blockinfo, 
    magmaDoubleComplex_ptr dx,
    magma_queue_t queue );

magma_int_t
magma_zbcsrvalcpy(
    magma_int_t size_b, 
    magma_int_t num_blocks, 
    magma_int_t num_zero_blocks, 
    magmaDoubleComplex_ptr *dAval, 
    magmaDoubleComplex_ptr *dBval,
    magmaDoubleComplex_ptr *dBval2,
    magma_queue_t queue );

magma_int_t
magma_zbcsrluegemm(
    magma_int_t size_b, 
    magma_int_t num_block_rows,
    magma_int_t kblocks,
    magmaDoubleComplex_ptr *dA, 
    magmaDoubleComplex_ptr *dB, 
    magmaDoubleComplex_ptr *dC,
    magma_queue_t queue );

magma_int_t
magma_zbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaDoubleComplex_ptr *dA, 
    magma_int_t *ipiv,
    magma_queue_t queue );

magma_int_t
magma_zbcsrblockinfo5(
    magma_int_t lustep,
    magma_int_t num_blocks, 
    magma_int_t c_blocks, 
    magma_int_t size_b,
    magma_index_t *blockinfo,
    magmaDoubleComplex_ptr dval,
    magmaDoubleComplex_ptr *AII,
    magma_queue_t queue );


magma_int_t
magma_ztestasync( 
    magma_z_matrix A,
    magma_queue_t queue );

//##################   stencil operators


magma_int_t
magma_zge3pt(
    magma_int_t m, 
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue );

//#############  Big data analytics
magma_int_t
magma_zjaccard_weights(
    magma_z_matrix A,
    magma_z_matrix *J,
    magma_queue_t queue );

magma_int_t
magma_zthrsholdselect(
    magma_int_t sampling,
    magma_int_t total_size,
    magma_int_t subset_size,
    magmaDoubleComplex *val,
    double *thrs,
    magma_queue_t queue );
 
#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
