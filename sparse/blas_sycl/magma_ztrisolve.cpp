/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Natalie Beams

       @precisions normal z -> s d c
*/
#include "magma_trisolve.h"

#define PRECISION_z

magma_int_t magma_ztrisolve_analysis(magma_z_matrix M, magma_solve_info_t *solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_queue_t queue)
{
    	// Not supported for SYCL currently (must be done inside the ztrisolve)
	// but we don't want it to be an error...
    return 0;
}

magma_int_t magma_ztrisolve(magma_z_matrix M, magma_solve_info_t solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_z_matrix b, magma_z_matrix x, magma_queue_t queue)
{
    magma_int_t info = 0;
    oneapi::mkl::sparse::matrix_handle_t M_handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&M_handle);
  
    oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), M_handle,
  		     M.num_rows, M.num_cols,
                     oneapi::mkl::index_base::zero, M.drow, M.dcol, MAGMA_Z_MKL_PTR(M.dval));
 
    oneapi::mkl::uplo fill_mode = upper_triangular ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;

    oneapi::mkl::diag diag_type = unit_diagonal ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

    magmaDoubleComplex one = MAGMA_Z_ONE;
    // TODO: can/should we handle conjtrans?
    oneapi::mkl::transpose M_op = transpose ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;

    // Optimize first
    oneapi::mkl::sparse::optimize_trsv(*queue->sycl_stream(),
                 fill_mode, M_op, diag_type, M_handle, {});
    // Now solve
    oneapi::mkl::sparse::trsv(*queue->sycl_stream(),
		   fill_mode, M_op, diag_type, M_handle, 
		   MAGMA_Z_MKL_PTR(b.dval), MAGMA_Z_MKL_PTR(x.dval), {});
cleanup:
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &M_handle);
    return info;
}
