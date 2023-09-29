/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

/**
    Purpose
    -------

    This is an interface to the MKL sparse routine csrmm computing the product
    of two sparse matrices stored in csr format.
    It requires that B be sorted, so the sort will be performed.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix

    @param[in]
    B           magma_z_matrix
                input matrix

    @param[out]
    AB          magma_z_matrix*
                output matrix AB = A * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcuspmm(
    magma_z_matrix A, magma_z_matrix B,
    magma_z_matrix *AB,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    oneapi::mkl::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_handle = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_handle = nullptr;
    oneapi::mkl::sparse::matmat_descr_t descr = nullptr;
    magma_z_matrix C={Magma_CSR};

    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR )
        && ( B.storage_type == Magma_CSR ) )
    {
      C.num_rows = A.num_rows;
      C.num_cols = B.num_cols;
      C.storage_type = A.storage_type;
      C.memory_location = A.memory_location;
      C.fill_mode = MagmaFull;
      CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));

      C.val = NULL;
      C.col = NULL;
      C.row = NULL;
      C.rowidx = NULL;
      C.blockinfo = NULL;
      C.diag = NULL;
      C.dval = NULL;
      C.dcol = NULL;
      C.drow = NULL;
      C.drowidx = NULL;
      C.ddiag = NULL;

      // Set up MKL sparse matrix handles
      oneapi::mkl::sparse::init_matrix_handle(&A_handle);
      oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), A_handle,
                       A.num_rows, A.num_cols,
                       oneapi::mkl::index_base::zero, A.drow, A.dcol, A.dval);

      oneapi::mkl::sparse::init_matrix_handle(&B_handle);
      oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), B_handle,
  		     B.num_rows, B.num_cols,
                       oneapi::mkl::index_base::zero, B.drow, B.dcol, B.dval);
      // For now, B must be sorted upon entry to matmat
      oneapi::mkl::sparse::sort_matrix(*queue->sycl_stream(), B_handle, {});

      oneapi::mkl::sparse::init_matrix_handle(&C_handle);
      oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), C_handle,
  		     C.num_rows, C.num_cols,
                       oneapi::mkl::index_base::zero, C.drow, C.dcol, C.dval);

      oneapi::mkl::sparse::init_matmat_descr(&descr);
      // TODO: other options?
      oneapi::mkl::sparse::matrix_view_descr viewA = oneapi::mkl::sparse::matrix_view_descr::general;
      oneapi::mkl::sparse::matrix_view_descr viewB = oneapi::mkl::sparse::matrix_view_descr::general;
      oneapi::mkl::sparse::matrix_view_descr viewC = oneapi::mkl::sparse::matrix_view_descr::general;
      oneapi::mkl::transpose opA = oneapi::mkl::transpose::nontrans;
      oneapi::mkl::transpose opB = oneapi::mkl::transpose::nontrans;
      oneapi::mkl::sparse::set_matmat_data(descr, viewA, opA, viewB, opB, viewC);

      // Following steps in the "simplified workflow" portion of:
      // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/oneapi-mkl-sparse-matmat.html
      std::int64_t *sizeTempBuffer;
      void *tempBuffer; 
      oneapi::mkl::sparse::matmat(*queue->sycl_stream(), A_handle, B_handle,
		                 C_handle, 
				 oneapi::mkl::sparse::matmat_request::work_estimation, 
                                 descr, sizeTempBuffer, tempBuffer, {});
      oneapi::mkl::sparse::matmat(*queue->sycl_stream(), A_handle, B_handle,
		                 C_handle, 
				 oneapi::mkl::sparse::matmat_request::compute,
                                 descr, sizeTempBuffer, tempBuffer, {});
      // NNZ will be in in sizeTempBuffer (host)
      oneapi::mkl::sparse::matmat(*queue->sycl_stream(), A_handle, B_handle,
		                 C_handle, 
				 oneapi::mkl::sparse::matmat_request::get_nnz,
                                 descr, sizeTempBuffer, tempBuffer, {});
      std::int64_t nnz = *sizeTempBuffer;
      C.nnz = nnz;
      CHECK( magma_index_malloc( &C.dcol, C.nnz ));
      CHECK( magma_zmalloc( &C.dval, C.nnz ));
      oneapi::mkl::sparse::matmat(*queue->sycl_stream(), A_handle, B_handle,
		                 C_handle, 
				 oneapi::mkl::sparse::matmat_request::finalize,
                                 descr, sizeTempBuffer, tempBuffer, {});

      magma_queue_sync( queue );
      CHECK( magma_zmtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
cleanup:
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &A_handle);
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &B_handle);
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &C_handle);
    oneapi::mkl::sparse::release_matmat_descr(&descr);      
    magma_zmfree( &C, queue );
    return info;
}
