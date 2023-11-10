/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define PRECISION_z
#define COMPLEX
#ifdef COMPLEX
#include <complex>
#endif

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#ifdef MAGMA_HAVE_HIP
  #define hipblasDoubleComplex hipDoubleComplex
#endif



void 
magma_zvalinit_kernel(  
    const magma_int_t num_el, 
    magmaDoubleComplex_ptr dval,
    sycl::nd_item<3> item_ct1) 
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    if (k < num_el) {
        dval[k] = zero;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dval        magmaDoubleComplex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zvalinit_gpu(
    magma_int_t num_el,
    magmaDoubleComplex_ptr dval,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zvalinit_kernel(num_el, dval, item_ct1);
                       });

    return MAGMA_SUCCESS;
}




void 
magma_zindexinit_kernel(  
    const magma_int_t num_el, 
    magmaIndex_ptr dind,
    sycl::nd_item<3> item_ct1) 
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (k < num_el) {
        dind[k] = 0;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dind        magmaIndex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zindexinit_gpu(
    magma_int_t num_el,
    magmaIndex_ptr dind,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zindexinit_kernel(num_el, dind, item_ct1);
                       });

    return MAGMA_SUCCESS;
}


void 
magma_zmatrixcup_count(  
    const magma_int_t num_rows,
    const magma_index_t* A_row,
    const magma_index_t* A_col,
    const magma_index_t* B_row,
    const magma_index_t* B_col,
    magma_index_t* inserted,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    if (row < num_rows) {
        int add = 0;
        int a = A_row[row];
        int b = B_row[row];
        int enda = A_row[ row+1 ];
        int endb = B_row[ row+1 ]; 
        int acol;
        int bcol;
        if (a<enda && b<endb) {
            do{
                acol = A_col[ a ];
                bcol = B_col[ b ];
                
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    add++;
                    a++;
                }
                else {
                    add++;
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                add++;
                b++;
            }while(b<endb);            
        }
        inserted[ row ] = add; 
    }
}


void 
magma_zmatrixcup_fill(  
    const magma_int_t num_rows,
    const magma_index_t* A_row,
    const magma_index_t* A_col,
    const magmaDoubleComplex* A_val,
    const magma_index_t* B_row,
    const magma_index_t* B_col,
    const magmaDoubleComplex* B_val,
    magma_index_t* U_row,
    magma_index_t* U_rowidx,
    magma_index_t* U_col,
    magmaDoubleComplex* U_val,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    if (row < num_rows) {
        int add = 0;
        int offset = U_row[row];
        int a = A_row[row];
        int b = B_row[row];
        int enda = A_row[ row+1 ];
        int endb = B_row[ row+1 ]; 
        int acol;
        int bcol;
        if (a<enda && b<endb) {
            do{
                acol = A_col[ a ];
                bcol = B_col[ b ];
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    U_col[ offset + add ] = acol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = A_val[ a ];
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    U_col[ offset + add ] = acol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = A_val[ a ];
                    add++;
                    a++;
                }
                else {
                    U_col[ offset + add ] = bcol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = B_val[ b ];
                    add++;
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A_col[ a ];
                U_col[ offset + add ] = acol;
                U_rowidx[ offset + add ] = row;
                U_val[ offset + add ] = A_val[ a ];
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                bcol = B_col[ b ];
                U_col[ offset + add ] = bcol;
                U_rowidx[ offset + add ] = row;
                U_val[ offset + add ] = B_val[ b ];
                add++;
                b++;
            }while(b<endb);            
        }
    }
}
    
    
    

/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.
    
    This is the GPU version of the operation.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in]
    B           magma_z_matrix
                Input matrix 2.

    @param[out]
    U           magma_z_matrix*
                U = A \cup B. If both matrices have a nonzero value 
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_cup_gpu(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    assert(A.num_rows == B.num_rows);
    magma_int_t num_rows = A.num_rows;
    U->num_rows = num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_DEV;
   

    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid11 = magma_ceildiv(num_rows, blocksize1 );
    int dimgrid12 = 1;
    int dimgrid13 = 1;
    sycl::range<3> grid1(dimgrid13, dimgrid12, dimgrid11);
    sycl::range<3> block1(1, blocksize2, blocksize1);

    magmaIndex_ptr inserted = NULL;
    CHECK(magma_index_malloc(&U->drow, num_rows+1));
    CHECK(magma_index_malloc(&inserted, num_rows));
    CHECK(magma_zindexinit_gpu(num_rows, inserted, queue));

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid1 * block1, block1),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zmatrixcup_count(num_rows, A.drow, A.dcol,
                                                  B.drow, B.dcol, inserted,
                                                  item_ct1);
                       });

    CHECK(magma_zget_row_ptr(num_rows, &U->nnz, inserted, U->drow, queue));
    

    CHECK(magma_zmalloc(&U->dval, U->nnz));
    CHECK(magma_index_malloc(&U->drowidx, U->nnz));
    CHECK(magma_index_malloc(&U->dcol, U->nnz));

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto U_drow_ct7 = U->drow;
        auto U_drowidx_ct8 = U->drowidx;
        auto U_dcol_ct9 = U->dcol;
        auto U_dval_ct10 = U->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid1 * block1, block1),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zmatrixcup_fill(
                                 num_rows, A.drow, A.dcol, A.dval, B.drow,
                                 B.dcol, B.dval, U_drow_ct7, U_drowidx_ct8,
                                 U_dcol_ct9, U_dval_ct10, item_ct1);
                         });
    });

cleanup:
    magma_free(inserted);
    return info;
}





/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.
    
    This is the GPU version of the operation.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in]
    B           magma_z_matrix
                Input matrix 2.

    @param[out]
    U           magma_z_matrix*
                U = A \cup B. If both matrices have a nonzero value 
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t magma_zcsr_sort_gpu(magma_z_matrix *A,
                                           magma_queue_t queue) {
    magma_int_t info = 0;
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&handle);

    oneapi::mkl::sparse::set_csr_data(*queue->sycl_stream(), handle,
		     A->num_rows, A->num_cols,
                     oneapi::mkl::index_base::zero, A->drow, A->dcol, A->dval);

    oneapi::mkl::sparse::sort_matrix(*queue->sycl_stream(), handle, {}); 
    
cleanup:
    oneapi::mkl::sparse::release_matrix_handle(*queue->sycl_stream(), &handle);

    return info;
}
