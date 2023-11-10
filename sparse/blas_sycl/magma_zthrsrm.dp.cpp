/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }
#define BLOCK_SIZE 128


void
zcompute_newval_kernel(
    magma_int_t num_rows,
    magma_index_t* Arow,
    magma_index_t* Brow,
    magma_index_t* Acol,
    magma_index_t* Browidx,
    magma_index_t* Bcol,
    magmaDoubleComplex* Aval,
    magmaDoubleComplex* Bval,
    sycl::nd_item<3> item_ct1)
{
    int tidx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    magma_index_t offset_new, offset_old, end_old;
    
    if (tidx < num_rows) {
        magma_int_t count = 0;
    
        offset_old = Arow[tidx];
        offset_new = Brow[tidx];
        end_old = Arow[tidx+1];
        
        for (int i = offset_old; i < end_old; i++) {
            if(Acol[i]>-1){
                Bcol[offset_new+count] = Acol[i];
                Bval[offset_new+count] = Aval[i];
                Browidx[offset_new + count] = tidx;
                count++;
            }
        }
    }
}

//kernel
void
zcompute_nnz_kernel(
    magma_int_t num_rows,
    magma_index_t* Arow,
    magma_index_t* Brow,
    magma_index_t* Acol,
    magmaDoubleComplex* Aval,
    double thrs,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (row < num_rows) {
        magma_int_t rm = 0;
        magma_int_t el = 0;
        
        for (int i = Arow[row]; i<Arow[row+1]; i++) {
            if (MAGMA_Z_ABS(Aval[i]) <= thrs ) {
                if (Acol[i] != row) {
                    Acol[i] = -1;//cheaperthanval
                    rm++;
                } else {
                    el++;
                }
            } else {
                el++;
            }
        }
        Brow[row] = el;
    }
}



/**
    Purpose
-------
    
    This routine selects a threshold separating the subset_size smallest
    magnitude elements from the rest.
    
    Arguments
    ---------
                
    @param[in]
    order       magma_int_t 
                dummy variable for now.
                
    @param[in,out]
    A           magma_z_matrix*  
                input/output matrix where elements are removed

    @param[out]
    thrs        double*  
                computed threshold

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/
extern "C" magma_int_t
magma_zthrsholdrm_gpu(
    magma_int_t order,
    magma_z_matrix* A,
    double* thrs,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    magma_int_t num_blocks = magma_ceildiv(A->num_rows,BLOCK_SIZE);
    magma_z_matrix B={Magma_CSR};
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.storage_type = A->storage_type;
    B.memory_location = Magma_DEV;
    
    magma_index_t *new_rownnz={NULL};

    sycl::range<3> block(1, 1, BLOCK_SIZE);
    sycl::range<3> grid(1, 1, num_blocks);

    magma_index_malloc(&new_rownnz,A->num_rows);
    magma_index_malloc(&B.drow,A->num_rows+1);

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct1 = A->drow;
        auto A_dcol_ct3 = A->dcol;
        auto A_dval_ct4 = A->dval;
        auto thrs_ct5 = *thrs;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             zcompute_nnz_kernel(
                                 A_num_rows_ct0, A_drow_ct1, new_rownnz,
                                 A_dcol_ct3, A_dval_ct4, thrs_ct5, item_ct1);
                         });
    });

    magma_zget_row_ptr(A->num_rows, &B.nnz, new_rownnz, B.drow, queue); 
    magma_zmalloc(&B.dval,B.nnz);
    magma_index_malloc(&B.rowidx,B.nnz);
    magma_index_malloc(&B.dcol,B.nnz);
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct1 = A->drow;
        auto A_dcol_ct3 = A->dcol;
        auto A_dval_ct6 = A->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             zcompute_newval_kernel(
                                 A_num_rows_ct0, A_drow_ct1, B.drow, A_dcol_ct3,
                                 B.drowidx, B.dcol, A_dval_ct6, B.dval,
                                 item_ct1);
                         });
    });

    //Rewrite the matrix with all the new values
    magma_zmatrix_swap(&B, A, queue);
    
    magma_zmfree(&B, queue);
    magma_free(new_rownnz);
    return info;
}
