/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#include <cuda_runtime.h>

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }
#define BLOCK_SIZE 128


__global__ void
zcompute_newval_kernel(
    magma_int_t num_rows,
    magma_index_t* Arow,
    magma_index_t* Brow,
    magma_index_t* Acol,
    magma_index_t* Browidx,
    magma_index_t* Bcol,
    magmaDoubleComplex* Aval,
    magmaDoubleComplex* Bval)
{
    int tidx = blockIdx.x*blockDim.x+threadIdx.x;
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
__global__ void
zcompute_nnz_kernel(
    magma_int_t num_rows,
    magma_index_t* Arow,
    magma_index_t* Brow,
    magma_index_t* Acol,
    magmaDoubleComplex* Aval,
    double thrs)
{
    int row= blockIdx.x*blockDim.x+threadIdx.x;
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

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(num_blocks, 1, 1 );
    
    magma_index_malloc(&new_rownnz,A->num_rows);
    magma_index_malloc(&B.drow,A->num_rows+1);

    zcompute_nnz_kernel<<<grid, block, 0, queue->cuda_stream()>>>
        (A->num_rows, A->drow, new_rownnz, A->dcol, A->dval,*thrs);

    magma_zget_row_ptr(A->num_rows, &B.nnz, new_rownnz, B.drow, queue); 
    magma_zmalloc(&B.dval,B.nnz);
    magma_index_malloc(&B.rowidx,B.nnz);
    magma_index_malloc(&B.dcol,B.nnz);
    zcompute_newval_kernel<<<grid, block, 0, queue->cuda_stream()>>>
        (A->num_rows, A->drow, B.drow, A->dcol,B.drowidx, B.dcol, A->dval, B.dval); 
   
    //Rewrite the matrix with all the new values
    magma_zmatrix_swap(&B, A, queue);
    
    magma_zmfree(&B, queue);
    magma_free(new_rownnz);
    return info;
}
