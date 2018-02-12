/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"
#include <cuda_runtime.h>
#define PRECISION_z


__global__ void 
magma_zget_row_ptr_kernel(   
    const magma_int_t num_rows, 
    magma_int_t* nnz,  
    const magma_index_t* __restrict__ rowidx, 
    magma_index_t* rowptr) 
{
    //int i, j;
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    //int nnz;
    /*magma_int_t nnz_per_row;
    if(k<num_rows){
#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        nnz_per_row =  __ldg( rowidx + k );
#else
        nnz_per_row =  rowidx[k];
#endif
        atomicAdd(&nnz,nnz_per_row);
    }
    if (k < 2)
    {
        if(k==1)

            {    
                rowptr[0] = 0;
                rowptr[1] = rowidx[0];
                for(int iter=2;iter<(num_rows+1)/2;++iter){
                    rowptr[iter] = rowptr[iter-1]+rowidx[iter-1];
                }           
            }
        else{
                rowptr[num_rows] = nnz;
                for(int iter=num_rows-1;iter>(num_rows+1)/2;iter--){
                    rowptr[iter] = rowptr[iter+1]-rowidx[iter];
                }           
        }
    }
    */
    //naive implementation for now.
        if (k==1) {    
                rowptr[0] = 0;
                for(int iter=1;iter<=num_rows;++iter){
                    rowptr[iter] = rowptr[iter-1]+rowidx[iter-1];
                }
                nnz[0] = rowptr[num_rows];
        }
} //kernel


extern "C" magma_int_t
magma_zget_row_ptr(
    const magma_int_t num_rows,
    magma_int_t *nnz,
    const magma_index_t* rowidx,
    magma_index_t* rowptr,
    magma_queue_t queue)
{
    /*
    int blocksize = 128;
    int gridsize = magma_ceildiv(num_rows, blocksize);
    magma_int_t *nnz_dev, *tnnz;
    magma_imalloc(&nnz_dev, 1);
    magma_imalloc_cpu(&tnnz, 1);
    
    dim3 block(blocksize,1,1);
    dim3 grid(gridsize,1,1);
    magma_zget_row_ptr_kernel<<<grid, block, 0, queue->cuda_stream()>>>
        (num_rows, nnz_dev, rowidx, rowptr);
        
    magma_igetvector(1,nnz_dev,1,tnnz,1,queue);
    *nnz = tnnz[0];

    
    magma_free(nnz_dev);
    magma_free_cpu(tnnz);
    */
    
    magma_index_t *hrowidx, *hrowptr;
    magma_index_malloc_cpu(&hrowidx, num_rows);
    magma_index_malloc_cpu(&hrowptr, num_rows+1);
    magma_index_getvector(num_rows,rowidx,1,hrowidx,1,queue);

    hrowptr[0] = 0;
    for(int iter=1;iter<=num_rows;++iter){
        hrowptr[iter] = hrowptr[iter-1]+hrowidx[iter-1];
    }
    
    *nnz = hrowptr[num_rows];
    
    magma_index_setvector(num_rows+1,hrowptr,1,rowptr,1,queue);
    
    magma_free_cpu(hrowidx);
    magma_free_cpu(hrowptr);
    
    return MAGMA_SUCCESS;
}




