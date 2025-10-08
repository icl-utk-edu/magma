/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define PRECISION_z

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }



__global__ void 
magma_zvalinit_kernel(  
    const magma_int_t num_el, 
    magmaDoubleComplex_ptr dval) 
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
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
    dim3 grid(dimgrid1, dimgrid2, dimgrid3);
    dim3 block(blocksize1, blocksize2, 1);
    magma_zvalinit_kernel<<< grid, block, 0, queue->cuda_stream() >>>
        (num_el, dval);

    return MAGMA_SUCCESS;
}




__global__ void 
magma_zindexinit_kernel(  
    const magma_int_t num_el, 
    magmaIndex_ptr dind) 
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
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
    dim3 grid(dimgrid1, dimgrid2, dimgrid3);
    dim3 block(blocksize1, blocksize2, 1);
    magma_zindexinit_kernel<<< grid, block, 0, queue->cuda_stream() >>>
        (num_el, dind);

    return MAGMA_SUCCESS;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a matrix $U = A \cup B$. If both matrices have a nonzero value
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
                $U = A \cup B$. If both matrices have a nonzero value
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zcsr_sort_gpu(
    magma_z_matrix *A,
    magma_queue_t queue)
{   
    magma_int_t info = 0;
    cusparseHandle_t handle=NULL;
    cusparseMatDescr_t descrA=NULL;
    
    magmaDoubleComplex_ptr tmp=NULL, csrVal_sorted=NULL;
    char* pBuffer;
    int *P;
    size_t pBufferSizeInBytes;
    
    CHECK_CUSPARSE( cusparseCreate( &handle ));
    CHECK_CUSPARSE( cusparseSetStream( handle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( cusparseSetMatType( descrA, 
        CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, 
        CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, 
        CUSPARSE_INDEX_BASE_ZERO ));
    
    CHECK(magma_zmalloc(&csrVal_sorted, A->nnz));
   
    // step 1: allocate buffer
    cusparseXcsrsort_bufferSizeExt(handle, A->num_rows, A->num_cols, 
        A->nnz, A->drow, A->dcol, &pBufferSizeInBytes);
    cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);
    
    // step 2: setup permutation vector P to identity
    cudaMalloc( (void**)&P, sizeof(int)*A->nnz);
    cusparseCreateIdentityPermutation(handle, A->nnz, P);
    
    // step 3: sort CSR format
    cusparseXcsrsort(handle, A->num_rows, A->num_cols, A->nnz, 
        descrA, A->drow, A->dcol, P, pBuffer);
    
    // step 4: gather sorted csrVal
#if CUDA_VERSION >= 12000
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, A->nnz, A->nnz,
                                        P, csrVal_sorted,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, A->nnz, A->dval, CUDA_C_64F) );
    CHECK_CUSPARSE( cusparseGather(handle, vec_values, vec_permutation) );
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) );
#else
    cusparseZgthr(handle, A->nnz, (cuDoubleComplex*)A->dval, (cuDoubleComplex*)csrVal_sorted, P, 
        CUSPARSE_INDEX_BASE_ZERO);
#endif

    SWAP(A->dval, csrVal_sorted);
    
cleanup:
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( handle );
    magma_free(csrVal_sorted);

    return info;
}
