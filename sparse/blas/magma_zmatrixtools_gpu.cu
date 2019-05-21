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


__global__ void 
magma_zmatrixcup_count(  
    const magma_int_t num_rows,
    const magma_index_t* A_row,
    const magma_index_t* A_col,
    const magma_index_t* B_row,
    const magma_index_t* B_col,
    magma_index_t* inserted)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
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


__global__ void 
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
    magmaDoubleComplex* U_val)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
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
    dim3 grid1(dimgrid11, dimgrid12, dimgrid13 );
    dim3 block1(blocksize1, blocksize2, 1 );
    
    magmaIndex_ptr inserted = NULL;
    CHECK(magma_index_malloc(&U->drow, num_rows+1));
    CHECK(magma_index_malloc(&inserted, num_rows));
    CHECK(magma_zindexinit_gpu(num_rows, inserted, queue));
    
    magma_zmatrixcup_count<<<grid1, block1, 0, queue->cuda_stream()>>>
        (num_rows, A.drow, A.dcol, B.drow, B.dcol, inserted);
    
    CHECK(magma_zget_row_ptr(num_rows, &U->nnz, inserted, U->drow, queue));
        
    CHECK(magma_zmalloc(&U->dval, U->nnz));
    CHECK(magma_index_malloc(&U->drowidx, U->nnz));
    CHECK(magma_index_malloc(&U->dcol, U->nnz));
    
    magma_zmatrixcup_fill<<<grid1, block1, 0, queue->cuda_stream()>>>
        (num_rows, A.drow, A.dcol, A.dval, B.drow, B.dcol, B.dval,
        U->drow, U->drowidx, U->dcol, U->dval);
    
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
    cusparseZgthr(handle, A->nnz, A->dval, csrVal_sorted, P, 
        CUSPARSE_INDEX_BASE_ZERO);
    
    SWAP(A->dval, csrVal_sorted);
    
cleanup:
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( handle );
    magma_free(csrVal_sorted);

    return info;
}