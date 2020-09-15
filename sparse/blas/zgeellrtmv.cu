/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
__global__ void 
zgeellrtmv_kernel_32( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment )
{
    int idx = blockIdx.y * gridDim.x * blockDim.x + 
              blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    int idb = threadIdx.x;   // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index
    
    extern __shared__ magmaDoubleComplex shared[];

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 16 ) {
            shared[idb] += shared[idb+16];
            if ( idp < 8 ) shared[idb] += shared[idb+8];
            if ( idp < 4 ) shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
__global__ void 
zgeellrtmv_kernel_16( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment )
{
    int idx = blockIdx.y * gridDim.x * blockDim.x + 
              blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    int idb = threadIdx.x;   // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index
    
    extern __shared__ magmaDoubleComplex shared[];

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 8 ) {
            shared[idb] += shared[idb+8];
            if ( idp < 4 ) shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}

//F. Vazquez, G. Ortega, J.J. Fernandez, E.M. Garzon, Almeria University
__global__ void 
zgeellrtmv_kernel_8( 
    int num_rows, 
    int num_cols,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowlength,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    int T,
    int alignment )
{
    int idx = blockIdx.y * gridDim.x * blockDim.x + 
              blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
    int idb = threadIdx.x;   // local thread index
    int idp = idb%T;  // number of threads assigned to one row
    int i = idx/T;  // row index
    
    extern __shared__ magmaDoubleComplex shared[];

    if (i < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = magma_ceildiv( drowlength[i], T );  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            // original code in paper (not working for me)
            //magmaDoubleComplex val = dval[ k*(T*alignment)+(i*T)+idp ];  
            //int col = dcolind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = dval[ k*(T)+(i*alignment)+idp ];
            int col = dcolind [ k*(T)+(i*alignment)+idp ];

            dot += val * dx[ col ];
        }
        shared[idb]  = dot;
        if ( idp < 4 ) {
            shared[idb] += shared[idb+4];
            if ( idp < 2 ) shared[idb] += shared[idb+2];
            if ( idp == 0 ) {
                dy[i] = (shared[idb]+shared[idb+1])*alpha + beta*dy [i];
            }
        }
    }
}



/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLRT. The ideas are taken from 
    "Improving the performance of the sparse matrix
    vector product with GPUs", (CIT 2010), 
    and modified to provide correct values.

    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A
    @param[in]
    m           magma_int_t
                number of rows 

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    nnz_per_row magma_int_t
                max number of nonzeros in a row

    @param[in]
    alpha       magmaDoubleComplex
                scalar alpha

    @param[in]
    dval        magmaDoubleComplex_ptr
                val array

    @param[in]
    dcolind     magmaIndex_ptr
                col indices  

    @param[in]
    drowlength  magmaIndex_ptr
                number of elements in each row

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar beta

    @param[out]
    dy          magmaDoubleComplex_ptr
                output vector y

    @param[in]
    blocksize   magma_int_t
                threads per block

    @param[in]
    alignment   magma_int_t
                threads assigned to each row

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
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
    magma_int_t alignment,
    magma_int_t blocksize,
    magma_queue_t queue )
{
    int num_blocks = magma_ceildiv( m, blocksize );

    magma_int_t num_threads = alignment*blocksize;
    magma_int_t threads = alignment*blocksize;

    int real_row_length = magma_roundup( nnz_per_row, alignment );

    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");

    int dimgrid1 = int( sqrt( double( num_blocks )));
    int dimgrid2 = magma_ceildiv( num_blocks, dimgrid1 );
    dim3 grid( dimgrid1, dimgrid2, 1);

    int Ms = alignment * blocksize * sizeof( magmaDoubleComplex );
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads, Ms);

    if ( alignment == 32 ) {
        zgeellrtmv_kernel_32<<< grid, threads, Ms, queue->cuda_stream() >>>
                 ( m, n, alpha, dval, dcolind, drowlength, dx, beta, dy, 
                                                 alignment, real_row_length );
    }
    else if ( alignment == 16 ) {
        zgeellrtmv_kernel_16<<< grid, threads, Ms, queue->cuda_stream() >>>
                 ( m, n, alpha, dval, dcolind, drowlength, dx, beta, dy, 
                                                 alignment, real_row_length );
    }
    else if ( alignment == 8 ) {
        zgeellrtmv_kernel_8<<< grid, threads, Ms, queue->cuda_stream() >>>
                 ( m, n, alpha, dval, dcolind, drowlength, dx, beta, dy, 
                                                 alignment, real_row_length );
    }
    else {
        printf("error: alignment %d not supported.\n", int(alignment) );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

    return MAGMA_SUCCESS;
}
