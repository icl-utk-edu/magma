/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_z

#define  Ablockinfo(i,j)  Ablockinfo[(i)*c_blocks   + (j)]
#define  Bblockinfo(i,j)  Bblockinfo[(i)*c_blocks   + (j)]
#define A(i,j) ((Ablockinfo(i,j)-1)*size_b*size_b)
#define B(i,j) ((Bblockinfo(i,j)-1)*size_b*size_b)

//============================================================

#define ldb m
#define lda m
#define ldc m


// every multiprocessor handles one BCSR-block
void 
zbcsrlupivloc_kernel( 
    int size_b,
    int kblocks,   
    double **A, 
    magma_int_t *ipiv,
    sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_group(2) < kblocks) {
        if (item_ct1.get_local_id(2) < size_b) {
            for( int i=0; i<size_b; i++){
                int dst = ipiv[i]-1;
                if( dst != i ){
                    double *A1 = A[item_ct1.get_group(2)] +
                                 item_ct1.get_local_id(2) * size_b + i;
                    double *A2 = A[item_ct1.get_group(2)] +
                                 item_ct1.get_local_id(2) * size_b + dst;
                    double tmp = *A2;
                    *A2 = *A1;
                    *A1 = tmp;
                }               
            }
        }
    }
}


/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine updates all blocks in
    the trailing matrix.
    
    Arguments
    ---------

    @param[in]
    size_b      magma_int_t
                blocksize in BCSR
    
    @param[in]
    kblocks     magma_int_t
                number of blocks
                
    @param[in]
    dA          magmaDoubleComplex_ptr *
                matrix in BCSR

    @param[in]
    ipiv        magmaInt_ptr
                array containing pivots
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrlupivloc(
    magma_int_t size_b, 
    magma_int_t kblocks,
    magmaDoubleComplex_ptr *dA,  
    magmaInt_ptr ipiv,
    magma_queue_t queue )
{
    #if defined(PRECISION_d)
    sycl::range<3> threads(1, 1, 64);

    sycl::range<3> grid(1, 1, kblocks);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zbcsrlupivloc_kernel(size_b, kblocks, dA, ipiv,
                                                item_ct1);
			   });
    #endif

    return MAGMA_SUCCESS;
}
