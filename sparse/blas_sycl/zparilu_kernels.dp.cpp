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

#define PRECISION_z


void 
magma_zparilu_csr_kernel(  
    const magma_int_t num_rows, 
    const magma_int_t nnz,  
    const magma_index_t *rowidxA, 
    const magma_index_t *colidxA,
    const magmaDoubleComplex * __restrict__ A, 
    const magma_index_t *rowptrL, 
    const magma_index_t *colidxL, 
    magmaDoubleComplex *valL, 
    const magma_index_t *rowptrU, 
    const magma_index_t *colidxU, 
    magmaDoubleComplex *valU,
    sycl::nd_item<3> item_ct1)
{
    int i, j;
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;
    if (k < nnz) {
       i = rowidxA[k];
        j = colidxA[k];
        s =  A[k];
        il = rowptrL[i];
        iu = rowptrU[j];

        while (il < rowptrL[i+1] && iu < rowptrU[j+1]) {
            sp = zero;
            jl = colidxL[il];
            ju = colidxU[iu];
            sp = (jl == ju) ? valL[il] * valU[iu] : sp;
            s = (jl == ju) ? s-sp : s;
            il = (jl <= ju) ? il+1 : il;
            iu = (jl >= ju) ? iu+1 : iu;
        }
        s += sp;  // undo the last operation (it must be the last)        
        if (i > j)      // modify l entry
            valL[il-1] =  s / valU[rowptrU[j+1]-1];
        else            // modify u entry
            valU[iu-1] = s;
    }
}




/**
    Purpose
    -------
    
    This routine iteratively computes an incomplete LU factorization.
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    This routine was used in the ISC 2015 paper:
    E. Chow et al.: "Asynchronous Iterative Algorithm for Computing Incomplete
                     Factorizations on GPUs", 
                     ISC High Performance 2015, LNCS 9137, pp. 1-16, 2015.
 
    The input format of the system matrix is COO, the lower triangular factor L 
    is stored in CSR, the upper triangular factor U is transposed, then also 
    stored in CSR (equivalent to CSC format for the non-transposed U).
    Every component of L and U is handled by one thread. 

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A determing initial guess & processing order

    @param[in,out]
    L           magma_z_matrix
                input/output matrix L containing the lower triangular factor 

    @param[in,out]
    U           magma_z_matrix
                input/output matrix U containing the upper triangular factor
                              
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zparilu_csr(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(A.nnz, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zparilu_csr_kernel(
                               A.num_rows, A.nnz, A.rowidx, A.col, A.val, L.row,
                               L.col, L.val, U.row, U.col, U.val, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
