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
magma_zparic_csr_kernel(    
    magma_int_t n, 
    magma_int_t nnz, 
    magma_index_t *Arowidx, 
    magma_index_t *Acolidx, 
    const magmaDoubleComplex * __restrict__  A_val,
    magma_index_t *rowptr, 
    magma_index_t *colidx, 
    magmaDoubleComplex *val ,
    sycl::nd_item<3> item_ct1)
{
    int i, j;
    int k = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
             item_ct1.get_local_id(2)); // % nnz;
    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;
    if ( k < nnz ) {     
        i = Arowidx[k];
        j = Acolidx[k];
        s = A_val[k];
        il = rowptr[i];
        iu = rowptr[j];
        while (il < rowptr[i+1] && iu < rowptr[j+1]) {
            sp = zero;
            jl = colidx[il];
            ju = colidx[iu];
            if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else {
                // we are going to modify this u entry
                sp = val[il] * val[iu];
                s -= sp;
                il++;
                iu++;
            }
        }
        s += sp; // undo the last operation (it must be the last)
        // modify entry
        if (i == j) // diagonal
            /*
            DPCT1064:154: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            val[il - 1] = MAGMA_Z_MAKE(sycl::sqrt(sycl::fabs(std::real(s))), 0.0);
        else  //sub-diagonal
            val[il-1] =  s / val[iu-1];
    }
}// kernel 


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
                     
    The input format of the initial guess matrix A is Magma_CSRCOO,
    A_CSR is CSR or CSRCOO format. 

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A - initial guess (lower triangular)

    @param[in,out]
    A_CSR       magma_z_matrix
                input/output matrix containing the IC approximation
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zparic_csr( 
    magma_z_matrix A,
    magma_z_matrix A_CSR,
    magma_queue_t queue )
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv( A.nnz, blocksize1 );
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zparic_csr_kernel(
                               A.num_rows, A.nnz, A.rowidx, A.col, A.val,
                               A_CSR.row, A_CSR.col, A_CSR.val, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
