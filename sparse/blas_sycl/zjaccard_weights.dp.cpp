/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"
#include <cmath>

#define PRECISION_z


void 
magma_zjaccardweights_kernel(   
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magma_int_t nnzJ,  
    magma_index_t *rowidxJ, 
    magma_index_t *colidxJ,
    magmaDoubleComplex *valJ, 
    magma_index_t *rowptrA, 
    magma_index_t *colidxA, 
    magmaDoubleComplex *valA ,
    sycl::nd_item<3> item_ct1) {
    int i, j;
    int k = item_ct1.get_local_range(2) * item_ct1.get_group_range(2) *
                item_ct1.get_group(1) +
            item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    /*
    DPCT1064:476: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex sum_i, sum_j, intersect;
    int il, iu, jl, ju;
    

    if (k < nnzJ)
    {
        i = rowidxJ[k];
        j = colidxJ[k];
        if( i != j ){
            il = rowptrA[i];
            iu = rowptrA[j];
            
            sum_i = zero;
            sum_j = zero;
            intersect = zero;

            /*
            DPCT1064:477: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            sum_i = MAGMA_Z_MAKE((double)rowptrA[i + 1] - rowptrA[i], 0.0);
            /*
            DPCT1064:478: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            sum_j = MAGMA_Z_MAKE((double)rowptrA[j + 1] - rowptrA[j], 0.0);

            while (il < rowptrA[i+1] && iu < rowptrA[j+1])
            {
            
                jl = colidxJ[il];
                ju = rowidxJ[iu];
            
                // avoid branching
                // if there are actual values:
                // intersect = ( jl == ju ) ? valJ[il] * valJ[iu] : sp;
                // else
                intersect = ( jl == ju ) ? intersect + one : intersect;
                il = ( jl <= ju ) ? il+1 : il;
                iu = ( ju <= jl ) ? iu+1 : iu;
            }

            /*
            DPCT1064:479: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            valJ[k] = MAGMA_Z_MAKE(
                std::real(intersect) / std::real(sum_i + sum_j - intersect), 0.0);
        } else {
            valJ[k] = MAGMA_Z_ONE;
        }
            
    }
}// end kernel 

/**
    Purpose
    -------

    Computes Jaccard weights for a matrix

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix

    @param[out]
    J           magma_z_matrix*
                Jaccard weights
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zjaccard_weights(
    magma_z_matrix A,
    magma_z_matrix *J,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t m = J->num_rows;
    magma_int_t n = J->num_rows;
    magma_int_t nnz = J->nnz;
    
    int blocksize1 = 32;
    int blocksize2 = 1;

    int dimgrid1 = sqrt( magma_ceildiv( nnz, blocksize1 ) );
    int dimgrid2 = magma_ceildiv(nnz, blocksize1*dimgrid1);
    int dimgrid3 = 1;
    // printf("thread block: ( %d x %d  ) x [%d x %d]\n", blocksize1, blocksize2, dimgrid1, dimgrid2);

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    // (spaces are added to prevent expansion from the script from messing up)
    // cudaFunc Set CacheConfig(cudaFuncCache PreferShared);

    /*
    DPCT1026:480: The call to cudaDeviceSetCacheConfig was removed because DPC++
    currently does not support setting cache config on devices.
    */

    sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
    sycl::range<3> block(1, blocksize2, blocksize1);

    /*
    DPCT1049:481: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto J_drowidx_ct3 = J->drowidx;
        auto J_dcol_ct4 = J->dcol;
        auto J_dval_ct5 = J->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zjaccardweights_kernel(
                                 m, n, nnz, J_drowidx_ct3, J_dcol_ct4,
                                 J_dval_ct5, A.drow, A.dcol, A.dval, item_ct1);
                         });
    });

    return info;
}
