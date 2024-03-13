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

#define BLOCK_SIZE 256


// kernel
void 
zdiagcheck_kernel( 
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex_ptr dval, 
    magmaIndex_ptr drowptr, 
    magmaIndex_ptr dcolind,
    magma_int_t * dinfo ,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int j;

    if(row<num_rows){
        int localinfo = 1;
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        // check whether there exists a nonzero diagonal entry
        for( j=start; j<end; j++){
            /*
            DPCT1064:802: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            if ((dcolind[j] == row) && (dval[j] != MAGMA_Z_ZERO)) {
                localinfo = 0;
            }
        }
        // set flag to 1
        if( localinfo == 1 ){
            dinfo[0] = -3009;
        }
    }
}



/**
    Purpose
    -------
    
    This routine checks for a CSR matrix whether there 
    exists a zero on the diagonal. This can be the diagonal entry missing
    or an explicit zero.
    
    Arguments
    ---------
                
    @param[in]
    dA          magma_z_matrix
                matrix in CSR format

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zdiagcheck(
    magma_z_matrix dA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t *hinfo = NULL;
    
    magma_int_t * dinfo = NULL;
    sycl::range<3> grid(1, 1, magma_ceildiv(dA.num_rows, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;
    
    CHECK( magma_imalloc( &dinfo, 1 ) );
    CHECK( magma_imalloc_cpu( &hinfo, 1 ) );
    hinfo[0] = 0;
    magma_isetvector( 1, hinfo, 1, dinfo, 1, queue );
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           zdiagcheck_kernel(dA.num_rows, dA.num_cols, dA.dval,
                                             dA.drow, dA.dcol, dinfo, item_ct1);
                       });
    info = hinfo[0];
    magma_igetvector( 1, dinfo, 1, hinfo, 1, queue ); 
    info = hinfo[0];
    
cleanup:
    magma_free( dinfo );
    magma_free_cpu( hinfo );

    return info;
}
