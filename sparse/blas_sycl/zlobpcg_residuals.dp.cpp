/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define PRECISION_z


// copied from dznrm2.cu in trunk/magmablas
// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
void sum_reduce( /*int n,*/ int i, magmaDouble_ptr  x ,
                            sycl::nd_item<3> item_ct1)
{
    /*
    DPCT1065:330: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    /*
    DPCT1065:331: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (i < 1024 && i + 1024 < n) {
            x[i] += x[i + 1024];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:332: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 512) {
        if (i < 512 && i + 512 < n) {
            x[i] += x[i + 512];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:333: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 256) {
        if (i < 256 && i + 256 < n) {
            x[i] += x[i + 256];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:334: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 128) {
        if (i < 128 && i + 128 < n) {
            x[i] += x[i + 128];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:335: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 64) {
        if (i < 64 && i + 64 < n) {
            x[i] += x[i + 64];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:336: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 32) {
        if (i < 32 && i + 32 < n) {
            x[i] += x[i + 32];
        } item_ct1.barrier();
    }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    /*
    DPCT1065:337: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 16) {
        if (i < 16 && i + 16 < n) {
            x[i] += x[i + 16];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:338: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 8) {
        if (i < 8 && i + 8 < n) {
            x[i] += x[i + 8];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:339: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 4) {
        if (i < 4 && i + 4 < n) {
            x[i] += x[i + 4];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:340: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 2) {
        if (i < 2 && i + 2 < n) {
            x[i] += x[i + 2];
        } item_ct1.barrier();
    }
    /*
    DPCT1065:341: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    if (n > 1) {
        if (i < 1 && i + 1 < n) {
            x[i] += x[i + 1];
        } item_ct1.barrier();
    }
}
// end sum_reduce



void
magma_zlobpcg_res_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    magmaDouble_ptr evals, 
    magmaDoubleComplex * X, 
    magmaDoubleComplex * R,
    magmaDouble_ptr res,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2); // global row index

    if ( row < num_rows) {
        for( int i=0; i < num_vecs; i++ ) {
            R[row + i * num_rows] =
                R[row + i * num_rows]
                /*
                DPCT1064:342: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                + MAGMA_Z_MAKE(-evals[i], 0.0) * X[row + i * num_rows];
        }
    }
}


/*
magmablas_dznrm2_kernel( 
    int m, 
    magmaDoubleComplex * da, 
    int ldda, 
    double * dxnorm )
{
    const int i = threadIdx.x;
    magmaDoubleComplex_ptr dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {
        #if (defined(PRECISION_s) || defined(PRECISION_d))
            re = dx[j];
            lsum += re*re;
        #else
            re = MAGMA_Z_REAL( dx[j] );
            double im = MAGMA_Z_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    sum_reduce< BLOCK_SIZE >( i, sum );
    
    if (i==0)
        res[blockIdx.x] = sqrt(sum[0]);
}
*/



/**
    Purpose
    -------
    
    This routine computes for Block-LOBPCG, the set of residuals. 
                            R = Ax - x evalues
    It replaces:
    for(int i=0; i < n; i++) {
        magma_zaxpy(m, MAGMA_Z_MAKE(-evalues[i],0),blockX+i*m,1,blockR+i*m,1);
    }
    The memory layout of x is:

        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    x = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows

    @param[in]
    num_vecs    magma_int_t
                number of vectors
                
    @param[in]
    evalues     magmaDouble_ptr 
                array of eigenvalues/approximations

    @param[in]
    X           magmaDoubleComplex_ptr 
                block of eigenvector approximations
                
    @param[in]
    R           magmaDoubleComplex_ptr 
                block of residuals

    @param[in]
    res         magmaDouble_ptr 
                array of residuals

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zlobpcg_res(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDouble_ptr evalues, 
    magmaDoubleComplex_ptr X,
    magmaDoubleComplex_ptr R, 
    magmaDouble_ptr res,
    magma_queue_t queue )
{
    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;

    sycl::range<3> threads(1, 1, block_size);
    sycl::range<3> grid(1, 1, magma_ceildiv(num_rows, block_size));

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_zlobpcg_res_kernel(num_rows, num_vecs, evalues,
                                                    X, R, res, item_ct1);
                       });

    return MAGMA_SUCCESS;
}
