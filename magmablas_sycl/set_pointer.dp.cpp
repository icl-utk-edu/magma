/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

/******************************************************************************/
static
void stepinit_ipiv_kernel(magma_int_t **ipiv_array, int pm,
                          sycl::nd_item<3> item_ct1)
{
    magma_int_t *ipiv = ipiv_array[item_ct1.get_group(2)];

    int tx = item_ct1.get_local_id(2);
#if 0
    // best case senario piv = i ==> no piv
    // set piv equal to myself piv[i]=i
    if (tx < pm)
    {
        ipiv[tx] = tx+1;
    }
#else
    //set piv from the last to the first shifted by 32 such a way that it simulate the worst case
    if (tx < pm)
    {
        int i, s;
        i = pm/32;
        i = (i == 1 ? 0 : i);
        s = tx%i;
        ipiv[tx] =  ( (pm - (s*32) ) - tx/i);
        //printf("voici s %d pm %d me %d  ipiv %d \n",s, pm, tx, ipiv[tx]);
    }
#endif
}


/******************************************************************************/
extern "C"
void stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount, magma_queue_t queue)

{
    /*
    DPCT1049:198: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount) *
                                             sycl::range<3>(1, 1, pm),
                                         sycl::range<3>(1, 1, pm)),
                       [=](sycl::nd_item<3> item_ct1) {
                           stepinit_ipiv_kernel(ipiv_array, pm, item_ct1);
                       });
}


/******************************************************************************/
static
void magma_iset_pointer_kernel(
    magma_int_t **output_array,
    magma_int_t *input,
    int lda,
    int row, int column, 
    int batchSize, sycl::nd_item<3> item_ct1)
{
    output_array[item_ct1.get_group(2)] =
        input + item_ct1.get_group(2) * batchSize + row + column * lda;
}


/******************************************************************************/
extern "C"
void magma_iset_pointer(
    magma_int_t **output_array,
    magma_int_t *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchSize,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batchSize)
    change into A_array[0] A_array[1],... A_array[batchCount-1], where the size of each A_array[i] is N
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount),
                                         sycl::range<3>(1, 1, 1)),
                       [=](sycl::nd_item<3> item_ct1) {
                           magma_iset_pointer_kernel(output_array, input, lda,
                                                     row, column, batchSize,
                                                     item_ct1);
                       });
}



/******************************************************************************/
void idisplace_pointers_kernel(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, sycl::nd_item<3> item_ct1)
{
    magma_int_t *inpt = input_array[item_ct1.get_group(2)];
    output_array[item_ct1.get_group(2)] = &inpt[row + column * lda];
    //printf("==> zdisplace_pointer_kernel input %p input_array %p output_array %p  \n",inpt, input_array[blockIdx.x],output_array[blockIdx.x]);
}


/******************************************************************************/
extern "C"
void magma_idisplace_pointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)

{
    /*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batchCount),
                                         sycl::range<3>(1, 1, 1)),
                       [=](sycl::nd_item<3> item_ct1) {
                           idisplace_pointers_kernel(output_array, input_array,
                                                     lda, row, column,
                                                     item_ct1);
                       });
}
