/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

extern "C" void
magma_zgbsv_native_work(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* dipiv,
        magmaDoubleComplex* dB, magma_int_t lddb,
        magma_int_t *info, void* device_work, magma_int_t* lwork,
        magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;

    if ( n < 0 )
        *info = -1;
    else if ( kl < 0 )
        *info = -2;
    else if ( ku < 0 )
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if ( ldda < (kl+kv+1) )
        *info = -6;
    else if ( lddb < n)
        *info = -9;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;
    }

    if( n == 0 ) return;

    // calculate the required workspace
    // [1] workspace of batched-strided gbsv
    magma_int_t gbsv_batch_lwork[1]  = {-1};
    magma_zgbsv_batched_strided_work(
        n, kl, ku, nrhs,
        NULL, ldda, ldda*n, NULL, n,
        NULL, lddb, lddb*nrhs,
        NULL, NULL, gbsv_batch_lwork, 1, queue);

    // [2] we need a "device_info" on device memory
    magma_int_t gbsv_native_lwork[1] = {0};
    gbsv_native_lwork[0] = gbsv_batch_lwork[0] + sizeof(magma_int_t);

    if(*lwork < 0) {
        // workspace query assumed
        *lwork = gbsv_native_lwork[0];
        *info  = 0;
        return;
    }

    if( *lwork < gbsv_native_lwork[0] ) {
        *info = -10;
        return;
    }

    magma_int_t* device_info = (magma_int_t*)((uint8_t*)device_work + gbsv_batch_lwork[0]);
    magma_zgbsv_batched_strided_work(
        n, kl, ku, nrhs,
        dA, ldda, ldda*n, dipiv, n,
        dB, lddb, lddb*nrhs, device_info,
        device_work, gbsv_native_lwork, 1, queue);

    // copy device_info to info
    magma_igetvector( 1, device_info, 1, info, 1, queue );

    return;
}

extern "C" magma_int_t
magma_zgbsv_native(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* dipiv,
        magmaDoubleComplex* dB, magma_int_t lddb,
        magma_int_t *info)
{
    magma_int_t kv    = kl + ku;

    if ( n < 0 )
        *info = -1;
    else if ( kl < 0 )
        *info = -2;
    else if ( ku < 0 )
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if ( ldda < (kl+kv+1) )
        *info = -6;
    else if ( lddb < n)
        *info = -9;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t cdev;
    magma_getdevice( &cdev );

    magma_queue_t queue;
    magma_queue_create( cdev, &queue );

    if( n == 0 ) return 0;

    magma_int_t lwork[1] = {-1};

    // query workspace
    magma_zgbsv_native_work(
        n, kl, ku, nrhs,
        NULL, ldda, NULL,
        NULL, lddb,
        info, NULL, lwork, queue);

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    magma_zgbsv_native_work(
        n, kl, ku, nrhs,
        dA, ldda, dipiv,
        dB, lddb,
        info, device_work, lwork, queue);

    magma_queue_sync( queue );
    magma_free(device_work);
    magma_queue_destroy( queue );

    return *info;
}
