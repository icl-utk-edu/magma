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
magma_zgbtrf_native_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab,
        magma_int_t* dipiv, magma_int_t *info,
        void* device_work, magma_int_t* lwork,
        magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;

    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( lddab < (kl+kv+1) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;
    }

    // calculate required workspace
    // [1] workspace of batched-strided gbtrf
    magma_int_t gbtrf_batch_lwork[1] = {-1};
    magma_zgbtrf_batched_strided_work(
        m, n, kl, ku,
        NULL, lddab, lddab*n, NULL, min(m,n),
        NULL, NULL, gbtrf_batch_lwork, 1, queue);

    // [3] we need a "device_info" on device memory
    magma_int_t gbtrf_native_lwork[1] = {0};
    gbtrf_native_lwork[0] = gbtrf_batch_lwork[0] + sizeof(magma_int_t);

    if(*lwork < 0) {
        // workspace query assumed
        *lwork = gbtrf_native_lwork[0];
        *info  = 0;
        return;
    }

    if( *lwork < gbtrf_native_lwork[0] ) {
        *info = -10;
        return;
    }

    magma_int_t* device_info = (magma_int_t*)((uint8_t*)device_work + gbtrf_batch_lwork[0]);
    magma_zgbtrf_batched_strided_work(
        m, n, kl, ku,
        dAB, lddab, lddab*n,
        dipiv, min(m,n),
        device_info,
        device_work, gbtrf_native_lwork,
        1, queue);

    // copy device_info to info
    magma_igetvector_async( 1, device_info, 1, info, 1, queue );

    return;
}

extern "C" magma_int_t
magma_zgbtrf_native(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab, magma_int_t* dipiv,
        magma_int_t *info)
{
    magma_int_t kv    = kl + ku;

    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( lddab < (kl+kv+1) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t cdev;
    magma_getdevice( &cdev );

    magma_queue_t queue;
    magma_queue_create( cdev, &queue );

    if( m == 0 || n == 0 ) return 0;

    magma_int_t lwork[1] = {-1};

    // query workspace
    magma_zgbtrf_native_work(
        m, n, kl, ku,
        NULL, lddab, NULL,
        info, NULL, lwork,
        queue);

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    magma_zgbtrf_native_work(
        m, n, kl, ku,
        dAB, lddab,
        dipiv, info,
        device_work, lwork,
        queue);
    magma_queue_sync( queue );

    magma_free(device_work);
    magma_queue_destroy( queue );

    return *info;
}
