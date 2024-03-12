/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// these are included already in magma_internal.h & other headers
//#include <cublas_v2.h>

#include "magma_internal.h"
#include "error.h"


#if defined(MAGMA_HAVE_SYCL)
#ifndef MAGMA_NO_V1

// -----------------------------------------------------------------------------
// globals
// see interface.cpp for definitions

#ifndef MAGMA_NO_V1
    extern magma_queue_t* g_null_queues;

    #ifdef HAVE_PTHREAD_KEY
    extern pthread_key_t g_magma_queue_key;
    #else
    extern magma_queue_t g_magma_queue;
    #endif
#endif // MAGMA_NO_V1


// -----------------------------------------------------------------------------
extern int g_magma_devices_cnt;


// =============================================================================
// device support

/***************************************************************************//**
    @deprecated
    Synchronize the current device.
    This functionality does not exist in OpenCL, so it is deprecated for CUDA, too.

    @ingroup magma_device
*******************************************************************************/
extern "C" void magma_device_sync() try {
    dpct::get_current_device().queues_wait_and_throw();
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
}

// =============================================================================
// queue support

/***************************************************************************//**
    @deprecated

    Sets the current global MAGMA v1 queue for kernels to execute in.
    In MAGMA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with MAGMA_NO_V1, this is not defined.

    @param[in]
    queue       Queue to set as current global MAGMA v1 queue.

    @return MAGMA_SUCCESS if successful

    @ingroup magma_queue
*******************************************************************************/
extern "C" magma_int_t
magmablasSetKernelStream( magma_queue_t queue )
{
    magma_int_t info = 0;
    #ifdef HAVE_PTHREAD_KEY
    info = pthread_setspecific( g_magma_queue_key, queue );
    #else
    g_magma_queue = queue;
    #endif
    return info;
}


/***************************************************************************//**
    @deprecated

    Gets the current global MAGMA v1 queue for kernels to execute in.
    In MAGMA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with MAGMA_NO_V1, this is not defined.

    @param[out]
    queue_ptr    On output, set to the current global MAGMA v1 queue.

    @return MAGMA_SUCCESS if successful

    @ingroup magma_queue
*******************************************************************************/
extern "C" magma_int_t
magmablasGetKernelStream( magma_queue_t *queue_ptr )
{
    #ifdef HAVE_PTHREAD_KEY
    *queue_ptr = (magma_queue_t) pthread_getspecific( g_magma_queue_key );
    #else
    *queue_ptr = g_magma_queue;
    #endif
    return 0;
}


/***************************************************************************//**
    @deprecated

    Gets the current global MAGMA v1 queue for kernels to execute in.
    Unlike magmablasGetKernelStream(), if the current queue is NULL,
    this will return a special MAGMA queue that has a NULL CUDA stream.
    This allows MAGMA v1 wrappers to call v2 kernels with a non-NULL queue.

    In MAGMA v2, all kernels take queue as an argument, so this is deprecated.
    If compiled with MAGMA_NO_V1, this is not defined.

    @return Current global MAGMA v1 queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
magma_queue_t magmablasGetQueue()
{
    magma_queue_t queue;
    #ifdef HAVE_PTHREAD_KEY
    queue = (magma_queue_t) pthread_getspecific( g_magma_queue_key );
    #else
    queue = g_magma_queue;
    #endif
    if ( queue == NULL ) {
        magma_device_t dev;
        magma_getdevice( &dev );
        if ( dev >= g_magma_devices_cnt || g_null_queues == NULL ) {
            fprintf( stderr, "Error: %s requires magma_init() to be called first for MAGMA v1 compatability.\n",
                     __func__ );
            return NULL;
        }
        // create queue w/ NULL stream first time that NULL queue is used
        if ( g_null_queues[dev] == NULL ) {
            #ifdef MAGMA_HAVE_SYCL
            magma_queue_create_from_sycl( dev, NULL, NULL, NULL, &g_null_queues[dev] );
            #endif
            //printf( "dev %lld create queue %p\n", (long long) dev, (void*) g_null_queues[dev] );
            assert( g_null_queues[dev] != NULL );
        }
        queue = g_null_queues[dev];
    }
    assert( queue != NULL );
    return queue;
}


/******************************************************************************/
// @deprecated
// MAGMA v1 version that doesn't take device ID.
extern "C" void magma_queue_create_v1_internal(magma_queue_t *queue_ptr,
                                               const char *func,
                                               const char *file, int line) try {
    int device;
    int err;
    err = device = dpct::dev_mgr::instance().current_device_id();
    check_xerror( err, func, file, line );
    MAGMA_UNUSED( err );

    magma_queue_create_internal( device, queue_ptr, func, file, line );
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif // not MAGMA_NO_V1
#endif // MAGMA_HAVE_CUDA
