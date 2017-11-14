Upgrading to MAGMA 2.0
----------------------

MAGMA 2.0 introduces queues to each magmablas function, and removes the
magmablas{Set, Get}KernelStream( ) functions. This eliminates a thread safety
issue. Previously, in this situation:

    thread 0                           thread 1
    --------                           --------
    magmablasSetKernelStream( q0 );
                                       magmablasSetKernelStream( q1 );
    magma_zgemm( ... );                magma_ztrmm( ... );

the magma_zgemm in thread 0 would erroneously execute on q1, instead of on q0 as
intended. This could even segfault, if q0 and q1 are on different CUDA devices.
One fix is to use thread-local storage, but this depends on the threading
environment, whether that is pthreads, OpenMP, Windows threads, or anything
else.

Our fix is to pass queues to each routine. This is similar to CUBLAS v2 handles.

    thread 0                           thread 1
    --------                           --------
    magma_zgemm( ..., q0 );            magma_ztrmm( ..., q1 );

This eliminates the global state and makes the code easier to follow, as each
call explicitly states what queue it is running on.

For backwards compatibility, magma.h continues to provide mostly the original
functionality. (The one significant change still being that MAGMA queues are
opaque structures now, not CUDA streams.) A new header, magma_v2.h, provides the
new functionality with queues passed to each routine.

Steps to take to update code that uses MAGMA:

1)  Change the header:

    #include <magma.h>             ==>    #include <magma_v2.h>

2)  Creating a MAGMA queue now takes a CUDA device index:

    magma_queue_create( &queue );  ==>    magma_queue_create( dev_id, &queue );

    This is for improved compatibility with OpenCL, which takes a device when
    creating a queue.

    The queue is now an opaque structure. (Previously it was identical to a CUDA
    stream.) You should not access its internal values except through the
    provided functions. You can query the queue for its device index:

    dev_id = magma_queue_get_device( queue );

    For non-portable, CUDA-only code, you can query the CUDA stream, CUBLAS
    handle, and CUSPARSE handle:

    stream   = magma_queue_get_cuda_stream(     queue );
    handle   = magma_queue_get_cublas_handle(   queue );
    sphandle = magma_queue_get_cusparse_handle( queue );

    Changing the CUDA stream associated with the CUBLAS or CUSPARSE handles
    will result in undefined behavior.

    If need be, you can create a queue from an existing CUDA stream, CUBLAS
    handle, and CUSPARSE handle:

    magma_queue_create_from_cuda( dev_id, cuda_stream, cublas_handle, cusparse_handle, &queue );

3)  NULL queues are not allowed. CUDA has a NULL stream which has extra implicit
    synchronization. This is not supported in OpenCL, so for portability, MAGMA
    no longer allows NULL queues. You may need to add synchronization:

    magma_queue_sync( queue );

    in appropriate places if your code relied on implicit NULL stream
    synchronization.

4)  BLAS functions add a queue argument:

    magma_zgemm     ( ... );       ==>    magma_zgemm     ( ..., queue );
    magmablas_zlaset( ... );       ==>    magmablas_zlaset( ..., queue );
    magma_zlarfb    ( ... );       ==>    magma_zlarfb    ( ..., queue );
    
    The magmablas{Set, Get}KernelStream( queue ) functions are now removed.

5)  Synchronous set, get, copy functions also take a queue:

    magma_zsetmatrix( ... );       ==>    magma_zsetmatrix( ..., queue );

    The semantics of magma_z{set, get, copy}matrix changed slightly. They take a
    queue, and synchronize that queue after the transfer. Previously, the v1
    interface also implicitly did a device synchronization before the transfer.
    This is not supported for OpenCL, so we changed for better compatibility.

    Asynchronous set, get, copy functions (magma_zsetmatrix_async) already took
    a queue, so remain unchanged.

6)  Most high-level functions interface remain unchanged:

    magma_zgetrf    ( m, n,  A, lda,  ipiv, &info );    ==>    no change
    magma_zgetrf_gpu( m, n, dA, ldda, ipiv, &info );    ==>    no change
