#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "magma_internal.h"
#include "error.h"

/***************************************************************************//**
    Prints error message to stderr.
    Used by the check_error() and check_xerror() macros.

    @param[in]
    err     Error code.

    @param[in]
    func    Function where error occurred; inserted by check_error().

    @param[in]
    file    File     where error occurred; inserted by check_error().

    @param[in]
    line    Line     where error occurred; inserted by check_error().

    @ingroup magma_error_internal
*******************************************************************************/
void magma_xerror( magma_int_t err, const char* func, const char* file, int line )
{
    if ( err != MAGMA_SUCCESS ) {
        fprintf( stderr, "MAGMA error: %s (%lld) in %s at %s:%d\n",
                 magma_strerror( err ), (long long) err, func, file, line );
    }
}

/***************************************************************************//**
    @return String describing MAGMA errors (magma_int_t).

    @param[in]
    err     Error code.

    @ingroup magma_error
*******************************************************************************/
extern "C"
const char* magma_strerror( magma_int_t err )
{
    // LAPACK-compliant errors
    if ( err > 0 ) {
        return "function-specific error, see documentation";
    }
    else if ( err < 0 && err > MAGMA_ERR ) {
        return "invalid argument";
    }
    // MAGMA-specific errors
    switch( err ) {
        case MAGMA_SUCCESS:
            return "success";

        case MAGMA_ERR:
            return "unknown error";

        case MAGMA_ERR_NOT_INITIALIZED:
            return "not initialized";

        case MAGMA_ERR_REINITIALIZED:
            return "reinitialized";

        case MAGMA_ERR_NOT_SUPPORTED:
            return "not supported";

        case MAGMA_ERR_ILLEGAL_VALUE:
            return "illegal value";

        case MAGMA_ERR_NOT_FOUND:
            return "not found";

        case MAGMA_ERR_ALLOCATION:
            return "allocation";

        case MAGMA_ERR_INTERNAL_LIMIT:
            return "internal limit";

        case MAGMA_ERR_UNALLOCATED:
            return "unallocated error";

        case MAGMA_ERR_FILESYSTEM:
            return "filesystem error";

        case MAGMA_ERR_UNEXPECTED:
            return "unexpected error";

        case MAGMA_ERR_SEQUENCE_FLUSHED:
            return "sequence flushed";

        case MAGMA_ERR_HOST_ALLOC:
            return "cannot allocate memory on CPU host";

        case MAGMA_ERR_DEVICE_ALLOC:
            return "cannot allocate memory on GPU device";

        case MAGMA_ERR_CUDASTREAM:
            return "CUDA stream error";

        case MAGMA_ERR_INVALID_PTR:
            return "invalid pointer";

        case MAGMA_ERR_UNKNOWN:
            return "unknown error";

        case MAGMA_ERR_NOT_IMPLEMENTED:
            return "not implemented";

        case MAGMA_ERR_NAN:
            return "NaN detected";

        // some MAGMA-sparse errors
        case MAGMA_SLOW_CONVERGENCE:
            return "stopping criterion not reached within iterations";

        case MAGMA_DIVERGENCE:
            return "divergence";

        case MAGMA_NOTCONVERGED :
            return "stopping criterion not reached within iterations";

        case MAGMA_NONSPD:
            return "not positive definite (SPD/HPD)";

        case MAGMA_ERR_BADPRECOND:
            return "bad preconditioner";

        default:
            return "unknown MAGMA error code";
    }
}
