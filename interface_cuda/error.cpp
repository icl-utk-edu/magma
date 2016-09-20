#include <cuda_runtime.h>

#include "magma_internal.h"
#include "error.h"

/***************************************************************************//**
    Prints error message to stderr.
    C++ function overloaded for different error types (CUDA,
    cuBLAS, MAGMA errors). Note CUDA and cuBLAS errors are enums,
    so can be differentiated.
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
void magma_xerror( cudaError_t err, const char* func, const char* file, int line )
{
    if ( err != cudaSuccess ) {
        fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
                 cudaGetErrorString( err ), err, func, file, line );
    }
}


/******************************************************************************/
/// @see magma_xerror
/// @ingroup magma_error_internal
void magma_xerror( cublasStatus_t err, const char* func, const char* file, int line )
{
    if ( err != CUBLAS_STATUS_SUCCESS ) {
        fprintf( stderr, "CUBLAS error: %s (%d) in %s at %s:%d\n",
                 magma_cublasGetErrorString( err ), err, func, file, line );
    }
}


/******************************************************************************/
/// @see magma_xerror
/// @ingroup magma_error_internal
void magma_xerror( magma_int_t err, const char* func, const char* file, int line )
{
    if ( err != MAGMA_SUCCESS ) {
        fprintf( stderr, "MAGMA error: %s (%lld) in %s at %s:%d\n",
                 magma_strerror( err ), (long long) err, func, file, line );
    }
}


/***************************************************************************//**
    @return String describing cuBLAS errors (cublasStatus_t).
    CUDA provides cudaGetErrorString, but not cublasGetErrorString.

    @param[in]
    err     Error code.

    @ingroup magma_error_internal
*******************************************************************************/
extern "C"
const char* magma_cublasGetErrorString( cublasStatus_t err )
{
    switch( err ) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";

        default:
            return "unknown CUBLAS error code";
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

        // map cusparse errors to magma errors
        case MAGMA_ERR_CUSPARSE_NOT_INITIALIZED:
            return "cusparse: not initialized";

        case MAGMA_ERR_CUSPARSE_ALLOC_FAILED:
            return "cusparse: allocation failed";

        case MAGMA_ERR_CUSPARSE_INVALID_VALUE:
            return "cusparse: invalid value";

        case MAGMA_ERR_CUSPARSE_ARCH_MISMATCH:
            return "cusparse: architecture mismatch";

        case MAGMA_ERR_CUSPARSE_MAPPING_ERROR:
            return "cusparse: mapping error";

        case MAGMA_ERR_CUSPARSE_EXECUTION_FAILED:
            return "cusparse: execution failed";

        case MAGMA_ERR_CUSPARSE_INTERNAL_ERROR:
            return "cusparse: internal error";

        case MAGMA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusparse: matrix type not supported";

        case MAGMA_ERR_CUSPARSE_ZERO_PIVOT:
            return "cusparse: zero pivot";

        default:
            return "unknown MAGMA error code";
    }
}
