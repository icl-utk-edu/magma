#ifndef ERROR_H
#define ERROR_H

#include "magma_types.h"

// overloaded C++ functions to deal with errors
void magma_xerror( cudaError_t    err, const char* func, const char* file, int line );
void magma_xerror( cublasStatus_t err, const char* func, const char* file, int line );
void magma_xerror( magma_int_t    err, const char* func, const char* file, int line );

#ifdef __cplusplus
extern "C" {
#endif

// cuda provides cudaGetErrorString,
// but not cublasGetErrorString, so provide our own.
// In magma.h, we also provide magma_strerror.
const char* magma_cublasGetErrorString( cublasStatus_t error );

#ifdef __cplusplus
}
#endif

#ifdef NDEBUG
#define check_error( err )                     ((void)0)
#define check_xerror( err, func, file, line )  ((void)0)
#else

/***************************************************************************//**
    Checks if err is not success, and prints an error message.
    Similar to assert(), if NDEBUG is defined, this does nothing.
    This version adds the current func, file, and line to the error message.

    @param[in]
    err     Error code.

    @ingroup magma_error_internal
*******************************************************************************/
#define check_error( err ) \
        magma_xerror( err, __func__, __FILE__, __LINE__ )

/***************************************************************************//**
    Checks if err is not success, and prints an error message.
    Similar to assert(), if NDEBUG is defined, this does nothing.
    This version takes func, file, and line as arguments to add to error message.

    @param[in]
    err     Error code.

    @param[in]
    func    Function where error occurred.

    @param[in]
    file    File     where error occurred.

    @param[in]
    line    Line     where error occurred.

    @ingroup magma_error_internal
*******************************************************************************/
#define check_xerror( err, func, file, line ) \
        magma_xerror( err, func, file, line )

#endif  // not NDEBUG

#endif // ERROR_H
