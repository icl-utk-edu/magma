/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#if defined(MAGMA_HAVE_CUDA)
#include <cuda.h>  // for CUDA_VERSION

#include "magmasparse_internal.h"

/**
    Maps a cuSPARSE error to a MAGMA error.
    
    @param[in]
    status      cuSPARSE error
    
    @return MAGMA error
    
    @ingroup magma_error_internal
    ********************************************************************/
extern "C" magma_int_t
cusparse2magma_error( cusparseStatus_t status )
{
    switch( status ) {
        case CUSPARSE_STATUS_SUCCESS:                   return MAGMA_SUCCESS;                                break;
        case CUSPARSE_STATUS_NOT_INITIALIZED:           return MAGMA_ERR_CUSPARSE_NOT_INITIALIZED;           break;
        case CUSPARSE_STATUS_ALLOC_FAILED:              return MAGMA_ERR_CUSPARSE_ALLOC_FAILED;              break;
        case CUSPARSE_STATUS_INVALID_VALUE:             return MAGMA_ERR_CUSPARSE_INVALID_VALUE;             break;
        case CUSPARSE_STATUS_ARCH_MISMATCH:             return MAGMA_ERR_CUSPARSE_ARCH_MISMATCH;             break;
        case CUSPARSE_STATUS_MAPPING_ERROR:             return MAGMA_ERR_CUSPARSE_MAPPING_ERROR;             break;
        case CUSPARSE_STATUS_EXECUTION_FAILED:          return MAGMA_ERR_CUSPARSE_EXECUTION_FAILED;          break;
        case CUSPARSE_STATUS_INTERNAL_ERROR:            return MAGMA_ERR_CUSPARSE_INTERNAL_ERROR;            break;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return MAGMA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED; break;
        
        // added in CUDA 6.0
        #if CUDA_VERSION >= 6000
        case CUSPARSE_STATUS_ZERO_PIVOT:                return MAGMA_ERR_CUSPARSE_ZERO_PIVOT;                break;
        #endif
        
        default:
            return MAGMA_ERR_UNKNOWN;
            break;
    }
}
#endif
