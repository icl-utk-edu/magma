/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"

// for every size [1:32], how many 1D configs can a warp hold?
#define NTCOL_1D_DEFAULT 32, 16, 10, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
const magma_int_t ntcol_1d_default[] = {NTCOL_1D_DEFAULT};

// =============================================================================
// GEMM
// =============================================================================
// Kepler (or older) 
const magma_int_t sgemm_batched_ntcol_300[] = {64,32,32,32,10,8,10,4,3,5,2,2,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t dgemm_batched_ntcol_300[] = {64,32,32,8,5,8,10,8,6,5,4,2,3,1,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t cgemm_batched_ntcol_300[] = {64,32,32,64,5,8,10,10,8,5,4,2,3,1,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t zgemm_batched_ntcol_300[] = {64,16,8,6,4,4,12,10,8,4,1,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

// Pascal (used also for maxwell) 
const magma_int_t sgemm_batched_ntcol_600[] = {64,64,64,32,14,13,9,5,7,3,5,3,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t dgemm_batched_ntcol_600[] = {64,64,32,16,16,8,10,1,8,8,2,2,3,5,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t cgemm_batched_ntcol_600[] = {64,32,64,32,32,8,13,1,3,8,2,2,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t zgemm_batched_ntcol_600[] = {64,14,14,14,32,8,16,2,4,2,4,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

// Volta  
const magma_int_t sgemm_batched_ntcol_700[] = {64,64,32,8,10,8,10,8,8,10,4,2,3,5,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t dgemm_batched_ntcol_700[] = {64,15,32,15,15,8,10,8,8,10,8,2,3,5,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t cgemm_batched_ntcol_700[] = {64,64,16,32,8,8,10,8,8,5,2,2,3,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const magma_int_t zgemm_batched_ntcol_700[] = {64,32,32,15,8,6,4,4,3,5,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

// =============================================================================
// LU
// =============================================================================
// Kepler (or older) 
const magma_int_t sgetrf_batched_ntcol_300[] = {32, 16, 8, 8, 8, 8, 8, 32, 8, 16, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4};
const magma_int_t dgetrf_batched_ntcol_300[] = {32, 16, 8, 16, 8, 4, 4, 32, 4, 4, 8, 8, 8, 8, 8, 8, 4, 4, 16, 16, 16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
const magma_int_t cgetrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};

// Pascal (used also for maxwell) 
const magma_int_t sgetrf_batched_ntcol_600[] = {8, 5, 3, 2, 3, 3, 3, 3, 32, 32, 32, 32, 2, 15, 16, 16, 16, 13, 12, 10, 11, 10, 10, 8, 9, 8, 2, 6, 6, 6, 1, 1};
const magma_int_t dgetrf_batched_ntcol_600[] = {8, 4, 2, 3, 2, 2, 2, 5, 3, 3, 5, 12, 3, 3, 9, 8, 6, 6, 6, 5, 6, 6, 6, 8, 8, 8, 7, 4, 4, 4, 4, 4};
const magma_int_t cgetrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};

// Volta  
const magma_int_t sgetrf_batched_ntcol_700[] = {8, 12, 3, 4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 8, 12, 12, 12, 10, 10, 10, 10, 8, 8, 8, 6, 6, 4, 6, 4, 1};
const magma_int_t dgetrf_batched_ntcol_700[] = {7, 4, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 9, 12, 8, 8, 10, 10, 10, 10, 8, 8, 8, 7, 6, 6, 6, 4, 4};
const magma_int_t cgetrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};

// =============================================================================
// QR
// =============================================================================
// Kepler (or older) 
const magma_int_t sgeqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
const magma_int_t dgeqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
const magma_int_t cgeqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgeqrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};

// Pascal (used also for maxwell) 
const magma_int_t sgeqrf_batched_ntcol_600[] = {16, 4, 3, 4, 3, 3, 6, 14, 32, 32, 32, 32, 16, 14, 16, 10, 9, 10, 9, 9, 6, 8, 6, 6, 8, 8, 2, 6, 6, 2, 4, 2};
const magma_int_t dgeqrf_batched_ntcol_600[] = {16, 4, 3, 3, 5, 5, 5, 10, 14, 10, 10, 10, 12, 12, 10, 4, 6, 6, 4, 8, 7, 7, 4, 4, 6, 2, 2, 1, 2, 2, 1, 1};
const magma_int_t cgeqrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgeqrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};

// Volta  
const magma_int_t sgeqrf_batched_ntcol_700[] = {32, 4, 3, 6, 5, 3, 2, 2, 3, 3, 3, 3, 3, 11, 3, 16, 12, 12, 12, 12, 10, 10, 10, 8, 8, 8, 1, 1, 1, 1, 1, 1};
const magma_int_t dgeqrf_batched_ntcol_700[] = {32, 4, 2, 2, 1, 1, 1, 5, 9, 16, 11, 8, 3, 3, 15, 16, 8, 8, 8, 4, 12, 4, 4, 6, 6, 4, 6, 4, 6, 4, 4, 4};
const magma_int_t cgeqrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgeqrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};

#ifdef __cplusplus
extern "C" {
#endif

// Definition of blocking sizes for NVIDIA cards
#ifdef HAVE_CUBLAS

// =============================================================================
/// @addtogroup magma_tuning
/// Tuning of the batched kernels that are invoked on extremely small sizes
/// @{

/***************************************************************************//**
    @return the ntcol value for very small xgemm_batched ( m = n = k)
*******************************************************************************/
magma_int_t magma_get_zgemm_batched_ntcol(magma_int_t m)
{
    magma_int_t* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)zgemm_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)zgemm_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)zgemm_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgemm_batched_ntcol
magma_int_t magma_get_cgemm_batched_ntcol(magma_int_t m)
{
    magma_int_t* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)cgemm_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)cgemm_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)cgemm_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgemm_batched_ntcol
magma_int_t magma_get_dgemm_batched_ntcol(magma_int_t m)
{
    magma_int_t* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)dgemm_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)dgemm_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)dgemm_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgemm_batched_ntcol
magma_int_t magma_get_sgemm_batched_ntcol(magma_int_t m)
{
    magma_int_t* ntcol_array; 

    if(m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)sgemm_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)sgemm_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)sgemm_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/***************************************************************************//**
    @return the ntcol value for very small xgetrf_batched ( m = n )
*******************************************************************************/
magma_int_t magma_get_zgetrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)zgetrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)zgetrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)zgetrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgetrf_batched_ntcol
magma_int_t magma_get_cgetrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)cgetrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)cgetrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)cgetrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgetrf_batched_ntcol
magma_int_t magma_get_dgetrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)dgetrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)dgetrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)dgetrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgetrf_batched_ntcol
magma_int_t magma_get_sgetrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/***************************************************************************//**
    @return the ntcol value for very small xgeqrf_batched ( m = n )
*******************************************************************************/
magma_int_t magma_get_zgeqrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)zgeqrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)zgeqrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)zgeqrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgeqrf_batched_ntcol
magma_int_t magma_get_cgeqrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)cgeqrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)cgeqrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)cgeqrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgeqrf_batched_ntcol
magma_int_t magma_get_dgeqrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)dgeqrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)dgeqrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)dgeqrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

/// @see magma_get_zgeqrf_batched_ntcol
magma_int_t magma_get_sgeqrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)sgeqrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)sgeqrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)sgeqrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

// =============================================================================
/// @}
// end group magma_tuning

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
