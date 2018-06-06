/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Definition of blocking sizes for NVIDIA cards
#ifdef HAVE_CUBLAS

// =============================================================================
/// @addtogroup magma_tuning
/// Optimal block sizes vary with GPU and, to a lesser extent, CPU.
/// Kepler tuning was on K20c   705 MHz with SandyBridge 2.6 GHz host (bunsen).
/// Fermi  tuning was on S2050 1147 MHz with AMD Opteron 2.4 GHz host (romulus).
/// @{


/******************************************************************************/
/// @return nb for spotrf based on n
magma_int_t magma_get_spotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  1500) nb = 256;
        else if (n < 10240) nb = 512;
        else                nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 256;
        else                nb = 512;
    }
    else {                     // 1.x
        if      (n <  3328) nb = 128;
        else if (n <  4256) nb = 224;
        else                nb = 288;
    }
    return nb;
}

/// @return nb for dpotrf based on n
magma_int_t magma_get_dpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  3072) nb = 256;
        else if (n < 10240) nb = 512;
        else                nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        if      (n <  3328) nb = 128;
        else if (n <  4256) nb = 128;
        else                nb = 256;
    }
    return nb;
}

/// @return nb for cpotrf based on n
magma_int_t magma_get_cpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  1500) nb = 192;
        else                nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for zpotrf based on n
magma_int_t magma_get_zpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  1500) nb = 192;
        else                nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}


/******************************************************************************/
/// @return nb for zpotrf_right based on n
magma_int_t magma_get_zpotrf_right_nb( magma_int_t n )
{
    return 128;
}

/// @return nb for cpotrf_right based on n
magma_int_t magma_get_cpotrf_right_nb( magma_int_t n )
{
    return 128;
}

/// @return nb for dpotrf_right based on n
magma_int_t magma_get_dpotrf_right_nb( magma_int_t n )
{
    return 320;
}

/// @return nb for spotrf_right based on n
magma_int_t magma_get_spotrf_right_nb( magma_int_t n )
{
    return 128;
}


/******************************************************************************/
/// @return nb for sgeqp3 based on m, n
magma_int_t magma_get_sgeqp3_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for dgeqp3 based on m, n
magma_int_t magma_get_dgeqp3_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for cgeqp3 based on m, n
magma_int_t magma_get_cgeqp3_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for zgeqp3 based on m, n
magma_int_t magma_get_zgeqp3_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}


/******************************************************************************/
/// @return nb for sgeqrf based on m, n
magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 96;
        else if (minmn <  7168) nb = 128;
        else if (minmn < 18432) nb = 256;
        else                    nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 64;
        else if (minmn <  8192) nb = 128;
        else                    nb = 256;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for dgeqrf based on m, n
magma_int_t magma_get_dgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  3072) nb = 64;
        else if (minmn < 10240) nb = 128;
        else                    nb = 256;
    }
    else {                     // 1.x and 2.x Fermi
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for cgeqrf based on m, n
magma_int_t magma_get_cgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x and 2.x Fermi
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for zgeqrf based on m, n
magma_int_t magma_get_zgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 64;
        else                    nb = 128;
    }
    return nb;
}


/******************************************************************************/
/// @return nb for sgeqlf based on m, n
magma_int_t magma_get_sgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_sgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for dgeqlf based on m, n
magma_int_t magma_get_dgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_dgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for cgeqlf based on m, n
magma_int_t magma_get_cgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  2048) nb = 32;
    else if (minmn <  4032) nb = 64;
    else                    nb = 128;
    return nb;
}

/// @return nb for zgeqlf based on m, n
magma_int_t magma_get_zgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  1024) nb = 64;
    else                    nb = 128;
    return nb;
}


/******************************************************************************/
/// @return nb for sgelqf based on m, n
magma_int_t magma_get_sgelqf_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_sgeqrf_nb( m, n );
}

/// @return nb for dgelqf based on m, n
magma_int_t magma_get_dgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_dgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for cgelqf based on m, n
magma_int_t magma_get_cgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  2048) nb = 32;
    else if (minmn <  4032) nb = 64;
    else                    nb = 128;
    return nb;
}

/// @return nb for zgelqf based on m, n
magma_int_t magma_get_zgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  1024) nb = 64;
    else                    nb = 128;
    return nb;
}


/******************************************************************************/
/// @return nb for sgetrf based on m, n
magma_int_t magma_get_sgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 600 ) {       // 6.x Pascal
        if      (minmn <  4096) nb = 128;
        else                    nb = 512;
    }
    else if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 256;
        else if (minmn < 18432) nb = 512;
        else                    nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for dgetrf based on m, n
magma_int_t magma_get_dgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  3072) nb = 128;
        else if (minmn <  8192) nb = 256;
        else                    nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for cgetrf based on m, n
magma_int_t magma_get_cgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn < 4096) nb = 64;
        else if (minmn < 8192) nb = 256;
        else                   nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for zgetrf based on m, n
magma_int_t magma_get_zgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn < 4096) nb = 64;
        else if (minmn < 8192) nb = 256;
        else                   nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn < 4096) nb = 64;
        else                   nb = 128;
    }
    else {                     // 1.x
        nb = 128;
    }
    return nb;
}


/******************************************************************************/
/// @return nb for native sgetrf based on m, n
magma_int_t magma_get_sgetrf_native_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <=  4096) nb = 64;
        else if (minmn <= 10240) nb = 128;
        else if (minmn <= 20480) nb = 512;
        else                     nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for native dgetrf based on m, n
magma_int_t magma_get_dgetrf_native_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <=  4096)  nb = 64;
        else if (minmn <=  10240) nb = 128;
        else if (minmn <=  20480) nb = 512;
        else                      nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for native cgetrf based on m, n
magma_int_t magma_get_cgetrf_native_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    // TODO: try all nb's (128,256,512, ... etc)
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <=  2048)  nb = 128;
        else                      nb = 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/// @return nb for native zgetrf based on m, n
magma_int_t magma_get_zgetrf_native_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    // TODO: try all nb's (128,256,512, ... etc)
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <=  4096)  nb = 32;
        else if (minmn <=  10240) nb = 64;
        else if (minmn <=  20480) nb = 256;
        else                      nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn < 4096) nb = 64;
        else                   nb = 128;
    }
    else {                     // 1.x
        nb = 128;
    }
    return nb;
}


/******************************************************************************/
/// @return nb for sgehrd based on n
magma_int_t magma_get_sgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        if      (n <  1024) nb = 32;
        else                nb = 96;
    }
    else {                     // 1.x
        if      (n <  1024) nb = 32;
        else                nb = 64;
    }
    return nb;
}

/// @return nb for dgehrd based on n
magma_int_t magma_get_dgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  2048) nb = 32;
    else                nb = 64;
    return nb;
}

/// @return nb for cgehrd based on n
magma_int_t magma_get_cgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  1024) nb = 32;
    else                nb = 64;
    return nb;
}

/// @return nb for zgehrd based on n
magma_int_t magma_get_zgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  2048) nb = 32;
    else                nb = 64;
    return nb;
}


/******************************************************************************/
// Currently, must be 64 due to zhemv_mgpu restrictions.

/// @return nb for ssytrd based on n
magma_int_t magma_get_ssytrd_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for dsytrd based on n
magma_int_t magma_get_dsytrd_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for chetrd based on n
magma_int_t magma_get_chetrd_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for zhetrd based on n
magma_int_t magma_get_zhetrd_nb( magma_int_t n )
{
    return 64;
}


/******************************************************************************/
/// @return nb for zhetrf based on n
magma_int_t magma_get_zhetrf_nb( magma_int_t n )
{
    return 256;
}

/// @return nb for chetrf based on n
magma_int_t magma_get_chetrf_nb( magma_int_t n )
{
    return 256;
}

/// @return nb for dsytrf based on n
magma_int_t magma_get_dsytrf_nb( magma_int_t n )
{
    return 96;
}

/// @return nb for ssytrf based on n
magma_int_t magma_get_ssytrf_nb( magma_int_t n )
{
    return 256;
}


/******************************************************************************/
/// @return nb for zhetrf_aasen based on n
magma_int_t magma_get_zhetrf_aasen_nb( magma_int_t n )
{
    return 256;
}

/// @return nb for chetrf_aasen based on n
magma_int_t magma_get_chetrf_aasen_nb( magma_int_t n )
{
    return 256;
}

/// @return nb for dsytrf_aasen based on n
magma_int_t magma_get_dsytrf_aasen_nb( magma_int_t n )
{
    return 256;
}

/// @return nb for ssytrf_aasen based on n
magma_int_t magma_get_ssytrf_aasen_nb( magma_int_t n )
{
    return 256;
}


/******************************************************************************/
/// @return nb for zhetrf_nopiv based on n
magma_int_t magma_get_zhetrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

/// @return nb for chetrf_nopiv based on n
magma_int_t magma_get_chetrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

/// @return nb for dsytrf_nopiv based on n
magma_int_t magma_get_dsytrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

/// @return nb for ssytrf_nopiv based on n
magma_int_t magma_get_ssytrf_nopiv_nb( magma_int_t n )
{
    return 320;
}


/******************************************************************************/
/// @return nb for sgebrd based on m, n
magma_int_t magma_get_sgebrd_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for dgebrd based on m, n
magma_int_t magma_get_dgebrd_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for cgebrd based on m, n
magma_int_t magma_get_cgebrd_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}

/// @return nb for zgebrd based on m, n
magma_int_t magma_get_zgebrd_nb( magma_int_t m, magma_int_t n )
{
    return 32;
}


/******************************************************************************/
/// @return nb for ssygst based on n
magma_int_t magma_get_ssygst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  4096) nb = 768;
        else                nb = 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 512;
        else                nb = 1024;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for dsygst based on n
magma_int_t magma_get_dsygst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for chegst based on n
magma_int_t magma_get_chegst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for zhegst based on n
magma_int_t magma_get_zhegst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}


/******************************************************************************/
/// @return nb for sgetri based on n
magma_int_t magma_get_sgetri_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for dgetri based on n
magma_int_t magma_get_dgetri_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for cgetri based on n
magma_int_t magma_get_cgetri_nb( magma_int_t n )
{
    return 64;
}

/// @return nb for zgetri based on n
magma_int_t magma_get_zgetri_nb( magma_int_t n )
{
    return 64;
}


/******************************************************************************/
/// @return nb for sgesvd based on m, n
magma_int_t magma_get_sgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_sgebrd_nb( m, n );
}

/// @return nb for dgesvd based on m, n
magma_int_t magma_get_dgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_dgebrd_nb( m, n );
}

/// @return nb for cgesvd based on m, n
magma_int_t magma_get_cgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_cgebrd_nb( m, n );
}

/// @return nb for zgesvd based on m, n
magma_int_t magma_get_zgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_zgebrd_nb( m, n );
}


/******************************************************************************/
/// @return nb for ssygst_m based on n
magma_int_t magma_get_ssygst_m_nb( magma_int_t n )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  4096) nb = 768;
        else                nb = 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 512;
        else                nb = 1024;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

/// @return nb for dsygst_m based on n
magma_int_t magma_get_dsygst_m_nb( magma_int_t n )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

/// @return nb for chegst_m based on n
magma_int_t magma_get_chegst_m_nb( magma_int_t n )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

/// @return nb for zhegst_m based on n
magma_int_t magma_get_zhegst_m_nb( magma_int_t n )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}


/******************************************************************************/
// TODO: these numbers seem very strange.
// Numbers in zbulge_back.cpp seem more realistic: 27 for Fermi + Westmere.

/// @return gpu over cpu performance for 2 stage TRD
magma_int_t magma_get_sbulge_gcperf( )
{
    magma_int_t perf;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        perf = 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        perf = 15000;
    }
    else {                     // 1.x
        perf = 10000;
    }
    return perf;
}

/// @return gpu over cpu performance for 2 stage TRD
magma_int_t magma_get_dbulge_gcperf( )
{
    magma_int_t perf;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        perf = 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        perf = 15000;
    }
    else {                     // 1.x
        perf = 10000;
    }
    return perf;
}

/// @return gpu over cpu performance for 2 stage TRD
magma_int_t magma_get_cbulge_gcperf( )
{
    magma_int_t perf;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        perf = 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        perf = 15000;
    }
    else {                     // 1.x
        perf = 10000;
    }
    return perf;
}

/// @return gpu over cpu performance for 2 stage TRD
magma_int_t magma_get_zbulge_gcperf( )
{
    magma_int_t perf;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        perf = 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        perf = 15000;
    }
    else {                     // 1.x
        perf = 10000;
    }
    return perf;
}


/******************************************************************************/
/// @return smlsiz for the divide and conquewr routine dlaex0 dstedx zstedx
magma_int_t magma_get_smlsize_divideconquer()
{
    return 128;
}



/******************************************************************************/
/// @return nb for 2 stage TRD
magma_int_t magma_get_sbulge_nb( magma_int_t n, magma_int_t nbthreads  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD
magma_int_t magma_get_dbulge_nb( magma_int_t n, magma_int_t nbthreads  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 128;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD
magma_int_t magma_get_cbulge_nb( magma_int_t n, magma_int_t nbthreads  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            nb = 128;
        else
            nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD
magma_int_t magma_get_zbulge_nb( magma_int_t n, magma_int_t nbthreads )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            nb = 128;
        else
            nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}


/******************************************************************************/
/// @return Vblksiz for 2 stage TRD
magma_int_t magma_get_sbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads  )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        size = min(nb, 128);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 64);
    }
    return size;
}

/// @return Vblksiz for 2 stage TRD
magma_int_t magma_get_dbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads  )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        size = min(nb, 64);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}

/// @return Vblksiz for 2 stage TRD
magma_int_t magma_get_cbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            size = min(nb, 48);
        else
            size = min(nb, 48);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}

/// @return Vblksiz for 2 stage TRD
magma_int_t magma_get_zbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            size = min(nb, 64);
        else
            size = min(nb, 32);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}


/******************************************************************************/
/// @return nb for 2 stage TRD_MGPU
magma_int_t magma_get_sbulge_mgpu_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD_MGPU
magma_int_t magma_get_dbulge_mgpu_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD_MGPU
magma_int_t magma_get_cbulge_mgpu_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/// @return nb for 2 stage TRD_MGPU
magma_int_t magma_get_zbulge_mgpu_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}


// =============================================================================
/// @}
// end group magma_tuning

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
