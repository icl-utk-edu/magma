#include "magma_v2.h"
#include "magma_mangling.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

// -----------------------------------------------------------------------------
// initialize
#define magmaf_init FORTRAN_NAME( magmaf_init, MAGMAF_INIT )
void magmaf_init()
{
    magma_init();
}

#define magmaf_finalize FORTRAN_NAME( magmaf_finalize, MAGMAF_FINALIZE )
void magmaf_finalize()
{
    magma_finalize();
}

// -----------------------------------------------------------------------------
// version
#define magmaf_version FORTRAN_NAME( magmaf_version, MAGMAF_VERSION )
void magmaf_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro )
{
    magma_version( major, minor, micro );
}

#define magmaf_print_environment FORTRAN_NAME( magmaf_print_environment, MAGMAF_PRINT_ENVIRONMENT )
void magmaf_print_environment()
{
    magma_print_environment();
}

// -----------------------------------------------------------------------------
// device support
#define magmaf_num_gpus FORTRAN_NAME( magmaf_num_gpus, MAGMAF_NUM_GPUS )
magma_int_t magmaf_num_gpus()
{
    return magma_num_gpus();
}

#define magmaf_getdevice_arch FORTRAN_NAME( magmaf_getdevice_arch, MAGMAF_GETDEVICE_ARCH )
magma_int_t magmaf_getdevice_arch()
{
    return magma_getdevice_arch();
}

#define magmaf_getdevice FORTRAN_NAME( magmaf_getdevice, MAGMAF_GETDEVICE )
void magmaf_getdevice( magma_int_t* dev )
{
    magma_getdevice( dev );
}

#define magmaf_setdevice FORTRAN_NAME( magmaf_setdevice, MAGMAF_SETDEVICE )
void magmaf_setdevice( magma_int_t* dev )
{
    magma_setdevice( *dev );
}

#define magmaf_mem_size FORTRAN_NAME( magmaf_mem_size, MAGMAF_MEM_SIZE )
size_t magmaf_mem_size( magma_queue_t* queue )
{
    return magma_mem_size( *queue );
}

// -----------------------------------------------------------------------------
// queue support
#define magmaf_queue_create FORTRAN_NAME( magmaf_queue_create, MAGMAF_QUEUE_CREATE )
void magmaf_queue_create( magma_int_t* dev, magma_queue_t* queue )
{
    magma_queue_create( *dev, queue );
}

#define magmaf_queue_destroy FORTRAN_NAME( magmaf_queue_destroy, MAGMAF_QUEUE_DESTROY )
void magmaf_queue_destroy( magma_queue_t* queue )
{
    magma_queue_destroy( *queue );
}

#define magmaf_queue_sync FORTRAN_NAME( magmaf_queue_sync, MAGMAF_QUEUE_SYNC )
void magmaf_queue_sync( magma_queue_t* queue )
{
    magma_queue_sync( *queue );
}

#define magmaf_queue_get_device FORTRAN_NAME( magmaf_queue_get_device, MAGMAF_QUEUE_GET_DEVICE )
magma_int_t magmaf_queue_get_device( magma_queue_t* queue )
{
    return magma_queue_get_device( *queue );
}

// -----------------------------------------------------------------------------
// GPU allocation

#define magmaf_malloc FORTRAN_NAME( magmaf_malloc, MAGMAF_MALLOC )
magma_int_t magmaf_malloc( void** ptr, magma_int_t* bytes )
{
    return magma_malloc( ptr, *bytes );
}

#define magmaf_smalloc FORTRAN_NAME( magmaf_smalloc, MAGMAF_smalloc )
magma_int_t magmaf_smalloc( float** ptr, magma_int_t* n )
{
    return magma_smalloc( ptr, *n );
}

#define magmaf_dmalloc FORTRAN_NAME( magmaf_dmalloc, MAGMAF_dmalloc )
magma_int_t magmaf_dmalloc( double** ptr, magma_int_t* n )
{
    return magma_dmalloc( ptr, *n );
}

#define magmaf_cmalloc FORTRAN_NAME( magmaf_cmalloc, MAGMAF_cmalloc )
magma_int_t magmaf_cmalloc( magmaFloatComplex** ptr, magma_int_t* n )
{
    return magma_cmalloc( ptr, *n );
}

#define magmaf_zmalloc FORTRAN_NAME( magmaf_zmalloc, MAGMAF_zmalloc )
magma_int_t magmaf_zmalloc( magmaDoubleComplex** ptr, magma_int_t* n )
{
    return magma_zmalloc( ptr, *n );
}

#define magmaf_free FORTRAN_NAME( magmaf_free, MAGMAF_free )
magma_int_t magmaf_free( void** ptr )
{
    return magma_free( *ptr );
}

// -----------------------------------------------------------------------------
// CPU regular (non-pinned) allocation

#define magmaf_malloc_cpu FORTRAN_NAME( magmaf_malloc_cpu, MAGMAF_MALLOC_CPU )
magma_int_t magmaf_malloc_cpu( void** ptr, magma_int_t* bytes )
{
    return magma_malloc_cpu( ptr, *bytes );
}

#define magmaf_smalloc_cpu FORTRAN_NAME( magmaf_smalloc_cpu, MAGMAF_smalloc_cpu )
magma_int_t magmaf_smalloc_cpu( float** ptr, magma_int_t* n )
{
    return magma_smalloc_cpu( ptr, *n );
}

#define magmaf_dmalloc_cpu FORTRAN_NAME( magmaf_dmalloc_cpu, MAGMAF_dmalloc_cpu )
magma_int_t magmaf_dmalloc_cpu( double** ptr, magma_int_t* n )
{
    return magma_dmalloc_cpu( ptr, *n );
}

#define magmaf_cmalloc_cpu FORTRAN_NAME( magmaf_cmalloc_cpu, MAGMAF_cmalloc_cpu )
magma_int_t magmaf_cmalloc_cpu( magmaFloatComplex** ptr, magma_int_t* n )
{
    return magma_cmalloc_cpu( ptr, *n );
}

#define magmaf_zmalloc_cpu FORTRAN_NAME( magmaf_zmalloc_cpu, MAGMAF_zmalloc_cpu )
magma_int_t magmaf_zmalloc_cpu( magmaDoubleComplex** ptr, magma_int_t* n )
{
    return magma_zmalloc_cpu( ptr, *n );
}

#define magmaf_free_cpu FORTRAN_NAME( magmaf_free_cpu, MAGMAF_free_cpu )
magma_int_t magmaf_free_cpu( void** ptr )
{
    return magma_free_cpu( *ptr );
}

// -----------------------------------------------------------------------------
// CPU pinned allocation

#define magmaf_malloc_pinned FORTRAN_NAME( magmaf_malloc_pinned, MAGMAF_MALLOC_PINNED )
magma_int_t magmaf_malloc_pinned( void** ptr, magma_int_t* bytes )
{
    return magma_malloc_pinned( ptr, *bytes );
}

#define magmaf_smalloc_pinned FORTRAN_NAME( magmaf_smalloc_pinned, MAGMAF_smalloc_pinned )
magma_int_t magmaf_smalloc_pinned( float** ptr, magma_int_t* n )
{
    return magma_smalloc_pinned( ptr, *n );
}

#define magmaf_dmalloc_pinned FORTRAN_NAME( magmaf_dmalloc_pinned, MAGMAF_dmalloc_pinned )
magma_int_t magmaf_dmalloc_pinned( double** ptr, magma_int_t* n )
{
    return magma_dmalloc_pinned( ptr, *n );
}

#define magmaf_cmalloc_pinned FORTRAN_NAME( magmaf_cmalloc_pinned, MAGMAF_cmalloc_pinned )
magma_int_t magmaf_cmalloc_pinned( magmaFloatComplex** ptr, magma_int_t* n )
{
    return magma_cmalloc_pinned( ptr, *n );
}

#define magmaf_zmalloc_pinned FORTRAN_NAME( magmaf_zmalloc_pinned, MAGMAF_zmalloc_pinned )
magma_int_t magmaf_zmalloc_pinned( magmaDoubleComplex** ptr, magma_int_t* n )
{
    return magma_zmalloc_pinned( ptr, *n );
}

#define magmaf_free_pinned FORTRAN_NAME( magmaf_free_pinned, MAGMAF_free_pinned )
magma_int_t magmaf_free_pinned( void** ptr )
{
    return magma_free_pinned( *ptr );
}

#ifdef __cplusplus
}
#endif
