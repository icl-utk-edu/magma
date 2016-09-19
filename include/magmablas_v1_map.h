/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMABLAS_V1_MAP_H
#define MAGMABLAS_V1_MAP_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

// =============================================================================
// map function names to old v1 routines

#include "magmablas_s_v1_map.h"
#include "magmablas_d_v1_map.h"
#include "magmablas_c_v1_map.h"
#include "magmablas_z_v1_map.h"
#include "magmablas_ds_v1_map.h"
#include "magmablas_zc_v1_map.h"

#undef magma_queue_create

#undef magma_setvector
#undef magma_getvector
#undef magma_copyvector
#undef magma_setmatrix
#undef magma_getmatrix
#undef magma_copymatrix
                                            
#undef magma_isetvector
#undef magma_igetvector
#undef magma_icopyvector
#undef magma_isetmatrix
#undef magma_igetmatrix
#undef magma_icopymatrix
                                            
#undef magma_index_setvector
#undef magma_index_getvector
#undef magma_index_copyvector
#undef magma_index_setmatrix
#undef magma_index_getmatrix
#undef magma_index_copymatrix

#define magma_queue_create                  magma_queue_create_v1

#define magma_setvector                     magma_setvector_v1
#define magma_getvector                     magma_getvector_v1
#define magma_copyvector                    magma_copyvector_v1
#define magma_setmatrix                     magma_setmatrix_v1
#define magma_getmatrix                     magma_getmatrix_v1
#define magma_copymatrix                    magma_copymatrix_v1
                                            
#define magma_isetvector                    magma_isetvector_v1
#define magma_igetvector                    magma_igetvector_v1
#define magma_icopyvector                   magma_icopyvector_v1
#define magma_isetmatrix                    magma_isetmatrix_v1
#define magma_igetmatrix                    magma_igetmatrix_v1
#define magma_icopymatrix                   magma_icopymatrix_v1
                                            
#define magma_index_setvector               magma_index_setvector_v1
#define magma_index_getvector               magma_index_getvector_v1
#define magma_index_copyvector              magma_index_copyvector_v1
#define magma_index_setmatrix               magma_index_setmatrix_v1
#define magma_index_getmatrix               magma_index_getmatrix_v1
#define magma_index_copymatrix              magma_index_copymatrix_v1

#endif // MAGMABLAS_V1_MAP_H
