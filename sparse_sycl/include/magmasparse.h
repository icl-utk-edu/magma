/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMASPARSE_H
#define MAGMASPARSE_H

#define MAGMA_SIVERGENCE  MAGMA_DIVERGENCE

/* ------------------------------------------------------------
 * MAGMASPARSE BLAS Functions
 * --------------------------------------------------------- */
//#include "magmasparseblas.h"

/* ------------------------------------------------------------
 * MAGMASPARSE functions
 * --------------------------------------------------------- */
#include "magmasparse_z.h"
#include "magmasparse_c.h"
#include "magmasparse_d.h"
#include "magmasparse_s.h"

// mixed precision
#include "magmasparse_zc.h"
#include "magmasparse_ds.h"

/* ------------------------------------------------------------
 * MAGMASPARSE types
 * --------------------------------------------------------- */
/*
#include "magmasparse_types_z.h"
#include "magmasparse_types_c.h"
#include "magmasparse_types_d.h"
#include "magmasparse_types_s.h"

*/
#include "magmasparse_types.h"
#endif /* MAGMASPARSE_H */
