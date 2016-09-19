#ifndef ZGEMM_FERMI_KERNELS_H
#define ZGEMM_FERMI_KERNELS_H

/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       See [zcds]gemm_fermi.cu for description of related files.
*/
#include "magma_internal.h"

// =============================================================================
#define COMPLEX
#define DOUBLE
#define TEXTURE_1D

#include "gemm_stencil_defs.h"

// =============================================================================
// currently, CPU driver assumes all transpose versions have same DIM_X, DIM_Y

// size of thread block for calculating C (innermost loop)
#define DIM_X  8
#define DIM_Y  8


// =============================================================================
// A x B
// size of work for a thread block
#define BLK_M_nn  24
#define BLK_N_nn  16

#define BLK_K  8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#undef  version
#define version trans_nn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A x B^T
// size of work for a thread block
#define BLK_M_nt  16
#define BLK_N_nt  24

#define BLK_M_nc  16
#define BLK_N_nc  24

#define BLK_K 8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#undef  version
#define version trans_nt
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_nc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A^T x B^T
// size of work for a thread block
#define BLK_M_tt  16
#define BLK_N_tt  24

#define BLK_M_tc  16
#define BLK_N_tc  24

#define BLK_M_ct  16
#define BLK_N_ct  24

#define BLK_M_cc  16
#define BLK_N_cc  24

#define BLK_K 8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 4
#define DIM_YA 16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#undef  version
#define version trans_tt
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_tc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_ct
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_cc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A^T x B
// size of work for a thread block
#define BLK_M_tn  24
#define BLK_N_tn  16

#define BLK_M_cn  24
#define BLK_N_cn  16

#define BLK_K  8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#undef  version
#define version trans_tn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_cn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef COMPLEX
#undef DOUBLE

#endif // ZGEMM_FERMI_KERNELS_H
