/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c


       @author Adrien REMY
*/
#include "magma_internal.h"
#include "zgerbt.h"

#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

/******************************************************************************/
static __device__ void
magmablas_zelementary_multiplication_devfunc(
    int Am, int An,
    magmaDoubleComplex *dA, int Ai, int Aj, int ldda,
    magmaDoubleComplex *du, int Ui,
    magmaDoubleComplex *dv, int Vi)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    __shared__ magmaDoubleComplex u1[block_height], u2[block_height], v1[block_width], v2[block_width];

    magmaDoubleComplex a00 = MAGMA_Z_ZERO, a10 = MAGMA_Z_ZERO, a01 = MAGMA_Z_ZERO, a11 = MAGMA_Z_ZERO;
    magmaDoubleComplex b1 = MAGMA_Z_ZERO, b2 = MAGMA_Z_ZERO, b3 = MAGMA_Z_ZERO, b4 = MAGMA_Z_ZERO;

    int Ar = Am;
    int Ac = An;

    // advance ptrs w.r.t offsets
    dA += Aj * ldda + Ai;
    du += Ui;
    dv += Vi;

    int Ar1 = (Ar+1) / 2;
    int Ar2 = Ar - Ar1;
    int Ac1 = (Ac+1) / 2;
    int Ac2 = Ac - Ac1;
    int Ul1 = Ar1;
    int Ul2 = Ar2;
    int Vl1 = Ac1;
    int Vl2 = Ac2;

    bool valid_00 = idx < Ar1 && idy < Ac1;
    bool valid_01 = idx < Ar1 && idy < Ac2;
    bool valid_10 = idx < Ar2 && idy < Ac1;
    bool valid_11 = idx < Ar2 && idy < Ac2;

    dA += idx + idy * ldda;
    du += idx;
    dv += idy;

    if(idx < Ul1) u1[tx] = du[0];
    if(idy < Vl1) v1[ty] = dv[0];
    if(idx < Ul2) u2[tx] = du[Ul1];
    if(idy < Vl2) v2[ty] = dv[Vl1];
    __syncthreads();

    if(valid_00) a00 = dA[0];
    if(valid_01) a01 = dA[ldda*Ac1];
    if(valid_10) a10 = dA[Ar1];
    if(valid_11) a11 = dA[ldda*Ac1+Ar1];
    __syncthreads();

    b1 = a00 + a01;
    b2 = a10 + a11;
    b3 = a00 - a01;
    b4 = a10 - a11;

    if(valid_00) dA[0]            = u1[tx] * v1[ty] * (b1 + b2);
    if(valid_01) dA[ldda*Ac1]     = u1[tx] * v2[ty] * (b3 + b4);
    if(valid_10) dA[Ar1]          = u2[tx] * v1[ty] * (b1 - b2);
    if(valid_11) dA[ldda*Ac1+Ar1] = u2[tx] * v2[ty] * (b3 - b4);
    __syncthreads();
}

/******************************************************************************/
__global__ void
magmablas_zelementary_multiplication_kernel(
    int Am, int An,
    magmaDoubleComplex *dA, int Ai, int Aj, int ldda,
    magmaDoubleComplex *du, int Ui,
    magmaDoubleComplex *dv, int Vi)
{
    magmablas_zelementary_multiplication_devfunc( Am, An, dA, Ai, Aj, ldda, du, Ui, dv, Vi );
}


/******************************************************************************/
__global__ void
magmablas_zelementary_multiplication_kernel_batched(
    int Am, int An,
    magmaDoubleComplex **dA_array, int Ai, int Aj, int ldda,
    magmaDoubleComplex *du, int Ui,
    magmaDoubleComplex *dv, int Vi)
{
    int batchid = blockIdx.z;
    magmablas_zelementary_multiplication_v2_devfunc( Am, An, dA_array[batchid], Ai, Aj, ldda, du, Ui, dv, Vi );
}


/******************************************************************************/
static __device__ void
magmablas_zapply_vector_devfunc(
    int n,
    magmaDoubleComplex *du, magmaDoubleComplex *db)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex a1, a2;
    if(n < 1) return;

    const int n1 = (n+1)/2;
    const int n2 = n - n1;

    if (idx < n1) {
        du += idx;
        db += idx;
    }

    a1 = (idx < n1) ? du[ 0] * db[ 0] : MAGMA_Z_ZERO;
    a2 = (idx < n2) ? du[n1] * db[n1] : MAGMA_Z_ZERO;

    if(idx < n1) db[ 0] = a1 + a2;
    if(idx < n2) db[n1] = a1 -a2;
}


/******************************************************************************/
__global__ void
magmablas_zapply_vector_kernel(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu, magmaDoubleComplex *db, int lddb, int offsetb)
{
    for(int i = 0; i < nrhs; i++) {
        magmablas_zapply_vector_devfunc(n, du+offsetu, db + i*lddb + offsetb);
    }
}


/******************************************************************************/
__global__ void
magmablas_zapply_vector_kernel_batched(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu, magmaDoubleComplex **db_array, int lddb, int offsetb )
{
    int batchid = blockIdx.y;
    for(int i = 0; i < nrhs; i++) {
        magmablas_zapply_vector_devfunc(n, du+offsetu, db_array[batchid]+ i * lddb + offsetb);
    }
}


/******************************************************************************/
static __device__ void
magmablas_zapply_transpose_vector_devfunc(
    int n,
    magmaDoubleComplex *du,magmaDoubleComplex *db )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n1  = (n + 1) / 2;
    const int n2  = n - n1;
    if(n < 2) return;

    magmaDoubleComplex a1,a2;

    if (idx < n1) {
        du += idx;
        db += idx;

        a1 = db[0] + ((idx < n2) ? db[n1] : MAGMA_Z_ZERO);
        a2 = db[0] - ((idx < n2) ? db[n1] : MAGMA_Z_ZERO);
    }

    if(idx < n1) db[ 0] = du[ 0] * a1;
    if(idx < n2) db[n1] = du[n1] * a2;
}


/******************************************************************************/
__global__ void
magmablas_zapply_transpose_vector_kernel(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu, magmaDoubleComplex *db, int lddb, int offsetb )
{
    for(int i = 0; i < nrhs; i++) {
        magmablas_zapply_transpose_vector_devfunc(n, du+offsetu, db+ i*lddb + offsetb);
    }
}


/******************************************************************************/
__global__ void
magmablas_zapply_transpose_vector_kernel_batched(
    int n, int nrhs,
    magmaDoubleComplex *du, int offsetu, magmaDoubleComplex **db_array, int lddb, int offsetb )
{
    int batchid = blockIdx.y;
    for(int i = 0; i < nrhs; i++) {
        magmablas_zapply_transpose_vector_devfunc(n, du+offsetu, db_array[batchid]+ i*lddb + offsetb);
    }
}
