/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

*/

#ifndef GESVJ_KERNELS_CUH
#define GESVJ_KERNELS_CUH

#include "atomics.cuh"

// See gesvj_update_vectors_template_device below
// Parameters: BLK_M, BLK_N, BLK_K, DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG
// Constraints:
// -- BLK_M, BLK_N, and BLK_K fully divisible by { DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG }
// -- BLK_N >= NB
// -- (DIM_X * DIM_Y) == (DIM_Y * DIM_XU) == (DIM_XG * DIM_YG)

#ifdef MAGMA_HAVE_CUDA
// Tuning for CUDA is done only for nb = {16,32} on a GH200 system
// TODO: expand tuning for other NB values
#define zgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define zgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define zgesvj_update_nb_16 32, 16,  8,  8,  8,  8,  8,  8,  8
#define zgesvj_update_nb_32 16, 32, 16,  8,  8,  8,  8,  8,  8

#define cgesvj_update_nb_4  32,  4,  4,  4, 4,  4,  4,  4, 4
#define cgesvj_update_nb_8  32,  8,  8,  8, 8,  8,  8,  8, 8
#define cgesvj_update_nb_16 64, 16, 16, 16, 8, 16,  8, 16, 8
#define cgesvj_update_nb_32 64, 32, 16, 16, 8, 16,  8, 16, 8

#define dgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define dgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define dgesvj_update_nb_16 64, 16,  8,  8,  8,  8,  8,  8,  8
#define dgesvj_update_nb_32 64, 32, 16, 16,  8, 16,  8, 16,  8

#define sgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define sgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_16 64, 16,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_32 64, 32, 16, 16,  8, 16,  8, 16,  8

#else
// Tuning for HIP is done only for nb = {16,32} on a MI300A system
// TODO: expand tuning for other NB values
#define zgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define zgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define zgesvj_update_nb_16 16, 16, 16, 16,  8, 16,  8, 16,  8
#define zgesvj_update_nb_32 16, 32, 16, 16, 16, 16, 16, 16, 16

#define cgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define cgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define cgesvj_update_nb_16 16, 16, 16, 16,  8, 16,  8, 16,  8
#define cgesvj_update_nb_32 16, 32, 16, 16, 16, 16, 16, 16, 16

#define dgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define dgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define dgesvj_update_nb_16 32, 16, 16, 16,  8, 16,  8, 16,  8
#define dgesvj_update_nb_32 64, 32, 16, 16, 16, 16, 16, 16, 16

#define sgesvj_update_nb_4  32,  4,  4,  4,  4,  4,  4,  4,  4
#define sgesvj_update_nb_8  32,  8,  8,  8,  8,  8,  8,  8,  8
#define sgesvj_update_nb_16 32, 16, 16, 16,  8, 16,  8, 16,  8
#define sgesvj_update_nb_32 32, 32, 16, 16,  8, 16,  8, 16,  8

#endif

/***************************************************************************//*
  Device function for gesvj update vectors
    - For a given pair of block-column [Ai Aj] in a Jacobi sweep, we compute
      the eigen-decomposition of the Gram matrix: [Ai Aj]^H [Ai Aj]
    - The block columns are updated using the eigenvectors matrix 'G':
      [Ai Aj] = [Ai Aj] x G
    - However, Ai and Aj are not necessarily adjacent in memory, so the above
      multiplication is done on several stages using the device function below
    - U0 and U1 represent the block columns Ai and Aj, each is of size M x NB
    - G is the matrix of eigenvectors (size 2NB x 2NB)
    - sU and sG are shared memory workspaces
    - U0 and U1 are updated inplace
    - rU0/rU1 is a register buffer for U0/U1 (of full width, so only partitioning horizontally)
    - The top or bottom half of G can be stored in shared memory (sG)

  The device function uses the same building block of magmablas GEMM code.
  It is basically two cascaded GEMM operations.
*/
template<typename T,
        const int BLK_M,  const int BLK_N, const int BLK_K,
        const int DIM_X,  const int DIM_Y,
        const int DIM_XU, const int DIM_YU,
        const int DIM_XG, const int DIM_YG >
static __device__
void gesvj_update_vectors_template_device (
    int M, int NB,
    T* __restrict__ U0, int LDU0,
    T* __restrict__ U1, int LDU1,
    T* __restrict__  G, int LDG,
    T* sU, int sldu,
    T* sG, int sldg )
{
    constexpr int THR_M  = BLK_M / DIM_X;
    constexpr int THR_N  = BLK_N / DIM_Y;
    constexpr int BLK_N2 = BLK_N * 2;

    const int Gn = NB * 2;

    int tx = threadIdx.x;  // thread's m dimension
    int ty = threadIdx.y;  // thread's n dimension

    int tid = DIM_X * ty + tx;    // thread's global number

    int txU = tid % DIM_XU;    // tx within A
    int tyU = tid / DIM_XU;    // ty within A

    int txG = tid % DIM_XG;    // tx within A
    int tyG = tid / DIM_XG;    // ty within A

    int bx = blockIdx.x;   // block's m dimension
    int by = blockIdx.y;   // block's n dimension

    // Registers for the innermost loop
    T rU0[THR_N][THR_M];
    T rU1[THR_N][THR_M];

    // Registers for the dev->shmem copy
    T ru[BLK_K/DIM_YU][BLK_M/DIM_XU];
    T rg[BLK_N2/DIM_YG][BLK_K/DIM_XG];

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dU0 = U0 + bx*BLK_M     + tyU*LDU0 + txU;
    ptrdiff_t boundU0 = (LDU0*(NB-1) + M) - ( bx*BLK_M  + tyU*LDU0 + txU ) - 1;

    const T *offs_dU1 = U1 + bx*BLK_M     + tyU*LDU1 + txU;
    ptrdiff_t boundU1 = (LDU1*(NB-1) + M) - ( bx*BLK_M  + tyU*LDU1 + txU ) - 1;

    const T *offs_dG  = G + by*BLK_N*LDG + tyG*LDG + txG;
    ptrdiff_t boundG  = (LDG*(Gn-1) + NB) - ( by*BLK_N*LDG + tyG*LDG + txG ) -1;

    // Zero rU0, rU1
    zero_rgArray2D<T, THR_M, THR_N>(rU0);
    zero_rgArray2D<T, THR_M, THR_N>(rU1);

    // 1st mul.: U0 x [G00 G01] -> [rU0 rU1]
    read_gm2sm_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU> (offs_dU0, LDU0, boundU0, sU, sldu, txU, tyU );
    read_gm2sm_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG> (offs_dG,  LDG,  boundG,  sG, sldg, txG, tyG );
    __syncthreads();

    int k = 0;
    for(k = 0; k < NB-BLK_K; k+=BLK_K) {
        offs_dU0 += BLK_K * LDU0;
        boundU0  -= BLK_K * LDU0;

        offs_dG += BLK_K;
        boundG  -= BLK_K;

        // prefetch
        read_gm2rg_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU>(offs_dU0, LDU0, boundU0, ru);
        read_gm2rg_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG>(offs_dG,  LDG,  boundG,  rg);

        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y> (sU, sldu, sG + 0  * sldg, sldg, rU0, tx, ty);
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y> (sU, sldu, sG + NB * sldg, sldg, rU1, tx, ty);
        __syncthreads();

        // mv prefetched blocks into sm
        write_rg2sm_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU>( ru, sU, sldu, txU, tyU );
        write_rg2sm_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG>( rg, sG, sldg, txG, tyG );
        __syncthreads();

    }

    // Multiply last full (BLK_K) or partial block of
    k = NB - k;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>(k, sU, sldu, sG +  0 * sldg, sldg, rU0, tx, ty);
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>(k, sU, sldu, sG + NB * sldg, sldg, rU1, tx, ty);
    __syncthreads();
    // end of 1st mul.

    // 2nd mul.: U1 x [G10 G11] -> [rU0 rU1]
    // adjust dG to the bottom half og G
    offs_dG = (G + NB) + by*BLK_N*LDG + tyG*LDG + txG;
    boundG  = (LDG*(Gn-1) + NB) - ( by*BLK_N*LDG + tyG*LDG + txG ) -1;
    read_gm2sm_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU> (offs_dU1, LDU1, boundU1, sU, sldu, txU, tyU );
    read_gm2sm_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG> (offs_dG,  LDG,  boundG,  sG, sldg, txG, tyG );
    __syncthreads();

    for(k = 0; k < NB-BLK_K; k+=BLK_K) {
        offs_dU1 += BLK_K * LDU1;
        boundU1  -= BLK_K * LDU1;

        offs_dG += BLK_K;
        boundG  -= BLK_K;

        // prefetch
        read_gm2rg_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU>(offs_dU1, LDU1, boundU1, ru);
        read_gm2rg_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG>(offs_dG,  LDG,  boundG,  rg);

        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y> (sU, sldu, sG +  0 * sldg, sldg, rU0, tx, ty);
        multiply_full_block<T, BLK_K, THR_M, THR_N, DIM_X, DIM_Y> (sU, sldu, sG + NB * sldg, sldg, rU1, tx, ty);
        __syncthreads();

        // mv prefetched blocks into sm
        write_rg2sm_notrans<T, BLK_M, BLK_K,  DIM_XU, DIM_YU>( ru, sU, sldu, txU, tyU );
        write_rg2sm_notrans<T, BLK_K, BLK_N2, DIM_XG, DIM_YG>( rg, sG, sldg, txG, tyG );
        __syncthreads();
    }

    k = NB - k;
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>(k, sU, sldu, sG +  0 * sldg, sldg, rU0, tx, ty);
    multiply_partial_block<T, THR_M, THR_N, DIM_X, DIM_Y>(k, sU, sldu, sG + NB * sldg, sldg, rU1, tx, ty);
    __syncthreads();

    // Store results regs->dev
    #pragma unroll
    for (int n = 0; n < THR_N; n++) {
        int coord_dUn = by*BLK_N + n*DIM_Y + ty;
        #pragma unroll
        for (int m = 0; m < THR_M; m++) {
            int coord_dUm = bx*BLK_M + m*DIM_X + tx;
            //if(bx == 0)printf("(%d, %d, %d, %d): (Cm, Cn) = (%d, %d) -- (m, nb) = (%d, %d)\n", bx, by, tx, ty, coord_dUm, coord_dUn, M, NB);
            if (coord_dUm < M && coord_dUn < NB) {
                ptrdiff_t offsU0 = (ptrdiff_t)coord_dUn*(ptrdiff_t)LDU0 + (ptrdiff_t)coord_dUm;
                ptrdiff_t offsU1 = (ptrdiff_t)coord_dUn*(ptrdiff_t)LDU1 + (ptrdiff_t)coord_dUm;

                U0[offsU0] = rU0[n][m];
                U1[offsU1] = rU1[n][m];
            }
        }
    }
}

/***************************************************************************//*
  Batch kernel for gesvj update vectors
    - See the above description of gesvj_update_vectors_template_device
*/
template<typename T,
        const int BLK_M,  const int BLK_N, const int BLK_K,
        const int DIM_X,  const int DIM_Y,
        const int DIM_XU, const int DIM_YU,
        const int DIM_XG, const int DIM_YG >
static __global__ __launch_bounds__(DIM_X*DIM_Y)
void gesvj_update_vectors_template_batched_kernel(
    int m, int nb,
    T **dU0array, int LDU0,
    T **dU1array, int LDU1,
    T **dGarray,  int LDG,
    magma_int_t *heevj_info,
    int *heevj_nsweeps)
{
    extern __shared__ T* sdata_gesvj_vec_update[];
    const int batchid = blockIdx.z;

    // ** self terminate if the corresponding heevj has zero info after one sweep
    // ** if either heevj_info or heevj_nsweeps are NULL, assume non-convergence and execute the vector updates
    //    (this can be done by setting info and nsweeps to any values other than those used to detect convergence)
    int local_heevj_info    = (heevj_info    == NULL) ? 1 : (int)heevj_info[batchid];
    int local_heevj_nsweeps = (heevj_nsweeps == NULL) ? 2 : (int)heevj_nsweeps[batchid];
    if(local_heevj_info == 0 && local_heevj_nsweeps == 1) return;

    const int sldu = SLDA(BLK_M);
    const int sldg = SLDB(BLK_K);
    T* sU = (T*)sdata_gesvj_vec_update;  // sU is sldu x BLK_K
    T* sG = sU + sldu * BLK_K;           // sG is sldg x  BLK_N * 2

    gesvj_update_vectors_template_device<T, BLK_M, BLK_N, BLK_K, DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG >
    (m, nb, dU0array[batchid], LDU0, dU1array[batchid], LDU1, dGarray[batchid], LDG, sU, sldu, sG, sldg );
}

/***************************************************************************//*
  Batch kernel driver for gesvj update vectors
*/
template<typename T,
        const int BLK_M,  const int BLK_N, const int BLK_K,
        const int DIM_X,  const int DIM_Y,
        const int DIM_XU, const int DIM_YU,
        const int DIM_XG, const int DIM_YG >
magma_int_t gesvj_update_vectors_template_batched_kernel_driver(
    magma_int_t m, magma_int_t nb,
    T **dU0array, magma_int_t lddu0,
    T **dU1array, magma_int_t lddu1,
    T **dGarray,  magma_int_t lddg,
    magma_int_t *heevj_info, int *heevj_nsweeps,
    magma_int_t batchCount, magma_queue_t queue )
{
    size_t shmem = 0;
    shmem += SLDA(BLK_M) * BLK_K  * sizeof(T);  // sA
    shmem += SLDB(BLK_K) * BLK_N * 2 * sizeof(T);  // sB

    // configure shmem
    #if CUDA_VERSION >= 9000
    int shmem_max;
    magma_device_t device;
    magma_getdevice( &device );
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= (size_t)shmem_max) {
        cudaFuncSetAttribute(
            gesvj_update_vectors_template_batched_kernel<T, BLK_M, BLK_N, BLK_K, DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG >,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #endif

    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), 1, ibatch );

        T **dU0array_i = dU0array + i;
        T **dU1array_i = dU1array + i;
        T **dGarray_i  = dGarray + i;
        void *kernel_args[] = {&m, &nb, &dU0array_i, &lddu0, &dU1array_i, &lddu1, &dGarray_i, &lddg, &heevj_info, &heevj_nsweeps};

        cudaError_t e = cudaLaunchKernel((void*)gesvj_update_vectors_template_batched_kernel<T, BLK_M, BLK_N, BLK_K, DIM_X, DIM_Y, DIM_XU, DIM_YU, DIM_XG, DIM_YG>,
                            dimGrid, dimBlock, kernel_args, shmem, queue->cuda_stream());

        if(e != cudaSuccess) {
            return -100;
        }
    }

    return 0;
}

/***************************************************************************//*
  A device function to generate a round-robin parallel ordering
    - The generation can be done independently by each GPU thread locally,
      i.e., no global generation is shared/queried by the threads
    - 'n'  : the number of columns/block-columns, must be even
    - 'col': the current column ID processed by the thread
    - Each thread needs to call this function twice to generate the next pair of columns
*/
static __host__  __device__ __inline__ int
get_next_column(int n, int col)
{
    int newcol = 0;
    newcol = (col % 2 == 0          ) ? -2 : 2;
    newcol = (col == n-1 || col == 2) ? -1 : newcol;
    newcol = (col == 0              ) ?  0 : newcol;
    return (col + newcol);
}

/***************************************************************************//*
  A kernel for setting up several pointer arrays for a given iteration
  (specified by i_gesvj_sweep), in the batch Jacobi SVD routine
    - dUi_array/dUo_array are the pointer arrays of the block columns of U (left vectors)
      dUi_array and dUo_array could be the same if inplace vector updates are used
    - dVi_array/dVo_array are the pointer arrays of the block columns of V (right vectors)
      dVi_array and dVo_array could be the same if inplace vector updates are used
    - 'nb' is the width of a block-column
    - 'i_gesvj_sweep' is the iteration ID inside a Jacobi sweep
    - 'nblk_col2' is the number of block-columns, must be even

  For computing the Gram matrices
    - 'dAgemm0_array' is the 'A' array of pointers for the batch GEMM computing the Gram matrices
    - 'dAgemm0_array' is the 'B' array of pointers for the batch GEMM computing the Gram matrices
    - The 'C' array of pointersfor the Gram matrices is not computed here. It needs to be generated
      once at the beginning of the batch SVD top-level routine

  For vector updates (if done using std batch GEMM)
    - 'dUjVj_input_array' contains two copies of Uj pointers followed by two copies of Vj pointers
    - 'dUkVk_input_array' contains two copies of Uk pointers followed by two copies of Vk pointers
    - 'dUjkVjk_output_array' contains the pointers of the block-columns of the left/right vectors
      to be updated
*/
template<typename T>
static __global__
void gesvj_setup_ptr_arrays_kernel(
        magma_vec_t jobv, int i_gesvj_sweep, int nb, int nblk_col2,
        T** dUi_array, T** dUo_array, int lddu,
        T** dVi_array, T** dVo_array, int lddv,
        T** dAgemm0_array, T** dAgemm1_array,
        T** dUjVj_input_array, T** dUkVk_input_array, T** dUjkVjk_output_array,
        int flat_batchCount)
{
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sub_batch = nblk_col2 / 2;

    const int pb_id  = gtx / sub_batch;
    const int tx     = gtx % sub_batch;

    int j = 2 * tx + 0;
    int k = 2 * tx + 1;
    for(int i = 0; i < i_gesvj_sweep; i++ ) {
        j = get_next_column(nblk_col2, j);
        k = get_next_column(nblk_col2, k);
    }

    if(gtx < flat_batchCount) {
        // base U ptr's for each problem [input & output]
        T* U  = (T*)dUi_array[pb_id];
        T* Uo = (T*)dUo_array[pb_id];

        // original (j, k) pairs for U
        T* Uj = U + j * nb * lddu;
        T* Uk = U + k * nb * lddu;

        // updated (j, k) pairs for U
        T* Ujo = Uo + j * nb * lddu;
        T* Uko = Uo + k * nb * lddu;

        // herk 1st third: for gemm Aj' x Aj
        dAgemm0_array[gtx + 0 * flat_batchCount]  = Uj;
        dAgemm1_array[gtx + 0 * flat_batchCount]  = Uj;

        // herk 2nd third: for gemm Ak' x Aj
        dAgemm0_array[gtx + 1 * flat_batchCount]  = Uk;
        dAgemm1_array[gtx + 1 * flat_batchCount]  = Uj;

        // herk 3rd third: for gemm Ak' x Ak
        dAgemm0_array[gtx + 2 * flat_batchCount]  = Uk;
        dAgemm1_array[gtx + 2 * flat_batchCount]  = Uk;

        // inputs for 1st update of U
        dUjVj_input_array[gtx + 0 * flat_batchCount] = Uj;
        dUjVj_input_array[gtx + 1 * flat_batchCount] = Uj;

        // inputs for 2nd update of U
        dUkVk_input_array[gtx + 0 * flat_batchCount] = Uk;
        dUkVk_input_array[gtx + 1 * flat_batchCount] = Uk;

        // output for U update
        dUjkVjk_output_array[gtx + 0 * flat_batchCount] = Ujo;
        dUjkVjk_output_array[gtx + 1 * flat_batchCount] = Uko;

        if(jobv == MagmaVec || jobv == MagmaSomeVec) {
            // base V ptr's for each problem [input & output]
            T* V  = (T*)dVi_array[pb_id];
            T* Vo  = dVo_array[pb_id];

            // original (j, k) pairs for V
            T* Vj = V + j * nb * lddv;
            T* Vk = V + k * nb * lddv;

            // updated (j, k) pairs for V
            T* Vjo = Vo + j * nb * lddv;
            T* Vko = Vo + k * nb * lddv;

            // inputs for 1st update of V
            dUjVj_input_array[gtx + 2 * flat_batchCount] = Vj;
            dUjVj_input_array[gtx + 3 * flat_batchCount] = Vj;

            // inputs for 2nd update of V
            dUkVk_input_array[gtx + 2 * flat_batchCount] = Vk;
            dUkVk_input_array[gtx + 3 * flat_batchCount] = Vk;

            // output for V update
            dUjkVjk_output_array[gtx + 2 * flat_batchCount] = Vjo;
            dUjkVjk_output_array[gtx + 3 * flat_batchCount] = Vko;
        }
    }
}

/***************************************************************************//*
  gesvj pointer setup kernel driver
    - See the above description of gesvj_setup_ptr_arrays_kernel
*/
template<typename T>
void gesvj_setup_ptr_arrays_kernel_batched(
        magma_vec_t jobv, int i_gesvj_sweep, int nb, int nblk_col2,
        T** dUi_array, T** dUo_array, int lddu,
        T** dVi_array, T** dVo_array, int lddv,
        T** dAgemm0_array, T** dAgemm1_array,
        T** dUjVj_input_array, T** dUkVk_input_array, T** dUjkVjk_output_array,
        int flat_batchCount, magma_queue_t queue)
{
    const int nthreads = 128;

    dim3 threads(nthreads, 1, 1);
    dim3 grid(magma_ceildiv(flat_batchCount, nthreads), 1, 1);
    gesvj_setup_ptr_arrays_kernel<T><<<grid, threads, 0, queue->cuda_stream()>>>
    (jobv, i_gesvj_sweep, nb, nblk_col2,
     dUi_array, dUo_array, lddu,
     dVi_array, dVo_array, lddv,
     dAgemm0_array, dAgemm1_array,
     dUjVj_input_array, dUkVk_input_array, dUjkVjk_output_array,
     flat_batchCount);
}

/******************************************************************************/
/*********************     GESVJ Finalize Values         **********************/
/******************************************************************************/
// device function to compute a x conj(a)
// TODO: move to a more generic header for potential use by other functions
template<typename T, typename TR>
__device__ __inline__ TR compute_a_x_conja(T a) {return a * a;}

template<>
__device__ __inline__ float compute_a_x_conja<magmaFloatComplex, float>(magmaFloatComplex a)
{
    return (MAGMA_C_REAL(a) * MAGMA_C_REAL(a) + MAGMA_C_IMAG(a) * MAGMA_C_IMAG(a));
}

template<>
__device__ __inline__ double compute_a_x_conja<magmaDoubleComplex, double>(magmaDoubleComplex a)
{
    return (MAGMA_Z_REAL(a) * MAGMA_Z_REAL(a) + MAGMA_Z_IMAG(a) * MAGMA_Z_IMAG(a));
}

/***************************************************************************//*
  Device function to compute the column-wise norm of a matrix
    - 'A' is of size m x n
    - dSigma is the output in global memory
    - sSigma is a shared memory workspace
*/
template<typename T, typename TR, int DIMX, int DIMY>
static __device__ __inline__
void gesvj_finalize_svd_device( int m, int n, T const* dA, int ldda, TR* dSigma, TR* sSigma)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int gty = bx * DIMY + ty;

    const T* dy = dA + gty * ldda;
    T  rA;
    TR rTmp = 0;

    const int m_ = m-DIMX;
    if(gty < n) {
        int i = 0;
        for(i = 0; i < m_; i+=DIMX) {
            rA    = dy[i + tx];
            rTmp += compute_a_x_conja<T, TR>( rA );
        }

        if( tx < m-i ) {
            rA    = dy[i + tx];
            rTmp += compute_a_x_conja<T, TR>( rA );
        }
    }

    // write partial squares into shared mem.
    sSigma[ty * DIMX + tx] = rTmp;
    __syncthreads();

    // reduction for each column in shared mem.
    // TODO: parallel reduction
    const int myn = min(DIMY, n - bx * DIMY);
    if(ty == 0 && tx < myn) {
        rTmp = 0;
        for(int i = 0; i < DIMX; i++) {
            rTmp += sSigma[tx * DIMX + i];
        }
        rTmp = sqrt(rTmp);
        dSigma[bx * DIMY + tx] = rTmp;
    }
}

/***************************************************************************//*
  Batch kernel to compute the column-wise norm of a matrix
    - See the above description of gesvj_finalize_svd_device
*/

template<typename T, typename TR, int DIMX, int DIMY>
__global__
void gesvj_finalize_svd_kernel_batched(
        int m, int n,
        T const* const* dA_array, int ldda, TR** dSigma_array)
{
    extern __shared__ TR sdata[];

    const int batchid = blockIdx.z;

    T const* dA = dA_array[batchid];
    TR*  dSigma = dSigma_array[batchid];

    TR* sSigma = (TR*)sdata;

    gesvj_finalize_svd_device<T, TR, DIMX, DIMY>(m, n, dA, ldda, dSigma, sSigma);

}

/***************************************************************************//*
  Batch kernel driver to compute the column-wise norm of a matrix
    - See the above description of gesvj_finalize_svd_device
*/
template<typename T, typename TR>
void gesvj_finalize_svd_batched_kernel_driver(
        magma_int_t m, magma_int_t n,
        T const* const* dA_array, magma_int_t ldda,
        TR** dSigma_array, magma_int_t batchCount, magma_queue_t queue)
{
    // DIMX must be >= DIMY
    constexpr int DIMX = 32;
    constexpr int DIMY = 8;

    const int BDIMX = magma_ceildiv(n, DIMY);
    const int BDIMY = 1;

    dim3 threads(DIMX, DIMY, 1);

    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += DIMX * DIMY * sizeof(TR);  // sSigma

    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( BDIMX, BDIMY, ibatch );
        gesvj_finalize_svd_kernel_batched<T, TR, DIMX, DIMY><<<grid, threads, shmem, queue->cuda_stream()>>>
        (m, n, dA_array + i, ldda, dSigma_array + i);
    }
}

/******************************************************************************/
/*********************     GESVJ Finalize Vectors        **********************/
/******************************************************************************/
// device function to reorder U & V (also scales U by 1/Sigma)
// works for m >= n (hence the name m_gte_n)
template<typename T, typename TR, int DIMX, int BLK_WIDTH>
static __device__ __inline__
void gesvj_finalize_vectors_m_gte_n_device(
        int need_u, int need_v,
        int m, int n,
        T const* dUi, int lddui,
        T const* dVi, int lddvi,
        T      * dUo, int ldduo,
        T      * dVo, int lddvo,
        TR* ds_sorted, magma_int_t* sort_index)
{
    __shared__ TR  ssigma[BLK_WIDTH];
    __shared__ int sindex[BLK_WIDTH];

    const int tx  = threadIdx.x;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int gty = by * BLK_WIDTH + tx;

    const int my_Um = min(DIMX,      m - bx * DIMX);
    const int my_Vm = min(DIMX,      n - bx * DIMX);
    const int myn   = min(BLK_WIDTH, n - by * BLK_WIDTH);

    // advance pointers
    dUi += bx * DIMX;
    dVi += bx * DIMX;
    dUo += bx * DIMX + by * BLK_WIDTH * ldduo;
    dVo += bx * DIMX + by * BLK_WIDTH * lddvo;

    // read sorted values & their indices
    TR  rsigma[BLK_WIDTH];
    int     ri[BLK_WIDTH];

    // defaults
    if(tx < BLK_WIDTH) {
        ssigma[tx] = 1.;  // should be ok since TR is real (init by one because we will divide by sigma)
        sindex[tx] = n-1; // by default, point to the last column of the non-ordered U/V
    }

    // read actual sigma/index
    if(tx < myn) {
        ssigma[tx] = ds_sorted[gty];
        sindex[tx] = (int)sort_index[gty];
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < BLK_WIDTH; j++) {
        rsigma[j] = 1. / ssigma[j];
        ri[j]     = sindex[j];
    }

    if(need_u == 1) {
        T rU[BLK_WIDTH];
        // left vectors are m x n for reduced svd
        if(tx < my_Um) {
            #pragma unroll
            for(int j = 0; j < BLK_WIDTH; j++) {
                rU[j] = dUi[ ri[j] * lddui + tx ];
            }

            // scale by 1/sigma (rsigma already has the reciprocal)
            #pragma unroll
            for(int j = 0; j < BLK_WIDTH; j++) {
                rU[j] *= rsigma[j];
            }

            // write back re-ordered
            if(myn == BLK_WIDTH) {
                #pragma unroll
                for(int j = 0; j < BLK_WIDTH; j++) {
                    dUo[ j * ldduo + tx ] = rU[j];
                }
            }
            else {
                #pragma unroll
                for(int j = 0; j < BLK_WIDTH; j++) {
                    if(j < myn) {
                        dUo[ j * ldduo + tx ] = rU[j];
                    }
                }
            }
        }
    }

    if(need_v == 1) {
        T rV[BLK_WIDTH];
        // right vectors are nxn for reduced svd
        if(tx < my_Vm) {
            #pragma unroll
            for(int j = 0; j < BLK_WIDTH; j++) {
                rV[j] = dVi[ ri[j] * lddvi + tx ];
            }

            // write back re-ordered
            if(myn == BLK_WIDTH) {
                #pragma unroll
                for(int j = 0; j < BLK_WIDTH; j++) {
                    dVo[ j * lddvo + tx ] = rV[j];
                }
            }
            else {
                #pragma unroll
                for(int j = 0; j < BLK_WIDTH; j++) {
                    if(j < myn) {
                        dVo[ j * lddvo + tx ] = rV[j];
                    }
                }
            }
        }
    }
}

/******************************************************************************/
// device function to reorder U & V (also scales U by 1/Sigma)
// works for m < n (hence the name m_lt_n)
// since zgesvj_batched works internally on m >= n only
// dUi (n x m) will be copied into dVo (n x m) and,
// dVi (m x m) will be copied into dUo (m x m)
template<typename T, typename TR, int DIMX, int DIMY>
static __device__ __inline__
void gesvj_finalize_vectors_m_lt_n_device(
        int need_u, int need_v,
        int m, int n,
        T const* dUi, int lddui,
        T const* dVi, int lddvi,
        T      * dUo, int ldduo,
        T      * dVo, int lddvo,
        TR* ds_sorted, magma_int_t* sort_index)
{
    __shared__ T   sA[DIMY][DIMX];
    __shared__ TR  ssigma[DIMY];
    __shared__ int sindex[DIMY];

    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * DIMX + tx;
    const int gty = by * DIMY + ty;
    const int gtx = bx * DIMY + tx;

    const int mym    = min(DIMY, m - by * DIMY);

    // defaults
    if(tid < DIMY) {
        ssigma[tid] = 1.;  // should be ok since TR is real (init by one because we will divide by sigma)
        sindex[tid] = n-1; // by default, point to the last column of the non-ordered U/V
    }

    // read actual sigma/index
    if(tid < mym) {
        ssigma[tid] = 1. /  ds_sorted[by * DIMY + tid];
        sindex[tid] = (int)sort_index[by * DIMY + tid];
    }
    __syncthreads();

    if(need_u == 1) {
        // copy dVi (m x m) into dUo (m x m)
        int my_sindex = sindex[ty];

        if(gtx < m && gty < m) {
            dUo[gty * ldduo + gtx] = dVi[my_sindex * lddvi + gtx];
        }
    }

    if(need_v == 1) {
        // dUi (n x m) will be conj-transposed into dVo (m x n)
        int my_sindex = sindex[ty];
        TR  my_sigma  = ssigma[ty];

        if(gtx < n && gty < m) {
            T rA = dUi[my_sindex * lddui + gtx];
            rA  *= ssigma[ty];
            dVo[gty * lddvo + gtx] = rA;
        }
    }
}

/******************************************************************************/
// kernel to finalize the singular vectors (m >= n)
template<typename T, typename TR, int DIMX, int BLK_WIDTH>
__global__
void gesvj_finalize_vectors_m_gte_n_kernel_batched(
        int need_u, int need_v,
        int m, int n,
        T const* const* dUi_array, int lddui,
        T const* const* dVi_array, int lddvi,
        T            ** dUo_array, int ldduo,
        T            ** dVo_array, int lddvo,
        TR           ** dSigma_array,
        magma_int_t  ** index_array)
{
    const int batchid = blockIdx.z;

    T const* dUi = dUi_array[batchid];
    T const* dVi = dVi_array[batchid];
    T      * dUo = dUo_array[batchid];
    T      * dVo = dVo_array[batchid];
    TR     * dS  = dSigma_array[batchid];
    magma_int_t *sort_index = index_array[batchid];

    gesvj_finalize_vectors_m_gte_n_device<T, TR, DIMX, BLK_WIDTH>
    (need_u, need_v, m, n, dUi, lddui, dVi, lddvi, dUo, ldduo, dVo, lddvo, dS, sort_index);
}

/******************************************************************************/
// kernel to finalize the singular vectors (m < n)
template<typename T, typename TR, int DIMX, int DIMY>
__global__
void gesvj_finalize_vectors_m_lt_n_kernel_batched(
        int need_u, int need_v,
        int m, int n,
        T const* const* dUi_array, int lddui,
        T const* const* dVi_array, int lddvi,
        T            ** dUo_array, int ldduo,
        T            ** dVo_array, int lddvo,
        TR           ** dSigma_array,
        magma_int_t  ** index_array)
{
    const int batchid = blockIdx.z;

    T const* dUi = dUi_array[batchid];
    T const* dVi = dVi_array[batchid];
    T      * dUo = dUo_array[batchid];
    T      * dVo = dVo_array[batchid];
    TR     * dS  = dSigma_array[batchid];
    magma_int_t *sort_index = index_array[batchid];

    gesvj_finalize_vectors_m_lt_n_device<T, TR, DIMX, DIMY>
    (need_u, need_v, m, n, dUi, lddui, dVi, lddvi, dUo, ldduo, dVo, lddvo, dS, sort_index);

}

/******************************************************************************/
// kernel driver to finalize the singular vectors
template<typename T, typename TR>
void gesvj_finalize_vectors_batched_kernel_driver(
        magma_vec_t jobu, magma_vec_t jobv,
        magma_int_t m, magma_int_t n,
        T const* const* dUi_array, int lddui,
        T const* const* dVi_array, int lddvi,
        T            ** dUo_array, int ldduo,
        T            ** dVo_array, int lddvo,
        TR           ** dSigma_array, magma_int_t  ** index_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    int need_u = (jobu == MagmaVec || jobu == MagmaSomeVec) ? 1 : 0;
    int need_v = (jobv == MagmaVec || jobv == MagmaSomeVec) ? 1 : 0;

    if(m >= n) {
        // DIMX must be >= DIMY
        constexpr int DIMX      = 32; // kernel assumes a 1D thread config
        constexpr int BLK_WIDTH =  8;

        const int BDIMX = magma_ceildiv(m, DIMX);
        const int BDIMY = magma_ceildiv(n, BLK_WIDTH);

        dim3 threads(DIMX, 1, 1);

        size_t shmem = 0;
        magma_int_t max_batchCount = queue->get_maxBatch();

        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( BDIMX, BDIMY, ibatch );
            gesvj_finalize_vectors_m_gte_n_kernel_batched<T, TR, DIMX, BLK_WIDTH><<<grid, threads, shmem, queue->cuda_stream()>>>
            ( need_u, need_v, m, n,
              dUi_array + i, lddui, dVi_array + i, lddvi,
              dUo_array + i, ldduo, dVo_array + i, lddvo,
              dSigma_array + i, index_array  + i);
        }
    }
    else {
        constexpr int DIMX = 16; // kernel assumes a 2D thread config
        constexpr int DIMY = 16;

        const int BDIMX = magma_ceildiv(n, DIMX);
        const int BDIMY = magma_ceildiv(m, DIMY);

        dim3 threads(DIMX, DIMX, 1);

        size_t shmem = 0;
        magma_int_t max_batchCount = queue->get_maxBatch();

        for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            dim3 grid( BDIMX, BDIMY, ibatch );
            gesvj_finalize_vectors_m_lt_n_kernel_batched<T, TR, DIMX, DIMY><<<grid, threads, shmem, queue->cuda_stream()>>>
            ( need_u, need_v, m, n,
              dUi_array + i, lddui, dVi_array + i, lddvi,
              dUo_array + i, ldduo, dVo_array + i, lddvo,
              dSigma_array + i, index_array  + i);
        }
    }
}
#endif //GESVJ_KERNELS_CUH
