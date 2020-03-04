/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#ifndef HEMM_TEMPLATE_KERNEL_VBATCHED_CUH
#define HEMM_TEMPLATE_KERNEL_VBATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "hemm_template_device.cuh"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_vbatched_ll_kernel(
    magma_int_t *M, magma_int_t *N, 
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC, 
    int specM, int specN)
{
    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < roffA || my_M < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max ( max(roffA, roffC), max(coffA, roffB) );
    my_N -= max( coffB, coffC );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );
    
    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= magma_ceildiv(my_M, BLK_M) ) return;
    if( blockIdx.y >= magma_ceildiv(my_N, BLK_N) ) return;
    
    hemm_template_device_ll
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( my_M, my_N, 
      Aarray[batchid] + (int)LDA[batchid] *  coffA + roffA, (int)LDA[batchid], 
      Barray[batchid] + (int)LDB[batchid] *  coffB + roffB, (int)LDB[batchid], 
      Carray[batchid] + (int)LDC[batchid] *  coffC + roffC, (int)LDC[batchid], 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_vbatched_lu_kernel(
    magma_int_t *M, magma_int_t *N, 
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC, 
    int specM, int specN)
{
    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M < roffA || my_M < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max ( max(roffA, roffC), max(coffA, roffB) );
    my_N -= max( coffB, coffC );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );
    
    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= magma_ceildiv(my_M, BLK_M) ) return;
    if( blockIdx.y >= magma_ceildiv(my_N, BLK_N) ) return;
    
    hemm_template_device_lu
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( my_M, my_N, 
      Aarray[batchid] + (int)LDA[batchid] *  coffA + roffA, (int)LDA[batchid], 
      Barray[batchid] + (int)LDB[batchid] *  coffB + roffB, (int)LDB[batchid], 
      Carray[batchid] + (int)LDC[batchid] *  coffC + roffC, (int)LDC[batchid], 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_vbatched_rl_kernel(
    magma_int_t *M, magma_int_t *N, 
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC, 
    int specM, int specN)
{
    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_N < roffA || my_N < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( roffB, roffC ); 
    my_N -= max( max(coffB, roffA), max(coffA, coffC) );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );
    
    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= magma_ceildiv(my_M, BLK_M) ) return;
    if( blockIdx.y >= magma_ceildiv(my_N, BLK_N) ) return;
    
    hemm_template_device_rl
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( my_M, my_N, 
      Aarray[batchid] + (int)LDA[batchid] *  coffA + roffA, (int)LDA[batchid], 
      Barray[batchid] + (int)LDB[batchid] *  coffB + roffB, (int)LDB[batchid], 
      Carray[batchid] + (int)LDC[batchid] *  coffC + roffC, (int)LDC[batchid], 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_vbatched_ru_kernel(
    magma_int_t *M, magma_int_t *N, 
    T const * const * Aarray, magma_int_t *LDA,
    T const * const * Barray, magma_int_t *LDB,
    T**       Carray, magma_int_t *LDC,
    T alpha, T beta, 
    int roffA, int coffA, int roffB, int coffB, int roffC, int coffC, 
    int specM, int specN)
{
    const int batchid = blockIdx.z;
    int my_M = (int)M[batchid];
    int my_N = (int)N[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_N < roffA || my_N < coffA ) return;
    if( my_M < roffB || my_N < coffB ) return;
    if( my_M < roffC || my_N < coffC ) return;
    // compute the maximum allowed value for m, n, k based on the input offsets
    my_M -= max( roffB, roffC ); 
    my_N -= max( max(coffB, roffA), max(coffA, coffC) );
    // check if the user forces values for m, n, and k
    my_M = ( specM <= 0 ) ? my_M : min( my_M, specM );
    my_N = ( specN <= 0 ) ? my_N : min( my_N, specN );
    
    if(my_M <= 0 || my_N <= 0 ) return;
    if( Aarray[batchid] == NULL || Barray[batchid] == NULL || Carray[batchid] == NULL ) return;
    if( blockIdx.x >= magma_ceildiv(my_M, BLK_M) ) return;
    if( blockIdx.y >= magma_ceildiv(my_N, BLK_N) ) return;
    
    hemm_template_device_ru
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( my_M, my_N, 
      Aarray[batchid] + (int)LDA[batchid] *  coffA + roffA, (int)LDA[batchid], 
      Barray[batchid] + (int)LDB[batchid] *  coffB + roffB, (int)LDB[batchid], 
      Carray[batchid] + (int)LDC[batchid] *  coffC + roffC, (int)LDC[batchid], 
      alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template_vbatched(
    magma_side_t side, magma_uplo_t uplo, 
    magma_int_t *m, magma_int_t *n, 
    T const * const * dA_array, magma_int_t *ldda,
    T const * const * dB_array, magma_int_t *lddb,
    T**       dC_array, magma_int_t *lddc,
    T alpha, T beta, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t specM, magma_int_t specN, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = 50000;
    dim3 threads(DIM, DIM, 1);
    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( magma_ceildiv(max_m, BLK_M), magma_ceildiv(max_n, BLK_N), ibatch );
                hemm_template_vbatched_ll_kernel <T, DIM, BLK_M, BLK_N, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
                ( m+i, n+i, dA_array+i, ldda+i, dB_array+i, lddb+i, dC_array+i, lddc+i, 
                  alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN );
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( magma_ceildiv(max_m, BLK_M), magma_ceildiv(max_n, BLK_N), ibatch );
                hemm_template_vbatched_lu_kernel <T, DIM, BLK_M, BLK_N, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
                ( m+i, n+i, dA_array+i, ldda+i, dB_array+i, lddb+i, dC_array+i, lddc+i, 
                  alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN );
            }
        }
    }else{
        if(uplo == MagmaLower){
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( magma_ceildiv(max_m, BLK_M), magma_ceildiv(max_n, BLK_N), ibatch );
                hemm_template_vbatched_rl_kernel <T, DIM, BLK_M, BLK_N, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
                ( m+i, n+i, dA_array+i, ldda+i, dB_array+i, lddb+i, dC_array+i, lddc+i, 
                  alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN );
            }
        }else{
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t ibatch = min(max_batchCount, batchCount-i);
                dim3 grid( magma_ceildiv(max_m, BLK_M), magma_ceildiv(max_n, BLK_N), ibatch );
                hemm_template_vbatched_ru_kernel <T, DIM, BLK_M, BLK_N, CONJA><<< grid, threads, 0, queue->cuda_stream() >>>
                ( m+i, n+i, dA_array+i, ldda+i, dB_array+i, lddb+i, dC_array+i, lddc+i, 
                  alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN );
            }
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_VBATCHED_CUH
