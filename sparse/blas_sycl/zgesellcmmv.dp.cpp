/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"
#include <cmath>

#define PRECISION_z

//#define TEXTURE


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning one thread to each row - 1D kernel
template<bool betazero>
void 
zgesellptmv2d_kernel_1( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    const magmaDoubleComplex * __restrict__ dval, 
    const magma_index_t * __restrict__ dcolind,
    const magma_index_t * __restrict__ drowptr,
    const magmaDoubleComplex *__restrict__ dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * __restrict__ dy,
    sycl::nd_item<3> item_ct1)
{
    // threads assigned to rows
    //int Idx = blockDim.x * blockIdx.x + threadIdx.x;
    //int offset = drowptr[ blockIdx.x ];
    //int border = (drowptr[ blockIdx.x+1 ]-offset)/blocksize;
    
    
    // T threads assigned to each row
    int idx = item_ct1.get_local_id(2); // local row
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2); // global block index
    int row = bdx * 256 + idx;  // global row index
    // int lblocksize = ( row + blocksize < num_rows) ? blocksize : ( num_rows - blocksize * (row/blocksize) );
    int lrow = item_ct1.get_local_id(2) % blocksize; // local row;

    if( row < num_rows ) {
        int offset = drowptr[ row/blocksize ];
        int border = (drowptr[ row/blocksize+1 ]-offset)/blocksize;

        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        for ( int n = 0; n < border; n++) { 
            int col = dcolind [ offset+ blocksize * n + lrow ];
            magmaDoubleComplex val = dval[ offset+ blocksize * n + lrow ];
            dot = dot + val * dx [ col ];
        }

        if (betazero) {
            dy[ row ] = dot * alpha;
        } else {
            dy[ row ] = dot * alpha + beta * dy [ row ];
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zgesellptmv2d_kernel_4( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex *  dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = dx[ i1 ];   
            x2 = dx[ i2 ]; 

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = dx[ dcolind[ block*kk] ];            
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        /*
        DPCT1065:198: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 2 ) {
            shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:199: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zgesellptmv2d_kernel_8( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex *  dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = dx[ i1 ];   
            x2 = dx[ i2 ]; 

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = dx[ dcolind[ block*kk] ];            
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        /*
        DPCT1065:200: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 4 ) {
            shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:201: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:202: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zgesellptmv2d_kernel_16( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex *  dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * dx[ col ];
        }
        shared[ldx]  = dot;

        /*
        DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 8 ) {
            shared[ldx]+=shared[ldx+blocksize*8];
            /*
            DPCT1065:204: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:205: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:206: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zgesellptmv2d_kernel_32( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex *  dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int ldx = idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index

    auto shared = (magmaDoubleComplex *)dpct_local;

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * dx[ col ];
        }
        shared[ldx]  = dot;

        /*
        DPCT1065:207: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if( idx < 16 ) {
            shared[ldx]+=shared[ldx+blocksize*16];
            /*
            DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];
            /*
            DPCT1065:209: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];
            /*
            DPCT1065:210: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];
            /*
            DPCT1065:211: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}



/************************* same but using texture mem *************************/

#if defined(PRECISION_d) && defined(TEXTURE)

__inline__ __device__ double 
read_from_tex( cudaTextureObject_t texdx, const int& i) {
  int2 temp = tex1Dfetch<int2>( texdx, i ); 
  return __hiloint2double(temp.y,temp.x);
}

// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_4_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = read_from_tex( texdx, dcolind[ block*kk] );
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 2 ) {
            shared[ldx]+=shared[ldx+blocksize*2];              
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_8_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = read_from_tex( texdx, dcolind[ block*kk] );
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 4 ) {
            shared[ldx]+=shared[ldx+blocksize*4];              
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_16_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * read_from_tex( texdx, col );
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 8 ) {
            shared[ldx]+=shared[ldx+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_32_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * read_from_tex( texdx, col );
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 16 ) {
            shared[ldx]+=shared[ldx+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}

#endif

/*********************     end of texture versions   **************************/

/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLP.
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 

    @param[in]
    blocksize   magma_int_t
                number of rows in one ELL-slice

    @param[in]
    slices      magma_int_t
                number of slices in matrix

    @param[in]
    alignment   magma_int_t
                number of threads assigned to one row

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in SELLP

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in SELLP

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of SELLP

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    // using a 2D thread grid

    int num_threads = blocksize*alignment;
    int nthreads_max = queue->sycl_stream()->get_device()
                            .get_info<sycl::info::device::max_work_group_size>();
    if ( num_threads > nthreads_max)
        printf("error: too many threads requested (%d) for this device (max %d).\n",
               num_threads, nthreads_max);
    
    int dimgrid1 = min( int( sqrt( double( slices ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( slices, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( slices, dimgrid1*dimgrid2 );
    int num_tx = blocksize;
    /*
    DPCT1083:214: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    int Ms = num_threads * sizeof(magmaDoubleComplex);

    // special case: alignment 1:
    if( alignment == 1 ){
        Ms = 0;
        num_tx = 256;
        int num_blocks = magma_ceildiv( n, 256 );
        dimgrid1 = num_blocks; //min( int( sqrt( double( num_blocks ))), 65535 );
        dimgrid2 = 1; //magma_ceildiv( num_blocks, dimgrid1 );
        dimgrid3 = 1;
        //blocksize = 256;
    }

    sycl::range<3> block(1, alignment, num_tx);

    if( dimgrid3 > 65535 ){
        printf("error: too many GPU thread blocks requested.\n");
    }

    sycl::range<3> grid(1, dimgrid2, dimgrid1);

    if ( alignment == 1) {
        /*
        DPCT1064:212: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (beta == MAGMA_Z_ZERO) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgesellptmv2d_kernel_1<true>(
                                   m, n, blocksize, alignment, alpha, dval,
                                   dcolind, drowptr, dx, beta, dy,
                                   item_ct1);
                           });
        } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgesellptmv2d_kernel_1<false>(
                                   m, n, blocksize, alignment, alpha, dval,
                                   dcolind, drowptr, dx, beta, dy,
                                   item_ct1);
                           });
        }
    }

    else if ( alignment == 4){
        /*
        DPCT1064:216: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (beta == MAGMA_Z_ZERO) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_4<true>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_4<false>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        }
    }

    else if ( alignment == 8){
        /*
        DPCT1064:219: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (beta == MAGMA_Z_ZERO) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_8<true>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_8<false>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        }
    }

    else if ( alignment == 16){
        /*
        DPCT1064:222: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (beta == MAGMA_Z_ZERO) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_16<true>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_16<false>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        }
    }

    else if ( alignment == 32){
        /*
        DPCT1064:225: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (beta == MAGMA_Z_ZERO) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_32<true>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        } else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1>
                    dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zgesellptmv2d_kernel_32<false>(
                                         m, n, blocksize, alignment, alpha,
                                         dval, dcolind, drowptr, dx, beta,
                                         dy, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
        }
    }

    else {
        printf("error: alignment %d not supported.\n", int(alignment) );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

    return MAGMA_SUCCESS;
}
