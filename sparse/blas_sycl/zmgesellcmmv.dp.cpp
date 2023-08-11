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


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning one thread to each row - 1D kernel
template<bool betazero>
void 
zmgesellptmv_kernel_1_3D( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(2); // local row
    int idy = item_ct1.get_local_id(1); // vector
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idx;  // global row index


    if (row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int offset = drowptr[ bdx ];
        int max_ = (drowptr[ bdx+1 ]-offset)/blocksize;  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            magmaDoubleComplex val = 
                        dval[ offset + idx + blocksize*k ];
            int col = 
                    dcolind[ offset + idx + blocksize*k ];

            dot += val * dx[ col*num_vecs+idy ];
        }
        if (betazero) {
            dy[ row+idy*num_rows ] = dot*alpha;
        } else {
            dy[ row+idy*num_rows ] = dot*alpha + beta*dy [ row+idy*num_rows ];
        }
    }
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zmgesellptmv_kernel_4_3D( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int idz = item_ct1.get_local_id(0); // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (row < num_rows ) {
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

            dot += val * dx[ col+vec ];
        }
        shared[ldz]  = dot;

        /*
        DPCT1065:532: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( idx < 2 ) {
            shared[ldz]+=shared[ldz+blocksize*2];
            /*
            DPCT1065:533: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx == 0 ) {
                if (betazero) {
                    dy[row+vec] = (shared[ldz]+shared[ldz+blocksize*1])*alpha; 
                } else {
                    dy[row+vec] = 
                    (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                                + beta*dy [row+vec];
                }
            }
        }
    }
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zmgesellptmv_kernel_8_3D( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    const magmaDoubleComplex * __restrict__ dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int idz = item_ct1.get_local_id(0); // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (row < num_rows ) {
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

            dot += val * dx[ col+vec ];
        }
        shared[ldz]  = dot;

        /*
        DPCT1065:534: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( idx < 4 ) {
            shared[ldz]+=shared[ldz+blocksize*4];
            /*
            DPCT1065:535: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];
            /*
            DPCT1065:536: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx == 0 ) {
                if (betazero) {
                    dy[row+vec] = (shared[ldz]+shared[ldz+blocksize*1])*alpha; 
                } else {
                    dy[row+vec] = 
                    (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                                + beta*dy [row+vec];
                }
            }
        }
    }
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zmgesellptmv_kernel_16_3D( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int idz = item_ct1.get_local_id(0); // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (row < num_rows ) {
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

            dot += val * dx[ col+vec ];
        }
        shared[ldz]  = dot;

        /*
        DPCT1065:537: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( idx < 8 ) {
            shared[ldz]+=shared[ldz+blocksize*8];
            /*
            DPCT1065:538: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];
            /*
            DPCT1065:539: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];
            /*
            DPCT1065:540: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx == 0 ) {
                if (betazero) {
                    dy[row+vec] = (shared[ldz]+shared[ldz+blocksize*1])*alpha; 
                } else {
                    dy[row+vec] = 
                    (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                                + beta*dy [row+vec];
                }
            }
        }
    }
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
void 
zmgesellptmv_kernel_32_3D( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int blocksize,
    int T,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy,
    sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
   // T threads assigned to each row
    int idx = item_ct1.get_local_id(1); // thread in row
    int idy = item_ct1.get_local_id(2); // local row
    int idz = item_ct1.get_local_id(0); // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = item_ct1.get_group(1) * item_ct1.get_group_range(2) +
              item_ct1.get_group(2);  // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    auto shared = (magmaDoubleComplex *)dpct_local;

    if (row < num_rows ) {
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

            dot += val * dx[ col+vec ];
        }
        shared[ldz]  = dot;

        /*
        DPCT1065:541: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( idx < 16 ) {
            shared[ldz]+=shared[ldz+blocksize*16];
            /*
            DPCT1065:542: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 8 ) shared[ldz]+=shared[ldz+blocksize*8];
            /*
            DPCT1065:543: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];
            /*
            DPCT1065:544: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];
            /*
            DPCT1065:545: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if ( idx == 0 ) {
                if (betazero) {
                    dy[row+vec] = (shared[ldz]+shared[ldz+blocksize*1])*alpha; 
                } else {
                    dy[row+vec] = 
                    (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                                + beta*dy [row+vec];
                }
            }
        }
    }
}

/************************* same but using texture mem *************************/



// SELLP SpMV kernel 2D grid - for large number of vectors
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
//template <bool betazero>
//void zmgesellptmv_kernel_1_3D_tex(
//    int num_rows, int num_cols, int num_vecs, int blocksize, int T,
//    magmaDoubleComplex alpha, magmaDoubleComplex *dval, magma_index_t *dcolind,
//    magma_index_t *drowptr,
//    /*
//    DPCT1050:881: The template argument of the image_accessor_ext could not be
//    deduced. You need to update this code.
//    */
//    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
//        texdx,
//    magmaDoubleComplex beta, magmaDoubleComplex *dy)
//{
//#if defined(PRECISION_d) && defined(TEXTURE) && (DPCT_COMPATIBILITY_TEMP >= 300)
//
//    int idx = threadIdx.x;      // local row
//    int idy = threadIdx.y;      // vector
//    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
//    int row = bdx * blocksize + idx;  // global row index
//
//    if (row < num_rows ) {
//        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
//        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
//        int offset = drowptr[ bdx ];
//        int max_ = (drowptr[ bdx+1 ]-offset)/blocksize;  
//            // number of elements each thread handles
//
//        for ( int k = 0; k < max_; k++ ) {
//            magmaDoubleComplex val = 
//                        dval[ offset + idx + blocksize*k ];
//            int col = 
//                    num_vecs * dcolind[ offset + idx + blocksize*k ];
//
//            int4 v = tex1Dfetch<int4>(texdx, col/2 + idy );
//            dot1 += val * __hiloint2double(v.y, v.x);
//            dot2 += val * __hiloint2double(v.w, v.z);
//        }
//        if (betazero) {
//            dy[row+num_rows*idy*2] = 
//                                dot1*alpha;
//            dy[row+num_rows*idy*2+num_rows] = 
//                                dot2*alpha;
//        } else {
//            dy[row+num_rows*idy*2] = 
//                                dot1*alpha
//                                + beta*dy [row*num_vecs+idy*2];
//            dy[row+num_rows*idy*2+num_rows] = 
//                                dot2*alpha
//                                + beta*dy [row*num_vecs+idy*2+1];
//        }
//    }
//#endif
//}
//
//
//// SELLP SpMV kernel 3D grid
//// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
//// A UNIFIED SPARSE MATRIX DATA FORMAT 
//// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
//// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
//template <bool betazero>
//void zmgesellptmv_kernel_4_3D_tex(
//    int num_rows, int num_cols, int num_vecs, int blocksize, int T,
//    magmaDoubleComplex alpha, magmaDoubleComplex *dval, magma_index_t *dcolind,
//    magma_index_t *drowptr,
//    /*
//    DPCT1050:882: The template argument of the image_accessor_ext could not be
//    deduced. You need to update this code.
//    */
//    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
//        texdx,
//    magmaDoubleComplex beta, magmaDoubleComplex *dy)
//{
//#if defined(PRECISION_d) && defined(TEXTURE) && (DPCT_COMPATIBILITY_TEMP >= 300)
//   // T threads assigned to each row
//    int idx = threadIdx.y;      // thread in row
//    int idy = threadIdx.x;      // local row
//    int idz = threadIdx.z;      // vector
//    int ldx = idx * blocksize + idy;
//    int ldz = idz * blocksize * T + idx * blocksize + idy;
//    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
//    int row = bdx * blocksize + idy;  // global row index
//    int sv = num_vecs/2 * blocksize * T;
//
//    extern __shared__ magmaDoubleComplex shared[];
//
//
//    if (row < num_rows ) {
//        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
//        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
//        int offset = drowptr[ bdx ];
//        int block = blocksize * T; // total number of threads
//
//        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
//            // number of elements each thread handles
//
//
//
//        for ( int k = 0; k < max_; k++ ) {
//            magmaDoubleComplex val = 
//                        dval[ offset + ldx + block*k ];
//            int col = 
//                    num_vecs * dcolind[ offset + ldx + block*k ];
//
//            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
//            dot1 += val * __hiloint2double(v.y, v.x);
//            dot2 += val * __hiloint2double(v.w, v.z);
//        }
//        shared[ldz]  = dot1;
//        shared[ldz+sv]  = dot2;
//
//        __syncthreads();
//        if ( idx < 2 ) {
//            shared[ldz]+=shared[ldz+blocksize*2];    
//            shared[ldz+sv]+=shared[ldz+sv+blocksize*2];               
//            __syncthreads();
//            if ( idx == 0 ) {
//                if (betazero) {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha;
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
//                } else {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2];
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2+1];
//                }
//            }
//        }
//    }
//#endif
//}
// 
//
//// SELLP SpMV kernel 3D grid
//// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
//// A UNIFIED SPARSE MATRIX DATA FORMAT 
//// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
//// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
//template <bool betazero>
//void zmgesellptmv_kernel_8_3D_tex(
//    int num_rows, int num_cols, int num_vecs, int blocksize, int T,
//    magmaDoubleComplex alpha, magmaDoubleComplex *dval, magma_index_t *dcolind,
//    magma_index_t *drowptr,
//    /*
//    DPCT1050:883: The template argument of the image_accessor_ext could not be
//    deduced. You need to update this code.
//    */
//    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
//        texdx,
//    magmaDoubleComplex beta, magmaDoubleComplex *dy)
//{
//#if defined(PRECISION_d) && defined(TEXTURE) && (DPCT_COMPATIBILITY_TEMP >= 300)
//   // T threads assigned to each row
//    int idx = threadIdx.y;      // thread in row
//    int idy = threadIdx.x;      // local row
//    int idz = threadIdx.z;      // vector
//    int ldx = idx * blocksize + idy;
//    int ldz = idz * blocksize * T + idx * blocksize + idy;
//    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
//    int row = bdx * blocksize + idy;  // global row index
//    int sv = num_vecs/2 * blocksize * T;
//
//    extern __shared__ magmaDoubleComplex shared[];
//
//
//    if (row < num_rows ) {
//        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
//        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
//        int offset = drowptr[ bdx ];
//        int block = blocksize * T; // total number of threads
//
//        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
//            // number of elements each thread handles
//
//
//
//        for ( int k = 0; k < max_; k++ ) {
//            magmaDoubleComplex val = 
//                        dval[ offset + ldx + block*k ];
//            int col = 
//                    num_vecs * dcolind[ offset + ldx + block*k ];
//
//            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
//            dot1 += val * __hiloint2double(v.y, v.x);
//            dot2 += val * __hiloint2double(v.w, v.z);
//        }
//        shared[ldz]  = dot1;
//        shared[ldz+sv]  = dot2;
//
//        __syncthreads();
//        if ( idx < 4 ) {
//            shared[ldz]+=shared[ldz+blocksize*4];    
//            shared[ldz+sv]+=shared[ldz+sv+blocksize*4];               
//            __syncthreads();
//            if ( idx < 2 ) {
//                shared[ldz]+=shared[ldz+blocksize*2];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
//            }
//            __syncthreads();
//            if ( idx == 0 ) {
//                if (betazero) {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha;
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
//                } else {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2];
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2+1];
//                }
//            }
//        }
//    }
//#endif
//}
//
//
//// SELLP SpMV kernel 3D grid
//// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
//// A UNIFIED SPARSE MATRIX DATA FORMAT 
//// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
//// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
//template <bool betazero>
//void zmgesellptmv_kernel_16_3D_tex(
//    int num_rows, int num_cols, int num_vecs, int blocksize, int T,
//    magmaDoubleComplex alpha, magmaDoubleComplex *dval, magma_index_t *dcolind,
//    magma_index_t *drowptr,
//    /*
//    DPCT1050:884: The template argument of the image_accessor_ext could not be
//    deduced. You need to update this code.
//    */
//    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
//        texdx,
//    magmaDoubleComplex beta, magmaDoubleComplex *dy)
//{
//#if defined(PRECISION_d) && defined(TEXTURE) && (DPCT_COMPATIBILITY_TEMP >= 300)
//   // T threads assigned to each row
//    int idx = threadIdx.y;      // thread in row
//    int idy = threadIdx.x;      // local row
//    int idz = threadIdx.z;      // vector
//    int ldx = idx * blocksize + idy;
//    int ldz = idz * blocksize * T + idx * blocksize + idy;
//    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
//    int row = bdx * blocksize + idy;  // global row index
//    int sv = num_vecs/2 * blocksize * T;
//
//    extern __shared__ magmaDoubleComplex shared[];
//
//
//    if (row < num_rows ) {
//        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
//        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
//        int offset = drowptr[ bdx ];
//        int block = blocksize * T; // total number of threads
//
//        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
//            // number of elements each thread handles
//
//
//
//        for ( int k = 0; k < max_; k++ ) {
//            magmaDoubleComplex val = 
//                        dval[ offset + ldx + block*k ];
//            int col = 
//                    num_vecs * dcolind[ offset + ldx + block*k ];
//
//            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
//            dot1 += val * __hiloint2double(v.y, v.x);
//            dot2 += val * __hiloint2double(v.w, v.z);
//        }
//        shared[ldz]  = dot1;
//        shared[ldz+sv]  = dot2;
//
//        __syncthreads();
//        if ( idx < 8 ) {
//            shared[ldz]+=shared[ldz+blocksize*8];    
//            shared[ldz+sv]+=shared[ldz+sv+blocksize*8];               
//            __syncthreads();
//            if ( idx < 4 ) {
//                shared[ldz]+=shared[ldz+blocksize*4];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
//            }
//            if ( idx < 2 ) {
//                shared[ldz]+=shared[ldz+blocksize*2];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
//            }
//            __syncthreads();
//            if ( idx == 0 ) {
//                if (betazero) {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha;
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
//                } else {
//                    dy[row+num_rows*idz*2] = 
//                    (shared[ldz]+shared[ldz+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2];
//                    dy[row+num_rows*idz*2+num_rows] = 
//                    (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2+1];
//                }
//            }
//        }
//    }
//#endif
//}
//
//
//// SELLP SpMV kernel 3D grid
//// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
//// A UNIFIED SPARSE MATRIX DATA FORMAT 
//// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
//// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
//template <bool betazero>
//void zmgesellptmv_kernel_32_3D_tex(
//    int num_rows, int num_cols, int num_vecs, int blocksize, int T,
//    magmaDoubleComplex alpha, magmaDoubleComplex *dval, magma_index_t *dcolind,
//    magma_index_t *drowptr,
//    /*
//    DPCT1050:885: The template argument of the image_accessor_ext could not be
//    deduced. You need to update this code.
//    */
//    dpct::image_accessor_ext<dpct_placeholder /*Fix the type manually*/, 1>
//        texdx,
//    magmaDoubleComplex beta, magmaDoubleComplex *dy)
//{
//#if defined(PRECISION_d) && defined(TEXTURE) && (DPCT_COMPATIBILITY_TEMP >= 300)
//   // T threads assigned to each row
//    int idx = threadIdx.y;      // thread in row
//    int idy = threadIdx.x;      // local row
//    int idz = threadIdx.z;      // vector
//    int ldx = idx * blocksize + idy;
//    int ldz = idz * blocksize * T + idx * blocksize + idy;
//    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
//    int row = bdx * blocksize + idy;  // global row index
//    int sv = num_vecs/2 * blocksize * T;
//
//    extern __shared__ magmaDoubleComplex shared[];
//
//
//    if (row < num_rows ) {
//        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
//        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
//        int offset = drowptr[ bdx ];
//        int block = blocksize * T; // total number of threads
//
//        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
//            // number of elements each thread handles
//
//
//
//        for ( int k = 0; k < max_; k++ ) {
//            magmaDoubleComplex val = 
//                        dval[ offset + ldx + block*k ];
//            int col = 
//                    num_vecs * dcolind[ offset + ldx + block*k ];
//
//            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
//            dot1 += val * __hiloint2double(v.y, v.x);
//            dot2 += val * __hiloint2double(v.w, v.z);
//        }
//        shared[ldz]  = dot1;
//        shared[ldz+sv]  = dot2;
//
//        __syncthreads();
//        if ( idx < 16 ) {
//            shared[ldz]+=shared[ldz+blocksize*16];    
//            shared[ldz+sv]+=shared[ldz+sv+blocksize*16];               
//            __syncthreads();
//            if ( idx < 8 ) {
//                shared[ldz]+=shared[ldz+blocksize*8];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*8];   
//            }
//            if ( idx < 4 ) {
//                shared[ldz]+=shared[ldz+blocksize*4];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
//            }
//            if ( idx < 2 ) {
//                shared[ldz]+=shared[ldz+blocksize*2];   
//                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
//            }
//            __syncthreads();
//            if ( idx == 0 ) {
//                if (betazero) {
//                    dy[row+num_rows*idz*2] = 
//                        (shared[ldz]+shared[ldz+blocksize*1])*alpha;
//                    dy[row+num_rows*idz*2+num_rows] = 
//                        (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
//                } else {
//                    dy[row+num_rows*idz*2] = 
//                        (shared[ldz]+shared[ldz+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2];
//                    dy[row+num_rows*idz*2+num_rows] = 
//                        (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
//                                                + beta*dy [row*num_vecs+idz*2+1];
//                }
//            }
//        }
//    }
//#endif
//}



/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A^t *  X + beta * Y on the GPU.
    Input format is SELLP. Note, that the input format for X is row-major
    while the output format for Y is column major!
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transpose A?

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 

    @param[in]
    num_vecs    magma_int_t
                number of columns in X and Y

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
magma_zmgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
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
    // using a 3D thread grid for small num_vecs, a 2D grid otherwise

    int texture=0, kepler=0, precision=0;

    magma_int_t arch = magma_getdevice_arch();
    if ( arch > 300 )
        kepler = 1;
           
    #if defined(PRECISION_d)
        precision = 1;
    #endif

    #if defined(TEXTURE)
        texture = 1;
    #endif

    if ( (texture==1) && (precision==1) && (kepler==1) ) {
        // Create channel.
        dpct::image_channel channel_desc;
        channel_desc = dpct::image_channel(
            32, 32, 32, 32, dpct::image_channel_data_type::signed_int);

        // Create resource descriptor.
        dpct::image_data resDescdx;
        memset(&resDescdx, 0, sizeof(resDescdx));

        resDescdx.set_data((void *)dx, m * num_vecs * sizeof(double),
                           channel_desc);

        // Specify texture object parameters.
        dpct::sampling_info texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.set(sycl::addressing_mode::clamp_to_edge);
        texDesc.set(sycl::filtering_mode::nearest);
        /*
        DPCT1007:547: Migration of struct cudaTextureDesc::readMode is not
        supported by the Intel(R) DPC++ Compatibility Tool.
        */
//        texDesc.readMode = cudaReadModeElementType;

        // Create texture object.
        dpct::image_wrapper_base_p texdx = 0;
        texdx = dpct::create_image_wrapper(resDescdx, texDesc);

        /*
        DPCT1026:546: The call to cudaDeviceSetSharedMemConfig was removed
        because DPC++ currently does not support configuring shared memory on
        devices.
        */

        if ( num_vecs%2 ==1 ) { // only multiple of 2 can be processed
            printf("error: number of vectors has to be multiple of 2.\n");
            return MAGMA_ERR_NOT_SUPPORTED;
        }
        if ( num_vecs > 8 ) // avoid running into memory problems
            alignment = 1; 

        int num_threads = (num_vecs/2) * blocksize*alignment;   
        
        // every thread handles two vectors
        if (  num_threads > 1024 )
            printf("error: too many threads requested.\n");

        sycl::range<3> block(num_vecs / 2, alignment, blocksize);

        int dimgrid1 = int( sqrt( double( slices )));
        int dimgrid2 = magma_ceildiv( slices, dimgrid1 );

        sycl::range<3> grid(1, dimgrid2, dimgrid1);
        /*
        DPCT1083:551: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        int Ms = num_vecs * blocksize * alignment * sizeof(magmaDoubleComplex);

//        if ( alignment == 1) {
//            sycl::range<3> block(1, num_vecs / 2, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:548: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:867: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_1_3D_tex<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//
//            else
//            /*
//            DPCT1049:549: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:868: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_1_3D_tex<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//        }
//        else if ( alignment == 4) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:550: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:869: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_4_3D_tex<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//            else
//            /*
//            DPCT1049:552: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:870: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_4_3D_tex<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//        }
//        else if ( alignment == 8) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:553: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:871: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_8_3D_tex<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//            else
//            /*
//            DPCT1049:554: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:872: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_8_3D_tex<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//        }
//        else if ( alignment == 16) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:555: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:873: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_16_3D_tex<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//            else
//            /*
//            DPCT1049:556: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:874: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_16_3D_tex<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//        }
//        else if ( alignment == 32) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:557: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:875: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_32_3D_tex<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//            else
//            /*
//            DPCT1049:558: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//            /*
//            DPCT1050:876: The template argument of the image_accessor_ext could
//            not be deduced. You need to update this code.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        auto texdx_acc = static_cast<dpct::image_wrapper<
//                            dpct_placeholder /*Fix the type manually*/, 1> *>(
//                                             texdx)
//                                             ->get_access(cgh);
//
//                        auto texdx_smpl = texdx->get_sampler();
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_32_3D_tex<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind,
//                                    dpct::image_accessor_ext<
//                                        dpct_placeholder /*Fix the type
//                                                            manually*/
//                                        ,
//                                        1>(texdx_smpl, texdx_acc),
//                                    texdx, beta, dy);
//                            });
//                    });
//        }
//        else {
//            printf("error: alignment %d not supported.\n", int(alignment) );
//            return MAGMA_ERR_NOT_SUPPORTED;
//        }
//    } else {
//        if ( num_vecs%2 ==1 ) { // only multiple of 2 can be processed
//            printf("error: number of vectors has to be multiple of 2.\n");
//            return MAGMA_ERR_NOT_SUPPORTED;
//        }
//
//        if ( num_vecs > 8 ) // avoid running into memory problems
//            alignment = 1;
//
//        int num_threads = num_vecs * blocksize*alignment;
//
//        // every thread handles two vectors
//        if (  num_threads > 1024 )
//            printf("error: too many threads requested.\n");
//
//        int dimgrid1 = int( sqrt( double( slices )));
//        int dimgrid2 = magma_ceildiv( slices, dimgrid1 );
//
//        sycl::range<3> grid(1, dimgrid2, dimgrid1);
//        /*
//        DPCT1083:562: The size of local memory in the migrated code may be
//        different from the original code. Check that the allocated memory size
//        in the migrated code is correct.
//        */
//        int Ms = num_threads * sizeof(magmaDoubleComplex);
//        if ( alignment == 1) {
//            sycl::range<3> block(1, num_vecs / 2, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:559: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->parallel_for(sycl::nd_range<3>(grid * block, block),
//                                   [=](sycl::nd_item<3> item_ct1) {
//                                       zmgesellptmv_kernel_1_3D<true>(
//                                           m, n, num_vecs, blocksize, alignment,
//                                           alpha, dval, dcolind, drowptr, dx,
//                                           beta, dy, item_ct1);
//                                   });
//            else
//            /*
//            DPCT1049:560: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->parallel_for(sycl::nd_range<3>(grid * block, block),
//                                   [=](sycl::nd_item<3> item_ct1) {
//                                       zmgesellptmv_kernel_1_3D<false>(
//                                           m, n, num_vecs, blocksize, alignment,
//                                           alpha, dval, dcolind, drowptr, dx,
//                                           beta, dy, item_ct1);
//                                   });
//        }
//        else if ( alignment == 4) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:561: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_4_3D<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//            else
//            /*
//            DPCT1049:563: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_4_3D<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//        }
//        else if ( alignment == 8) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:564: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_8_3D<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//            else
//            /*
//            DPCT1049:565: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_8_3D<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//        }
//        else if ( alignment == 16) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:566: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_16_3D<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//            else
//            /*
//            DPCT1049:567: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_16_3D<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//        }
//        else if ( alignment == 32) {
//            sycl::range<3> block(num_vecs / 2, alignment, blocksize);
//            if (beta == MAGMA_Z_ZERO)
//            /*
//            DPCT1049:568: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_32_3D<true>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//            else
//            /*
//            DPCT1049:569: The work-group size passed to the SYCL kernel may
//            exceed the limit. To get the device limit, query
//            info::device::max_work_group_size. Adjust the work-group size if
//            needed.
//            */
//                ((sycl::queue *)(queue->sycl_stream()))
//                    ->submit([&](sycl::handler &cgh) {
//                        sycl::accessor<uint8_t, 1,
//                                       sycl::access_mode::read_write,
//                                       sycl::access::target::local>
//                            dpct_local_acc_ct1(sycl::range<1>(Ms), cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(grid * block, block),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                zmgesellptmv_kernel_32_3D<false>(
//                                    m, n, num_vecs, blocksize, alignment, alpha,
//                                    dval, dcolind, drowptr, dx, beta, dy,
//                                    item_ct1, dpct_local_acc_ct1.get_pointer());
//                            });
//                    });
//        }
//        else {
//            printf("error: alignment %d not supported.\n", int(alignment) );
//            return MAGMA_ERR_NOT_SUPPORTED;
//        }
    }

    return MAGMA_SUCCESS;
}
