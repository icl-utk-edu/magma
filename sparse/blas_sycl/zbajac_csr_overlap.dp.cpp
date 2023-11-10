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

#define PRECISION_z
#define BLOCKSIZE 256

void magma_zk_testLocking(unsigned int* locks, int n,
                          const sycl::nd_item<3> &item_ct1) {
    int id = item_ct1.get_local_id(2) % n;
    bool leaveLoop = false;
    while (!leaveLoop) {
        if (dpct::atomic_exchange<sycl::access::address_space::generic_space>(
                &(locks[id]), 1u) == 0u) {
            //critical section
            leaveLoop = true;
            dpct::atomic_exchange<sycl::access::address_space::generic_space>(
                &(locks[id]), 0u);
        }
    }
}

void
magma_zbajac_csr_o_ls_kernel1(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD, 
                            magma_index_t * rowD, 
                            magma_index_t * colD, 
                            magmaDoubleComplex * valR, 
                            magma_index_t * rowR,
                            magma_index_t * colR, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag = item_ct1.get_group(2) * item_ct1.get_local_range(2);
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    int i, j, start, end;
    //bool leaveLoop = false;
    

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero;

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }
        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}


void
magma_zbajac_csr_o_ls_kernel2(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD0, 
                            magma_index_t * rowD0, 
                            magma_index_t * colD0, 
                            magmaDoubleComplex * valR0, 
                            magma_index_t * rowR0,
                            magma_index_t * colR0, 
                            magmaDoubleComplex * valD1, 
                            magma_index_t * rowD1, 
                            magma_index_t * colD1, 
                            magmaDoubleComplex * valR1, 
                            magma_index_t * rowR1,
                            magma_index_t * colR1, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag = item_ct1.get_group(2) * item_ct1.get_local_range(2) / 2 -
                  item_ct1.get_local_range(2) / 2;
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) / 2 +
                item_ct1.get_local_id(2) - item_ct1.get_local_range(2) / 2;
    int i, j, start, end;
    //bool leaveLoop = false;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }
    
    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}


void
magma_zbajac_csr_o_ls_kernel4(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD0, magma_index_t * rowD0, magma_index_t * colD0, magmaDoubleComplex * valR0, magma_index_t * rowR0, magma_index_t * colR0, 
                            magmaDoubleComplex * valD1, magma_index_t * rowD1, magma_index_t * colD1, magmaDoubleComplex * valR1, magma_index_t * rowR1, magma_index_t * colR1, 
                            magmaDoubleComplex * valD2, magma_index_t * rowD2, magma_index_t * colD2, magmaDoubleComplex * valR2, magma_index_t * rowR2, magma_index_t * colR2, 
                            magmaDoubleComplex * valD3, magma_index_t * rowD3, magma_index_t * colD3, magmaDoubleComplex * valR3, magma_index_t * rowR3, magma_index_t * colR3, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap;
    int index =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap + item_ct1.get_local_id(2);
    int i, j, start, end;
    //bool leaveLoop = false;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3; rowD = rowD3;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2; rowD = rowD2;
    } else if (item_ct1.get_group(2) % matrices == 2) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 3) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}


void
magma_zbajac_csr_o_ls_kernel8(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD0, magma_index_t * rowD0, magma_index_t * colD0, magmaDoubleComplex * valR0, magma_index_t * rowR0, magma_index_t * colR0, 
                            magmaDoubleComplex * valD1, magma_index_t * rowD1, magma_index_t * colD1, magmaDoubleComplex * valR1, magma_index_t * rowR1, magma_index_t * colR1, 
                            magmaDoubleComplex * valD2, magma_index_t * rowD2, magma_index_t * colD2, magmaDoubleComplex * valR2, magma_index_t * rowR2, magma_index_t * colR2, 
                            magmaDoubleComplex * valD3, magma_index_t * rowD3, magma_index_t * colD3, magmaDoubleComplex * valR3, magma_index_t * rowR3, magma_index_t * colR3, 
                            magmaDoubleComplex * valD4, magma_index_t * rowD4, magma_index_t * colD4, magmaDoubleComplex * valR4, magma_index_t * rowR4, magma_index_t * colR4, 
                            magmaDoubleComplex * valD5, magma_index_t * rowD5, magma_index_t * colD5, magmaDoubleComplex * valR5, magma_index_t * rowR5, magma_index_t * colR5, 
                            magmaDoubleComplex * valD6, magma_index_t * rowD6, magma_index_t * colD6, magmaDoubleComplex * valR6, magma_index_t * rowR6, magma_index_t * colR6, 
                            magmaDoubleComplex * valD7, magma_index_t * rowD7, magma_index_t * colD7, magmaDoubleComplex * valR7, magma_index_t * rowR7, magma_index_t * colR7, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap;
    int index =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap + item_ct1.get_local_id(2);
    int i, j, start, end;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7; rowD = rowD7;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6; rowD = rowD6;
    } else if (item_ct1.get_group(2) % matrices == 2) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5; rowD = rowD5;
    } else if (item_ct1.get_group(2) % matrices == 3) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4; rowD = rowD4;
    } else if (item_ct1.get_group(2) % matrices == 4) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3; rowD = rowD3;
    } else if (item_ct1.get_group(2) % matrices == 5) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2; rowD = rowD2;
    } else if (item_ct1.get_group(2) % matrices == 6) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 7) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}


void
magma_zbajac_csr_o_ls_kernel16(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex *valD0 , magma_index_t *rowD0 , magma_index_t *colD0 , magmaDoubleComplex *valR0 , magma_index_t *rowR0 , magma_index_t *colR0 , 
                            magmaDoubleComplex *valD1 , magma_index_t *rowD1 , magma_index_t *colD1 , magmaDoubleComplex *valR1 , magma_index_t *rowR1 , magma_index_t *colR1 , 
                            magmaDoubleComplex *valD2 , magma_index_t *rowD2 , magma_index_t *colD2 , magmaDoubleComplex *valR2 , magma_index_t *rowR2 , magma_index_t *colR2 , 
                            magmaDoubleComplex *valD3 , magma_index_t *rowD3 , magma_index_t *colD3 , magmaDoubleComplex *valR3 , magma_index_t *rowR3 , magma_index_t *colR3 , 
                            magmaDoubleComplex *valD4 , magma_index_t *rowD4 , magma_index_t *colD4 , magmaDoubleComplex *valR4 , magma_index_t *rowR4 , magma_index_t *colR4 , 
                            magmaDoubleComplex *valD5 , magma_index_t *rowD5 , magma_index_t *colD5 , magmaDoubleComplex *valR5 , magma_index_t *rowR5 , magma_index_t *colR5 , 
                            magmaDoubleComplex *valD6 , magma_index_t *rowD6 , magma_index_t *colD6 , magmaDoubleComplex *valR6 , magma_index_t *rowR6 , magma_index_t *colR6 , 
                            magmaDoubleComplex *valD7 , magma_index_t *rowD7 , magma_index_t *colD7 , magmaDoubleComplex *valR7 , magma_index_t *rowR7 , magma_index_t *colR7 , 
                            magmaDoubleComplex *valD8 , magma_index_t *rowD8 , magma_index_t *colD8 , magmaDoubleComplex *valR8 , magma_index_t *rowR8 , magma_index_t *colR8 , 
                            magmaDoubleComplex *valD9 , magma_index_t *rowD9 , magma_index_t *colD9 , magmaDoubleComplex *valR9 , magma_index_t *rowR9 , magma_index_t *colR9 , 
                            magmaDoubleComplex *valD10, magma_index_t *rowD10, magma_index_t *colD10, magmaDoubleComplex *valR10, magma_index_t *rowR10, magma_index_t *colR10,
                            magmaDoubleComplex *valD11, magma_index_t *rowD11, magma_index_t *colD11, magmaDoubleComplex *valR11, magma_index_t *rowR11, magma_index_t *colR11,
                            magmaDoubleComplex *valD12, magma_index_t *rowD12, magma_index_t *colD12, magmaDoubleComplex *valR12, magma_index_t *rowR12, magma_index_t *colR12, 
                            magmaDoubleComplex *valD13, magma_index_t *rowD13, magma_index_t *colD13, magmaDoubleComplex *valR13, magma_index_t *rowR13, magma_index_t *colR13, 
                            magmaDoubleComplex *valD14, magma_index_t *rowD14, magma_index_t *colD14, magmaDoubleComplex *valR14, magma_index_t *rowR14, magma_index_t *colR14, 
                            magmaDoubleComplex *valD15, magma_index_t *rowD15, magma_index_t *colD15, magmaDoubleComplex *valR15, magma_index_t *rowR15, magma_index_t *colR15,  
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap;
    int index =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap + item_ct1.get_local_id(2);
    int i, j, start, end;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR15; valD = valD15; colR = colR15; rowR = rowR15;
        colD = colD15;
        rowD = rowD15;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR14; valD = valD14; colR = colR14; rowR = rowR14;
        colD = colD14;
        rowD = rowD14;
    } else if (item_ct1.get_group(2) % matrices == 2) {
        valR = valR13; valD = valD13; colR = colR13; rowR = rowR13;
        colD = colD13;
        rowD = rowD13;
    } else if (item_ct1.get_group(2) % matrices == 3) {
        valR = valR12; valD = valD12; colR = colR12; rowR = rowR12;
        colD = colD12;
        rowD = rowD12;
    } else if (item_ct1.get_group(2) % matrices == 4) {
        valR = valR11; valD = valD11; colR = colR11; rowR = rowR11;
        colD = colD11;
        rowD = rowD11;
    } else if (item_ct1.get_group(2) % matrices == 5) {
        valR = valR10; valD = valD10; colR = colR10; rowR = rowR10;
        colD = colD10;
        rowD = rowD10;
    } else if (item_ct1.get_group(2) % matrices == 6) {
        valR = valR9; valD = valD9; colR = colR9; rowR = rowR9; colD = colD9;
        rowD = rowD9;
    } else if (item_ct1.get_group(2) % matrices == 7) {
        valR = valR8; valD = valD8; colR = colR8; rowR = rowR8; colD = colD8;
        rowD = rowD8;
    } else if (item_ct1.get_group(2) % matrices == 8) {
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7;
        rowD = rowD7;
    } else if (item_ct1.get_group(2) % matrices == 9) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6;
        rowD = rowD6;
    } else if (item_ct1.get_group(2) % matrices == 10) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5;
        rowD = rowD5;
    } else if (item_ct1.get_group(2) % matrices == 11) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4;
        rowD = rowD4;
    } else if (item_ct1.get_group(2) % matrices == 12) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3;
        rowD = rowD3;
    } else if (item_ct1.get_group(2) % matrices == 13) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2;
        rowD = rowD2;
    } else if (item_ct1.get_group(2) % matrices == 14) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1;
        rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 15) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0;
        rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:60: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}

void
magma_zbajac_csr_o_ls_kernel32(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex *valD0 , magma_index_t *rowD0 , magma_index_t *colD0 , magmaDoubleComplex *valR0 , magma_index_t *rowR0 , magma_index_t *colR0 , 
                            magmaDoubleComplex *valD1 , magma_index_t *rowD1 , magma_index_t *colD1 , magmaDoubleComplex *valR1 , magma_index_t *rowR1 , magma_index_t *colR1 , 
                            magmaDoubleComplex *valD2 , magma_index_t *rowD2 , magma_index_t *colD2 , magmaDoubleComplex *valR2 , magma_index_t *rowR2 , magma_index_t *colR2 , 
                            magmaDoubleComplex *valD3 , magma_index_t *rowD3 , magma_index_t *colD3 , magmaDoubleComplex *valR3 , magma_index_t *rowR3 , magma_index_t *colR3 , 
                            magmaDoubleComplex *valD4 , magma_index_t *rowD4 , magma_index_t *colD4 , magmaDoubleComplex *valR4 , magma_index_t *rowR4 , magma_index_t *colR4 , 
                            magmaDoubleComplex *valD5 , magma_index_t *rowD5 , magma_index_t *colD5 , magmaDoubleComplex *valR5 , magma_index_t *rowR5 , magma_index_t *colR5 , 
                            magmaDoubleComplex *valD6 , magma_index_t *rowD6 , magma_index_t *colD6 , magmaDoubleComplex *valR6 , magma_index_t *rowR6 , magma_index_t *colR6 , 
                            magmaDoubleComplex *valD7 , magma_index_t *rowD7 , magma_index_t *colD7 , magmaDoubleComplex *valR7 , magma_index_t *rowR7 , magma_index_t *colR7 , 
                            magmaDoubleComplex *valD8 , magma_index_t *rowD8 , magma_index_t *colD8 , magmaDoubleComplex *valR8 , magma_index_t *rowR8 , magma_index_t *colR8 , 
                            magmaDoubleComplex *valD9 , magma_index_t *rowD9 , magma_index_t *colD9 , magmaDoubleComplex *valR9 , magma_index_t *rowR9 , magma_index_t *colR9 , 
                            magmaDoubleComplex *valD10, magma_index_t *rowD10, magma_index_t *colD10, magmaDoubleComplex *valR10, magma_index_t *rowR10, magma_index_t *colR10,
                            magmaDoubleComplex *valD11, magma_index_t *rowD11, magma_index_t *colD11, magmaDoubleComplex *valR11, magma_index_t *rowR11, magma_index_t *colR11,
                            magmaDoubleComplex *valD12, magma_index_t *rowD12, magma_index_t *colD12, magmaDoubleComplex *valR12, magma_index_t *rowR12, magma_index_t *colR12, 
                            magmaDoubleComplex *valD13, magma_index_t *rowD13, magma_index_t *colD13, magmaDoubleComplex *valR13, magma_index_t *rowR13, magma_index_t *colR13, 
                            magmaDoubleComplex *valD14, magma_index_t *rowD14, magma_index_t *colD14, magmaDoubleComplex *valR14, magma_index_t *rowR14, magma_index_t *colR14, 
                            magmaDoubleComplex *valD15, magma_index_t *rowD15, magma_index_t *colD15, magmaDoubleComplex *valR15, magma_index_t *rowR15, magma_index_t *colR15, 
                            magmaDoubleComplex *valD16, magma_index_t *rowD16, magma_index_t *colD16, magmaDoubleComplex *valR16, magma_index_t *rowR16, magma_index_t *colR16, 
                            magmaDoubleComplex *valD17, magma_index_t *rowD17, magma_index_t *colD17, magmaDoubleComplex *valR17, magma_index_t *rowR17, magma_index_t *colR17, 
                            magmaDoubleComplex *valD18, magma_index_t *rowD18, magma_index_t *colD18, magmaDoubleComplex *valR18, magma_index_t *rowR18, magma_index_t *colR18, 
                            magmaDoubleComplex *valD19, magma_index_t *rowD19, magma_index_t *colD19, magmaDoubleComplex *valR19, magma_index_t *rowR19, magma_index_t *colR19, 
                            magmaDoubleComplex *valD20, magma_index_t *rowD20, magma_index_t *colD20, magmaDoubleComplex *valR20, magma_index_t *rowR20, magma_index_t *colR20, 
                            magmaDoubleComplex *valD21, magma_index_t *rowD21, magma_index_t *colD21, magmaDoubleComplex *valR21, magma_index_t *rowR21, magma_index_t *colR21, 
                            magmaDoubleComplex *valD22, magma_index_t *rowD22, magma_index_t *colD22, magmaDoubleComplex *valR22, magma_index_t *rowR22, magma_index_t *colR22, 
                            magmaDoubleComplex *valD23, magma_index_t *rowD23, magma_index_t *colD23, magmaDoubleComplex *valR23, magma_index_t *rowR23, magma_index_t *colR23, 
                            magmaDoubleComplex *valD24, magma_index_t *rowD24, magma_index_t *colD24, magmaDoubleComplex *valR24, magma_index_t *rowR24, magma_index_t *colR24, 
                            magmaDoubleComplex *valD25, magma_index_t *rowD25, magma_index_t *colD25, magmaDoubleComplex *valR25, magma_index_t *rowR25, magma_index_t *colR25, 
                            magmaDoubleComplex *valD26, magma_index_t *rowD26, magma_index_t *colD26, magmaDoubleComplex *valR26, magma_index_t *rowR26, magma_index_t *colR26, 
                            magmaDoubleComplex *valD27, magma_index_t *rowD27, magma_index_t *colD27, magmaDoubleComplex *valR27, magma_index_t *rowR27, magma_index_t *colR27, 
                            magmaDoubleComplex *valD28, magma_index_t *rowD28, magma_index_t *colD28, magmaDoubleComplex *valR28, magma_index_t *rowR28, magma_index_t *colR28, 
                            magmaDoubleComplex *valD29, magma_index_t *rowD29, magma_index_t *colD29, magmaDoubleComplex *valR29, magma_index_t *rowR29, magma_index_t *colR29, 
                            magmaDoubleComplex *valD30, magma_index_t *rowD30, magma_index_t *colD30, magmaDoubleComplex *valR30, magma_index_t *rowR30, magma_index_t *colR30, 
                            magmaDoubleComplex *valD31, magma_index_t *rowD31, magma_index_t *colD31, magmaDoubleComplex *valR31, magma_index_t *rowR31, magma_index_t *colR31, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap;
    int index =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap + item_ct1.get_local_id(2);
    int i, j, start, end;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR31; valD = valD31; colR = colR31; rowR = rowR31;
        colD = colD31;
        rowD = rowD31;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR30; valD = valD30; colR = colR30; rowR = rowR30;
        colD = colD30;
        rowD = rowD30;
    } else if (item_ct1.get_group(2) % matrices == 2) {
        valR = valR29; valD = valD29; colR = colR29; rowR = rowR29;
        colD = colD29;
        rowD = rowD29;
    } else if (item_ct1.get_group(2) % matrices == 3) {
        valR = valR28; valD = valD28; colR = colR28; rowR = rowR28;
        colD = colD28;
        rowD = rowD28;
    } else if (item_ct1.get_group(2) % matrices == 4) {
        valR = valR27; valD = valD27; colR = colR27; rowR = rowR27;
        colD = colD27;
        rowD = rowD27;
    } else if (item_ct1.get_group(2) % matrices == 5) {
        valR = valR26; valD = valD26; colR = colR26; rowR = rowR26;
        colD = colD26;
        rowD = rowD26;
    } else if (item_ct1.get_group(2) % matrices == 6) {
        valR = valR25; valD = valD25; colR = colR25; rowR = rowR25;
        colD = colD25;
        rowD = rowD25;
    } else if (item_ct1.get_group(2) % matrices == 7) {
        valR = valR24; valD = valD24; colR = colR24; rowR = rowR24;
        colD = colD24;
        rowD = rowD24;
    } else if (item_ct1.get_group(2) % matrices == 8) {
        valR = valR23; valD = valD23; colR = colR23; rowR = rowR23;
        colD = colD23;
        rowD = rowD23;
    } else if (item_ct1.get_group(2) % matrices == 9) {
        valR = valR22; valD = valD22; colR = colR22; rowR = rowR22;
        colD = colD22;
        rowD = rowD22;
    } else if (item_ct1.get_group(2) % matrices == 10) {
        valR = valR21; valD = valD21; colR = colR21; rowR = rowR21;
        colD = colD21;
        rowD = rowD21;
    } else if (item_ct1.get_group(2) % matrices == 11) {
        valR = valR20; valD = valD20; colR = colR20; rowR = rowR20;
        colD = colD20;
        rowD = rowD20;
    } else if (item_ct1.get_group(2) % matrices == 12) {
        valR = valR19; valD = valD19; colR = colR19; rowR = rowR19;
        colD = colD19;
        rowD = rowD19;
    } else if (item_ct1.get_group(2) % matrices == 13) {
        valR = valR18; valD = valD18; colR = colR18; rowR = rowR18;
        colD = colD18;
        rowD = rowD18;
    } else if (item_ct1.get_group(2) % matrices == 14) {
        valR = valR17; valD = valD17; colR = colR17; rowR = rowR17;
        colD = colD17;
        rowD = rowD17;
    } else if (item_ct1.get_group(2) % matrices == 15) {
        valR = valR16; valD = valD16; colR = colR16; rowR = rowR16;
        colD = colD16;
        rowD = rowD16;
    } else if (item_ct1.get_group(2) % matrices == 16) {
        valR = valR15; valD = valD15; colR = colR15; rowR = rowR15;
        colD = colD15;
        rowD = rowD15;
    } else if (item_ct1.get_group(2) % matrices == 17) {
        valR = valR14; valD = valD14; colR = colR14; rowR = rowR14;
        colD = colD14;
        rowD = rowD14;
    } else if (item_ct1.get_group(2) % matrices == 18) {
        valR = valR13; valD = valD13; colR = colR13; rowR = rowR13;
        colD = colD13;
        rowD = rowD13;
    } else if (item_ct1.get_group(2) % matrices == 19) {
        valR = valR12; valD = valD12; colR = colR12; rowR = rowR12;
        colD = colD12;
        rowD = rowD12;
    } else if (item_ct1.get_group(2) % matrices == 20) {
        valR = valR11; valD = valD11; colR = colR11; rowR = rowR11;
        colD = colD11;
        rowD = rowD11;
    } else if (item_ct1.get_group(2) % matrices == 21) {
        valR = valR10; valD = valD10; colR = colR10; rowR = rowR10;
        colD = colD10;
        rowD = rowD10;
    } else if (item_ct1.get_group(2) % matrices == 22) {
        valR = valR9; valD = valD9; colR = colR9; rowR = rowR9; colD = colD9;
        rowD = rowD9;
    } else if (item_ct1.get_group(2) % matrices == 23) {
        valR = valR8; valD = valD8; colR = colR8; rowR = rowR8; colD = colD8;
        rowD = rowD8;
    } else if (item_ct1.get_group(2) % matrices == 24) {
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7;
        rowD = rowD7;
    } else if (item_ct1.get_group(2) % matrices == 25) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6;
        rowD = rowD6;
    } else if (item_ct1.get_group(2) % matrices == 26) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5;
        rowD = rowD5;
    } else if (item_ct1.get_group(2) % matrices == 27) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4;
        rowD = rowD4;
    } else if (item_ct1.get_group(2) % matrices == 28) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3;
        rowD = rowD3;
    } else if (item_ct1.get_group(2) % matrices == 29) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2;
        rowD = rowD2;
    } else if (item_ct1.get_group(2) % matrices == 30) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1;
        rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 31) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0;
        rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:61: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}

void
magma_zbajac_csr_o_ls_kernel64(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex *valD0 , magma_index_t *rowD0 , magma_index_t *colD0 , magmaDoubleComplex *valR0 , magma_index_t *rowR0 , magma_index_t *colR0 , 
                            magmaDoubleComplex *valD1 , magma_index_t *rowD1 , magma_index_t *colD1 , magmaDoubleComplex *valR1 , magma_index_t *rowR1 , magma_index_t *colR1 , 
                            magmaDoubleComplex *valD2 , magma_index_t *rowD2 , magma_index_t *colD2 , magmaDoubleComplex *valR2 , magma_index_t *rowR2 , magma_index_t *colR2 , 
                            magmaDoubleComplex *valD3 , magma_index_t *rowD3 , magma_index_t *colD3 , magmaDoubleComplex *valR3 , magma_index_t *rowR3 , magma_index_t *colR3 , 
                            magmaDoubleComplex *valD4 , magma_index_t *rowD4 , magma_index_t *colD4 , magmaDoubleComplex *valR4 , magma_index_t *rowR4 , magma_index_t *colR4 , 
                            magmaDoubleComplex *valD5 , magma_index_t *rowD5 , magma_index_t *colD5 , magmaDoubleComplex *valR5 , magma_index_t *rowR5 , magma_index_t *colR5 , 
                            magmaDoubleComplex *valD6 , magma_index_t *rowD6 , magma_index_t *colD6 , magmaDoubleComplex *valR6 , magma_index_t *rowR6 , magma_index_t *colR6 , 
                            magmaDoubleComplex *valD7 , magma_index_t *rowD7 , magma_index_t *colD7 , magmaDoubleComplex *valR7 , magma_index_t *rowR7 , magma_index_t *colR7 , 
                            magmaDoubleComplex *valD8 , magma_index_t *rowD8 , magma_index_t *colD8 , magmaDoubleComplex *valR8 , magma_index_t *rowR8 , magma_index_t *colR8 , 
                            magmaDoubleComplex *valD9 , magma_index_t *rowD9 , magma_index_t *colD9 , magmaDoubleComplex *valR9 , magma_index_t *rowR9 , magma_index_t *colR9 , 
                            magmaDoubleComplex *valD10, magma_index_t *rowD10, magma_index_t *colD10, magmaDoubleComplex *valR10, magma_index_t *rowR10, magma_index_t *colR10,
                            magmaDoubleComplex *valD11, magma_index_t *rowD11, magma_index_t *colD11, magmaDoubleComplex *valR11, magma_index_t *rowR11, magma_index_t *colR11,
                            magmaDoubleComplex *valD12, magma_index_t *rowD12, magma_index_t *colD12, magmaDoubleComplex *valR12, magma_index_t *rowR12, magma_index_t *colR12, 
                            magmaDoubleComplex *valD13, magma_index_t *rowD13, magma_index_t *colD13, magmaDoubleComplex *valR13, magma_index_t *rowR13, magma_index_t *colR13, 
                            magmaDoubleComplex *valD14, magma_index_t *rowD14, magma_index_t *colD14, magmaDoubleComplex *valR14, magma_index_t *rowR14, magma_index_t *colR14, 
                            magmaDoubleComplex *valD15, magma_index_t *rowD15, magma_index_t *colD15, magmaDoubleComplex *valR15, magma_index_t *rowR15, magma_index_t *colR15, 
                            magmaDoubleComplex *valD16, magma_index_t *rowD16, magma_index_t *colD16, magmaDoubleComplex *valR16, magma_index_t *rowR16, magma_index_t *colR16, 
                            magmaDoubleComplex *valD17, magma_index_t *rowD17, magma_index_t *colD17, magmaDoubleComplex *valR17, magma_index_t *rowR17, magma_index_t *colR17, 
                            magmaDoubleComplex *valD18, magma_index_t *rowD18, magma_index_t *colD18, magmaDoubleComplex *valR18, magma_index_t *rowR18, magma_index_t *colR18, 
                            magmaDoubleComplex *valD19, magma_index_t *rowD19, magma_index_t *colD19, magmaDoubleComplex *valR19, magma_index_t *rowR19, magma_index_t *colR19, 
                            magmaDoubleComplex *valD20, magma_index_t *rowD20, magma_index_t *colD20, magmaDoubleComplex *valR20, magma_index_t *rowR20, magma_index_t *colR20, 
                            magmaDoubleComplex *valD21, magma_index_t *rowD21, magma_index_t *colD21, magmaDoubleComplex *valR21, magma_index_t *rowR21, magma_index_t *colR21, 
                            magmaDoubleComplex *valD22, magma_index_t *rowD22, magma_index_t *colD22, magmaDoubleComplex *valR22, magma_index_t *rowR22, magma_index_t *colR22, 
                            magmaDoubleComplex *valD23, magma_index_t *rowD23, magma_index_t *colD23, magmaDoubleComplex *valR23, magma_index_t *rowR23, magma_index_t *colR23, 
                            magmaDoubleComplex *valD24, magma_index_t *rowD24, magma_index_t *colD24, magmaDoubleComplex *valR24, magma_index_t *rowR24, magma_index_t *colR24, 
                            magmaDoubleComplex *valD25, magma_index_t *rowD25, magma_index_t *colD25, magmaDoubleComplex *valR25, magma_index_t *rowR25, magma_index_t *colR25, 
                            magmaDoubleComplex *valD26, magma_index_t *rowD26, magma_index_t *colD26, magmaDoubleComplex *valR26, magma_index_t *rowR26, magma_index_t *colR26, 
                            magmaDoubleComplex *valD27, magma_index_t *rowD27, magma_index_t *colD27, magmaDoubleComplex *valR27, magma_index_t *rowR27, magma_index_t *colR27, 
                            magmaDoubleComplex *valD28, magma_index_t *rowD28, magma_index_t *colD28, magmaDoubleComplex *valR28, magma_index_t *rowR28, magma_index_t *colR28, 
                            magmaDoubleComplex *valD29, magma_index_t *rowD29, magma_index_t *colD29, magmaDoubleComplex *valR29, magma_index_t *rowR29, magma_index_t *colR29, 
                            magmaDoubleComplex *valD30, magma_index_t *rowD30, magma_index_t *colD30, magmaDoubleComplex *valR30, magma_index_t *rowR30, magma_index_t *colR30, 
                            magmaDoubleComplex *valD31, magma_index_t *rowD31, magma_index_t *colD31, magmaDoubleComplex *valR31, magma_index_t *rowR31, magma_index_t *colR31, 
                            magmaDoubleComplex *valD32, magma_index_t *rowD32, magma_index_t *colD32, magmaDoubleComplex *valR32, magma_index_t *rowR32, magma_index_t *colR32, 
                            magmaDoubleComplex *valD33, magma_index_t *rowD33, magma_index_t *colD33, magmaDoubleComplex *valR33, magma_index_t *rowR33, magma_index_t *colR33, 
                            magmaDoubleComplex *valD34, magma_index_t *rowD34, magma_index_t *colD34, magmaDoubleComplex *valR34, magma_index_t *rowR34, magma_index_t *colR34, 
                            magmaDoubleComplex *valD35, magma_index_t *rowD35, magma_index_t *colD35, magmaDoubleComplex *valR35, magma_index_t *rowR35, magma_index_t *colR35, 
                            magmaDoubleComplex *valD36, magma_index_t *rowD36, magma_index_t *colD36, magmaDoubleComplex *valR36, magma_index_t *rowR36, magma_index_t *colR36, 
                            magmaDoubleComplex *valD37, magma_index_t *rowD37, magma_index_t *colD37, magmaDoubleComplex *valR37, magma_index_t *rowR37, magma_index_t *colR37, 
                            magmaDoubleComplex *valD38, magma_index_t *rowD38, magma_index_t *colD38, magmaDoubleComplex *valR38, magma_index_t *rowR38, magma_index_t *colR38, 
                            magmaDoubleComplex *valD39, magma_index_t *rowD39, magma_index_t *colD39, magmaDoubleComplex *valR39, magma_index_t *rowR39, magma_index_t *colR39, 
                            magmaDoubleComplex *valD40, magma_index_t *rowD40, magma_index_t *colD40, magmaDoubleComplex *valR40, magma_index_t *rowR40, magma_index_t *colR40, 
                            magmaDoubleComplex *valD41, magma_index_t *rowD41, magma_index_t *colD41, magmaDoubleComplex *valR41, magma_index_t *rowR41, magma_index_t *colR41, 
                            magmaDoubleComplex *valD42, magma_index_t *rowD42, magma_index_t *colD42, magmaDoubleComplex *valR42, magma_index_t *rowR42, magma_index_t *colR42, 
                            magmaDoubleComplex *valD43, magma_index_t *rowD43, magma_index_t *colD43, magmaDoubleComplex *valR43, magma_index_t *rowR43, magma_index_t *colR43, 
                            magmaDoubleComplex *valD44, magma_index_t *rowD44, magma_index_t *colD44, magmaDoubleComplex *valR44, magma_index_t *rowR44, magma_index_t *colR44, 
                            magmaDoubleComplex *valD45, magma_index_t *rowD45, magma_index_t *colD45, magmaDoubleComplex *valR45, magma_index_t *rowR45, magma_index_t *colR45, 
                            magmaDoubleComplex *valD46, magma_index_t *rowD46, magma_index_t *colD46, magmaDoubleComplex *valR46, magma_index_t *rowR46, magma_index_t *colR46, 
                            magmaDoubleComplex *valD47, magma_index_t *rowD47, magma_index_t *colD47, magmaDoubleComplex *valR47, magma_index_t *rowR47, magma_index_t *colR47, 
                            magmaDoubleComplex *valD48, magma_index_t *rowD48, magma_index_t *colD48, magmaDoubleComplex *valR48, magma_index_t *rowR48, magma_index_t *colR48, 
                            magmaDoubleComplex *valD49, magma_index_t *rowD49, magma_index_t *colD49, magmaDoubleComplex *valR49, magma_index_t *rowR49, magma_index_t *colR49, 
                            magmaDoubleComplex *valD50, magma_index_t *rowD50, magma_index_t *colD50, magmaDoubleComplex *valR50, magma_index_t *rowR50, magma_index_t *colR50,
                            magmaDoubleComplex *valD51, magma_index_t *rowD51, magma_index_t *colD51, magmaDoubleComplex *valR51, magma_index_t *rowR51, magma_index_t *colR51,
                            magmaDoubleComplex *valD52, magma_index_t *rowD52, magma_index_t *colD52, magmaDoubleComplex *valR52, magma_index_t *rowR52, magma_index_t *colR52, 
                            magmaDoubleComplex *valD53, magma_index_t *rowD53, magma_index_t *colD53, magmaDoubleComplex *valR53, magma_index_t *rowR53, magma_index_t *colR53, 
                            magmaDoubleComplex *valD54, magma_index_t *rowD54, magma_index_t *colD54, magmaDoubleComplex *valR54, magma_index_t *rowR54, magma_index_t *colR54, 
                            magmaDoubleComplex *valD55, magma_index_t *rowD55, magma_index_t *colD55, magmaDoubleComplex *valR55, magma_index_t *rowR55, magma_index_t *colR55, 
                            magmaDoubleComplex *valD56, magma_index_t *rowD56, magma_index_t *colD56, magmaDoubleComplex *valR56, magma_index_t *rowR56, magma_index_t *colR56, 
                            magmaDoubleComplex *valD57, magma_index_t *rowD57, magma_index_t *colD57, magmaDoubleComplex *valR57, magma_index_t *rowR57, magma_index_t *colR57, 
                            magmaDoubleComplex *valD58, magma_index_t *rowD58, magma_index_t *colD58, magmaDoubleComplex *valR58, magma_index_t *rowR58, magma_index_t *colR58, 
                            magmaDoubleComplex *valD59, magma_index_t *rowD59, magma_index_t *colD59, magmaDoubleComplex *valR59, magma_index_t *rowR59, magma_index_t *colR59, 
                            magmaDoubleComplex *valD60, magma_index_t *rowD60, magma_index_t *colD60, magmaDoubleComplex *valR60, magma_index_t *rowR60, magma_index_t *colR60, 
                            magmaDoubleComplex *valD61, magma_index_t *rowD61, magma_index_t *colD61, magmaDoubleComplex *valR61, magma_index_t *rowR61, magma_index_t *colR61, 
                            magmaDoubleComplex *valD62, magma_index_t *rowD62, magma_index_t *colD62, magmaDoubleComplex *valR62, magma_index_t *rowR62, magma_index_t *colR62, 
                            magmaDoubleComplex *valD63, magma_index_t *rowD63, magma_index_t *colD63, magmaDoubleComplex *valR63, magma_index_t *rowR63, magma_index_t *colR63, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x ,
                            const sycl::nd_item<3> &item_ct1,
                            magmaDoubleComplex *local_x)
{
    int inddiag =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap;
    int index =
        item_ct1.get_group(2) * (item_ct1.get_local_range(2) - overlap) -
        overlap + item_ct1.get_local_id(2);
    int i, j, start, end;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;

    if (item_ct1.get_group(2) % matrices == 0) {
        valR = valR63; valD = valD63; colR = colR63; rowR = rowR63;
        colD = colD63;
        rowD = rowD63;
    } else if (item_ct1.get_group(2) % matrices == 1) {
        valR = valR62; valD = valD62; colR = colR62; rowR = rowR62;
        colD = colD62;
        rowD = rowD62;
    } else if (item_ct1.get_group(2) % matrices == 2) {
        valR = valR61; valD = valD61; colR = colR61; rowR = rowR61;
        colD = colD61;
        rowD = rowD61;
    } else if (item_ct1.get_group(2) % matrices == 3) {
        valR = valR60; valD = valD60; colR = colR60; rowR = rowR60;
        colD = colD60;
        rowD = rowD60;
    } else if (item_ct1.get_group(2) % matrices == 4) {
        valR = valR59; valD = valD59; colR = colR59; rowR = rowR59;
        colD = colD59;
        rowD = rowD59;
    } else if (item_ct1.get_group(2) % matrices == 5) {
        valR = valR58; valD = valD58; colR = colR58; rowR = rowR58;
        colD = colD58;
        rowD = rowD58;
    } else if (item_ct1.get_group(2) % matrices == 6) {
        valR = valR57; valD = valD57; colR = colR57; rowR = rowR57;
        colD = colD57;
        rowD = rowD57;
    } else if (item_ct1.get_group(2) % matrices == 7) {
        valR = valR56; valD = valD56; colR = colR56; rowR = rowR56;
        colD = colD56;
        rowD = rowD56;
    } else if (item_ct1.get_group(2) % matrices == 8) {
        valR = valR55; valD = valD55; colR = colR55; rowR = rowR55;
        colD = colD55;
        rowD = rowD55;
    } else if (item_ct1.get_group(2) % matrices == 9) {
        valR = valR54; valD = valD54; colR = colR54; rowR = rowR54;
        colD = colD54;
        rowD = rowD54;
    } else if (item_ct1.get_group(2) % matrices == 10) {
        valR = valR53; valD = valD53; colR = colR53; rowR = rowR53;
        colD = colD53;
        rowD = rowD53;
    } else if (item_ct1.get_group(2) % matrices == 11) {
        valR = valR52; valD = valD52; colR = colR52; rowR = rowR52;
        colD = colD52;
        rowD = rowD52;
    } else if (item_ct1.get_group(2) % matrices == 12) {
        valR = valR51; valD = valD51; colR = colR51; rowR = rowR51;
        colD = colD51;
        rowD = rowD51;
    } else if (item_ct1.get_group(2) % matrices == 13) {
        valR = valR50; valD = valD50; colR = colR50; rowR = rowR50;
        colD = colD50;
        rowD = rowD50;
    } else if (item_ct1.get_group(2) % matrices == 14) {
        valR = valR49; valD = valD49; colR = colR49; rowR = rowR49;
        colD = colD49;
        rowD = rowD49;
    } else if (item_ct1.get_group(2) % matrices == 15) {
        valR = valR48; valD = valD48; colR = colR48; rowR = rowR48;
        colD = colD48;
        rowD = rowD48;
    } else if (item_ct1.get_group(2) % matrices == 16) {
        valR = valR47; valD = valD47; colR = colR47; rowR = rowR47;
        colD = colD47;
        rowD = rowD47;
    } else if (item_ct1.get_group(2) % matrices == 17) {
        valR = valR46; valD = valD46; colR = colR46; rowR = rowR46;
        colD = colD46;
        rowD = rowD46;
    } else if (item_ct1.get_group(2) % matrices == 18) {
        valR = valR45; valD = valD45; colR = colR45; rowR = rowR45;
        colD = colD45;
        rowD = rowD45;
    } else if (item_ct1.get_group(2) % matrices == 19) {
        valR = valR44; valD = valD44; colR = colR44; rowR = rowR44;
        colD = colD44;
        rowD = rowD44;
    } else if (item_ct1.get_group(2) % matrices == 20) {
        valR = valR43; valD = valD43; colR = colR43; rowR = rowR43;
        colD = colD43;
        rowD = rowD43;
    } else if (item_ct1.get_group(2) % matrices == 21) {
        valR = valR42; valD = valD42; colR = colR42; rowR = rowR42;
        colD = colD42;
        rowD = rowD42;
    } else if (item_ct1.get_group(2) % matrices == 22) {
        valR = valR41; valD = valD41; colR = colR41; rowR = rowR41;
        colD = colD41;
        rowD = rowD41;
    } else if (item_ct1.get_group(2) % matrices == 23) {
        valR = valR40; valD = valD40; colR = colR40; rowR = rowR40;
        colD = colD40;
        rowD = rowD40;
    } else if (item_ct1.get_group(2) % matrices == 24) {
        valR = valR39; valD = valD39; colR = colR39; rowR = rowR39;
        colD = colD39;
        rowD = rowD39;
    } else if (item_ct1.get_group(2) % matrices == 25) {
        valR = valR38; valD = valD38; colR = colR38; rowR = rowR38;
        colD = colD38;
        rowD = rowD38;
    } else if (item_ct1.get_group(2) % matrices == 26) {
        valR = valR37; valD = valD37; colR = colR37; rowR = rowR37;
        colD = colD37;
        rowD = rowD37;
    } else if (item_ct1.get_group(2) % matrices == 27) {
        valR = valR36; valD = valD36; colR = colR36; rowR = rowR36;
        colD = colD36;
        rowD = rowD36;
    } else if (item_ct1.get_group(2) % matrices == 28) {
        valR = valR35; valD = valD35; colR = colR35; rowR = rowR35;
        colD = colD35;
        rowD = rowD35;
    } else if (item_ct1.get_group(2) % matrices == 29) {
        valR = valR34; valD = valD34; colR = colR34; rowR = rowR34;
        colD = colD34;
        rowD = rowD34;
    } else if (item_ct1.get_group(2) % matrices == 30) {
        valR = valR33; valD = valD33; colR = colR33; rowR = rowR33;
        colD = colD33;
        rowD = rowD33;
    } else if (item_ct1.get_group(2) % matrices == 31) {
        valR = valR32; valD = valD32; colR = colR32; rowR = rowR32;
        colD = colD32;
        rowD = rowD32;
    } else if (item_ct1.get_group(2) % matrices == 32) {
        valR = valR31; valD = valD31; colR = colR31; rowR = rowR31;
        colD = colD31;
        rowD = rowD31;
    } else if (item_ct1.get_group(2) % matrices == 33) {
        valR = valR30; valD = valD30; colR = colR30; rowR = rowR30;
        colD = colD30;
        rowD = rowD30;
    } else if (item_ct1.get_group(2) % matrices == 34) {
        valR = valR29; valD = valD29; colR = colR29; rowR = rowR29;
        colD = colD29;
        rowD = rowD29;
    } else if (item_ct1.get_group(2) % matrices == 35) {
        valR = valR28; valD = valD28; colR = colR28; rowR = rowR28;
        colD = colD28;
        rowD = rowD28;
    } else if (item_ct1.get_group(2) % matrices == 36) {
        valR = valR27; valD = valD27; colR = colR27; rowR = rowR27;
        colD = colD27;
        rowD = rowD27;
    } else if (item_ct1.get_group(2) % matrices == 37) {
        valR = valR26; valD = valD26; colR = colR26; rowR = rowR26;
        colD = colD26;
        rowD = rowD26;
    } else if (item_ct1.get_group(2) % matrices == 38) {
        valR = valR25; valD = valD25; colR = colR25; rowR = rowR25;
        colD = colD25;
        rowD = rowD25;
    } else if (item_ct1.get_group(2) % matrices == 39) {
        valR = valR24; valD = valD24; colR = colR24; rowR = rowR24;
        colD = colD24;
        rowD = rowD24;
    } else if (item_ct1.get_group(2) % matrices == 40) {
        valR = valR23; valD = valD23; colR = colR23; rowR = rowR23;
        colD = colD23;
        rowD = rowD23;
    } else if (item_ct1.get_group(2) % matrices == 41) {
        valR = valR22; valD = valD22; colR = colR22; rowR = rowR22;
        colD = colD22;
        rowD = rowD22;
    } else if (item_ct1.get_group(2) % matrices == 42) {
        valR = valR21; valD = valD21; colR = colR21; rowR = rowR21;
        colD = colD21;
        rowD = rowD21;
    } else if (item_ct1.get_group(2) % matrices == 43) {
        valR = valR20; valD = valD20; colR = colR20; rowR = rowR20;
        colD = colD20;
        rowD = rowD20;
    } else if (item_ct1.get_group(2) % matrices == 44) {
        valR = valR19; valD = valD19; colR = colR19; rowR = rowR19;
        colD = colD19;
        rowD = rowD19;
    } else if (item_ct1.get_group(2) % matrices == 45) {
        valR = valR18; valD = valD18; colR = colR18; rowR = rowR18;
        colD = colD18;
        rowD = rowD18;
    } else if (item_ct1.get_group(2) % matrices == 46) {
        valR = valR17; valD = valD17; colR = colR17; rowR = rowR17;
        colD = colD17;
        rowD = rowD17;
    } else if (item_ct1.get_group(2) % matrices == 47) {
        valR = valR16; valD = valD16; colR = colR16; rowR = rowR16;
        colD = colD16;
        rowD = rowD16;
    } else if (item_ct1.get_group(2) % matrices == 48) {
        valR = valR15; valD = valD15; colR = colR15; rowR = rowR15;
        colD = colD15;
        rowD = rowD15;
    } else if (item_ct1.get_group(2) % matrices == 49) {
        valR = valR14; valD = valD14; colR = colR14; rowR = rowR14;
        colD = colD14;
        rowD = rowD14;
    } else if (item_ct1.get_group(2) % matrices == 50) {
        valR = valR13; valD = valD13; colR = colR13; rowR = rowR13;
        colD = colD13;
        rowD = rowD13;
    } else if (item_ct1.get_group(2) % matrices == 51) {
        valR = valR12; valD = valD12; colR = colR12; rowR = rowR12;
        colD = colD12;
        rowD = rowD12;
    } else if (item_ct1.get_group(2) % matrices == 52) {
        valR = valR11; valD = valD11; colR = colR11; rowR = rowR11;
        colD = colD11;
        rowD = rowD11;
    } else if (item_ct1.get_group(2) % matrices == 53) {
        valR = valR10; valD = valD10; colR = colR10; rowR = rowR10;
        colD = colD10;
        rowD = rowD10;
    } else if (item_ct1.get_group(2) % matrices == 54) {
        valR = valR9; valD = valD9; colR = colR9; rowR = rowR9; colD = colD9;
        rowD = rowD9;
    } else if (item_ct1.get_group(2) % matrices == 55) {
        valR = valR8; valD = valD8; colR = colR8; rowR = rowR8; colD = colD8;
        rowD = rowD8;
    } else if (item_ct1.get_group(2) % matrices == 56) {
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7;
        rowD = rowD7;
    } else if (item_ct1.get_group(2) % matrices == 57) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6;
        rowD = rowD6;
    } else if (item_ct1.get_group(2) % matrices == 58) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5;
        rowD = rowD5;
    } else if (item_ct1.get_group(2) % matrices == 59) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4;
        rowD = rowD4;
    } else if (item_ct1.get_group(2) % matrices == 60) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3;
        rowD = rowD3;
    } else if (item_ct1.get_group(2) % matrices == 61) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2;
        rowD = rowD2;
    } else if (item_ct1.get_group(2) % matrices == 62) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1;
        rowD = rowD1;
    } else if (item_ct1.get_group(2) % matrices == 63) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0;
        rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        bl = b[index];

        if( start != end ){
            #pragma unroll
            for( i=start; i<end; i++ )
                 v += valR[i] * x[ colR[i] ];
        }

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations

        local_x[item_ct1.get_local_id(2)] =
            x[index] + (v - tmp) / (valD[start]);
        /*
        DPCT1065:62: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

#pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[item_ct1.get_local_id(2)] += (v - tmp) / (valD[start]);
        }
        if (item_ct1.get_local_id(2) >=
            overlap) { // only write back the lower subdomain
            x[index] = local_x[item_ct1.get_local_id(2)];
        }
    }
}


/**
    Purpose
    -------
    
    This routine is a block-asynchronous Jacobi iteration 
    with directed restricted additive Schwarz overlap (top-down) performing s
    local Jacobi-updates within the block. Input format is two CSR matrices,
    one containing the diagonal blocks, one containing the rest.

    Arguments
    ---------

    @param[in]
    localiters  magma_int_t
                number of local Jacobi-like updates

    @param[in]
    matrices    magma_int_t
                number of sub-matrices

    @param[in]
    overlap     magma_int_t
                size of the overlap
                
    @param[in]
    D           magma_z_matrix*
                set of matrices with diagonal blocks

    @param[in]
    R           magma_z_matrix*
                set of matrices with non-diagonal parts

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in]
    x           magma_z_matrix*
                iterate/solution

    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbajac_csr_overlap(
    magma_int_t localiters,
    magma_int_t matrices,
    magma_int_t overlap,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue )
{
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;
    int size = D[0].num_rows;
    int min_nnz=100;
        
    for(int i=0; i<matrices; i++){
       min_nnz = min(min_nnz, R[i].nnz);   
    }
    
    if ( min_nnz > -1 ){ 
        if ( matrices == 1 ){
            int dimgrid1 = magma_ceildiv( size  , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto x_dval_ct11 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel1(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, b.dval, x_dval_ct11, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 2) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto x_dval_ct17 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel2(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, b.dval, x_dval_ct17, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 4){
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto D_dval_ct16 = D[2].dval;
                    auto D_drow_ct17 = D[2].drow;
                    auto D_dcol_ct18 = D[2].dcol;
                    auto R_dval_ct19 = R[2].dval;
                    auto R_drow_ct20 = R[2].drow;
                    auto R_dcol_ct21 = R[2].dcol;
                    auto D_dval_ct22 = D[3].dval;
                    auto D_drow_ct23 = D[3].drow;
                    auto D_dcol_ct24 = D[3].dcol;
                    auto R_dval_ct25 = R[3].dval;
                    auto R_drow_ct26 = R[3].drow;
                    auto R_dcol_ct27 = R[3].dcol;
                    auto x_dval_ct29 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel4(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, D_dval_ct16, D_drow_ct17,
                                D_dcol_ct18, R_dval_ct19, R_drow_ct20,
                                R_dcol_ct21, D_dval_ct22, D_drow_ct23,
                                D_dcol_ct24, R_dval_ct25, R_drow_ct26,
                                R_dcol_ct27, b.dval, x_dval_ct29, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 8) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto D_dval_ct16 = D[2].dval;
                    auto D_drow_ct17 = D[2].drow;
                    auto D_dcol_ct18 = D[2].dcol;
                    auto R_dval_ct19 = R[2].dval;
                    auto R_drow_ct20 = R[2].drow;
                    auto R_dcol_ct21 = R[2].dcol;
                    auto D_dval_ct22 = D[3].dval;
                    auto D_drow_ct23 = D[3].drow;
                    auto D_dcol_ct24 = D[3].dcol;
                    auto R_dval_ct25 = R[3].dval;
                    auto R_drow_ct26 = R[3].drow;
                    auto R_dcol_ct27 = R[3].dcol;
                    auto D_dval_ct28 = D[4].dval;
                    auto D_drow_ct29 = D[4].drow;
                    auto D_dcol_ct30 = D[4].dcol;
                    auto R_dval_ct31 = R[4].dval;
                    auto R_drow_ct32 = R[4].drow;
                    auto R_dcol_ct33 = R[4].dcol;
                    auto D_dval_ct34 = D[5].dval;
                    auto D_drow_ct35 = D[5].drow;
                    auto D_dcol_ct36 = D[5].dcol;
                    auto R_dval_ct37 = R[5].dval;
                    auto R_drow_ct38 = R[5].drow;
                    auto R_dcol_ct39 = R[5].dcol;
                    auto D_dval_ct40 = D[6].dval;
                    auto D_drow_ct41 = D[6].drow;
                    auto D_dcol_ct42 = D[6].dcol;
                    auto R_dval_ct43 = R[6].dval;
                    auto R_drow_ct44 = R[6].drow;
                    auto R_dcol_ct45 = R[6].dcol;
                    auto D_dval_ct46 = D[7].dval;
                    auto D_drow_ct47 = D[7].drow;
                    auto D_dcol_ct48 = D[7].dcol;
                    auto R_dval_ct49 = R[7].dval;
                    auto R_drow_ct50 = R[7].drow;
                    auto R_dcol_ct51 = R[7].dcol;
                    auto x_dval_ct53 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel8(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, D_dval_ct16, D_drow_ct17,
                                D_dcol_ct18, R_dval_ct19, R_drow_ct20,
                                R_dcol_ct21, D_dval_ct22, D_drow_ct23,
                                D_dcol_ct24, R_dval_ct25, R_drow_ct26,
                                R_dcol_ct27, D_dval_ct28, D_drow_ct29,
                                D_dcol_ct30, R_dval_ct31, R_drow_ct32,
                                R_dcol_ct33, D_dval_ct34, D_drow_ct35,
                                D_dcol_ct36, R_dval_ct37, R_drow_ct38,
                                R_dcol_ct39, D_dval_ct40, D_drow_ct41,
                                D_dcol_ct42, R_dval_ct43, R_drow_ct44,
                                R_dcol_ct45, D_dval_ct46, D_drow_ct47,
                                D_dcol_ct48, R_dval_ct49, R_drow_ct50,
                                R_dcol_ct51, b.dval, x_dval_ct53, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 16) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto D_dval_ct16 = D[2].dval;
                    auto D_drow_ct17 = D[2].drow;
                    auto D_dcol_ct18 = D[2].dcol;
                    auto R_dval_ct19 = R[2].dval;
                    auto R_drow_ct20 = R[2].drow;
                    auto R_dcol_ct21 = R[2].dcol;
                    auto D_dval_ct22 = D[3].dval;
                    auto D_drow_ct23 = D[3].drow;
                    auto D_dcol_ct24 = D[3].dcol;
                    auto R_dval_ct25 = R[3].dval;
                    auto R_drow_ct26 = R[3].drow;
                    auto R_dcol_ct27 = R[3].dcol;
                    auto D_dval_ct28 = D[4].dval;
                    auto D_drow_ct29 = D[4].drow;
                    auto D_dcol_ct30 = D[4].dcol;
                    auto R_dval_ct31 = R[4].dval;
                    auto R_drow_ct32 = R[4].drow;
                    auto R_dcol_ct33 = R[4].dcol;
                    auto D_dval_ct34 = D[5].dval;
                    auto D_drow_ct35 = D[5].drow;
                    auto D_dcol_ct36 = D[5].dcol;
                    auto R_dval_ct37 = R[5].dval;
                    auto R_drow_ct38 = R[5].drow;
                    auto R_dcol_ct39 = R[5].dcol;
                    auto D_dval_ct40 = D[6].dval;
                    auto D_drow_ct41 = D[6].drow;
                    auto D_dcol_ct42 = D[6].dcol;
                    auto R_dval_ct43 = R[6].dval;
                    auto R_drow_ct44 = R[6].drow;
                    auto R_dcol_ct45 = R[6].dcol;
                    auto D_dval_ct46 = D[7].dval;
                    auto D_drow_ct47 = D[7].drow;
                    auto D_dcol_ct48 = D[7].dcol;
                    auto R_dval_ct49 = R[7].dval;
                    auto R_drow_ct50 = R[7].drow;
                    auto R_dcol_ct51 = R[7].dcol;
                    auto D_dval_ct52 = D[8].dval;
                    auto D_drow_ct53 = D[8].drow;
                    auto D_dcol_ct54 = D[8].dcol;
                    auto R_dval_ct55 = R[8].dval;
                    auto R_drow_ct56 = R[8].drow;
                    auto R_dcol_ct57 = R[8].dcol;
                    auto D_dval_ct58 = D[9].dval;
                    auto D_drow_ct59 = D[9].drow;
                    auto D_dcol_ct60 = D[9].dcol;
                    auto R_dval_ct61 = R[9].dval;
                    auto R_drow_ct62 = R[9].drow;
                    auto R_dcol_ct63 = R[9].dcol;
                    auto D_dval_ct64 = D[10].dval;
                    auto D_drow_ct65 = D[10].drow;
                    auto D_dcol_ct66 = D[10].dcol;
                    auto R_dval_ct67 = R[10].dval;
                    auto R_drow_ct68 = R[10].drow;
                    auto R_dcol_ct69 = R[10].dcol;
                    auto D_dval_ct70 = D[11].dval;
                    auto D_drow_ct71 = D[11].drow;
                    auto D_dcol_ct72 = D[11].dcol;
                    auto R_dval_ct73 = R[11].dval;
                    auto R_drow_ct74 = R[11].drow;
                    auto R_dcol_ct75 = R[11].dcol;
                    auto D_dval_ct76 = D[12].dval;
                    auto D_drow_ct77 = D[12].drow;
                    auto D_dcol_ct78 = D[12].dcol;
                    auto R_dval_ct79 = R[12].dval;
                    auto R_drow_ct80 = R[12].drow;
                    auto R_dcol_ct81 = R[12].dcol;
                    auto D_dval_ct82 = D[13].dval;
                    auto D_drow_ct83 = D[13].drow;
                    auto D_dcol_ct84 = D[13].dcol;
                    auto R_dval_ct85 = R[13].dval;
                    auto R_drow_ct86 = R[13].drow;
                    auto R_dcol_ct87 = R[13].dcol;
                    auto D_dval_ct88 = D[14].dval;
                    auto D_drow_ct89 = D[14].drow;
                    auto D_dcol_ct90 = D[14].dcol;
                    auto R_dval_ct91 = R[14].dval;
                    auto R_drow_ct92 = R[14].drow;
                    auto R_dcol_ct93 = R[14].dcol;
                    auto D_dval_ct94 = D[15].dval;
                    auto D_drow_ct95 = D[15].drow;
                    auto D_dcol_ct96 = D[15].dcol;
                    auto R_dval_ct97 = R[15].dval;
                    auto R_drow_ct98 = R[15].drow;
                    auto R_dcol_ct99 = R[15].dcol;
                    auto x_dval_ct101 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel16(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, D_dval_ct16, D_drow_ct17,
                                D_dcol_ct18, R_dval_ct19, R_drow_ct20,
                                R_dcol_ct21, D_dval_ct22, D_drow_ct23,
                                D_dcol_ct24, R_dval_ct25, R_drow_ct26,
                                R_dcol_ct27, D_dval_ct28, D_drow_ct29,
                                D_dcol_ct30, R_dval_ct31, R_drow_ct32,
                                R_dcol_ct33, D_dval_ct34, D_drow_ct35,
                                D_dcol_ct36, R_dval_ct37, R_drow_ct38,
                                R_dcol_ct39, D_dval_ct40, D_drow_ct41,
                                D_dcol_ct42, R_dval_ct43, R_drow_ct44,
                                R_dcol_ct45, D_dval_ct46, D_drow_ct47,
                                D_dcol_ct48, R_dval_ct49, R_drow_ct50,
                                R_dcol_ct51, D_dval_ct52, D_drow_ct53,
                                D_dcol_ct54, R_dval_ct55, R_drow_ct56,
                                R_dcol_ct57, D_dval_ct58, D_drow_ct59,
                                D_dcol_ct60, R_dval_ct61, R_drow_ct62,
                                R_dcol_ct63, D_dval_ct64, D_drow_ct65,
                                D_dcol_ct66, R_dval_ct67, R_drow_ct68,
                                R_dcol_ct69, D_dval_ct70, D_drow_ct71,
                                D_dcol_ct72, R_dval_ct73, R_drow_ct74,
                                R_dcol_ct75, D_dval_ct76, D_drow_ct77,
                                D_dcol_ct78, R_dval_ct79, R_drow_ct80,
                                R_dcol_ct81, D_dval_ct82, D_drow_ct83,
                                D_dcol_ct84, R_dval_ct85, R_drow_ct86,
                                R_dcol_ct87, D_dval_ct88, D_drow_ct89,
                                D_dcol_ct90, R_dval_ct91, R_drow_ct92,
                                R_dcol_ct93, D_dval_ct94, D_drow_ct95,
                                D_dcol_ct96, R_dval_ct97, R_drow_ct98,
                                R_dcol_ct99, b.dval, x_dval_ct101, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 32) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            /*
            DPCT1042:954: The size of the arguments passed to the SYCL kernel
            exceeds the minimum size limit (1024) for a non-custom SYCL device.
            You can get the hardware argument size limit by querying
            info::device::max_parameter_size. You may need to rewrite this code
            if the size of the arguments exceeds the hardware limit.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto D_dval_ct16 = D[2].dval;
                    auto D_drow_ct17 = D[2].drow;
                    auto D_dcol_ct18 = D[2].dcol;
                    auto R_dval_ct19 = R[2].dval;
                    auto R_drow_ct20 = R[2].drow;
                    auto R_dcol_ct21 = R[2].dcol;
                    auto D_dval_ct22 = D[3].dval;
                    auto D_drow_ct23 = D[3].drow;
                    auto D_dcol_ct24 = D[3].dcol;
                    auto R_dval_ct25 = R[3].dval;
                    auto R_drow_ct26 = R[3].drow;
                    auto R_dcol_ct27 = R[3].dcol;
                    auto D_dval_ct28 = D[4].dval;
                    auto D_drow_ct29 = D[4].drow;
                    auto D_dcol_ct30 = D[4].dcol;
                    auto R_dval_ct31 = R[4].dval;
                    auto R_drow_ct32 = R[4].drow;
                    auto R_dcol_ct33 = R[4].dcol;
                    auto D_dval_ct34 = D[5].dval;
                    auto D_drow_ct35 = D[5].drow;
                    auto D_dcol_ct36 = D[5].dcol;
                    auto R_dval_ct37 = R[5].dval;
                    auto R_drow_ct38 = R[5].drow;
                    auto R_dcol_ct39 = R[5].dcol;
                    auto D_dval_ct40 = D[6].dval;
                    auto D_drow_ct41 = D[6].drow;
                    auto D_dcol_ct42 = D[6].dcol;
                    auto R_dval_ct43 = R[6].dval;
                    auto R_drow_ct44 = R[6].drow;
                    auto R_dcol_ct45 = R[6].dcol;
                    auto D_dval_ct46 = D[7].dval;
                    auto D_drow_ct47 = D[7].drow;
                    auto D_dcol_ct48 = D[7].dcol;
                    auto R_dval_ct49 = R[7].dval;
                    auto R_drow_ct50 = R[7].drow;
                    auto R_dcol_ct51 = R[7].dcol;
                    auto D_dval_ct52 = D[8].dval;
                    auto D_drow_ct53 = D[8].drow;
                    auto D_dcol_ct54 = D[8].dcol;
                    auto R_dval_ct55 = R[8].dval;
                    auto R_drow_ct56 = R[8].drow;
                    auto R_dcol_ct57 = R[8].dcol;
                    auto D_dval_ct58 = D[9].dval;
                    auto D_drow_ct59 = D[9].drow;
                    auto D_dcol_ct60 = D[9].dcol;
                    auto R_dval_ct61 = R[9].dval;
                    auto R_drow_ct62 = R[9].drow;
                    auto R_dcol_ct63 = R[9].dcol;
                    auto D_dval_ct64 = D[10].dval;
                    auto D_drow_ct65 = D[10].drow;
                    auto D_dcol_ct66 = D[10].dcol;
                    auto R_dval_ct67 = R[10].dval;
                    auto R_drow_ct68 = R[10].drow;
                    auto R_dcol_ct69 = R[10].dcol;
                    auto D_dval_ct70 = D[11].dval;
                    auto D_drow_ct71 = D[11].drow;
                    auto D_dcol_ct72 = D[11].dcol;
                    auto R_dval_ct73 = R[11].dval;
                    auto R_drow_ct74 = R[11].drow;
                    auto R_dcol_ct75 = R[11].dcol;
                    auto D_dval_ct76 = D[12].dval;
                    auto D_drow_ct77 = D[12].drow;
                    auto D_dcol_ct78 = D[12].dcol;
                    auto R_dval_ct79 = R[12].dval;
                    auto R_drow_ct80 = R[12].drow;
                    auto R_dcol_ct81 = R[12].dcol;
                    auto D_dval_ct82 = D[13].dval;
                    auto D_drow_ct83 = D[13].drow;
                    auto D_dcol_ct84 = D[13].dcol;
                    auto R_dval_ct85 = R[13].dval;
                    auto R_drow_ct86 = R[13].drow;
                    auto R_dcol_ct87 = R[13].dcol;
                    auto D_dval_ct88 = D[14].dval;
                    auto D_drow_ct89 = D[14].drow;
                    auto D_dcol_ct90 = D[14].dcol;
                    auto R_dval_ct91 = R[14].dval;
                    auto R_drow_ct92 = R[14].drow;
                    auto R_dcol_ct93 = R[14].dcol;
                    auto D_dval_ct94 = D[15].dval;
                    auto D_drow_ct95 = D[15].drow;
                    auto D_dcol_ct96 = D[15].dcol;
                    auto R_dval_ct97 = R[15].dval;
                    auto R_drow_ct98 = R[15].drow;
                    auto R_dcol_ct99 = R[15].dcol;
                    auto D_dval_ct100 = D[16].dval;
                    auto D_drow_ct101 = D[16].drow;
                    auto D_dcol_ct102 = D[16].dcol;
                    auto R_dval_ct103 = R[16].dval;
                    auto R_drow_ct104 = R[16].drow;
                    auto R_dcol_ct105 = R[16].dcol;
                    auto D_dval_ct106 = D[17].dval;
                    auto D_drow_ct107 = D[17].drow;
                    auto D_dcol_ct108 = D[17].dcol;
                    auto R_dval_ct109 = R[17].dval;
                    auto R_drow_ct110 = R[17].drow;
                    auto R_dcol_ct111 = R[17].dcol;
                    auto D_dval_ct112 = D[18].dval;
                    auto D_drow_ct113 = D[18].drow;
                    auto D_dcol_ct114 = D[18].dcol;
                    auto R_dval_ct115 = R[18].dval;
                    auto R_drow_ct116 = R[18].drow;
                    auto R_dcol_ct117 = R[18].dcol;
                    auto D_dval_ct118 = D[19].dval;
                    auto D_drow_ct119 = D[19].drow;
                    auto D_dcol_ct120 = D[19].dcol;
                    auto R_dval_ct121 = R[19].dval;
                    auto R_drow_ct122 = R[19].drow;
                    auto R_dcol_ct123 = R[19].dcol;
                    auto D_dval_ct124 = D[20].dval;
                    auto D_drow_ct125 = D[20].drow;
                    auto D_dcol_ct126 = D[20].dcol;
                    auto R_dval_ct127 = R[20].dval;
                    auto R_drow_ct128 = R[20].drow;
                    auto R_dcol_ct129 = R[20].dcol;
                    auto D_dval_ct130 = D[21].dval;
                    auto D_drow_ct131 = D[21].drow;
                    auto D_dcol_ct132 = D[21].dcol;
                    auto R_dval_ct133 = R[21].dval;
                    auto R_drow_ct134 = R[21].drow;
                    auto R_dcol_ct135 = R[21].dcol;
                    auto D_dval_ct136 = D[22].dval;
                    auto D_drow_ct137 = D[22].drow;
                    auto D_dcol_ct138 = D[22].dcol;
                    auto R_dval_ct139 = R[22].dval;
                    auto R_drow_ct140 = R[22].drow;
                    auto R_dcol_ct141 = R[22].dcol;
                    auto D_dval_ct142 = D[23].dval;
                    auto D_drow_ct143 = D[23].drow;
                    auto D_dcol_ct144 = D[23].dcol;
                    auto R_dval_ct145 = R[23].dval;
                    auto R_drow_ct146 = R[23].drow;
                    auto R_dcol_ct147 = R[23].dcol;
                    auto D_dval_ct148 = D[24].dval;
                    auto D_drow_ct149 = D[24].drow;
                    auto D_dcol_ct150 = D[24].dcol;
                    auto R_dval_ct151 = R[24].dval;
                    auto R_drow_ct152 = R[24].drow;
                    auto R_dcol_ct153 = R[24].dcol;
                    auto D_dval_ct154 = D[25].dval;
                    auto D_drow_ct155 = D[25].drow;
                    auto D_dcol_ct156 = D[25].dcol;
                    auto R_dval_ct157 = R[25].dval;
                    auto R_drow_ct158 = R[25].drow;
                    auto R_dcol_ct159 = R[25].dcol;
                    auto D_dval_ct160 = D[26].dval;
                    auto D_drow_ct161 = D[26].drow;
                    auto D_dcol_ct162 = D[26].dcol;
                    auto R_dval_ct163 = R[26].dval;
                    auto R_drow_ct164 = R[26].drow;
                    auto R_dcol_ct165 = R[26].dcol;
                    auto D_dval_ct166 = D[27].dval;
                    auto D_drow_ct167 = D[27].drow;
                    auto D_dcol_ct168 = D[27].dcol;
                    auto R_dval_ct169 = R[27].dval;
                    auto R_drow_ct170 = R[27].drow;
                    auto R_dcol_ct171 = R[27].dcol;
                    auto D_dval_ct172 = D[28].dval;
                    auto D_drow_ct173 = D[28].drow;
                    auto D_dcol_ct174 = D[28].dcol;
                    auto R_dval_ct175 = R[28].dval;
                    auto R_drow_ct176 = R[28].drow;
                    auto R_dcol_ct177 = R[28].dcol;
                    auto D_dval_ct178 = D[29].dval;
                    auto D_drow_ct179 = D[29].drow;
                    auto D_dcol_ct180 = D[29].dcol;
                    auto R_dval_ct181 = R[29].dval;
                    auto R_drow_ct182 = R[29].drow;
                    auto R_dcol_ct183 = R[29].dcol;
                    auto D_dval_ct184 = D[30].dval;
                    auto D_drow_ct185 = D[30].drow;
                    auto D_dcol_ct186 = D[30].dcol;
                    auto R_dval_ct187 = R[30].dval;
                    auto R_drow_ct188 = R[30].drow;
                    auto R_dcol_ct189 = R[30].dcol;
                    auto D_dval_ct190 = D[31].dval;
                    auto D_drow_ct191 = D[31].drow;
                    auto D_dcol_ct192 = D[31].dcol;
                    auto R_dval_ct193 = R[31].dval;
                    auto R_drow_ct194 = R[31].drow;
                    auto R_dcol_ct195 = R[31].dcol;
                    auto x_dval_ct197 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel32(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, D_dval_ct16, D_drow_ct17,
                                D_dcol_ct18, R_dval_ct19, R_drow_ct20,
                                R_dcol_ct21, D_dval_ct22, D_drow_ct23,
                                D_dcol_ct24, R_dval_ct25, R_drow_ct26,
                                R_dcol_ct27, D_dval_ct28, D_drow_ct29,
                                D_dcol_ct30, R_dval_ct31, R_drow_ct32,
                                R_dcol_ct33, D_dval_ct34, D_drow_ct35,
                                D_dcol_ct36, R_dval_ct37, R_drow_ct38,
                                R_dcol_ct39, D_dval_ct40, D_drow_ct41,
                                D_dcol_ct42, R_dval_ct43, R_drow_ct44,
                                R_dcol_ct45, D_dval_ct46, D_drow_ct47,
                                D_dcol_ct48, R_dval_ct49, R_drow_ct50,
                                R_dcol_ct51, D_dval_ct52, D_drow_ct53,
                                D_dcol_ct54, R_dval_ct55, R_drow_ct56,
                                R_dcol_ct57, D_dval_ct58, D_drow_ct59,
                                D_dcol_ct60, R_dval_ct61, R_drow_ct62,
                                R_dcol_ct63, D_dval_ct64, D_drow_ct65,
                                D_dcol_ct66, R_dval_ct67, R_drow_ct68,
                                R_dcol_ct69, D_dval_ct70, D_drow_ct71,
                                D_dcol_ct72, R_dval_ct73, R_drow_ct74,
                                R_dcol_ct75, D_dval_ct76, D_drow_ct77,
                                D_dcol_ct78, R_dval_ct79, R_drow_ct80,
                                R_dcol_ct81, D_dval_ct82, D_drow_ct83,
                                D_dcol_ct84, R_dval_ct85, R_drow_ct86,
                                R_dcol_ct87, D_dval_ct88, D_drow_ct89,
                                D_dcol_ct90, R_dval_ct91, R_drow_ct92,
                                R_dcol_ct93, D_dval_ct94, D_drow_ct95,
                                D_dcol_ct96, R_dval_ct97, R_drow_ct98,
                                R_dcol_ct99, D_dval_ct100, D_drow_ct101,
                                D_dcol_ct102, R_dval_ct103, R_drow_ct104,
                                R_dcol_ct105, D_dval_ct106, D_drow_ct107,
                                D_dcol_ct108, R_dval_ct109, R_drow_ct110,
                                R_dcol_ct111, D_dval_ct112, D_drow_ct113,
                                D_dcol_ct114, R_dval_ct115, R_drow_ct116,
                                R_dcol_ct117, D_dval_ct118, D_drow_ct119,
                                D_dcol_ct120, R_dval_ct121, R_drow_ct122,
                                R_dcol_ct123, D_dval_ct124, D_drow_ct125,
                                D_dcol_ct126, R_dval_ct127, R_drow_ct128,
                                R_dcol_ct129, D_dval_ct130, D_drow_ct131,
                                D_dcol_ct132, R_dval_ct133, R_drow_ct134,
                                R_dcol_ct135, D_dval_ct136, D_drow_ct137,
                                D_dcol_ct138, R_dval_ct139, R_drow_ct140,
                                R_dcol_ct141, D_dval_ct142, D_drow_ct143,
                                D_dcol_ct144, R_dval_ct145, R_drow_ct146,
                                R_dcol_ct147, D_dval_ct148, D_drow_ct149,
                                D_dcol_ct150, R_dval_ct151, R_drow_ct152,
                                R_dcol_ct153, D_dval_ct154, D_drow_ct155,
                                D_dcol_ct156, R_dval_ct157, R_drow_ct158,
                                R_dcol_ct159, D_dval_ct160, D_drow_ct161,
                                D_dcol_ct162, R_dval_ct163, R_drow_ct164,
                                R_dcol_ct165, D_dval_ct166, D_drow_ct167,
                                D_dcol_ct168, R_dval_ct169, R_drow_ct170,
                                R_dcol_ct171, D_dval_ct172, D_drow_ct173,
                                D_dcol_ct174, R_dval_ct175, R_drow_ct176,
                                R_dcol_ct177, D_dval_ct178, D_drow_ct179,
                                D_dcol_ct180, R_dval_ct181, R_drow_ct182,
                                R_dcol_ct183, D_dval_ct184, D_drow_ct185,
                                D_dcol_ct186, R_dval_ct187, R_drow_ct188,
                                R_dcol_ct189, D_dval_ct190, D_drow_ct191,
                                D_dcol_ct192, R_dval_ct193, R_drow_ct194,
                                R_dcol_ct195, b.dval, x_dval_ct197, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else if (matrices == 64) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            sycl::range<3> grid(dimgrid3, dimgrid2, dimgrid1);
            sycl::range<3> block(1, blocksize2, blocksize1);
            /*
            DPCT1042:956: The size of the arguments passed to the SYCL kernel
            exceeds the minimum size limit (1024) for a non-custom SYCL device.
            You can get the hardware argument size limit by querying
            info::device::max_parameter_size. You may need to rewrite this code
            if the size of the arguments exceeds the hardware limit.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<magmaDoubleComplex, 1> local_x_acc_ct1(
                        sycl::range<1>(BLOCKSIZE), cgh);

                    auto D_dval_ct4 = D[0].dval;
                    auto D_drow_ct5 = D[0].drow;
                    auto D_dcol_ct6 = D[0].dcol;
                    auto R_dval_ct7 = R[0].dval;
                    auto R_drow_ct8 = R[0].drow;
                    auto R_dcol_ct9 = R[0].dcol;
                    auto D_dval_ct10 = D[1].dval;
                    auto D_drow_ct11 = D[1].drow;
                    auto D_dcol_ct12 = D[1].dcol;
                    auto R_dval_ct13 = R[1].dval;
                    auto R_drow_ct14 = R[1].drow;
                    auto R_dcol_ct15 = R[1].dcol;
                    auto D_dval_ct16 = D[2].dval;
                    auto D_drow_ct17 = D[2].drow;
                    auto D_dcol_ct18 = D[2].dcol;
                    auto R_dval_ct19 = R[2].dval;
                    auto R_drow_ct20 = R[2].drow;
                    auto R_dcol_ct21 = R[2].dcol;
                    auto D_dval_ct22 = D[3].dval;
                    auto D_drow_ct23 = D[3].drow;
                    auto D_dcol_ct24 = D[3].dcol;
                    auto R_dval_ct25 = R[3].dval;
                    auto R_drow_ct26 = R[3].drow;
                    auto R_dcol_ct27 = R[3].dcol;
                    auto D_dval_ct28 = D[4].dval;
                    auto D_drow_ct29 = D[4].drow;
                    auto D_dcol_ct30 = D[4].dcol;
                    auto R_dval_ct31 = R[4].dval;
                    auto R_drow_ct32 = R[4].drow;
                    auto R_dcol_ct33 = R[4].dcol;
                    auto D_dval_ct34 = D[5].dval;
                    auto D_drow_ct35 = D[5].drow;
                    auto D_dcol_ct36 = D[5].dcol;
                    auto R_dval_ct37 = R[5].dval;
                    auto R_drow_ct38 = R[5].drow;
                    auto R_dcol_ct39 = R[5].dcol;
                    auto D_dval_ct40 = D[6].dval;
                    auto D_drow_ct41 = D[6].drow;
                    auto D_dcol_ct42 = D[6].dcol;
                    auto R_dval_ct43 = R[6].dval;
                    auto R_drow_ct44 = R[6].drow;
                    auto R_dcol_ct45 = R[6].dcol;
                    auto D_dval_ct46 = D[7].dval;
                    auto D_drow_ct47 = D[7].drow;
                    auto D_dcol_ct48 = D[7].dcol;
                    auto R_dval_ct49 = R[7].dval;
                    auto R_drow_ct50 = R[7].drow;
                    auto R_dcol_ct51 = R[7].dcol;
                    auto D_dval_ct52 = D[8].dval;
                    auto D_drow_ct53 = D[8].drow;
                    auto D_dcol_ct54 = D[8].dcol;
                    auto R_dval_ct55 = R[8].dval;
                    auto R_drow_ct56 = R[8].drow;
                    auto R_dcol_ct57 = R[8].dcol;
                    auto D_dval_ct58 = D[9].dval;
                    auto D_drow_ct59 = D[9].drow;
                    auto D_dcol_ct60 = D[9].dcol;
                    auto R_dval_ct61 = R[9].dval;
                    auto R_drow_ct62 = R[9].drow;
                    auto R_dcol_ct63 = R[9].dcol;
                    auto D_dval_ct64 = D[10].dval;
                    auto D_drow_ct65 = D[10].drow;
                    auto D_dcol_ct66 = D[10].dcol;
                    auto R_dval_ct67 = R[10].dval;
                    auto R_drow_ct68 = R[10].drow;
                    auto R_dcol_ct69 = R[10].dcol;
                    auto D_dval_ct70 = D[11].dval;
                    auto D_drow_ct71 = D[11].drow;
                    auto D_dcol_ct72 = D[11].dcol;
                    auto R_dval_ct73 = R[11].dval;
                    auto R_drow_ct74 = R[11].drow;
                    auto R_dcol_ct75 = R[11].dcol;
                    auto D_dval_ct76 = D[12].dval;
                    auto D_drow_ct77 = D[12].drow;
                    auto D_dcol_ct78 = D[12].dcol;
                    auto R_dval_ct79 = R[12].dval;
                    auto R_drow_ct80 = R[12].drow;
                    auto R_dcol_ct81 = R[12].dcol;
                    auto D_dval_ct82 = D[13].dval;
                    auto D_drow_ct83 = D[13].drow;
                    auto D_dcol_ct84 = D[13].dcol;
                    auto R_dval_ct85 = R[13].dval;
                    auto R_drow_ct86 = R[13].drow;
                    auto R_dcol_ct87 = R[13].dcol;
                    auto D_dval_ct88 = D[14].dval;
                    auto D_drow_ct89 = D[14].drow;
                    auto D_dcol_ct90 = D[14].dcol;
                    auto R_dval_ct91 = R[14].dval;
                    auto R_drow_ct92 = R[14].drow;
                    auto R_dcol_ct93 = R[14].dcol;
                    auto D_dval_ct94 = D[15].dval;
                    auto D_drow_ct95 = D[15].drow;
                    auto D_dcol_ct96 = D[15].dcol;
                    auto R_dval_ct97 = R[15].dval;
                    auto R_drow_ct98 = R[15].drow;
                    auto R_dcol_ct99 = R[15].dcol;
                    auto D_dval_ct100 = D[16].dval;
                    auto D_drow_ct101 = D[16].drow;
                    auto D_dcol_ct102 = D[16].dcol;
                    auto R_dval_ct103 = R[16].dval;
                    auto R_drow_ct104 = R[16].drow;
                    auto R_dcol_ct105 = R[16].dcol;
                    auto D_dval_ct106 = D[17].dval;
                    auto D_drow_ct107 = D[17].drow;
                    auto D_dcol_ct108 = D[17].dcol;
                    auto R_dval_ct109 = R[17].dval;
                    auto R_drow_ct110 = R[17].drow;
                    auto R_dcol_ct111 = R[17].dcol;
                    auto D_dval_ct112 = D[18].dval;
                    auto D_drow_ct113 = D[18].drow;
                    auto D_dcol_ct114 = D[18].dcol;
                    auto R_dval_ct115 = R[18].dval;
                    auto R_drow_ct116 = R[18].drow;
                    auto R_dcol_ct117 = R[18].dcol;
                    auto D_dval_ct118 = D[19].dval;
                    auto D_drow_ct119 = D[19].drow;
                    auto D_dcol_ct120 = D[19].dcol;
                    auto R_dval_ct121 = R[19].dval;
                    auto R_drow_ct122 = R[19].drow;
                    auto R_dcol_ct123 = R[19].dcol;
                    auto D_dval_ct124 = D[20].dval;
                    auto D_drow_ct125 = D[20].drow;
                    auto D_dcol_ct126 = D[20].dcol;
                    auto R_dval_ct127 = R[20].dval;
                    auto R_drow_ct128 = R[20].drow;
                    auto R_dcol_ct129 = R[20].dcol;
                    auto D_dval_ct130 = D[21].dval;
                    auto D_drow_ct131 = D[21].drow;
                    auto D_dcol_ct132 = D[21].dcol;
                    auto R_dval_ct133 = R[21].dval;
                    auto R_drow_ct134 = R[21].drow;
                    auto R_dcol_ct135 = R[21].dcol;
                    auto D_dval_ct136 = D[22].dval;
                    auto D_drow_ct137 = D[22].drow;
                    auto D_dcol_ct138 = D[22].dcol;
                    auto R_dval_ct139 = R[22].dval;
                    auto R_drow_ct140 = R[22].drow;
                    auto R_dcol_ct141 = R[22].dcol;
                    auto D_dval_ct142 = D[23].dval;
                    auto D_drow_ct143 = D[23].drow;
                    auto D_dcol_ct144 = D[23].dcol;
                    auto R_dval_ct145 = R[23].dval;
                    auto R_drow_ct146 = R[23].drow;
                    auto R_dcol_ct147 = R[23].dcol;
                    auto D_dval_ct148 = D[24].dval;
                    auto D_drow_ct149 = D[24].drow;
                    auto D_dcol_ct150 = D[24].dcol;
                    auto R_dval_ct151 = R[24].dval;
                    auto R_drow_ct152 = R[24].drow;
                    auto R_dcol_ct153 = R[24].dcol;
                    auto D_dval_ct154 = D[25].dval;
                    auto D_drow_ct155 = D[25].drow;
                    auto D_dcol_ct156 = D[25].dcol;
                    auto R_dval_ct157 = R[25].dval;
                    auto R_drow_ct158 = R[25].drow;
                    auto R_dcol_ct159 = R[25].dcol;
                    auto D_dval_ct160 = D[26].dval;
                    auto D_drow_ct161 = D[26].drow;
                    auto D_dcol_ct162 = D[26].dcol;
                    auto R_dval_ct163 = R[26].dval;
                    auto R_drow_ct164 = R[26].drow;
                    auto R_dcol_ct165 = R[26].dcol;
                    auto D_dval_ct166 = D[27].dval;
                    auto D_drow_ct167 = D[27].drow;
                    auto D_dcol_ct168 = D[27].dcol;
                    auto R_dval_ct169 = R[27].dval;
                    auto R_drow_ct170 = R[27].drow;
                    auto R_dcol_ct171 = R[27].dcol;
                    auto D_dval_ct172 = D[28].dval;
                    auto D_drow_ct173 = D[28].drow;
                    auto D_dcol_ct174 = D[28].dcol;
                    auto R_dval_ct175 = R[28].dval;
                    auto R_drow_ct176 = R[28].drow;
                    auto R_dcol_ct177 = R[28].dcol;
                    auto D_dval_ct178 = D[29].dval;
                    auto D_drow_ct179 = D[29].drow;
                    auto D_dcol_ct180 = D[29].dcol;
                    auto R_dval_ct181 = R[29].dval;
                    auto R_drow_ct182 = R[29].drow;
                    auto R_dcol_ct183 = R[29].dcol;
                    auto D_dval_ct184 = D[30].dval;
                    auto D_drow_ct185 = D[30].drow;
                    auto D_dcol_ct186 = D[30].dcol;
                    auto R_dval_ct187 = R[30].dval;
                    auto R_drow_ct188 = R[30].drow;
                    auto R_dcol_ct189 = R[30].dcol;
                    auto D_dval_ct190 = D[31].dval;
                    auto D_drow_ct191 = D[31].drow;
                    auto D_dcol_ct192 = D[31].dcol;
                    auto R_dval_ct193 = R[31].dval;
                    auto R_drow_ct194 = R[31].drow;
                    auto R_dcol_ct195 = R[31].dcol;
                    auto D_dval_ct196 = D[32].dval;
                    auto D_drow_ct197 = D[32].drow;
                    auto D_dcol_ct198 = D[32].dcol;
                    auto R_dval_ct199 = R[32].dval;
                    auto R_drow_ct200 = R[32].drow;
                    auto R_dcol_ct201 = R[32].dcol;
                    auto D_dval_ct202 = D[33].dval;
                    auto D_drow_ct203 = D[33].drow;
                    auto D_dcol_ct204 = D[33].dcol;
                    auto R_dval_ct205 = R[33].dval;
                    auto R_drow_ct206 = R[33].drow;
                    auto R_dcol_ct207 = R[33].dcol;
                    auto D_dval_ct208 = D[34].dval;
                    auto D_drow_ct209 = D[34].drow;
                    auto D_dcol_ct210 = D[34].dcol;
                    auto R_dval_ct211 = R[34].dval;
                    auto R_drow_ct212 = R[34].drow;
                    auto R_dcol_ct213 = R[34].dcol;
                    auto D_dval_ct214 = D[35].dval;
                    auto D_drow_ct215 = D[35].drow;
                    auto D_dcol_ct216 = D[35].dcol;
                    auto R_dval_ct217 = R[35].dval;
                    auto R_drow_ct218 = R[35].drow;
                    auto R_dcol_ct219 = R[35].dcol;
                    auto D_dval_ct220 = D[36].dval;
                    auto D_drow_ct221 = D[36].drow;
                    auto D_dcol_ct222 = D[36].dcol;
                    auto R_dval_ct223 = R[36].dval;
                    auto R_drow_ct224 = R[36].drow;
                    auto R_dcol_ct225 = R[36].dcol;
                    auto D_dval_ct226 = D[37].dval;
                    auto D_drow_ct227 = D[37].drow;
                    auto D_dcol_ct228 = D[37].dcol;
                    auto R_dval_ct229 = R[37].dval;
                    auto R_drow_ct230 = R[37].drow;
                    auto R_dcol_ct231 = R[37].dcol;
                    auto D_dval_ct232 = D[38].dval;
                    auto D_drow_ct233 = D[38].drow;
                    auto D_dcol_ct234 = D[38].dcol;
                    auto R_dval_ct235 = R[38].dval;
                    auto R_drow_ct236 = R[38].drow;
                    auto R_dcol_ct237 = R[38].dcol;
                    auto D_dval_ct238 = D[39].dval;
                    auto D_drow_ct239 = D[39].drow;
                    auto D_dcol_ct240 = D[39].dcol;
                    auto R_dval_ct241 = R[39].dval;
                    auto R_drow_ct242 = R[39].drow;
                    auto R_dcol_ct243 = R[39].dcol;
                    auto D_dval_ct244 = D[40].dval;
                    auto D_drow_ct245 = D[40].drow;
                    auto D_dcol_ct246 = D[40].dcol;
                    auto R_dval_ct247 = R[40].dval;
                    auto R_drow_ct248 = R[40].drow;
                    auto R_dcol_ct249 = R[40].dcol;
                    auto D_dval_ct250 = D[41].dval;
                    auto D_drow_ct251 = D[41].drow;
                    auto D_dcol_ct252 = D[41].dcol;
                    auto R_dval_ct253 = R[41].dval;
                    auto R_drow_ct254 = R[41].drow;
                    auto R_dcol_ct255 = R[41].dcol;
                    auto D_dval_ct256 = D[42].dval;
                    auto D_drow_ct257 = D[42].drow;
                    auto D_dcol_ct258 = D[42].dcol;
                    auto R_dval_ct259 = R[42].dval;
                    auto R_drow_ct260 = R[42].drow;
                    auto R_dcol_ct261 = R[42].dcol;
                    auto D_dval_ct262 = D[43].dval;
                    auto D_drow_ct263 = D[43].drow;
                    auto D_dcol_ct264 = D[43].dcol;
                    auto R_dval_ct265 = R[43].dval;
                    auto R_drow_ct266 = R[43].drow;
                    auto R_dcol_ct267 = R[43].dcol;
                    auto D_dval_ct268 = D[44].dval;
                    auto D_drow_ct269 = D[44].drow;
                    auto D_dcol_ct270 = D[44].dcol;
                    auto R_dval_ct271 = R[44].dval;
                    auto R_drow_ct272 = R[44].drow;
                    auto R_dcol_ct273 = R[44].dcol;
                    auto D_dval_ct274 = D[45].dval;
                    auto D_drow_ct275 = D[45].drow;
                    auto D_dcol_ct276 = D[45].dcol;
                    auto R_dval_ct277 = R[45].dval;
                    auto R_drow_ct278 = R[45].drow;
                    auto R_dcol_ct279 = R[45].dcol;
                    auto D_dval_ct280 = D[46].dval;
                    auto D_drow_ct281 = D[46].drow;
                    auto D_dcol_ct282 = D[46].dcol;
                    auto R_dval_ct283 = R[46].dval;
                    auto R_drow_ct284 = R[46].drow;
                    auto R_dcol_ct285 = R[46].dcol;
                    auto D_dval_ct286 = D[47].dval;
                    auto D_drow_ct287 = D[47].drow;
                    auto D_dcol_ct288 = D[47].dcol;
                    auto R_dval_ct289 = R[47].dval;
                    auto R_drow_ct290 = R[47].drow;
                    auto R_dcol_ct291 = R[47].dcol;
                    auto D_dval_ct292 = D[48].dval;
                    auto D_drow_ct293 = D[48].drow;
                    auto D_dcol_ct294 = D[48].dcol;
                    auto R_dval_ct295 = R[48].dval;
                    auto R_drow_ct296 = R[48].drow;
                    auto R_dcol_ct297 = R[48].dcol;
                    auto D_dval_ct298 = D[49].dval;
                    auto D_drow_ct299 = D[49].drow;
                    auto D_dcol_ct300 = D[49].dcol;
                    auto R_dval_ct301 = R[49].dval;
                    auto R_drow_ct302 = R[49].drow;
                    auto R_dcol_ct303 = R[49].dcol;
                    auto D_dval_ct304 = D[50].dval;
                    auto D_drow_ct305 = D[50].drow;
                    auto D_dcol_ct306 = D[50].dcol;
                    auto R_dval_ct307 = R[50].dval;
                    auto R_drow_ct308 = R[50].drow;
                    auto R_dcol_ct309 = R[50].dcol;
                    auto D_dval_ct310 = D[51].dval;
                    auto D_drow_ct311 = D[51].drow;
                    auto D_dcol_ct312 = D[51].dcol;
                    auto R_dval_ct313 = R[51].dval;
                    auto R_drow_ct314 = R[51].drow;
                    auto R_dcol_ct315 = R[51].dcol;
                    auto D_dval_ct316 = D[52].dval;
                    auto D_drow_ct317 = D[52].drow;
                    auto D_dcol_ct318 = D[52].dcol;
                    auto R_dval_ct319 = R[52].dval;
                    auto R_drow_ct320 = R[52].drow;
                    auto R_dcol_ct321 = R[52].dcol;
                    auto D_dval_ct322 = D[53].dval;
                    auto D_drow_ct323 = D[53].drow;
                    auto D_dcol_ct324 = D[53].dcol;
                    auto R_dval_ct325 = R[53].dval;
                    auto R_drow_ct326 = R[53].drow;
                    auto R_dcol_ct327 = R[53].dcol;
                    auto D_dval_ct328 = D[54].dval;
                    auto D_drow_ct329 = D[54].drow;
                    auto D_dcol_ct330 = D[54].dcol;
                    auto R_dval_ct331 = R[54].dval;
                    auto R_drow_ct332 = R[54].drow;
                    auto R_dcol_ct333 = R[54].dcol;
                    auto D_dval_ct334 = D[55].dval;
                    auto D_drow_ct335 = D[55].drow;
                    auto D_dcol_ct336 = D[55].dcol;
                    auto R_dval_ct337 = R[55].dval;
                    auto R_drow_ct338 = R[55].drow;
                    auto R_dcol_ct339 = R[55].dcol;
                    auto D_dval_ct340 = D[56].dval;
                    auto D_drow_ct341 = D[56].drow;
                    auto D_dcol_ct342 = D[56].dcol;
                    auto R_dval_ct343 = R[56].dval;
                    auto R_drow_ct344 = R[56].drow;
                    auto R_dcol_ct345 = R[56].dcol;
                    auto D_dval_ct346 = D[57].dval;
                    auto D_drow_ct347 = D[57].drow;
                    auto D_dcol_ct348 = D[57].dcol;
                    auto R_dval_ct349 = R[57].dval;
                    auto R_drow_ct350 = R[57].drow;
                    auto R_dcol_ct351 = R[57].dcol;
                    auto D_dval_ct352 = D[58].dval;
                    auto D_drow_ct353 = D[58].drow;
                    auto D_dcol_ct354 = D[58].dcol;
                    auto R_dval_ct355 = R[58].dval;
                    auto R_drow_ct356 = R[58].drow;
                    auto R_dcol_ct357 = R[58].dcol;
                    auto D_dval_ct358 = D[59].dval;
                    auto D_drow_ct359 = D[59].drow;
                    auto D_dcol_ct360 = D[59].dcol;
                    auto R_dval_ct361 = R[59].dval;
                    auto R_drow_ct362 = R[59].drow;
                    auto R_dcol_ct363 = R[59].dcol;
                    auto D_dval_ct364 = D[60].dval;
                    auto D_drow_ct365 = D[60].drow;
                    auto D_dcol_ct366 = D[60].dcol;
                    auto R_dval_ct367 = R[60].dval;
                    auto R_drow_ct368 = R[60].drow;
                    auto R_dcol_ct369 = R[60].dcol;
                    auto D_dval_ct370 = D[61].dval;
                    auto D_drow_ct371 = D[61].drow;
                    auto D_dcol_ct372 = D[61].dcol;
                    auto R_dval_ct373 = R[61].dval;
                    auto R_drow_ct374 = R[61].drow;
                    auto R_dcol_ct375 = R[61].dcol;
                    auto D_dval_ct376 = D[62].dval;
                    auto D_drow_ct377 = D[62].drow;
                    auto D_dcol_ct378 = D[62].dcol;
                    auto R_dval_ct379 = R[62].dval;
                    auto R_drow_ct380 = R[62].drow;
                    auto R_dcol_ct381 = R[62].dcol;
                    auto D_dval_ct382 = D[63].dval;
                    auto D_drow_ct383 = D[63].drow;
                    auto D_dcol_ct384 = D[63].dcol;
                    auto R_dval_ct385 = R[63].dval;
                    auto R_drow_ct386 = R[63].drow;
                    auto R_dcol_ct387 = R[63].dcol;
                    auto x_dval_ct389 = x->dval;

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            magma_zbajac_csr_o_ls_kernel64(
                                localiters, size, matrices, overlap, D_dval_ct4,
                                D_drow_ct5, D_dcol_ct6, R_dval_ct7, R_drow_ct8,
                                R_dcol_ct9, D_dval_ct10, D_drow_ct11,
                                D_dcol_ct12, R_dval_ct13, R_drow_ct14,
                                R_dcol_ct15, D_dval_ct16, D_drow_ct17,
                                D_dcol_ct18, R_dval_ct19, R_drow_ct20,
                                R_dcol_ct21, D_dval_ct22, D_drow_ct23,
                                D_dcol_ct24, R_dval_ct25, R_drow_ct26,
                                R_dcol_ct27, D_dval_ct28, D_drow_ct29,
                                D_dcol_ct30, R_dval_ct31, R_drow_ct32,
                                R_dcol_ct33, D_dval_ct34, D_drow_ct35,
                                D_dcol_ct36, R_dval_ct37, R_drow_ct38,
                                R_dcol_ct39, D_dval_ct40, D_drow_ct41,
                                D_dcol_ct42, R_dval_ct43, R_drow_ct44,
                                R_dcol_ct45, D_dval_ct46, D_drow_ct47,
                                D_dcol_ct48, R_dval_ct49, R_drow_ct50,
                                R_dcol_ct51, D_dval_ct52, D_drow_ct53,
                                D_dcol_ct54, R_dval_ct55, R_drow_ct56,
                                R_dcol_ct57, D_dval_ct58, D_drow_ct59,
                                D_dcol_ct60, R_dval_ct61, R_drow_ct62,
                                R_dcol_ct63, D_dval_ct64, D_drow_ct65,
                                D_dcol_ct66, R_dval_ct67, R_drow_ct68,
                                R_dcol_ct69, D_dval_ct70, D_drow_ct71,
                                D_dcol_ct72, R_dval_ct73, R_drow_ct74,
                                R_dcol_ct75, D_dval_ct76, D_drow_ct77,
                                D_dcol_ct78, R_dval_ct79, R_drow_ct80,
                                R_dcol_ct81, D_dval_ct82, D_drow_ct83,
                                D_dcol_ct84, R_dval_ct85, R_drow_ct86,
                                R_dcol_ct87, D_dval_ct88, D_drow_ct89,
                                D_dcol_ct90, R_dval_ct91, R_drow_ct92,
                                R_dcol_ct93, D_dval_ct94, D_drow_ct95,
                                D_dcol_ct96, R_dval_ct97, R_drow_ct98,
                                R_dcol_ct99, D_dval_ct100, D_drow_ct101,
                                D_dcol_ct102, R_dval_ct103, R_drow_ct104,
                                R_dcol_ct105, D_dval_ct106, D_drow_ct107,
                                D_dcol_ct108, R_dval_ct109, R_drow_ct110,
                                R_dcol_ct111, D_dval_ct112, D_drow_ct113,
                                D_dcol_ct114, R_dval_ct115, R_drow_ct116,
                                R_dcol_ct117, D_dval_ct118, D_drow_ct119,
                                D_dcol_ct120, R_dval_ct121, R_drow_ct122,
                                R_dcol_ct123, D_dval_ct124, D_drow_ct125,
                                D_dcol_ct126, R_dval_ct127, R_drow_ct128,
                                R_dcol_ct129, D_dval_ct130, D_drow_ct131,
                                D_dcol_ct132, R_dval_ct133, R_drow_ct134,
                                R_dcol_ct135, D_dval_ct136, D_drow_ct137,
                                D_dcol_ct138, R_dval_ct139, R_drow_ct140,
                                R_dcol_ct141, D_dval_ct142, D_drow_ct143,
                                D_dcol_ct144, R_dval_ct145, R_drow_ct146,
                                R_dcol_ct147, D_dval_ct148, D_drow_ct149,
                                D_dcol_ct150, R_dval_ct151, R_drow_ct152,
                                R_dcol_ct153, D_dval_ct154, D_drow_ct155,
                                D_dcol_ct156, R_dval_ct157, R_drow_ct158,
                                R_dcol_ct159, D_dval_ct160, D_drow_ct161,
                                D_dcol_ct162, R_dval_ct163, R_drow_ct164,
                                R_dcol_ct165, D_dval_ct166, D_drow_ct167,
                                D_dcol_ct168, R_dval_ct169, R_drow_ct170,
                                R_dcol_ct171, D_dval_ct172, D_drow_ct173,
                                D_dcol_ct174, R_dval_ct175, R_drow_ct176,
                                R_dcol_ct177, D_dval_ct178, D_drow_ct179,
                                D_dcol_ct180, R_dval_ct181, R_drow_ct182,
                                R_dcol_ct183, D_dval_ct184, D_drow_ct185,
                                D_dcol_ct186, R_dval_ct187, R_drow_ct188,
                                R_dcol_ct189, D_dval_ct190, D_drow_ct191,
                                D_dcol_ct192, R_dval_ct193, R_drow_ct194,
                                R_dcol_ct195, D_dval_ct196, D_drow_ct197,
                                D_dcol_ct198, R_dval_ct199, R_drow_ct200,
                                R_dcol_ct201, D_dval_ct202, D_drow_ct203,
                                D_dcol_ct204, R_dval_ct205, R_drow_ct206,
                                R_dcol_ct207, D_dval_ct208, D_drow_ct209,
                                D_dcol_ct210, R_dval_ct211, R_drow_ct212,
                                R_dcol_ct213, D_dval_ct214, D_drow_ct215,
                                D_dcol_ct216, R_dval_ct217, R_drow_ct218,
                                R_dcol_ct219, D_dval_ct220, D_drow_ct221,
                                D_dcol_ct222, R_dval_ct223, R_drow_ct224,
                                R_dcol_ct225, D_dval_ct226, D_drow_ct227,
                                D_dcol_ct228, R_dval_ct229, R_drow_ct230,
                                R_dcol_ct231, D_dval_ct232, D_drow_ct233,
                                D_dcol_ct234, R_dval_ct235, R_drow_ct236,
                                R_dcol_ct237, D_dval_ct238, D_drow_ct239,
                                D_dcol_ct240, R_dval_ct241, R_drow_ct242,
                                R_dcol_ct243, D_dval_ct244, D_drow_ct245,
                                D_dcol_ct246, R_dval_ct247, R_drow_ct248,
                                R_dcol_ct249, D_dval_ct250, D_drow_ct251,
                                D_dcol_ct252, R_dval_ct253, R_drow_ct254,
                                R_dcol_ct255, D_dval_ct256, D_drow_ct257,
                                D_dcol_ct258, R_dval_ct259, R_drow_ct260,
                                R_dcol_ct261, D_dval_ct262, D_drow_ct263,
                                D_dcol_ct264, R_dval_ct265, R_drow_ct266,
                                R_dcol_ct267, D_dval_ct268, D_drow_ct269,
                                D_dcol_ct270, R_dval_ct271, R_drow_ct272,
                                R_dcol_ct273, D_dval_ct274, D_drow_ct275,
                                D_dcol_ct276, R_dval_ct277, R_drow_ct278,
                                R_dcol_ct279, D_dval_ct280, D_drow_ct281,
                                D_dcol_ct282, R_dval_ct283, R_drow_ct284,
                                R_dcol_ct285, D_dval_ct286, D_drow_ct287,
                                D_dcol_ct288, R_dval_ct289, R_drow_ct290,
                                R_dcol_ct291, D_dval_ct292, D_drow_ct293,
                                D_dcol_ct294, R_dval_ct295, R_drow_ct296,
                                R_dcol_ct297, D_dval_ct298, D_drow_ct299,
                                D_dcol_ct300, R_dval_ct301, R_drow_ct302,
                                R_dcol_ct303, D_dval_ct304, D_drow_ct305,
                                D_dcol_ct306, R_dval_ct307, R_drow_ct308,
                                R_dcol_ct309, D_dval_ct310, D_drow_ct311,
                                D_dcol_ct312, R_dval_ct313, R_drow_ct314,
                                R_dcol_ct315, D_dval_ct316, D_drow_ct317,
                                D_dcol_ct318, R_dval_ct319, R_drow_ct320,
                                R_dcol_ct321, D_dval_ct322, D_drow_ct323,
                                D_dcol_ct324, R_dval_ct325, R_drow_ct326,
                                R_dcol_ct327, D_dval_ct328, D_drow_ct329,
                                D_dcol_ct330, R_dval_ct331, R_drow_ct332,
                                R_dcol_ct333, D_dval_ct334, D_drow_ct335,
                                D_dcol_ct336, R_dval_ct337, R_drow_ct338,
                                R_dcol_ct339, D_dval_ct340, D_drow_ct341,
                                D_dcol_ct342, R_dval_ct343, R_drow_ct344,
                                R_dcol_ct345, D_dval_ct346, D_drow_ct347,
                                D_dcol_ct348, R_dval_ct349, R_drow_ct350,
                                R_dcol_ct351, D_dval_ct352, D_drow_ct353,
                                D_dcol_ct354, R_dval_ct355, R_drow_ct356,
                                R_dcol_ct357, D_dval_ct358, D_drow_ct359,
                                D_dcol_ct360, R_dval_ct361, R_drow_ct362,
                                R_dcol_ct363, D_dval_ct364, D_drow_ct365,
                                D_dcol_ct366, R_dval_ct367, R_drow_ct368,
                                R_dcol_ct369, D_dval_ct370, D_drow_ct371,
                                D_dcol_ct372, R_dval_ct373, R_drow_ct374,
                                R_dcol_ct375, D_dval_ct376, D_drow_ct377,
                                D_dcol_ct378, R_dval_ct379, R_drow_ct380,
                                R_dcol_ct381, D_dval_ct382, D_drow_ct383,
                                D_dcol_ct384, R_dval_ct385, R_drow_ct386,
                                R_dcol_ct387, b.dval, x_dval_ct389, item_ct1,
                                local_x_acc_ct1.get_pointer());
                        });
                });
        }
        else {
           printf("error: invalid matrix count.\n");
        }
    }
    else {
            printf("error: all elements in diagonal block.\n");
    }
    return MAGMA_SUCCESS;
}
