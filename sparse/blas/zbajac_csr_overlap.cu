/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define PRECISION_z
#define BLOCKSIZE 256

__global__ void magma_zk_testLocking(unsigned int* locks, int n) {
    int id = threadIdx.x % n;
    bool leaveLoop = false;
    while (!leaveLoop) {
        if (atomicExch(&(locks[id]), 1u) == 0u) {
            //critical section
            leaveLoop = true;
            atomicExch(&(locks[id]),0u);
        }
    }
}

/*
__global__ void
magma_zbajac_csr_o_ls_kernel(int localiters, int n, 
                             int matrices, int overlap, 
                             magma_z_matrix *D, magma_z_matrix *R,
                             const magmaDoubleComplex *  __restrict__ b,                            
                             magmaDoubleComplex * x )
{
   // int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
   // int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
        int inddiag =  blockIdx.x*blockDim.x/2-blockDim.x/2;
    int index   = blockIdx.x*blockDim.x/2+threadIdx.x-blockDim.x/2;
    int i, j, start, end;
    
     __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    //valR = R[ (1+blockIdx.x-1)%matrices ].dval;
    //colR = R[ (1+blockIdx.x-1)%matrices ].dcol;
    //rowR = R[ (1+blockIdx.x-1)%matrices ].drow;
    //valD = D[ (1+blockIdx.x-1)%matrices ].dval;
    //colD = D[ (1+blockIdx.x-1)%matrices ].dcol;
    //rowD = D[ (1+blockIdx.x-1)%matrices ].drow;
    
    if (blockIdx.x%2 == 1) {
        valR = R[0].dval;
        valD = D[0].dval;
        colR = R[0].dcol;
        rowR = R[0].drow;
        colD = D[0].dcol;
        rowD = D[0].drow;
    }
    else {
        valR = R[1].dval;
        valD = D[1].dval;
        colR = R[1].dcol;
        rowR = R[1].drow;
        colD = D[1].dcol;
        rowD = D[1].drow;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];
printf("bdx:%d idx:%d  start:%d  end:%d\n", blockIdx.x, threadIdx.x, start, end);

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif

        #pragma unroll
        for (i = start; i < end; i++)
             v += valR[i] * x[ colR[i] ];
        
        start = rowD[index];
        end   = rowD[index+1];
        
        #pragma unroll
        for (i = start; i < end; i++)
            tmp += valD[i] * x[ colD[i] ];
        
        v =  bl - v;
        
        // add more local iterations            
        
        local_x[threadIdx.x] = x[index]; //+ ( v - tmp); // / (valD[start]);
        __syncthreads();
        
        #pragma unroll
        for (j = 0; j < localiters-1; j++)
        {
            tmp = zero;
            #pragma unroll
            for (i = start; i < end; i++)
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if (threadIdx.x > overlap) { // RAS
            x[index] = local_x[threadIdx.x];
        }
    }
}

*/

__global__ void
magma_zbajac_csr_o_ls_kernel1(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD, 
                            magma_index_t * rowD, 
                            magma_index_t * colD, 
                            magmaDoubleComplex * valR, 
                            magma_index_t * rowR,
                            magma_index_t * colR, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*blockDim.x;
    int index   =  blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;
    //bool leaveLoop = false;
    

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];

        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero; 


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif

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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}


__global__ void
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
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*blockDim.x/2-blockDim.x/2;
    int index   = blockIdx.x*blockDim.x/2+threadIdx.x-blockDim.x/2;
    int i, j, start, end;
    //bool leaveLoop = false;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if (blockIdx.x%matrices == 0) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    }
    else if (blockIdx.x%matrices == 1) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }
    
    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}


__global__ void
magma_zbajac_csr_o_ls_kernel4(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD0, magma_index_t * rowD0, magma_index_t * colD0, magmaDoubleComplex * valR0, magma_index_t * rowR0, magma_index_t * colR0, 
                            magmaDoubleComplex * valD1, magma_index_t * rowD1, magma_index_t * colD1, magmaDoubleComplex * valR1, magma_index_t * rowR1, magma_index_t * colR1, 
                            magmaDoubleComplex * valD2, magma_index_t * rowD2, magma_index_t * colD2, magmaDoubleComplex * valR2, magma_index_t * rowR2, magma_index_t * colR2, 
                            magmaDoubleComplex * valD3, magma_index_t * rowD3, magma_index_t * colD3, magmaDoubleComplex * valR3, magma_index_t * rowR3, magma_index_t * colR3, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
    int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
    int i, j, start, end;
    //bool leaveLoop = false;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if ( blockIdx.x%matrices==0 ) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3; rowD = rowD3;
    }else if ( blockIdx.x%matrices==1 ) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2; rowD = rowD2;
    }else if ( blockIdx.x%matrices==2 ) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    }else if ( blockIdx.x%matrices==3 ) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}


__global__ void
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
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
    int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
    int i, j, start, end;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if( blockIdx.x%matrices==0 ){
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7; rowD = rowD7;
    }else if ( blockIdx.x%matrices==1 ) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6; rowD = rowD6;
    }else if ( blockIdx.x%matrices==2 ) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5; rowD = rowD5;
    }else if ( blockIdx.x%matrices==3 ) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4; rowD = rowD4;
    }else if ( blockIdx.x%matrices==4 ) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3; rowD = rowD3;
    }else if ( blockIdx.x%matrices==5 ) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2; rowD = rowD2;
    }else if ( blockIdx.x%matrices==6 ) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    }else if ( blockIdx.x%matrices==7 ) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}


__global__ void
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
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
    int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
    int i, j, start, end;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if      ( blockIdx.x%matrices==0  ) { valR = valR15; valD = valD15; colR = colR15; rowR = rowR15; colD = colD15; rowD = rowD15; }
    else if ( blockIdx.x%matrices==1  ) { valR = valR14; valD = valD14; colR = colR14; rowR = rowR14; colD = colD14; rowD = rowD14; }
    else if ( blockIdx.x%matrices==2  ) { valR = valR13; valD = valD13; colR = colR13; rowR = rowR13; colD = colD13; rowD = rowD13; }
    else if ( blockIdx.x%matrices==3  ) { valR = valR12; valD = valD12; colR = colR12; rowR = rowR12; colD = colD12; rowD = rowD12; }
    else if ( blockIdx.x%matrices==4  ) { valR = valR11; valD = valD11; colR = colR11; rowR = rowR11; colD = colD11; rowD = rowD11; }
    else if ( blockIdx.x%matrices==5  ) { valR = valR10; valD = valD10; colR = colR10; rowR = rowR10; colD = colD10; rowD = rowD10; }
    else if ( blockIdx.x%matrices==6  ) { valR = valR9;  valD = valD9;  colR = colR9;  rowR = rowR9;  colD = colD9;  rowD = rowD9;  }
    else if ( blockIdx.x%matrices==7  ) { valR = valR8;  valD = valD8;  colR = colR8;  rowR = rowR8;  colD = colD8;  rowD = rowD8;  }
    else if ( blockIdx.x%matrices==8  ) { valR = valR7;  valD = valD7;  colR = colR7;  rowR = rowR7;  colD = colD7;  rowD = rowD7;  }
    else if ( blockIdx.x%matrices==9  ) { valR = valR6;  valD = valD6;  colR = colR6;  rowR = rowR6;  colD = colD6;  rowD = rowD6;  }
    else if ( blockIdx.x%matrices==10 ) { valR = valR5;  valD = valD5;  colR = colR5;  rowR = rowR5;  colD = colD5;  rowD = rowD5;  }
    else if ( blockIdx.x%matrices==11 ) { valR = valR4;  valD = valD4;  colR = colR4;  rowR = rowR4;  colD = colD4;  rowD = rowD4;  }
    else if ( blockIdx.x%matrices==12 ) { valR = valR3;  valD = valD3;  colR = colR3;  rowR = rowR3;  colD = colD3;  rowD = rowD3;  }
    else if ( blockIdx.x%matrices==13 ) { valR = valR2;  valD = valD2;  colR = colR2;  rowR = rowR2;  colD = colD2;  rowD = rowD2;  }
    else if ( blockIdx.x%matrices==14 ) { valR = valR1;  valD = valD1;  colR = colR1;  rowR = rowR1;  colD = colD1;  rowD = rowD1;  }
    else if ( blockIdx.x%matrices==15 ) { valR = valR0;  valD = valD0;  colR = colR0;  rowR = rowR0;  colD = colD0;  rowD = rowD0;  }


    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}

__global__ void
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
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
    int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
    int i, j, start, end;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if      ( blockIdx.x%matrices==0  ) { valR = valR31; valD = valD31; colR = colR31; rowR = rowR31; colD = colD31; rowD = rowD31; }
    else if ( blockIdx.x%matrices==1  ) { valR = valR30; valD = valD30; colR = colR30; rowR = rowR30; colD = colD30; rowD = rowD30; }
    else if ( blockIdx.x%matrices==2  ) { valR = valR29; valD = valD29; colR = colR29; rowR = rowR29; colD = colD29; rowD = rowD29; }
    else if ( blockIdx.x%matrices==3  ) { valR = valR28; valD = valD28; colR = colR28; rowR = rowR28; colD = colD28; rowD = rowD28; }
    else if ( blockIdx.x%matrices==4  ) { valR = valR27; valD = valD27; colR = colR27; rowR = rowR27; colD = colD27; rowD = rowD27; }
    else if ( blockIdx.x%matrices==5  ) { valR = valR26; valD = valD26; colR = colR26; rowR = rowR26; colD = colD26; rowD = rowD26; }
    else if ( blockIdx.x%matrices==6  ) { valR = valR25; valD = valD25; colR = colR25; rowR = rowR25; colD = colD25; rowD = rowD25; }
    else if ( blockIdx.x%matrices==7  ) { valR = valR24; valD = valD24; colR = colR24; rowR = rowR24; colD = colD24; rowD = rowD24; }
    else if ( blockIdx.x%matrices==8  ) { valR = valR23; valD = valD23; colR = colR23; rowR = rowR23; colD = colD23; rowD = rowD23; }
    else if ( blockIdx.x%matrices==9  ) { valR = valR22; valD = valD22; colR = colR22; rowR = rowR22; colD = colD22; rowD = rowD22; }
    else if ( blockIdx.x%matrices==10 ) { valR = valR21; valD = valD21; colR = colR21; rowR = rowR21; colD = colD21; rowD = rowD21; }
    else if ( blockIdx.x%matrices==11 ) { valR = valR20; valD = valD20; colR = colR20; rowR = rowR20; colD = colD20; rowD = rowD20; }
    else if ( blockIdx.x%matrices==12 ) { valR = valR19; valD = valD19; colR = colR19; rowR = rowR19; colD = colD19; rowD = rowD19; }
    else if ( blockIdx.x%matrices==13 ) { valR = valR18; valD = valD18; colR = colR18; rowR = rowR18; colD = colD18; rowD = rowD18; }
    else if ( blockIdx.x%matrices==14 ) { valR = valR17; valD = valD17; colR = colR17; rowR = rowR17; colD = colD17; rowD = rowD17; }
    else if ( blockIdx.x%matrices==15 ) { valR = valR16; valD = valD16; colR = colR16; rowR = rowR16; colD = colD16; rowD = rowD16; }
    else if ( blockIdx.x%matrices==16 ) { valR = valR15; valD = valD15; colR = colR15; rowR = rowR15; colD = colD15; rowD = rowD15; }
    else if ( blockIdx.x%matrices==17 ) { valR = valR14; valD = valD14; colR = colR14; rowR = rowR14; colD = colD14; rowD = rowD14; }
    else if ( blockIdx.x%matrices==18 ) { valR = valR13; valD = valD13; colR = colR13; rowR = rowR13; colD = colD13; rowD = rowD13; }
    else if ( blockIdx.x%matrices==19 ) { valR = valR12; valD = valD12; colR = colR12; rowR = rowR12; colD = colD12; rowD = rowD12; }
    else if ( blockIdx.x%matrices==20 ) { valR = valR11; valD = valD11; colR = colR11; rowR = rowR11; colD = colD11; rowD = rowD11; }
    else if ( blockIdx.x%matrices==21 ) { valR = valR10; valD = valD10; colR = colR10; rowR = rowR10; colD = colD10; rowD = rowD10; }
    else if ( blockIdx.x%matrices==22 ) { valR = valR9;  valD = valD9;  colR = colR9;  rowR = rowR9;  colD = colD9;  rowD = rowD9;  }
    else if ( blockIdx.x%matrices==23 ) { valR = valR8;  valD = valD8;  colR = colR8;  rowR = rowR8;  colD = colD8;  rowD = rowD8;  }
    else if ( blockIdx.x%matrices==24 ) { valR = valR7;  valD = valD7;  colR = colR7;  rowR = rowR7;  colD = colD7;  rowD = rowD7;  }
    else if ( blockIdx.x%matrices==25 ) { valR = valR6;  valD = valD6;  colR = colR6;  rowR = rowR6;  colD = colD6;  rowD = rowD6;  }
    else if ( blockIdx.x%matrices==26 ) { valR = valR5;  valD = valD5;  colR = colR5;  rowR = rowR5;  colD = colD5;  rowD = rowD5;  }
    else if ( blockIdx.x%matrices==27 ) { valR = valR4;  valD = valD4;  colR = colR4;  rowR = rowR4;  colD = colD4;  rowD = rowD4;  }
    else if ( blockIdx.x%matrices==28 ) { valR = valR3;  valD = valD3;  colR = colR3;  rowR = rowR3;  colD = colD3;  rowD = rowD3;  }
    else if ( blockIdx.x%matrices==29 ) { valR = valR2;  valD = valD2;  colR = colR2;  rowR = rowR2;  colD = colD2;  rowD = rowD2;  }
    else if ( blockIdx.x%matrices==30 ) { valR = valR1;  valD = valD1;  colR = colR1;  rowR = rowR1;  colD = colD1;  rowD = rowD1;  }
    else if ( blockIdx.x%matrices==31 ) { valR = valR0;  valD = valD0;  colR = colR0;  rowR = rowR0;  colD = colD0;  rowD = rowD0;  }
    

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }
}

__global__ void
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
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
    int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
    int i, j, start, end;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if      ( blockIdx.x%matrices==0  ) { valR = valR63; valD = valD63; colR = colR63; rowR = rowR63; colD = colD63; rowD = rowD63; }
    else if ( blockIdx.x%matrices==1  ) { valR = valR62; valD = valD62; colR = colR62; rowR = rowR62; colD = colD62; rowD = rowD62; }
    else if ( blockIdx.x%matrices==2  ) { valR = valR61; valD = valD61; colR = colR61; rowR = rowR61; colD = colD61; rowD = rowD61; }
    else if ( blockIdx.x%matrices==3  ) { valR = valR60; valD = valD60; colR = colR60; rowR = rowR60; colD = colD60; rowD = rowD60; }
    else if ( blockIdx.x%matrices==4  ) { valR = valR59; valD = valD59; colR = colR59; rowR = rowR59; colD = colD59; rowD = rowD59; }
    else if ( blockIdx.x%matrices==5  ) { valR = valR58; valD = valD58; colR = colR58; rowR = rowR58; colD = colD58; rowD = rowD58; }
    else if ( blockIdx.x%matrices==6  ) { valR = valR57; valD = valD57; colR = colR57; rowR = rowR57; colD = colD57; rowD = rowD57; }
    else if ( blockIdx.x%matrices==7  ) { valR = valR56; valD = valD56; colR = colR56; rowR = rowR56; colD = colD56; rowD = rowD56; }
    else if ( blockIdx.x%matrices==8  ) { valR = valR55; valD = valD55; colR = colR55; rowR = rowR55; colD = colD55; rowD = rowD55; }
    else if ( blockIdx.x%matrices==9  ) { valR = valR54; valD = valD54; colR = colR54; rowR = rowR54; colD = colD54; rowD = rowD54; }
    else if ( blockIdx.x%matrices==10 ) { valR = valR53; valD = valD53; colR = colR53; rowR = rowR53; colD = colD53; rowD = rowD53; }
    else if ( blockIdx.x%matrices==11 ) { valR = valR52; valD = valD52; colR = colR52; rowR = rowR52; colD = colD52; rowD = rowD52; }
    else if ( blockIdx.x%matrices==12 ) { valR = valR51; valD = valD51; colR = colR51; rowR = rowR51; colD = colD51; rowD = rowD51; }
    else if ( blockIdx.x%matrices==13 ) { valR = valR50; valD = valD50; colR = colR50; rowR = rowR50; colD = colD50; rowD = rowD50; }
    else if ( blockIdx.x%matrices==14 ) { valR = valR49; valD = valD49; colR = colR49; rowR = rowR49; colD = colD49; rowD = rowD49; }
    else if ( blockIdx.x%matrices==15 ) { valR = valR48; valD = valD48; colR = colR48; rowR = rowR48; colD = colD48; rowD = rowD48; }
    else if ( blockIdx.x%matrices==16 ) { valR = valR47; valD = valD47; colR = colR47; rowR = rowR47; colD = colD47; rowD = rowD47; }
    else if ( blockIdx.x%matrices==17 ) { valR = valR46; valD = valD46; colR = colR46; rowR = rowR46; colD = colD46; rowD = rowD46; }
    else if ( blockIdx.x%matrices==18 ) { valR = valR45; valD = valD45; colR = colR45; rowR = rowR45; colD = colD45; rowD = rowD45; }
    else if ( blockIdx.x%matrices==19 ) { valR = valR44; valD = valD44; colR = colR44; rowR = rowR44; colD = colD44; rowD = rowD44; }
    else if ( blockIdx.x%matrices==20 ) { valR = valR43; valD = valD43; colR = colR43; rowR = rowR43; colD = colD43; rowD = rowD43; }
    else if ( blockIdx.x%matrices==21 ) { valR = valR42; valD = valD42; colR = colR42; rowR = rowR42; colD = colD42; rowD = rowD42; }
    else if ( blockIdx.x%matrices==22 ) { valR = valR41; valD = valD41; colR = colR41; rowR = rowR41; colD = colD41; rowD = rowD41; }
    else if ( blockIdx.x%matrices==23 ) { valR = valR40; valD = valD40; colR = colR40; rowR = rowR40; colD = colD40; rowD = rowD40; }
    else if ( blockIdx.x%matrices==24 ) { valR = valR39; valD = valD39; colR = colR39; rowR = rowR39; colD = colD39; rowD = rowD39; }
    else if ( blockIdx.x%matrices==25 ) { valR = valR38; valD = valD38; colR = colR38; rowR = rowR38; colD = colD38; rowD = rowD38; }
    else if ( blockIdx.x%matrices==26 ) { valR = valR37; valD = valD37; colR = colR37; rowR = rowR37; colD = colD37; rowD = rowD37; }
    else if ( blockIdx.x%matrices==27 ) { valR = valR36; valD = valD36; colR = colR36; rowR = rowR36; colD = colD36; rowD = rowD36; }
    else if ( blockIdx.x%matrices==28 ) { valR = valR35; valD = valD35; colR = colR35; rowR = rowR35; colD = colD35; rowD = rowD35; }
    else if ( blockIdx.x%matrices==29 ) { valR = valR34; valD = valD34; colR = colR34; rowR = rowR34; colD = colD34; rowD = rowD34; }
    else if ( blockIdx.x%matrices==30 ) { valR = valR33; valD = valD33; colR = colR33; rowR = rowR33; colD = colD33; rowD = rowD33; }
    else if ( blockIdx.x%matrices==31 ) { valR = valR32; valD = valD32; colR = colR32; rowR = rowR32; colD = colD32; rowD = rowD32; }
    else if ( blockIdx.x%matrices==32 ) { valR = valR31; valD = valD31; colR = colR31; rowR = rowR31; colD = colD31; rowD = rowD31; }
    else if ( blockIdx.x%matrices==33 ) { valR = valR30; valD = valD30; colR = colR30; rowR = rowR30; colD = colD30; rowD = rowD30; }
    else if ( blockIdx.x%matrices==34 ) { valR = valR29; valD = valD29; colR = colR29; rowR = rowR29; colD = colD29; rowD = rowD29; }
    else if ( blockIdx.x%matrices==35 ) { valR = valR28; valD = valD28; colR = colR28; rowR = rowR28; colD = colD28; rowD = rowD28; }
    else if ( blockIdx.x%matrices==36 ) { valR = valR27; valD = valD27; colR = colR27; rowR = rowR27; colD = colD27; rowD = rowD27; }
    else if ( blockIdx.x%matrices==37 ) { valR = valR26; valD = valD26; colR = colR26; rowR = rowR26; colD = colD26; rowD = rowD26; }
    else if ( blockIdx.x%matrices==38 ) { valR = valR25; valD = valD25; colR = colR25; rowR = rowR25; colD = colD25; rowD = rowD25; }
    else if ( blockIdx.x%matrices==39 ) { valR = valR24; valD = valD24; colR = colR24; rowR = rowR24; colD = colD24; rowD = rowD24; }
    else if ( blockIdx.x%matrices==40 ) { valR = valR23; valD = valD23; colR = colR23; rowR = rowR23; colD = colD23; rowD = rowD23; }
    else if ( blockIdx.x%matrices==41 ) { valR = valR22; valD = valD22; colR = colR22; rowR = rowR22; colD = colD22; rowD = rowD22; }
    else if ( blockIdx.x%matrices==42 ) { valR = valR21; valD = valD21; colR = colR21; rowR = rowR21; colD = colD21; rowD = rowD21; }
    else if ( blockIdx.x%matrices==43 ) { valR = valR20; valD = valD20; colR = colR20; rowR = rowR20; colD = colD20; rowD = rowD20; }
    else if ( blockIdx.x%matrices==44 ) { valR = valR19; valD = valD19; colR = colR19; rowR = rowR19; colD = colD19; rowD = rowD19; }
    else if ( blockIdx.x%matrices==45 ) { valR = valR18; valD = valD18; colR = colR18; rowR = rowR18; colD = colD18; rowD = rowD18; }
    else if ( blockIdx.x%matrices==46 ) { valR = valR17; valD = valD17; colR = colR17; rowR = rowR17; colD = colD17; rowD = rowD17; }
    else if ( blockIdx.x%matrices==47 ) { valR = valR16; valD = valD16; colR = colR16; rowR = rowR16; colD = colD16; rowD = rowD16; }
    else if ( blockIdx.x%matrices==48 ) { valR = valR15; valD = valD15; colR = colR15; rowR = rowR15; colD = colD15; rowD = rowD15; }
    else if ( blockIdx.x%matrices==49 ) { valR = valR14; valD = valD14; colR = colR14; rowR = rowR14; colD = colD14; rowD = rowD14; }
    else if ( blockIdx.x%matrices==50 ) { valR = valR13; valD = valD13; colR = colR13; rowR = rowR13; colD = colD13; rowD = rowD13; }
    else if ( blockIdx.x%matrices==51 ) { valR = valR12; valD = valD12; colR = colR12; rowR = rowR12; colD = colD12; rowD = rowD12; }
    else if ( blockIdx.x%matrices==52 ) { valR = valR11; valD = valD11; colR = colR11; rowR = rowR11; colD = colD11; rowD = rowD11; }
    else if ( blockIdx.x%matrices==53 ) { valR = valR10; valD = valD10; colR = colR10; rowR = rowR10; colD = colD10; rowD = rowD10; }
    else if ( blockIdx.x%matrices==54 ) { valR = valR9;  valD = valD9;  colR = colR9;  rowR = rowR9;  colD = colD9;  rowD = rowD9;  }
    else if ( blockIdx.x%matrices==55 ) { valR = valR8;  valD = valD8;  colR = colR8;  rowR = rowR8;  colD = colD8;  rowD = rowD8;  }
    else if ( blockIdx.x%matrices==56 ) { valR = valR7;  valD = valD7;  colR = colR7;  rowR = rowR7;  colD = colD7;  rowD = rowD7;  }
    else if ( blockIdx.x%matrices==57 ) { valR = valR6;  valD = valD6;  colR = colR6;  rowR = rowR6;  colD = colD6;  rowD = rowD6;  }
    else if ( blockIdx.x%matrices==58 ) { valR = valR5;  valD = valD5;  colR = colR5;  rowR = rowR5;  colD = colD5;  rowD = rowD5;  }
    else if ( blockIdx.x%matrices==59 ) { valR = valR4;  valD = valD4;  colR = colR4;  rowR = rowR4;  colD = colD4;  rowD = rowD4;  }
    else if ( blockIdx.x%matrices==60 ) { valR = valR3;  valD = valD3;  colR = colR3;  rowR = rowR3;  colD = colD3;  rowD = rowD3;  }
    else if ( blockIdx.x%matrices==61 ) { valR = valR2;  valD = valD2;  colR = colR2;  rowR = rowR2;  colD = colD2;  rowD = rowD2;  }
    else if ( blockIdx.x%matrices==62 ) { valR = valR1;  valD = valD1;  colR = colR1;  rowR = rowR1;  colD = colD1;  rowD = rowD1;  }
    else if ( blockIdx.x%matrices==63 ) { valR = valR0;  valD = valD0;  colR = colR0;  rowR = rowR0;  colD = colD0;  rowD = rowD0;  }
    

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


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
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
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
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel1<<< grid, block, 0, queue->cuda_stream() >>>
            ( localiters, size, matrices, overlap,
            D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
            b.dval, x->dval );  
        }
        else if (matrices == 2) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel2<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
                    D[1].dval, D[1].drow, D[1].dcol, R[1].dval, R[1].drow, R[1].dcol,
                    b.dval, x->dval );  
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue->cuda_stream() >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
        }
        else if (matrices == 4){
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel4<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
                    D[1].dval, D[1].drow, D[1].dcol, R[1].dval, R[1].drow, R[1].dcol,
                    D[2].dval, D[2].drow, D[2].dcol, R[2].dval, R[2].drow, R[2].dcol,
                    D[3].dval, D[3].drow, D[3].dcol, R[3].dval, R[3].drow, R[3].dcol,
                    b.dval, x->dval );  
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue->cuda_stream() >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
        }
        else if (matrices == 8) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel8<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
                    D[1].dval, D[1].drow, D[1].dcol, R[1].dval, R[1].drow, R[1].dcol,
                    D[2].dval, D[2].drow, D[2].dcol, R[2].dval, R[2].drow, R[2].dcol,
                    D[3].dval, D[3].drow, D[3].dcol, R[3].dval, R[3].drow, R[3].dcol,
                    D[4].dval, D[4].drow, D[4].dcol, R[4].dval, R[4].drow, R[4].dcol,
                    D[5].dval, D[5].drow, D[5].dcol, R[5].dval, R[5].drow, R[5].dcol,
                    D[6].dval, D[6].drow, D[6].dcol, R[6].dval, R[6].drow, R[6].dcol,
                    D[7].dval, D[7].drow, D[7].dcol, R[7].dval, R[7].drow, R[7].dcol,
                    b.dval, x->dval );  
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue->cuda_stream() >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
        }
        else if (matrices == 16) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel16<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[ 0].dval, D[ 0].drow, D[ 0].dcol, R[ 0].dval, R[ 0].drow, R[ 0].dcol, 
                    D[ 1].dval, D[ 1].drow, D[ 1].dcol, R[ 1].dval, R[ 1].drow, R[ 1].dcol,
                    D[ 2].dval, D[ 2].drow, D[ 2].dcol, R[ 2].dval, R[ 2].drow, R[ 2].dcol,
                    D[ 3].dval, D[ 3].drow, D[ 3].dcol, R[ 3].dval, R[ 3].drow, R[ 3].dcol,
                    D[ 4].dval, D[ 4].drow, D[ 4].dcol, R[ 4].dval, R[ 4].drow, R[ 4].dcol,
                    D[ 5].dval, D[ 5].drow, D[ 5].dcol, R[ 5].dval, R[ 5].drow, R[ 5].dcol,
                    D[ 6].dval, D[ 6].drow, D[ 6].dcol, R[ 6].dval, R[ 6].drow, R[ 6].dcol,
                    D[ 7].dval, D[ 7].drow, D[ 7].dcol, R[ 7].dval, R[ 7].drow, R[ 7].dcol,
                    D[ 8].dval, D[ 8].drow, D[ 8].dcol, R[ 8].dval, R[ 8].drow, R[ 8].dcol, 
                    D[ 9].dval, D[ 9].drow, D[ 9].dcol, R[ 9].dval, R[ 9].drow, R[ 9].dcol,
                    D[10].dval, D[10].drow, D[10].dcol, R[10].dval, R[10].drow, R[10].dcol,
                    D[11].dval, D[11].drow, D[11].dcol, R[11].dval, R[11].drow, R[11].dcol,
                    D[12].dval, D[12].drow, D[12].dcol, R[12].dval, R[12].drow, R[12].dcol,
                    D[13].dval, D[13].drow, D[13].dcol, R[13].dval, R[13].drow, R[13].dcol,
                    D[14].dval, D[14].drow, D[14].dcol, R[14].dval, R[14].drow, R[14].dcol,
                    D[15].dval, D[15].drow, D[15].dcol, R[15].dval, R[15].drow, R[15].dcol,
                    b.dval, x->dval );  
        }
        else if (matrices == 32) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel32<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[ 0].dval, D[ 0].drow, D[ 0].dcol, R[ 0].dval, R[ 0].drow, R[ 0].dcol, 
                    D[ 1].dval, D[ 1].drow, D[ 1].dcol, R[ 1].dval, R[ 1].drow, R[ 1].dcol,
                    D[ 2].dval, D[ 2].drow, D[ 2].dcol, R[ 2].dval, R[ 2].drow, R[ 2].dcol,
                    D[ 3].dval, D[ 3].drow, D[ 3].dcol, R[ 3].dval, R[ 3].drow, R[ 3].dcol,
                    D[ 4].dval, D[ 4].drow, D[ 4].dcol, R[ 4].dval, R[ 4].drow, R[ 4].dcol,
                    D[ 5].dval, D[ 5].drow, D[ 5].dcol, R[ 5].dval, R[ 5].drow, R[ 5].dcol,
                    D[ 6].dval, D[ 6].drow, D[ 6].dcol, R[ 6].dval, R[ 6].drow, R[ 6].dcol,
                    D[ 7].dval, D[ 7].drow, D[ 7].dcol, R[ 7].dval, R[ 7].drow, R[ 7].dcol,
                    D[ 8].dval, D[ 8].drow, D[ 8].dcol, R[ 8].dval, R[ 8].drow, R[ 8].dcol, 
                    D[ 9].dval, D[ 9].drow, D[ 9].dcol, R[ 9].dval, R[ 9].drow, R[ 9].dcol,
                    D[10].dval, D[10].drow, D[10].dcol, R[10].dval, R[10].drow, R[10].dcol,
                    D[11].dval, D[11].drow, D[11].dcol, R[11].dval, R[11].drow, R[11].dcol,
                    D[12].dval, D[12].drow, D[12].dcol, R[12].dval, R[12].drow, R[12].dcol,
                    D[13].dval, D[13].drow, D[13].dcol, R[13].dval, R[13].drow, R[13].dcol,
                    D[14].dval, D[14].drow, D[14].dcol, R[14].dval, R[14].drow, R[14].dcol,
                    D[15].dval, D[15].drow, D[15].dcol, R[15].dval, R[15].drow, R[15].dcol,
                    D[16].dval, D[16].drow, D[16].dcol, R[16].dval, R[16].drow, R[16].dcol,
                    D[17].dval, D[17].drow, D[17].dcol, R[17].dval, R[17].drow, R[17].dcol,
                    D[18].dval, D[18].drow, D[18].dcol, R[18].dval, R[18].drow, R[18].dcol, 
                    D[19].dval, D[19].drow, D[19].dcol, R[19].dval, R[19].drow, R[19].dcol,
                    D[20].dval, D[20].drow, D[20].dcol, R[20].dval, R[20].drow, R[20].dcol,
                    D[21].dval, D[21].drow, D[21].dcol, R[21].dval, R[21].drow, R[21].dcol,
                    D[22].dval, D[22].drow, D[22].dcol, R[22].dval, R[22].drow, R[22].dcol,
                    D[23].dval, D[23].drow, D[23].dcol, R[23].dval, R[23].drow, R[23].dcol,
                    D[24].dval, D[24].drow, D[24].dcol, R[24].dval, R[24].drow, R[24].dcol,
                    D[25].dval, D[25].drow, D[25].dcol, R[25].dval, R[25].drow, R[25].dcol,
                    D[26].dval, D[26].drow, D[26].dcol, R[26].dval, R[26].drow, R[26].dcol,
                    D[27].dval, D[27].drow, D[27].dcol, R[27].dval, R[27].drow, R[27].dcol,
                    D[28].dval, D[28].drow, D[28].dcol, R[28].dval, R[28].drow, R[28].dcol, 
                    D[29].dval, D[29].drow, D[29].dcol, R[29].dval, R[29].drow, R[29].dcol,
                    D[30].dval, D[30].drow, D[30].dcol, R[30].dval, R[30].drow, R[30].dcol,
                    D[31].dval, D[31].drow, D[31].dcol, R[31].dval, R[31].drow, R[31].dcol,
                    b.dval, x->dval );  
        }
        else if (matrices == 64) {
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel64<<< grid, block, 0, queue->cuda_stream() >>>
                ( localiters, size, matrices, overlap,
                    D[ 0].dval, D[ 0].drow, D[ 0].dcol, R[ 0].dval, R[ 0].drow, R[ 0].dcol, 
                    D[ 1].dval, D[ 1].drow, D[ 1].dcol, R[ 1].dval, R[ 1].drow, R[ 1].dcol,
                    D[ 2].dval, D[ 2].drow, D[ 2].dcol, R[ 2].dval, R[ 2].drow, R[ 2].dcol,
                    D[ 3].dval, D[ 3].drow, D[ 3].dcol, R[ 3].dval, R[ 3].drow, R[ 3].dcol,
                    D[ 4].dval, D[ 4].drow, D[ 4].dcol, R[ 4].dval, R[ 4].drow, R[ 4].dcol,
                    D[ 5].dval, D[ 5].drow, D[ 5].dcol, R[ 5].dval, R[ 5].drow, R[ 5].dcol,
                    D[ 6].dval, D[ 6].drow, D[ 6].dcol, R[ 6].dval, R[ 6].drow, R[ 6].dcol,
                    D[ 7].dval, D[ 7].drow, D[ 7].dcol, R[ 7].dval, R[ 7].drow, R[ 7].dcol,
                    D[ 8].dval, D[ 8].drow, D[ 8].dcol, R[ 8].dval, R[ 8].drow, R[ 8].dcol, 
                    D[ 9].dval, D[ 9].drow, D[ 9].dcol, R[ 9].dval, R[ 9].drow, R[ 9].dcol,
                    D[10].dval, D[10].drow, D[10].dcol, R[10].dval, R[10].drow, R[10].dcol,
                    D[11].dval, D[11].drow, D[11].dcol, R[11].dval, R[11].drow, R[11].dcol,
                    D[12].dval, D[12].drow, D[12].dcol, R[12].dval, R[12].drow, R[12].dcol,
                    D[13].dval, D[13].drow, D[13].dcol, R[13].dval, R[13].drow, R[13].dcol,
                    D[14].dval, D[14].drow, D[14].dcol, R[14].dval, R[14].drow, R[14].dcol,
                    D[15].dval, D[15].drow, D[15].dcol, R[15].dval, R[15].drow, R[15].dcol,
                    D[16].dval, D[16].drow, D[16].dcol, R[16].dval, R[16].drow, R[16].dcol,
                    D[17].dval, D[17].drow, D[17].dcol, R[17].dval, R[17].drow, R[17].dcol,
                    D[18].dval, D[18].drow, D[18].dcol, R[18].dval, R[18].drow, R[18].dcol, 
                    D[19].dval, D[19].drow, D[19].dcol, R[19].dval, R[19].drow, R[19].dcol,
                    D[20].dval, D[20].drow, D[20].dcol, R[20].dval, R[20].drow, R[20].dcol,
                    D[21].dval, D[21].drow, D[21].dcol, R[21].dval, R[21].drow, R[21].dcol,
                    D[22].dval, D[22].drow, D[22].dcol, R[22].dval, R[22].drow, R[22].dcol,
                    D[23].dval, D[23].drow, D[23].dcol, R[23].dval, R[23].drow, R[23].dcol,
                    D[24].dval, D[24].drow, D[24].dcol, R[24].dval, R[24].drow, R[24].dcol,
                    D[25].dval, D[25].drow, D[25].dcol, R[25].dval, R[25].drow, R[25].dcol,
                    D[26].dval, D[26].drow, D[26].dcol, R[26].dval, R[26].drow, R[26].dcol,
                    D[27].dval, D[27].drow, D[27].dcol, R[27].dval, R[27].drow, R[27].dcol,
                    D[28].dval, D[28].drow, D[28].dcol, R[28].dval, R[28].drow, R[28].dcol, 
                    D[29].dval, D[29].drow, D[29].dcol, R[29].dval, R[29].drow, R[29].dcol,
                    D[30].dval, D[30].drow, D[30].dcol, R[30].dval, R[30].drow, R[30].dcol,
                    D[31].dval, D[31].drow, D[31].dcol, R[31].dval, R[31].drow, R[31].dcol,
                    D[32].dval, D[32].drow, D[32].dcol, R[32].dval, R[32].drow, R[32].dcol,
                    D[33].dval, D[33].drow, D[33].dcol, R[33].dval, R[33].drow, R[33].dcol,
                    D[34].dval, D[34].drow, D[34].dcol, R[34].dval, R[34].drow, R[34].dcol,
                    D[35].dval, D[35].drow, D[35].dcol, R[35].dval, R[35].drow, R[35].dcol,
                    D[36].dval, D[36].drow, D[36].dcol, R[36].dval, R[36].drow, R[36].dcol,
                    D[37].dval, D[37].drow, D[37].dcol, R[37].dval, R[37].drow, R[37].dcol,
                    D[38].dval, D[38].drow, D[38].dcol, R[38].dval, R[38].drow, R[38].dcol,
                    D[39].dval, D[39].drow, D[39].dcol, R[39].dval, R[39].drow, R[39].dcol,
                    D[40].dval, D[40].drow, D[40].dcol, R[40].dval, R[40].drow, R[40].dcol,
                    D[41].dval, D[41].drow, D[41].dcol, R[41].dval, R[41].drow, R[41].dcol,
                    D[42].dval, D[42].drow, D[42].dcol, R[42].dval, R[42].drow, R[42].dcol,
                    D[43].dval, D[43].drow, D[43].dcol, R[43].dval, R[43].drow, R[43].dcol,
                    D[44].dval, D[44].drow, D[44].dcol, R[44].dval, R[44].drow, R[44].dcol,
                    D[45].dval, D[45].drow, D[45].dcol, R[45].dval, R[45].drow, R[45].dcol,
                    D[46].dval, D[46].drow, D[46].dcol, R[46].dval, R[46].drow, R[46].dcol,
                    D[47].dval, D[47].drow, D[47].dcol, R[47].dval, R[47].drow, R[47].dcol,
                    D[48].dval, D[48].drow, D[48].dcol, R[48].dval, R[48].drow, R[48].dcol,
                    D[49].dval, D[49].drow, D[49].dcol, R[49].dval, R[49].drow, R[49].dcol,
                    D[50].dval, D[50].drow, D[50].dcol, R[50].dval, R[50].drow, R[50].dcol,
                    D[51].dval, D[51].drow, D[51].dcol, R[51].dval, R[51].drow, R[51].dcol,
                    D[52].dval, D[52].drow, D[52].dcol, R[52].dval, R[52].drow, R[52].dcol,
                    D[53].dval, D[53].drow, D[53].dcol, R[53].dval, R[53].drow, R[53].dcol,
                    D[54].dval, D[54].drow, D[54].dcol, R[54].dval, R[54].drow, R[54].dcol,
                    D[55].dval, D[55].drow, D[55].dcol, R[55].dval, R[55].drow, R[55].dcol,
                    D[56].dval, D[56].drow, D[56].dcol, R[56].dval, R[56].drow, R[56].dcol,
                    D[57].dval, D[57].drow, D[57].dcol, R[57].dval, R[57].drow, R[57].dcol,
                    D[58].dval, D[58].drow, D[58].dcol, R[58].dval, R[58].drow, R[58].dcol,
                    D[59].dval, D[59].drow, D[59].dcol, R[59].dval, R[59].drow, R[59].dcol,
                    D[60].dval, D[60].drow, D[60].dcol, R[60].dval, R[60].drow, R[60].dcol,
                    D[61].dval, D[61].drow, D[61].dcol, R[61].dval, R[61].drow, R[61].dcol,
                    D[62].dval, D[62].drow, D[62].dcol, R[62].dval, R[62].drow, R[62].dcol,
                    D[63].dval, D[63].drow, D[63].dcol, R[63].dval, R[63].drow, R[63].dcol,
                    b.dval, x->dval );  
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue->cuda_stream() >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
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
