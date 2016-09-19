/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"
#include <cuda_profiler_api.h>

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4

#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION > 6000) // only for cuda>6000
// #if (CUDA_ARCH >= 300)
         


__device__ void                                                                      
magma_zlowerisai_regs1_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 1 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 1 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 1; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 1;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%1 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%1);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs1_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 1 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 1 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 1; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 1-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 1-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%1 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%1);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs2_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 2 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 2 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 2; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 2;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%2 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%2);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs2_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 2 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 2 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 2; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 2-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 2-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%2 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%2);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs3_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 3 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 3 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 3; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 3;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%3 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%3);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs3_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 3 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 3 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 3; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 3-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 3-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%3 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%3);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs4_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 4 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 4 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 4; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 4;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%4 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%4);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs4_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 4 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 4 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 4; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 4-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 4-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%4 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%4);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs5_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 5 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 5 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 5; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 5;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%5 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%5);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs5_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 5 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 5 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 5; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 5-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 5-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%5 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%5);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs6_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 6 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 6 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 6; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 6;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%6 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%6);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs6_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 6 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 6 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 6; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 6-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 6-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%6 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%6);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs7_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 7 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 7 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 7; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 7;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%7 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%7);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs7_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 7 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 7 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 7; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 7-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 7-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%7 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%7);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs8_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 8 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 8 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 8; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 8;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%8 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%8);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs8_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 8 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 8 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 8; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 8-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 8-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%8 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%8);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs9_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 9 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 9 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 9; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 9;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%9 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%9);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs9_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 9 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 9 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 9; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 9-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 9-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%9 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%9);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs10_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 10 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 10 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 10; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 10;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%10 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%10);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs10_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 10 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 10 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 10; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 10-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 10-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%10 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%10);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs11_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 11 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 11 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 11; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 11;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%11 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%11);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs11_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 11 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 11 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 11; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 11-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 11-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%11 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%11);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs12_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 12 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 12 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 12; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 12;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%12 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%12);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs12_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 12 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 12 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 12; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 12-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 12-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%12 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%12);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs13_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 13 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 13 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 13; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 13;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%13 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%13);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs13_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 13 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 13 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 13; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 13-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 13-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%13 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%13);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs14_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 14 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 14 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 14; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 14;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%14 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%14);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs14_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 14 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 14 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 14; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 14-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 14-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%14 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%14);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs15_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 15 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 15 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 15; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 15;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%15 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%15);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs15_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 15 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 15 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 15; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 15-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 15-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%15 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%15);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs16_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 16 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 16 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 16; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 16;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%16 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%16);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs16_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 16 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 16 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 16; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 16-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 16-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%16 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%16);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs17_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 17 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 17 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 17; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 17;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%17 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%17);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs17_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 17 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 17 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 17; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 17-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 17-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%17 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%17);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs18_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 18 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 18 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 18; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 18;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%18 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%18);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs18_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 18 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 18 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 18; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 18-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 18-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%18 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%18);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs19_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 19 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 19 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 19; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 19;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%19 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%19);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs19_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 19 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 19 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 19; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 19-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 19-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%19 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%19);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs20_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 20 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 20 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 20; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 20;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%20 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%20);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs20_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 20 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 20 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 20; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 20-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 20-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%20 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%20);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs21_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 21 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 21 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 21; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 21;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%21 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%21);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs21_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 21 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 21 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 21; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 21-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 21-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%21 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%21);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs22_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 22 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 22 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 22; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 22;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%22 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%22);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs22_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 22 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 22 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 22; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 22-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 22-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%22 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%22);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs23_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 23 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 23 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 23; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 23;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%23 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%23);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs23_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 23 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 23 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 23; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 23-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 23-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%23 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%23);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs24_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 24 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 24 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 24; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 24;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%24 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%24);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs24_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 24 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 24 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 24; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 24-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 24-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%24 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%24);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs25_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 25 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 25 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 25; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 25;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%25 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%25);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs25_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 25 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 25 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 25; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 25-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 25-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%25 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%25);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs26_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 26 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 26 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 26; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 26;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%26 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%26);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs26_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 26 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 26 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 26; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 26-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 26-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%26 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%26);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs27_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 27 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 27 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 27; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 27;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%27 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%27);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs27_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 27 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 27 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 27; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 27-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 27-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%27 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%27);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs28_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 28 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 28 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 28; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 28;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%28 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%28);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs28_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 28 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 28 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 28; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 28-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 28-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%28 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%28);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs29_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 29 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 29 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 29; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 29;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%29 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%29);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs29_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 29 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 29 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 29; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 29-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 29-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%29 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%29);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs30_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 30 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 30 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 30; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 30;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%30 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%30);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs30_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 30 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 30 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 30; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 30-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 30-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%30 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%30);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs31_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 31 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 31 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 31; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 31;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%31 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%31);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs31_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 31 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 31 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 31; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 31-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 31-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%31 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%31);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs32_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 32 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 32 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 32; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 32;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%32 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%32);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs32_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 32 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row+1 ];                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 32 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 32; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t ];                                                               
    int alim = Arow[ t+1 ];                                                          
    int l = mstart;                                                                  
    int idx = 0;                                                                     
    while( k < alim && l < mlim ){ // stop once this column is done                  
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k++;                                                                     
            l++;                                                                     
            idx++;                                                                   
        } else if( acol < mcol ){// need to check next element                       
            k++;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l++; // check next elment in the sparsity pattern                        
            idx++; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 32-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 32-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%32 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%32);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    




                                                                                                
__global__                                                                                              
void magma_zlowerisai_regs_switch(                                                                      
magma_int_t num_rows,                                                                                   
const magma_index_t * __restrict__ Arow,                                                                
const magma_index_t * __restrict__ Acol,                                                                
const magmaDoubleComplex * __restrict__ Aval,                                                           
magma_index_t *Mrow,                                                                                    
magma_index_t *Mcol,                                                                                    
magmaDoubleComplex *Mval )                                                                              
{                                                                                                       
                                                                                                        
                                                                                                        
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                                      
    if( row < num_rows ){                                                                               
    int N = Mrow[ row+1 ] - Mrow[ row ];                                                                
    switch( N ) {                                                                                       
        case  1:                                                                                       
            magma_zlowerisai_regs1_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  2:                                                                                       
            magma_zlowerisai_regs2_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  3:                                                                                       
            magma_zlowerisai_regs3_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  4:                                                                                       
            magma_zlowerisai_regs4_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  5:                                                                                       
            magma_zlowerisai_regs5_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  6:                                                                                       
            magma_zlowerisai_regs6_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  7:                                                                                       
            magma_zlowerisai_regs7_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  8:                                                                                       
            magma_zlowerisai_regs8_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  9:                                                                                       
            magma_zlowerisai_regs9_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  10:                                                                                       
            magma_zlowerisai_regs10_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  11:                                                                                       
            magma_zlowerisai_regs11_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  12:                                                                                       
            magma_zlowerisai_regs12_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  13:                                                                                       
            magma_zlowerisai_regs13_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  14:                                                                                       
            magma_zlowerisai_regs14_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  15:                                                                                       
            magma_zlowerisai_regs15_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  16:                                                                                       
            magma_zlowerisai_regs16_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  17:                                                                                       
            magma_zlowerisai_regs17_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  18:                                                                                       
            magma_zlowerisai_regs18_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  19:                                                                                       
            magma_zlowerisai_regs19_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  20:                                                                                       
            magma_zlowerisai_regs20_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  21:                                                                                       
            magma_zlowerisai_regs21_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  22:                                                                                       
            magma_zlowerisai_regs22_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  23:                                                                                       
            magma_zlowerisai_regs23_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  24:                                                                                       
            magma_zlowerisai_regs24_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  25:                                                                                       
            magma_zlowerisai_regs25_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  26:                                                                                       
            magma_zlowerisai_regs26_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  27:                                                                                       
            magma_zlowerisai_regs27_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  28:                                                                                       
            magma_zlowerisai_regs28_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  29:                                                                                       
            magma_zlowerisai_regs29_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  30:                                                                                       
            magma_zlowerisai_regs30_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  31:                                                                                       
            magma_zlowerisai_regs31_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  32:                                                                                       
            magma_zlowerisai_regs32_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        default:                                                                                        
            printf("% error: size out of range: %d\n", N); break;                                    
    }                                                                                                   
    }                                                                                                   
}                                                                                                       
                                                                                                        




                                                                                                
__global__                                                                                              
void magma_zupperisai_regs_switch(                                                                      
magma_int_t num_rows,                                                                                   
const magma_index_t * __restrict__ Arow,                                                                
const magma_index_t * __restrict__ Acol,                                                                
const magmaDoubleComplex * __restrict__ Aval,                                                           
magma_index_t *Mrow,                                                                                    
magma_index_t *Mcol,                                                                                    
magmaDoubleComplex *Mval )                                                                              
{                                                                                                       
                                                                                                        
                                                                                                        
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                                      
    if( row < num_rows ){                                                                               
    int N = Mrow[ row+1 ] - Mrow[ row ];                                                                
    switch( N ) {                                                                                       
        case  1:                                                                                       
            magma_zupperisai_regs1_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  2:                                                                                       
            magma_zupperisai_regs2_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  3:                                                                                       
            magma_zupperisai_regs3_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  4:                                                                                       
            magma_zupperisai_regs4_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  5:                                                                                       
            magma_zupperisai_regs5_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  6:                                                                                       
            magma_zupperisai_regs6_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  7:                                                                                       
            magma_zupperisai_regs7_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  8:                                                                                       
            magma_zupperisai_regs8_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  9:                                                                                       
            magma_zupperisai_regs9_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  10:                                                                                       
            magma_zupperisai_regs10_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  11:                                                                                       
            magma_zupperisai_regs11_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  12:                                                                                       
            magma_zupperisai_regs12_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  13:                                                                                       
            magma_zupperisai_regs13_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  14:                                                                                       
            magma_zupperisai_regs14_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  15:                                                                                       
            magma_zupperisai_regs15_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  16:                                                                                       
            magma_zupperisai_regs16_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  17:                                                                                       
            magma_zupperisai_regs17_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  18:                                                                                       
            magma_zupperisai_regs18_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  19:                                                                                       
            magma_zupperisai_regs19_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  20:                                                                                       
            magma_zupperisai_regs20_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  21:                                                                                       
            magma_zupperisai_regs21_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  22:                                                                                       
            magma_zupperisai_regs22_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  23:                                                                                       
            magma_zupperisai_regs23_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  24:                                                                                       
            magma_zupperisai_regs24_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  25:                                                                                       
            magma_zupperisai_regs25_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  26:                                                                                       
            magma_zupperisai_regs26_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  27:                                                                                       
            magma_zupperisai_regs27_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  28:                                                                                       
            magma_zupperisai_regs28_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  29:                                                                                       
            magma_zupperisai_regs29_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  30:                                                                                       
            magma_zupperisai_regs30_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  31:                                                                                       
            magma_zupperisai_regs31_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  32:                                                                                       
            magma_zupperisai_regs32_kernel (                                                            
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        default:                                                                                        
            printf("% error: size out of range: %d\n", N); break;                                    
    }                                                                                                   
    }                                                                                                   
}                                                                                                       
                                                                                                                                                                                                                
         
__device__ void                                                                      
magma_zlowerisai_regs1_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 1 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 1 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 1; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 1-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 1;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%1 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%1);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs1_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 1 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 1 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 1; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 1-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 1-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 1-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%1 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%1);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs2_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 2 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 2 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 2; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 2-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 2;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%2 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%2);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs2_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 2 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 2 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 2; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 2-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 2-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 2-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%2 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%2);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs3_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 3 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 3 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 3; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 3-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 3;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%3 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%3);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs3_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 3 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 3 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 3; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 3-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 3-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 3-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%3 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%3);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs4_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 4 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 4 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 4; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 4-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 4;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%4 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%4);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs4_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 4 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 4 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 4; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 4-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 4-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 4-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%4 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%4);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs5_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 5 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 5 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 5; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 5-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 5;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%5 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%5);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs5_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 5 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 5 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 5; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 5-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 5-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 5-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%5 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%5);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs6_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 6 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 6 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 6; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 6-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 6;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%6 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%6);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs6_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 6 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 6 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 6; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 6-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 6-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 6-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%6 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%6);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs7_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 7 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 7 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 7; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 7-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 7;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%7 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%7);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs7_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 7 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 7 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 7; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 7-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 7-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 7-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%7 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%7);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs8_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 8 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 8 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 8; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 8-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 8;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%8 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%8);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs8_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 8 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 8 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 8; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 8-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 8-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 8-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%8 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%8);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs9_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 9 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 9 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 9; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 9-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 9;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%9 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%9);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs9_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 9 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 9 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 9; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 9-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 9-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 9-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%9 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%9);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs10_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 10 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 10 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 10; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 10-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 10;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%10 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%10);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs10_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 10 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 10 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 10; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 10-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 10-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 10-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%10 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%10);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs11_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 11 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 11 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 11; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 11-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 11;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%11 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%11);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs11_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 11 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 11 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 11; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 11-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 11-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 11-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%11 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%11);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs12_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 12 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 12 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 12; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 12-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 12;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%12 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%12);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs12_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 12 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 12 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 12; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 12-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 12-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 12-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%12 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%12);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs13_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 13 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 13 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 13; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 13-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 13;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%13 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%13);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs13_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 13 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 13 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 13; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 13-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 13-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 13-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%13 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%13);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs14_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 14 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 14 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 14; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 14-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 14;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%14 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%14);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs14_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 14 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 14 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 14; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 14-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 14-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 14-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%14 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%14);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs15_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 15 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 15 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 15; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 15-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 15;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%15 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%15);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs15_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 15 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 15 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 15; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 15-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 15-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 15-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%15 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%15);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs16_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 16 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 16 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 16; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 16-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 16;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%16 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%16);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs16_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 16 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 16 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 16; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 16-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 16-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 16-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%16 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%16);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs17_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 17 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 17 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 17; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 17-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 17;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%17 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%17);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs17_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 17 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 17 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 17; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 17-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 17-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 17-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%17 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%17);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs18_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 18 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 18 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 18; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 18-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 18;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%18 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%18);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs18_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 18 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 18 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 18; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 18-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 18-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 18-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%18 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%18);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs19_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 19 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 19 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 19; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 19-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 19;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%19 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%19);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs19_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 19 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 19 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 19; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 19-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 19-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 19-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%19 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%19);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs20_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 20 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 20 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 20; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 20-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 20;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%20 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%20);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs20_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 20 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 20 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 20; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 20-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 20-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 20-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%20 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%20);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs21_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 21 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 21 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 21; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 21-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 21;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%21 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%21);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs21_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 21 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 21 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 21; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 21-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 21-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 21-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%21 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%21);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs22_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 22 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 22 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 22; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 22-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 22;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%22 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%22);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs22_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 22 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 22 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 22; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 22-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 22-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 22-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%22 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%22);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs23_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 23 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 23 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 23; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 23-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 23;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%23 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%23);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs23_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 23 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 23 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 23; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 23-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 23-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 23-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%23 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%23);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs24_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 24 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 24 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 24; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 24-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 24;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%24 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%24);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs24_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 24 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 24 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 24; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 24-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 24-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 24-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%24 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%24);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs25_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 25 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 25 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 25; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 25-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 25;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%25 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%25);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs25_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 25 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 25 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 25; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 25-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 25-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 25-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%25 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%25);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs26_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 26 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 26 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 26; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 26-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 26;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%26 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%26);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs26_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 26 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 26 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 26; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 26-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 26-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 26-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%26 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%26);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs27_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 27 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 27 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 27; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 27-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 27;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%27 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%27);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs27_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 27 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 27 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 27; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 27-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 27-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 27-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%27 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%27);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs28_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 28 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 28 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 28; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 28-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 28;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%28 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%28);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs28_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 28 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 28 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 28; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 28-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 28-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 28-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%28 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%28);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs29_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 29 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 29 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 29; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 29-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 29;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%29 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%29);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs29_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 29 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 29 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 29; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 29-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 29-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 29-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%29 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%29);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs30_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 30 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 30 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 30; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 30-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 30;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%30 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%30);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs30_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 30 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 30 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 30; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 30-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 30-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 30-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%30 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%30);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs31_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 31 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 31 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 31; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 31-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 31;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%31 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%31);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs31_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 31 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 31 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 31; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 31-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 31-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 31-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%31 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%31);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zlowerisai_regs32_inv_kernel(                                                  
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 32 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 32 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 32; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 32-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                                  
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 0; k < 32;  k++)                                                    
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%32 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex top = __shfl(rB, k%32);                                 
         if ( tid > k)                                                               
             rB -= (top*rA);                                                         
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// lower kernel for size                                                                                     
                                                                                    
__device__ void                                                                      
magma_zupperisai_regs32_inv_kernel(                                                      
magma_int_t num_rows,                                                                
const magma_index_t * __restrict__ Arow,                                             
const magma_index_t * __restrict__ Acol,                                             
const magmaDoubleComplex * __restrict__ Aval,                                        
magma_index_t *Mrow,                                                                 
magma_index_t *Mcol,                                                                 
magmaDoubleComplex *Mval )                                                           
{                                                                                    
#ifdef REAL                                                                          
    int tid = threadIdx.x;                                                           
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                   
                                                                                     
    if( tid >= 32 )                                                                  
        return;                                                                      
                                                                                     
    if( row >= num_rows )                                                            
        return;                                                                      
                                                                                     
    // only if within the size                                                       
    int mstart = Mrow[ row ];                                                        
    int mlim = Mrow[ row ]-1;                                                        
                                                                                     
    magmaDoubleComplex rB;      // registers for trsv                                
    magmaDoubleComplex dA[ 32 ];  // registers for trisystem                         
    magmaDoubleComplex rA;                                                           
                                                                                     
    // set dA to 0                                                                   
    #pragma unroll                                                                   
     for( int j = 0; j < 32; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;                                                     
     }                                                                               
                                                                                     
    // generate the triangular systems                                               
    int t = Mcol[ mstart + tid ];                                                    
    int k = Arow[ t+1 ] - 1;                                                         
    int alim = Arow[ t ]-1;                                                          
    int l = Mrow[ row+1 ]-1;                                                         
    int idx = 32-1;                                                                  
    while( k > alim && l > mlim  ){ // stop once this column is done                 
        int mcol =  Mcol[ l ];                                                       
        int acol = Acol[k];                                                          
        if( mcol == acol ){ //match                                                  
            dA[ idx ] = Aval[ k ];                                                   
            k--;                                                                     
            l--;                                                                     
            idx--;                                                                   
        } else if( acol > mcol ){// need to check next element                       
            k--;                                                                     
        } else { // element does not exist, i.e. l < LC.col[k]                       
            l--; // check next elment in the sparsity pattern                        
            idx--; // leave this element equal zero                                  
        }                                                                            
    }                                                                                
                                                                                     
    // second: solve the triangular systems - in registers                           
    // we know how RHS looks like                                                    
    rB = ( tid == 32-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                               
                                                                                     
                                                                                     
        // Triangular solve in regs.                                                 
    #pragma unroll                                                                   
    for (int k = 32-1; k >-1; k--)                                                   
    {                                                                                
        rA = dA[ k ];                                                                
         if (k%32 == tid)                                                           
             rB /= rA;                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%32);                              
         if ( tid < k)                                                               
             rB -= (bottom*rA);                                                      
    }                                                                                
                                                                                     
    // Drop B to dev memory - in ISAI preconditioner M                               
    Mval[ mstart + tid ] = rB;                                                       
                                                                                     
#endif                                                                               
                                                                                     
}// upper kernel for size                                                                                     
                                                                                    




                                                                                                
__global__                                                                                              
void magma_zlowerisai_regs_inv_switch(                                                                  
magma_int_t num_rows,                                                                                   
const magma_index_t * __restrict__ Arow,                                                                
const magma_index_t * __restrict__ Acol,                                                                
const magmaDoubleComplex * __restrict__ Aval,                                                           
magma_index_t *Mrow,                                                                                    
magma_index_t *Mcol,                                                                                    
magmaDoubleComplex *Mval )                                                                              
{                                                                                                       
                                                                                                        
                                                                                                        
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                                      
    if( row < num_rows ){                                                                               
    int N = Mrow[ row+1 ] - Mrow[ row ];                                                                
    switch( N ) {                                                                                       
        case  1:                                                                                       
            magma_zlowerisai_regs1_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  2:                                                                                       
            magma_zlowerisai_regs2_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  3:                                                                                       
            magma_zlowerisai_regs3_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  4:                                                                                       
            magma_zlowerisai_regs4_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  5:                                                                                       
            magma_zlowerisai_regs5_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  6:                                                                                       
            magma_zlowerisai_regs6_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  7:                                                                                       
            magma_zlowerisai_regs7_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  8:                                                                                       
            magma_zlowerisai_regs8_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  9:                                                                                       
            magma_zlowerisai_regs9_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  10:                                                                                       
            magma_zlowerisai_regs10_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  11:                                                                                       
            magma_zlowerisai_regs11_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  12:                                                                                       
            magma_zlowerisai_regs12_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  13:                                                                                       
            magma_zlowerisai_regs13_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  14:                                                                                       
            magma_zlowerisai_regs14_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  15:                                                                                       
            magma_zlowerisai_regs15_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  16:                                                                                       
            magma_zlowerisai_regs16_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  17:                                                                                       
            magma_zlowerisai_regs17_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  18:                                                                                       
            magma_zlowerisai_regs18_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  19:                                                                                       
            magma_zlowerisai_regs19_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  20:                                                                                       
            magma_zlowerisai_regs20_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  21:                                                                                       
            magma_zlowerisai_regs21_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  22:                                                                                       
            magma_zlowerisai_regs22_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  23:                                                                                       
            magma_zlowerisai_regs23_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  24:                                                                                       
            magma_zlowerisai_regs24_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  25:                                                                                       
            magma_zlowerisai_regs25_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  26:                                                                                       
            magma_zlowerisai_regs26_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  27:                                                                                       
            magma_zlowerisai_regs27_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  28:                                                                                       
            magma_zlowerisai_regs28_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  29:                                                                                       
            magma_zlowerisai_regs29_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  30:                                                                                       
            magma_zlowerisai_regs30_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  31:                                                                                       
            magma_zlowerisai_regs31_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  32:                                                                                       
            magma_zlowerisai_regs32_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        default:                                                                                        
            printf("% error: size out of range: %d\n", N); break;                                    
    }                                                                                                   
    }                                                                                                   
}                                                                                                       
                                                                                                        




                                                                                                
__global__                                                                                              
void magma_zupperisai_regs_inv_switch(                                                                  
magma_int_t num_rows,                                                                                   
const magma_index_t * __restrict__ Arow,                                                                
const magma_index_t * __restrict__ Acol,                                                                
const magmaDoubleComplex * __restrict__ Aval,                                                           
magma_index_t *Mrow,                                                                                    
magma_index_t *Mcol,                                                                                    
magmaDoubleComplex *Mval )                                                                              
{                                                                                                       
                                                                                                        
                                                                                                        
    int row = blockIdx.y * gridDim.x + blockIdx.x;                                                      
    if( row < num_rows ){                                                                               
    int N = Mrow[ row+1 ] - Mrow[ row ];                                                                
    switch( N ) {                                                                                       
        case  1:                                                                                       
            magma_zupperisai_regs1_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  2:                                                                                       
            magma_zupperisai_regs2_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  3:                                                                                       
            magma_zupperisai_regs3_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  4:                                                                                       
            magma_zupperisai_regs4_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  5:                                                                                       
            magma_zupperisai_regs5_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  6:                                                                                       
            magma_zupperisai_regs6_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  7:                                                                                       
            magma_zupperisai_regs7_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  8:                                                                                       
            magma_zupperisai_regs8_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  9:                                                                                       
            magma_zupperisai_regs9_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  10:                                                                                       
            magma_zupperisai_regs10_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  11:                                                                                       
            magma_zupperisai_regs11_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  12:                                                                                       
            magma_zupperisai_regs12_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  13:                                                                                       
            magma_zupperisai_regs13_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  14:                                                                                       
            magma_zupperisai_regs14_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  15:                                                                                       
            magma_zupperisai_regs15_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  16:                                                                                       
            magma_zupperisai_regs16_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  17:                                                                                       
            magma_zupperisai_regs17_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  18:                                                                                       
            magma_zupperisai_regs18_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  19:                                                                                       
            magma_zupperisai_regs19_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  20:                                                                                       
            magma_zupperisai_regs20_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  21:                                                                                       
            magma_zupperisai_regs21_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  22:                                                                                       
            magma_zupperisai_regs22_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  23:                                                                                       
            magma_zupperisai_regs23_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  24:                                                                                       
            magma_zupperisai_regs24_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  25:                                                                                       
            magma_zupperisai_regs25_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  26:                                                                                       
            magma_zupperisai_regs26_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  27:                                                                                       
            magma_zupperisai_regs27_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  28:                                                                                       
            magma_zupperisai_regs28_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  29:                                                                                       
            magma_zupperisai_regs29_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  30:                                                                                       
            magma_zupperisai_regs30_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  31:                                                                                       
            magma_zupperisai_regs31_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        case  32:                                                                                       
            magma_zupperisai_regs32_inv_kernel (                                                        
                num_rows, Arow, Acol, Aval, Mrow, Mcol, Mval ); break;                                  
        default:                                                                                        
            printf("% error: size out of range: %d\n", N); break;                                    
    }                                                                                                   
    }                                                                                                   
}                                                                                                       
                                                                                                        
// #endif
#endif

/**
    Purpose
    -------
    This routine is designet to combine all kernels into one.

    Arguments
    ---------
    

    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular
                
    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix
                
    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not
                
    @param[in]
    L           magma_z_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.
                
    @param[in,out]
    M           magma_z_matrix*
                SPAI preconditioner CSR col-major
                
    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.
                
    @param[out]
    locations   magma_int_t*
                Array indicating the locations.
                
    @param[out]
    trisystems  magmaDoubleComplex*
                trisystems
                
    @param[out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zisai_generator_regs(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue )
{
    magma_int_t info = 0;  
    
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    
    
    // routine 1
    int r1bs1 = 32;
    int r1bs2 = 1;
    int r1dg1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int r1dg2 = min(magma_ceildiv( M->num_rows, r1dg1 ), 65535);
    int r1dg3 = magma_ceildiv( M->num_rows, r1dg1*r1dg2 );
    //printf(" grid: %d x %d x %d\n", r1dg1, r1dg2, r1dg3 );
    dim3 r1block( r1bs1, r1bs2, 1 );
    dim3 r1grid( r1dg1, r1dg2, r1dg3 );
    
    int r2bs1 = 256;
    int r2bs2 = 1;
    int r2dg1 = magma_ceildiv( L.num_rows, r2bs1 );
    int r2dg2 = 1;
    int r2dg3 = 1;
    dim3 r2block( r2bs1, r2bs2, 1 );
    dim3 r2grid( r2dg1, r2dg2, r2dg3 );
    
#if (CUDA_VERSION > 6000)
// #if (CUDA_ARCH >= 300)

    if( uplotype == MagmaLower ){//printf("in here lower new kernel\n");
        //cudaProfilerStart();
        magma_zlowerisai_regs_inv_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(                                                                           
            L.num_rows,                                                                                   
            L.row,
            L.col,
            L.val,
            M->row,
            M->col,
            M->val );
         //cudaProfilerStop(); 
         //exit(-1);
    } else {// printf("in here upper new kernel\n");
        magma_zupperisai_regs_inv_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(                                                                           
            L.num_rows,                                                                                   
            L.row,
            L.col,
            L.val,
            M->row,
            M->col,
            M->val );
    }
// #endif
#else
   printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
   info = MAGMA_ERR_NOT_SUPPORTED; 
#endif
    
    return info;
}

