/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date


   @author Ahmad Abdelfattah
   @author Azzam Haidar

   @precisions normal z -> s d c
 */

#ifndef MAGMABLAS_ZGETF2_NOPIV_DEVICES_Z_H
#define MAGMABLAS_ZGETF2_NOPIV_DEVICES_Z_H

/******************************************************************************/
template<int WIDTH>
static __inline__
void
zgetf2_nopiv_fused_device( int m, int minmn, magmaDoubleComplex rA[WIDTH], double tol,
                     magmaDoubleComplex* swork, int &linfo, int gbstep, int &rowid,
		     const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);

    magmaDoubleComplex reg       = MAGMA_Z_ZERO;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(swork);
    double* dsx = (double*)(sx + item_ct1.get_local_range(1) * WIDTH);
    sx    += ty * WIDTH;
    rowid = tx;

    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        dsx[ rowid ] = sycl::fabs(MAGMA_Z_REAL( rA[i] )) + sycl::fabs(MAGMA_Z_IMAG( rA[i] ));
        item_ct1.barrier();
        rx_abs_max = dsx[i];
        
        // If a non-zero tolerance is specified, replace the small diagonal elements 
        // and increment the info to indicate the number of replacements 
        if(rx_abs_max < tol) {
            if(tx == i)
            {
                int sign = (MAGMA_Z_REAL( rA[i] ) < 0 ? -1 : 1);
                rA[i] = MAGMA_Z_MAKE(sign * tol, 0);
            }
            rx_abs_max = tol;
            linfo++;
            item_ct1.barrier();
        }
        
        // If the tolerance is zero, the above condition is never satisfied, so the info
        // will be the first singularity 
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;

        if( rowid == i ) {
            #pragma unroll
            for(int j = 0; j < WIDTH; j++){
                sx[j] = rA[j];
            }
        }
        item_ct1.barrier();

        reg = (rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < WIDTH; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
    }
}


#endif // MAGMABLAS_ZGETF2_NOPIV_DEVICES_Z_H
