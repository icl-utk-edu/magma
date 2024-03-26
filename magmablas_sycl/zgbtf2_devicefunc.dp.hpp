#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date


   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
 */

#ifndef MAGMABLAS_ZGBTF2_DEVICES_Z_H
#define MAGMABLAS_ZGBTF2_DEVICES_Z_H

////////////////////////////////////////////////////////////////////////////////
// reads an entire band matrix from global mem. to shared mem.
__inline__ void
read_sAB(
    int mband, int n, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = ty_; j < n; j += groups) {
            int col_start = kl + max(ku-j,0);
            int col_end   = kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j) = dAB(i,j);
            }
        }
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
// writes an entire band matrix from shared mem. to global mem.
__inline__ void
write_sAB(
    int mband, int n, int kl, int ku,
    magmaDoubleComplex *sAB, int sldab,
    magmaDoubleComplex *dAB, int lddab,
    int ntx, int tx)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = max(ntx, groups * mband);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = ty_; j < n; j += groups) {
            for(int i = tx_; i < mband; i+=tpg) {
                dAB(i,j) = sAB(i,j);
            }
        }
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
// read from column jstart to column jend (inclusive) from dAB to sAB
// jstart and jend are global column indices with respect to dAB
__inline__ void
read_sAB_updated_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            int col_start = 0;       //kl + max(ku-j,0);
            int col_end   = mband-1; //kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j-jstart) = dAB(i,j);
            }
        }
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
// read from column jstart to column jend (inclusive) from dAB to sAB
// jstart and jend are global column indices with respect to dAB
__inline__ void
read_sAB_new_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            int col_start = kl + max(ku-j,0);
            int col_end   = kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j-jstart) = dAB(i,j);
            }
        }
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
// writes selected columns (jstart to jend, inclusive) of a band matrix
__inline__ void
write_sAB_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *sAB, int sldab,
    magmaDoubleComplex *dAB, int lddab,
    int ntx, int tx)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            for(int i = tx_; i < mband; i+=tpg) {
                dAB(i,j) = sAB(i,j-jstart);
            }
        }
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
// reads the entire matrix of right hand sides (for fused gbsv)
__inline__ void
read_sB(
    int n, int nrhs,
    magmaDoubleComplex *dB, int lddb,
    magmaDoubleComplex *sB, int sldb,
    int ntx, int tx )
{
#define sB(i,j)        sB[(j)*sldb + (i)]
#define dB(i,j)        dB[(j)*lddb + (i)]

    const int tpg    = min(ntx, n);
    const int groups = max(1, ntx / n);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % n;
    const int ty_    = tx / n;

    if(tx < active) {
        for(int j = ty_; j < nrhs; j += groups) {
            for(int i = tx_; i < n; i+=tpg) {
                sB(i,j) = dB(i,j);
            }
        }
    }

#undef sB
#undef dB
}

////////////////////////////////////////////////////////////////////////////////
// writes the entire matrix of solutions (x) for fused gbsv
__inline__ void
write_sB(
    int n, int nrhs,
    magmaDoubleComplex *sB, int sldb,
    magmaDoubleComplex *dB, int lddb,
    int ntx, int tx )
{
#define sB(i,j)        sB[(j)*sldb + (i)]
#define dB(i,j)        dB[(j)*lddb + (i)]

    const int tpg    = min(ntx, n);
    const int groups = max(1, ntx / n);
    const int active = min(ntx, groups * tpg);
    const int tx_    = tx % n;
    const int ty_    = tx / n;

    if(tx < active) {
        for(int j = ty_; j < nrhs; j += groups) {
            for(int i = tx_; i < n; i+=tpg) {
                dB(i,j) = sB(i,j);
            }
        }
    }

#undef sB
#undef dB
}



#endif  //#define MAGMABLAS_ZGBTF2_DEVICES_Z_H
