/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca
  
       @precisions normal z -> s d c
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "magma_bulge.h"
#include "magma_zbulgeinc.h"

// =============================================================================
#ifdef HAVE_clBLAS
#define dE(i,j)  (dE, (dE_offset + (i) + ldde*(j)))
#else
#define dE(i,j)  (dE + (i) + ldde*(j))
#endif
#define V(j)     (V+(j))
#define T(j)     (T+(j))

/***************************************************************************//**
    Parallel apply Q2 from bulgechasing symetric matrices - static scheduling
    Lower case is treated
    
    side == magmaLeft:
        meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so
        traverse Vs in reverse order (forward) from q_n to q_1 Also
        E is splitten by block of col over cores because each apply
        consist in a block of row (horizontal block)
    
    
    side == magmaRight:
        meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so
        traverse Vs in normal order (forward) from q_1 to q_n Also
        E is splitten by block of row over core because each apply
        consist in a block of col (vertical block)
    
*******************************************************************************/
extern "C" magma_int_t
magma_zbulge_applyQ_v2(
    magma_side_t side,
    magma_int_t NE, magma_int_t N,
    magma_int_t NB, magma_int_t Vblksiz,
    magmaDoubleComplex_ptr dE, magma_int_t ldde,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t *info)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t Vm, Vn, mt, nt;
    magma_int_t myrow, mycol, blkj, blki;
    magma_int_t blkid,vpos,tpos;
    magma_int_t firstrow, nbcolinvolvd;
    magma_int_t versionL  = 113;
    magma_int_t versionR  = 92;
    magma_int_t Vchunksiz = 10;
    *info=0;

    /* Quick return */
    if ( NE == 0 ) {
        return *info;
    }
    if ( N == 0 ) {
        return *info;
    }
    if ( NB == 0 ) {
        return *info;
    }
    /* ==========================================
     * some infos for developer
     * Initialisation and checking nb of cores
     * ==========================================*/
    /* we have 2 algo for left (113 114) and 2 algo for right (91 92)
     * which correspond to versionL versionR.
     * They are very similar (detail explained in tech report and matlab code)
     * however version 114 and 92 improve locality.
     * while version 113 is used in case WNATZ=1 (construct Q2) which allow
     * the construction to be done in an optimized way taking into
     * consideration that the matrix is Identity so making less flops.
     *
    */

    // Initialize streaming and events
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    magma_event_t myevent[2];
    cudaEventCreateWithFlags(&myevent[0],cudaEventDisableTiming);
    cudaEventCreateWithFlags(&myevent[1],cudaEventDisableTiming);
    //magma_event_create(&myevent[0]);
    //magma_event_create(&myevent[1]);


    // Azzam 21/11/2012
    // NOTE THAT dwork was of size 2*NE*Vblksiz+...
    // but I am thinking why not modifing it to NE*Vblksiz+...
    // BUT NO because the 2* is used because of making 2 streams working and so
    // they might be using dwork in parallel
    magmaDoubleComplex *dwork, *dwork0, *dwork1, *dwvt0, *dwvt1;
    magmaDoubleComplex *dT0, *dV0, *dT1, *dV1;
    magma_int_t lddv = ldv;
    magma_int_t lddt = ldt;
    magma_int_t lddw = 0;
    magma_int_t lddwork  = magma_roundup( NE, 32 );
    magma_int_t dwVTsiz  = lddv*Vblksiz; // lddv*lddv + lddv*lddwork; (v2) // lddv*Vblksiz; (v1,v3)
    magma_int_t dworksiz = lddwork*Vblksiz;  // lddv*Vblksiz; (v2)   // NE*Vblksiz=lddwork*Vblksiz; (v1,v3)

    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, 2*dworksiz + 2*dwVTsiz +  2*Vchunksiz* (Vblksiz* (lddv+lddt)) )) {
        printf ("!!!!  magma_zbulge_applyQ magma_alloc failed for: dwork\n" );
        return MAGMA_ERR_ALLOCATION;
    }
    dwork0 = dwork;               // size = dworksiz;
    dwork1 = dwork0 + dworksiz;   // size = dworksiz;
    dwvt0  = dwork + 2*dworksiz;  // size = dwVTsiz;
    dwvt1  = dwvt0 + dwVTsiz;     // size = dwVTsiz;
    dV0    = dwork + 2*dworksiz + 2*dwVTsiz;
    dT0    = dV0 + Vchunksiz*Vblksiz*lddv;
    dV1    = dT0 + Vchunksiz*Vblksiz*lddt;
    dT1    = dV1 + Vchunksiz*Vblksiz*lddv;


    // make overlapped copy
    magma_int_t ncpy = 0;
    magma_int_t copyed=0, copyst=0;
    magma_int_t blkcnt,nothing, mysiz, flip, vld,tld, locpos;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &nothing);

    flip = 0;

    // performance loss if the reflector are applied to a big number of eigenvectors (~10000)
    // => apply the reflectors to blocks of eigenvectors.
    magma_int_t sz_bl = NE;
    //magma_int_t nr_bl = magma_ceildiv(NE,10000);        //nr of blocks
    //magma_int_t sz_bl = magma_ceildiv(NE,nr_bl*64)*64; //maximum size of blocks (to have blocks of around the same size and multiple of 64)
    magma_int_t ib;                                      //size of current block


    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

    #ifdef ENABLE_DEBUG
    printf("  APPLY Q_v22 GPU with  N %5lld, NE %5lld,  NB %5lld, Vblksiz %5lld, versionL %5lld versionR %5lld  SIDE %5d\n",
           (long long) N, (long long) NE, (long long) NB, (long long) Vblksiz,
           (long long) versionL, (long long) versionR, side );
    #endif

    /*
     * MagmaLeft
     */
    if (side == MagmaLeft) {
        /*
         * Version 113:
         * loop over the block_col (nt) and for each find the
         * number of tiles (mt) in this block_col. then loop over mt, find
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E.
         */
        if ( versionL == 113 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=nt-1; blkj >= 0; blkj--) {
                /* the index of the first row on the top of block (blkj) */
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if ( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=mt; blki > 0; blki--) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow     = firstrow + (mt-blki)*NB;
                    mycol     = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }
                    /*calculate the pointer to the Vs and the Ts.
                     * Note that Vs and Ts have special storage done
                     * by the bulgechasing function*/
                    //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d vpos %d\n",blkj,blki,Vm, Vn,mycol,vpos);
                    magma_bulge_findpos113(N, NB, Vblksiz, mycol, myrow, &blkid);
               
                    // COPY Vchunksiz Vs and Vchunksiz Ts to GPU and store it in dV0/dV1 and dT0/dT1
                    if (ncpy == 0) {
                        // flip = 1 for this.
                        copyst = 0;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        if (mysiz > 0) {
                            ncpy = 1;
                            flip = 1;
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1, vld, queues[1]);
                            magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1, tld, queues[1]);
                            //printf("doing the first copy   of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);
                        }
                    }
                   
                    if (blkid == copyst) {
                        flip   = ncpy % 2;
                        copyst = copyed;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        //printf(" get to copy blkid %d blkid+(2*Vchunksiz) %d copyst %d copyed %d\n",blkid,blkid+(Vchunksiz),copyst,copyed);
                        if (mysiz > 0) {
                            ncpy = ncpy + 1;
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            if (flip == 0) { // now I am working on dV0 so copy the next and put it on dV1
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);
                                magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1, vld, queues[1]);
                                magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1, tld, queues[1]);
                            } else { // now I am working on dV1 so copy the next and put it on dV0
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV0 dT0\n",mysiz,copyst,copyed,vpos,tpos);
                                magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV0, vld, queues[0]);
                                magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT0, tld, queues[0]);
                            }
                        }
                    }

                    if ((Vm > 0) && (Vn > 0)) {
                        locpos = blkid%Vchunksiz;
                        magma_int_t lcvpos   = locpos*Vblksiz*lddv;
                        magma_int_t lctpos   = locpos*Vblksiz*lddt;
                        //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d locvpos %5d loctpos %5d  blkid %2d  using data in dV%1d dT%1d\n",blkj,blki,Vm, Vn,mycol,lcvpos,lctpos, blkid,flip,flip);
                        if (flip == 0) {
                            magma_queue_wait_event( queues[0], myevent[1] );
                            for (magma_int_t i=0; i < NE; i += sz_bl) {
                                ib = min(sz_bl, NE-i);
                                lddw = min(lddwork,sz_bl);
                                //magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0+lcvpos, lddv, dT0+lctpos, lddt, dE(myrow,i), ldde, dwork0, lddw, queues[0] );
                                magma_zlarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0+lcvpos, lddv, dT0+lctpos, lddt, dE(myrow,i), ldde, dwork0, lddw, dwvt0, lddv, queues[0] );
                            }
                            magma_event_record( myevent[0], queues[0] );
                        } else {
                            magma_queue_wait_event( queues[1], myevent[0] );
                            for (magma_int_t i=0; i < NE; i += sz_bl) {
                                ib = min(sz_bl, NE-i);
                                lddw = min(lddwork,sz_bl);
                                //magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1+lcvpos, lddv, dT1+lctpos, lddt, dE(myrow,i), ldde, dwork1, lddw, queues[1] );
                                magma_zlarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1+lcvpos, lddv, dT1+lctpos, lddt, dE(myrow,i), ldde, dwork1, lddw, dwvt1, lddv, queues[1] );
                            }
                            magma_event_record( myevent[1], queues[1] );
                        }
                    }  // end for (Vm &Vn) > 0
                } // end for blki
            } // end for blkj
        } // end if version=113
        /*
         * Version 114:
         * loop over the block_row (mt) and for each find diagonally the
         * number of tiles (nt) in this block_row. then loop over nt, find
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E.
         */
        else {
            mt    = magma_ceildiv((N-1),NB);
            for (blki = mt; blki > 0; blki--) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = nt-1; blkj >= 0; blkj--) {
                    /* the index of the first row of the first col meaning
                     * the block on the top left (blki) */
                    firstrow = (mt-blki)*NB+1;
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }

                    if ((Vm > 0) && (Vn > 0)) {
                        /* calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, queues[0]);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, queues[0]);
                        //printf("voici blki %d  rownbm %d mycol %d  coled %d  blkid %d vpos %d  tpos %d\n", blki, rownbm, mycol, coled, blkid, vpos, tpos);
                        for (magma_int_t i=0; i < NE; i += sz_bl) {
                            ib = min(sz_bl, NE-i);
                            magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0, lddv, dT0, lddt, dE(myrow,i), ldde, dwork, NE, queues[0] );
                        }
                    } // end for (Vm &Vn) > 0
                } // end for blkj
            } // end for blki
        } // end version 114
    } // end LEFT
    /*
     * MagmaRight
     */
    else {
        /*
         * Version 91:
         */
        if ( versionR == 91 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=0; blkj < nt; blkj++) {
                /* the index of the first myrow on the top of block (blkj) */
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if ( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=1; blki <= mt; blki++) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow  = firstrow + (mt-blki)*NB;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( (blkj == nt-1) && (blki == mt) ) {
                        Vn = min( Vblksiz, Vm );
                    } else {
                        Vn = min( Vblksiz, Vm-1 );
                    }
                    mycol     = blkj*Vblksiz;
                    if ((Vm > 0) && (Vn > 0)) {
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, queues[0]);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, queues[0]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE, queues[0] );
                    } // end for (Vm &Vn) > 0
                } // end for blki
            } // end fo blkj
        } // end of version 91
        /*
         * Version 92:
         */
        else {
            mt    = magma_ceildiv((N-1),NB);
            for (blki = 1; blki <= mt; blki++) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = 0; blkj < nt; blkj++) {
                    /* the index of the first row of the first col meaning
                     * the block on the top left (blki) */
                    firstrow = (mt-blki)*NB+1;
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }
                    if ((Vm > 0) && (Vn > 0)) {
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, queues[0]);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, queues[0]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE, queues[0] );
                    } // end for (Vm &Vn) > 0
                } //end for blkj
            } // end for blki
        } //end of version 92
    } // end RIGHT


    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( myevent[0] );
    magma_event_destroy( myevent[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_free(dwork);

    return *info;
}

#undef V
#undef T
#undef dE
