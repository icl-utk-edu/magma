/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include "magmasparse_internal.h"
#include <cuda.h>  // for CUDA_VERSION


#define PRECISION_z

#ifdef PRECISION_d

typedef struct SpaFmt {
/*--------------------------------------------- 
| C-style CSR format - used internally
| for all matrices in CSR format 
|---------------------------------------------*/
  int n;
  int *nzcount;  /* length of each row */
  int **ja;      /* pointer-to-pointer to store column indices  */
  double **ma;   /* pointer-to-pointer to store nonzero entries */
} SparMat, *csptr;

/*-------------------- end protos*/


typedef struct ILUfac {
    int n;
    csptr L;      /* L part elements                            */
    double *D;    /* diagonal elements                          */
    csptr U;      /* U part elements                            */
    int *work;    /* working buffer */
} ILUSpar, LDUmat, *iluptr;


void *Malloc( int nbytes, const char *msg )
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL)
    printf( "Not enough mem for %s. Requested size: %d bytes", msg, nbytes );

  return ptr;
}


int setupCS(csptr amat, int len, int job)
{
/*----------------------------------------------------------------------
| Initialize SpaFmt structs.
|----------------------------------------------------------------------
| on entry:
|==========
| ( amat )  =  Pointer to a SpaFmt struct.
|     len   =  size of matrix
|     job   =  0: pattern only
|              1: data and pattern
|
| On return:
|===========
|
|  amat->n
|      ->*nzcount
|      ->**ja
|      ->**ma
|
| integer value returned:
|             0   --> successful return.
|             1   --> memory allocation error.
|--------------------------------------------------------------------*/
   amat->n = len;
   amat->nzcount = (int *)Malloc( len*sizeof(int), "setupCS" );
   amat->ja = (int **) Malloc( len*sizeof(int *), "setupCS" );
   if( job == 1 ) 
       amat->ma = (double **) Malloc( len*sizeof(double *), "setupCS" );
   else
       amat->ma = NULL;
   return 0;
}
/*---------------------------------------------------------------------
|     end of setupCS
|--------------------------------------------------------------------*/



int CSRcs( int n, double *a, int *ja, int *ia, csptr mat, int rsa )
{
/*----------------------------------------------------------------------
| Convert CSR matrix to SpaFmt struct
|----------------------------------------------------------------------
| on entry:
|==========
| a, ja, ia  = Matrix stored in CSR format (with FORTRAN indexing).
| rsa        = source file is symmetric HB matrix 
|
| On return:
|===========
|
| ( mat )  =  Matrix stored as SpaFmt struct. (C indexing)
|
|       integer value returned:
|             0   --> successful return.
|             1   --> memory allocation error.
|--------------------------------------------------------------------*/
  int i, j, j1, len, col, nnz, info=0;
  double *bra;
  int *bja;
  /*    setup data structure for mat (csptr) struct */
  info = setupCS( mat, n, 1 );
  if( rsa ) { /* RSA HB matrix */
    for( j = 0; j < n; j++ ) {
      len = ia[j+1] - ia[j];
      mat->nzcount[j] = len;
    }
    for( j = 0; j < n; j++ ) {
      for( j1 = ia[j]-1; j1 < ia[j+1]-1; j1++ ) {
        col = ja[j1] - 1;
        if( col != j ) mat->nzcount[col]++;
      }
    }
    for( j = 0; j < n; j++ ) {
      nnz = mat->nzcount[j];
      mat->ja[j] = (int *)Malloc( nnz * sizeof(int), "CSRcs" );
      mat->ma[j] = (double *)Malloc( nnz * sizeof(double), "CSRcs" );
      mat->nzcount[j] = 0;
    }
    for( j = 0; j < n; j++ ) {
      for( j1 = ia[j]-1; j1 < ia[j+1]-1; j1++ ) {
        col = ja[j1] - 1;
        mat->ja[j][mat->nzcount[j]] = col;
        mat->ma[j][mat->nzcount[j]] = a[j1];
        mat->nzcount[j]++;
        if( col != j ) {
          mat->ja[col][mat->nzcount[col]] = j;
          mat->ma[col][mat->nzcount[col]] = a[j1];
          mat->nzcount[col]++;
        }
      }
    }
    return 0;
  }

  for (j=0; j<n; j++) {
    len = ia[j+1] - ia[j];
    mat->nzcount[j] = len;
    if (len > 0) {
      bja = (int *) Malloc( len*sizeof(int), "CSRcs" );
      bra = (double *) Malloc( len*sizeof(double), "CSRcs" );
      i = 0;
      for (j1=ia[j]; j1<ia[j+1]; j1++) {
        bja[i] = ja[j1] ;
        bra[i] = a[j1] ;
        i++;
      }
      mat->ja[j] = bja;
      mat->ma[j] = bra;
    }
  }    
  return 0;
}
/*---------------------------------------------------------------------
|     end of CSRcs
|--------------------------------------------------------------------*/


int setupILU( iluptr lu, int n )
{
/*----------------------------------------------------------------------
| Initialize ILUSpar structs.
|----------------------------------------------------------------------
| on entry:
|==========
|   ( lu )  =  Pointer to a ILUSpar struct.
|       n   =  size of matrix
|
| On return:
|===========
|
|    lu->n
|      ->L     L matrix, SpaFmt format
|      ->D     Diagonals
|      ->U     U matrix, SpaFmt format
|      ->work  working buffer of length n
|      ->bf    buffer
|
| integer value returned:
|             0   --> successful return.
|            -1   --> memory allocation error.
|--------------------------------------------------------------------*/
    lu->n  = n;
    lu->D = (double *)Malloc( sizeof(double) * n, "setupILU" );
    lu->L = (csptr)Malloc( sizeof(SparMat), "setupILU" );
    setupCS( lu->L, n, 1 );
    lu->U = (csptr)Malloc( sizeof(SparMat), "setupILU" );
    setupCS( lu->U, n, 1 );
    lu->work = (int *)Malloc( sizeof(int) * n, "setupILU" );
    return 0;
}
/*---------------------------------------------------------------------
|     end of setupILU
|--------------------------------------------------------------------*/




int qsplit(double *a, int *ind, int n, int Ncut)
{
/*----------------------------------------------------------------------
|     does a quick-sort split of a real array.
|     on input a[0 : (n-1)] is a real array
|     on output is permuted such that its elements satisfy:
|
|     abs(a[i]) >= abs(a[Ncut-1]) for i < Ncut-1 and
|     abs(a[i]) <= abs(a[Ncut-1]) for i > Ncut-1
|
|     ind[0 : (n-1)] is an integer array permuted in the same way as a.
|---------------------------------------------------------------------*/
   double tmp, abskey;
   int j, itmp, first, mid, last, ncut;
   ncut = Ncut - 1;

   first = 0;
   last = n-1;
   if (ncut<first || ncut>last) return 0;
/* outer loop -- while mid != ncut */
do{
   mid = first;
   abskey = fabs(a[mid]);
  for (j=first+1; j<=last; j++) {
     if (fabs(a[j]) > abskey) {
        mid = mid+1;
     tmp = a[mid];
     itmp = ind[mid];
     a[mid] = a[j];
     ind[mid] = ind[j];
     a[j]  = tmp;
     ind[j] = itmp;
      }
   }
/*-------------------- interchange */
   tmp = a[mid];
   a[mid] = a[first];
   a[first]  = tmp;
   itmp = ind[mid];
   ind[mid] = ind[first];
   ind[first] = itmp;
/*-------------------- test for while loop */
   if (mid == ncut) break;
   if (mid > ncut) 
      last = mid-1;
   else
      first = mid+1;
   }while(mid != ncut);
   
   return 0;
}
/*--------------- end of qsplitC ----------------------------------------
|---------------------------------------------------------------------*/


#endif


extern "C" magma_int_t
magma_zilut_saad( 
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue ) {
//     csptr csmat, iluptr lu, int lfil, double tol, FILE *fp )


/*----------------------------------------------------------------------------
 * ILUT preconditioner
 * incomplete LU factorization with dual truncation mechanism
 * NOTE : no pivoting implemented as yet in GE for diagonal elements
 *----------------------------------------------------------------------------
 * Parameters
 *----------------------------------------------------------------------------
 * on entry:
 * =========
 * csmat    = matrix stored in SpaFmt format -- see heads.h for details
 * lu       = pointer to a ILUSpar struct -- see heads.h for details
 * lfil     = integer. The fill-in parameter. Each column of L and
 *            each column of U will have a maximum of lfil elements.
 *            WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO
 *            EARLIER VERSIONS. 
 *            lfil must be .ge. 0.
 * tol      = real*8. Sets the threshold for dropping small terms in the
 *            factorization. See below for details on dropping strategy.
 * fp       = file pointer for error log ( might be stdout )
 *
 * on return:
 * ==========
 * ierr     = return value.
 *            ierr  = 0   --> successful return.
 *            ierr  = -1  --> Illegal value for lfil
 *            ierr  = -2  --> zero diagonal or zero col encountered
 * lu->n    = dimension of the matrix
 *   ->L    = L part -- stored in SpaFmt format
 *   ->D    = Diagonals
 *   ->U    = U part -- stored in SpaFmt format
 *----------------------------------------------------------------------------
 * Notes:
 * ======
 * All the diagonals of the input matrix must not be zero
 *----------------------------------------------------------------------------
 * Dual drop-off strategy works as follows. 
 *
 * 1) Theresholding in L and U as set by tol. Any element whose size
 *    is less than some tolerance (relative to the norm of current
 *    row in u) is dropped.
 *
 * 2) Keeping only the largest lfil elements in the i-th column of L
 *    and the largest lfil elements in the i-th column of U.
 *
 * Flexibility: one can use tol=0 to get a strategy based on keeping the
 * largest elements in each column of L and U. Taking tol .ne. 0 but lfil=n
 * will give the usual threshold strategy (however, fill-in is then
 * impredictible).
 *--------------------------------------------------------------------------*/
 
 magma_int_t info = 0;
 
#ifdef PRECISION_d

  magma_int_t timing = 1;
  
  real_Double_t start, end;

  csptr csmat = NULL;
  csmat = (csptr)Malloc( sizeof(SparMat), "main:csmat"); 
  CSRcs( A.num_rows, A.val, A.col, A.row, csmat, 0 );
  start = magma_sync_wtime( queue );
  
  
  iluptr lu = NULL;      /* a temporary lu matrix           */
  lu = (iluptr)Malloc( sizeof(ILUSpar), "main" );
  int lfil = magma_ceildiv((A.nnz - A.num_rows )*precond->atol,(2*A.num_rows));
  double tol = 0.0;
  int n = csmat->n; 
  int len, lenu, lenl;
  int nzcount, *ja, *jbuf, *iw, i, j, k;
  int col, jpos, jrow, upos;
  double t, tnorm, tolnorm, fact, lxu, *wn, *ma, *w;
  csptr L, U;
  double *D;
  if( lfil < 0 ) {
    printf( "ilut: Illegal value for lfil.\n" );
    return -1;
  }    
  setupILU( lu, n );
  L = lu->L;
  U = lu->U;
  D = lu->D;
  iw = (int *)Malloc( n*sizeof(int), "ilut" );
  jbuf = (int *)Malloc( n*sizeof(int), "ilut" );
  wn = (double *)Malloc( n * sizeof(double), "ilut" );
  w = (double *)Malloc( n * sizeof(double), "ilut" );  
  /* set indicator array jw to -1 */
  for( i = 0; i < n; i++ ) iw[i] = -1;
  /* beginning of main loop */
  for( i = 0; i < n; i++ ) {
    nzcount = csmat->nzcount[i];
    ja = csmat->ja[i];
    ma = csmat->ma[i];
    tnorm = 0;
    for( j = 0; j < nzcount; j++ ) {
      tnorm += fabs( ma[j] );
    }
    if( tnorm == 0.0 ) {
      printf( "ilut: zero row encountered.\n" );
      return -2;
    }
    tnorm /= (double)nzcount;
    tolnorm = tol * tnorm;
    /* unpack L-part and U-part of column of A in arrays w */
    lenu = 0;
    lenl = 0;
    jbuf[i] = i;
    w[i] = 0;
    iw[i] = i;
    for( j = 0; j < nzcount; j++ ) {
      col = ja[j];
      t = ma[j];
      if( col < i ) {
        iw[col] = lenl;
        jbuf[lenl] = col;
        w[lenl] = t;
        lenl++;
      } else if( col == i ) {
        w[i] = t;
      } else {
        lenu++;
        jpos = i + lenu;
        iw[col] = jpos;
        jbuf[jpos] = col;
        w[jpos] = t;
      }
    }
    j = -1;
    len = 0;
    /* eliminate previous rows */
    while( ++j < lenl ) {
/*----------------------------------------------------------------------------
 *  in order to do the elimination in the correct order we must select the
 *  smallest column index among jbuf[k], k = j+1, ..., lenl
 *--------------------------------------------------------------------------*/
      jrow = jbuf[j];
      jpos = j;
      /* determine smallest column index */
      for( k = j + 1; k < lenl; k++ ) {
        if( jbuf[k] < jrow ) {
          jrow = jbuf[k];
          jpos = k;
        }
      }
      if( jpos != j ) {
        col = jbuf[j];
        jbuf[j] = jbuf[jpos];
        jbuf[jpos] = col;
        iw[jrow] = j;
        iw[col]  = jpos;
        t = w[j];
        w[j] = w[jpos];
        w[jpos] = t;
      }

      /* get the multiplier */
      fact = w[j] * D[jrow];
      w[j] = fact;
      /* zero out element in row by resetting iw(n+jrow) to -1 */
      iw[jrow] = -1;

      /* combine current row and row jrow */
      nzcount = U->nzcount[jrow];
      ja = U->ja[jrow];
      ma = U->ma[jrow];
      for( k = 0; k < nzcount; k++ ) {
        col = ja[k];
        jpos = iw[col];
        lxu = - fact * ma[k];
        /* if fill-in element is small then disregard */
        if( fabs( lxu ) < tolnorm && jpos == -1 ) continue;

        if( col < i ) {
          /* dealing with lower part */
          if( jpos == -1 ) {
            /* this is a fill-in element */
            jbuf[lenl] = col;
            iw[col] = lenl;
            w[lenl] = lxu;
            lenl++;
          } else {
            w[jpos] += lxu;
          }

        } else {
          /* dealing with upper part */
//          if( jpos == -1 ) {
      if( jpos == -1 && fabs(lxu) > tolnorm) {
            /* this is a fill-in element */
            lenu++;
            upos = i + lenu;
            jbuf[upos] = col;
            iw[col] = upos;
            w[upos] = lxu;
          } else {
            w[jpos] += lxu;
          }
        }
      }
    }

    /* restore iw */
    iw[i] = -1;
    for( j = 0; j < lenu; j++ ) {
      iw[jbuf[i+j+1]] = -1;
    }

/*---------- case when diagonal is zero */
    if( w[i] == 0.0 ) {
      printf( "zero diagonal encountered.\n" );
      for( j = i; j < n; j++ ) {
        L->ja[j] = NULL; 
        L->ma[j] = NULL;
        U->ja[j] = NULL; 
        U->ma[j] = NULL;
      }
      return -2;
    }
/*-----------Update diagonal */    
    D[i] = 1 / w[i];

    /* update L-matrix */
//    len = min( lenl, lfil );
    len = lenl < lfil ? lenl : lfil;
    for( j = 0; j < lenl; j++ ) {
      wn[j] = fabs( w[j] );
      iw[j] = j;
    }
    qsplit( wn, iw, lenl, len );
    L->nzcount[i] = len;
    if( len > 0 ) {
      ja = L->ja[i] = (int *)Malloc( len*sizeof(int), "ilut" );
      ma = L->ma[i] = (double *)Malloc( len*sizeof(double), "ilut" );
    }
    for( j = 0; j < len; j++ ) {
      jpos = iw[j];
      ja[j] = jbuf[jpos];
      ma[j] = w[jpos];
    }
    for( j = 0; j < lenl; j++ ) iw[j] = -1;

    /* update U-matrix */
//    len = min( lenu, lfil );
    len = lenu < lfil ? lenu : lfil;
    for( j = 0; j < lenu; j++ ) {
      wn[j] = fabs( w[i+j+1] );
      iw[j] = i+j+1;
    }
    qsplit( wn, iw, lenu, len );
    U->nzcount[i] = len;
    if( len > 0 ) {
      ja = U->ja[i] = (int *)Malloc( len*sizeof(int), "ilut" );
      ma = U->ma[i] = (double *)Malloc( len*sizeof(double), "ilut" );
    }
    for( j = 0; j < len; j++ ) {
      jpos = iw[j];
      ja[j] = jbuf[jpos];
      ma[j] = w[jpos];
    }
    for( j = 0; j < lenu; j++ ) {
      iw[j] = -1;
    }
  }
  int nzcounts = 0;
  int nzcountL=0, nzcountU=0;
  for(int z=0; z<n; z++){
    nzcounts = nzcounts +   L->nzcount[z] + U->nzcount[z]; 
    nzcountL = nzcountL +   L->nzcount[z];
    nzcountU = nzcountU +   U->nzcount[z];
  }
  printf("ilut_fill_ratio = %.6f;\n", (double)(nzcounts+n)/(double)(A.nnz)); 
  printf("%% L:%d U:%d D:%d = %d vs. %lld\n",
         nzcountL, nzcountU, n, nzcountL + nzcountU + n, (long long) A.nnz);

  free( iw );
  free( jbuf );
  free( wn );
  free(w);
  precond->Lma = L->ma;
  precond->Lja = L->ja;
  precond->Lnz = L->nzcount;
  precond->L.num_rows = L->n;
  precond->L.memory_location = Magma_CPU;
  precond->L.storage_type = Magma_CSR;
  
  precond->Uma = U->ma;
  precond->Uja = U->ja;
  precond->Unz = U->nzcount;
  precond->U.num_rows = U->n;
  precond->U.memory_location = Magma_CPU;
  precond->U.storage_type = Magma_CSR;
  
  precond->d.val = D;
  precond->d.num_rows = U->n;
  precond->d.memory_location = Magma_CPU;
  precond->d.storage_type = Magma_CSR;
  
  end = magma_sync_wtime( queue );
  if( timing == 1 ){
      printf(" ilut_runtime = %.4e;\n", end-start);
  }
    
    
#endif
  return info;
}


extern "C" magma_int_t
magma_zilut_saad_apply( 
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ) {
//magma_zilut_lutsolC( double *y, double *x, iluptr lu )
// {
 /*----------------------------------------------------------------------
  *    performs a forward followed by a backward solve
  *    for LU matrix as produced by ilut
  *    y  = right-hand-side
  *    x  = solution on return
  *    lu = LU matrix as produced by ilut.
  *--------------------------------------------------------------------*/
     magma_int_t info = 0;
     
 #ifdef PRECISION_d
     magma_z_matrix x_h, b_h;
     magma_zmtransfer( *x, &x_h, Magma_DEV, Magma_CPU, queue );
     magma_zmtransfer( b, &b_h, Magma_DEV, Magma_CPU, queue );
     magma_zmfree( x, queue );
     
     int n = precond->L.num_rows, i, j, nzcount, *ja;
     double *ma;
     csptr L, U;
     L = (csptr)Malloc( sizeof(SparMat), "main:csmat"); 
     U = (csptr)Malloc( sizeof(SparMat), "main:csmat"); 
     
     L->nzcount = precond->Lnz;
     L->ja = precond->Lja;
     L->ma = precond->Lma;
     L->n = precond->L.num_rows;
     
     U->nzcount = precond->Unz;
     U->ja = precond->Uja;
     U->ma = precond->Uma;
     U->n = precond->U.num_rows;
     /* Block L solve */
     
     for( i = 0; i < n; i++ ) {
         x_h.val[i] = b_h.val[i];
         nzcount = L->nzcount[i];
         ja = L->ja[i];
         ma = L->ma[i];
         for( j = 0; j < nzcount; j++ ) {
             x_h.val[i] -= x_h.val[ja[j]] * ma[j];
         }
     }
     /* Block -- U solve */
     for( i = n-1; i >= 0; i-- ) {
         nzcount = U->nzcount[i];
         ja = U->ja[i];
         ma = U->ma[i];
         for( j = 0; j < nzcount; j++ ) {
             x_h.val[i] -= x_h.val[ja[j]] * ma[j];
         }
         x_h.val[i] *= precond->d.val[i];
     }
      
     magma_zmtransfer( x_h, x, Magma_CPU, Magma_DEV, queue );
     magma_zmfree( &x_h, queue );
     magma_zmfree( &b_h, queue );
#endif
     return info;
 }

