/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       Azzam: this FGMRES code is for testing purpose.
       It is based on Yousef Saad FGMRES implementation,
       all credit will go to Saad. 
*/
#ifdef MAGMA_USE_CUDAMEMSET
#include <cuda_runtime_api.h>
#endif

#include "magma_internal.h"



#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define WARP      32
#define HALFWARP  16
#define BLOCKDIM  512
#define ALIGNMENT 512
#define MAXTHREADS (30 * 1024 * 5)
// number of half warp per block
#define NHW 32
// for LU kernel
#define BLOCKDIM2 512
// BLOCKDIM2/HALFWARP
#define NHW2 32 
#define ZERO 0.0
#define TRUE  1
#define FALSE 0
#define EPSILON   1.0e-18
#define EPSMAC    1.0e-16
/*--- max # of diags in DIA */
#define MAXDIAG 60
/*--- max # of color in 
 *--- multi-color reordering*/
#define MAXCOL 100



//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)
#define PRINT 1
#define PRINT2 1
#define DEBUG 0

/*--------------------------------------*
|      FGMRES method
| n     : matrix dimension
| kdim  : max Krylov subspace dimension
| maxiter: max iter number
| tol   : relative tolerance
| dA   : pointer to A on device
| ldda  : leading dimension of dA
| dLU_dprec  : (factorized) A on device (result of GETRF())
| lddludp : leading dimension of dLU_dprec
| ipiv  : pivoting information (device or host??)
| dX   : vector x on device
| dB   : right hand side x on device
| rnorm0: initial residual norm
| niters: number of iterations carried out
| queue : magam queue
 *--------------------------------------*/
#if 0
extern "C" magma_int_t fgmres(magma_int_t n, magma_int_t restrt, magma_int_t maxiter, double tol,
	    double *dA, magma_int_t ldda, 
        double *dLU_dprec, magma_int_t lddludp,
	    magma_int_t *ipiv, 
	    double *dX, double *dB,
	    double *rnorm0, magma_int_t *niters,
	    magma_queue_t queue)
#else
extern "C" magma_int_t
magma_dfgmres_plu_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, 
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t maxiter_inner, magma_int_t restrt_inner,
    magma_int_t userinitguess, 
    double tol, double innertol,
	double *rnorm0, magma_int_t *niters,
    magma_refinement_t solver_type,
    char *algoname, magma_int_t is_inner,
    magma_queue_t queue)
#endif
{
    magma_int_t tot_inner_iter=0;
    char algoname_outer[80], algoname_inner[80];
    /*-----------------------------------------*/
    magma_int_t i,j,k,k1,ii,out_flag,in_flag, info;
    double gam, eps1, *d_VV, *d_w;
    /*------------------------------------------*/
    /*------------------------------------------*
      |        Initialization                     |
      *------------------------------------------*/
    // return;
 
    // ALIGNMENT OF DEVICE MEM
    magma_int_t m = ALIGNMENT/sizeof(double);
    magma_int_t n2 = (n+m-1)/m*m;

    /*-----------------------------*
      |         DEVICE MEM           |
      *-----------------------------*/
    /* Allocate Device mem for matrix Vm */
    magma_malloc((void**)&d_VV, (restrt+1)*n2*sizeof(double));
    /* work array */
    magma_malloc((void**)&d_w, restrt*n2*sizeof(double));
   
    /*-----------------------------*
      |           HOST MEM          |
      *-----------------------------*/
    /* Allocate Host mem for Hessenberg matrix H */
    double *HH;
    magma_dmalloc_cpu( &HH, restrt*(restrt+1));  
    /* Givens rotation */
    double *c;
    magma_dmalloc_cpu(&c, restrt);
    double *s;
    magma_dmalloc_cpu(&s, restrt);
    double *rs;
    magma_dmalloc_cpu(&rs, restrt+1);
 
  
    /*-----------------------------------*
      |          Iteration                 |
      *------------------------------------*/
    out_flag = TRUE;
    magma_int_t iters = 0;
    double ro, t;

    /* outer loop */
    while (out_flag)
	{
	    magma_dgemv(MagmaNoTrans, n, n, -1.0, dA, ldda, dX, 1,
			0, d_VV, 1, queue);

	    magma_daxpy(n, 1.0, dB, 1, d_VV, 1, queue);

	    ro = magma_ddot(n, d_VV, 1, d_VV, 1, queue);
	    ro = sqrt(ro);

	    if (fabs(ro-ZERO) <= EPSILON) {
		out_flag = FALSE;
		break;
	    }   
	    t = 1.0 / ro;    
        
	    magma_dscal(n, t, d_VV, 1, queue);
	    if (iters == 0) {
		*rnorm0 = ro;
		MAGMA_PRINTF("Initial Residual: %e\n", ro);
		eps1 = tol*ro;      
	    }

	    rs[0] = ro;
	    i = -1;
	    in_flag = TRUE;
    
	    /* Inner loop */
	    while (in_flag) {
		i++;
		iters ++;
        magma_int_t  inner_iter=0;
      
        if( solver_type == Magma_REFINE_IRGMSTRS || solver_type == Magma_REFINE_GMSTRS )
        {
            magma_dsgetrs_gpu( MagmaNoTrans, n, 1, dLU_sprec, lddlusp, dipiv, &d_VV[i*n2], n2, &d_w[i*n2], n2, dSX, &info );
        }
        else if( solver_type == Magma_REFINE_IRGMDTRS || solver_type == Magma_REFINE_GMDTRS )
        {
		    magma_dcopy( n, &d_VV[i*n2], 1, &d_w[i*n2], 1, queue);
		    magma_dgetrs_gpu( MagmaNoTrans, n, 1, dLU_dprec, lddludp, ipiv, &d_w[i*n2], n2, &info );
        }
        else if( solver_type == Magma_REFINE_GMGMDTRS || solver_type == Magma_REFINE_GMGMSTRS )
        {
            double inner_norm;
            //double *inner_dX=NULL;
            //cudaMalloc((void**)&inner_dX, n2*sizeof(double));
            //cudaMemset(inner_dX, 0, n2*sizeof(double));
            #ifdef MAGMA_USE_CUDAMEMSET
            cudaMemset(&d_w[i*n2], 0, n2*sizeof(double));
            #else
            magmablas_dlaset(MagmaFull, n2, 1, 0.0, 0.0, &d_w[i*n2], n2, queue);
            #endif
            magma_dfgmres_plu_gpu(MagmaNoTrans, n, 1, dA, ldda,
                    dLU_sprec, lddlusp, dLU_dprec, lddludp,
                    ipiv, dipiv, &d_VV[i*n2], n2, &d_w[i*n2], n2, dSX, 
                    restrt_inner, restrt_inner, restrt_inner, restrt_inner,
                    0, innertol, innertol, &inner_norm, &inner_iter,
                    (solver_type == Magma_REFINE_GMGMSTRS ? Magma_REFINE_GMSTRS : Magma_REFINE_GMDTRS), 
                    algoname, 1, queue);
        //magmablas_dlacpy( MagmaFull, n, 1, inner_dX, n2, &d_w[i*n2], n2, queue );
        //magma_queue_sync(queue);
        //cudaFree(inner_dX);
            tot_inner_iter += inner_iter;
            if(!is_inner)
                MAGMA_PRINTF("%s_GMRES_outer_iter %lld uses inner_iter %lld \n", algoname, (long long) iters, (long long) inner_iter );
        }


		magma_dgemv( MagmaNoTrans, n, n, 1.0, dA, ldda,
			     &d_w[i*n2], 1, 0.0, &d_VV[(i+1)*n2], 1, queue);
			     
		for (j=0; j<=i; j++) {
		    HH[IDX2C(i,j,restrt+1)] = 
			magma_ddot( n, &d_VV[j*n2], 1, &d_VV[(i+1)*n2], 1, queue);

		    magma_daxpy( n, -HH[IDX2C(i,j,restrt+1)], &d_VV[j*n2], 1,
				 &d_VV[(i+1)*n2], 1, queue );
		}

		t = magma_ddot( n, &d_VV[(i+1)*n2], 1, &d_VV[(i+1)*n2], 1, queue );
		t = sqrt(t);

		HH[IDX2C(i,i+1,restrt+1)] = t;
            
		if (fabs(t-ZERO) > EPSILON) {
		    t = 1.0 / t;
		    magma_dscal( n, t, &d_VV[(i+1)*n2], 1, queue );
		}
      
		if (i !=0 )
		    for (k=1; k<=i; k++) {
			k1 = k-1;

			t  = HH[IDX2C(i,k1,restrt+1)];

			HH[IDX2C(i,k1,restrt+1)] =
			    c[k1]*t + s[k1]*HH[IDX2C(i,k,restrt+1)];

			HH[IDX2C(i,k, restrt+1)] =
			    -s[k1]*t + c[k1]*HH[IDX2C(i,k,restrt+1)];
		    }

		double Hii  = HH[IDX2C(i,i,restrt+1)];
		double Hii1 = HH[IDX2C(i,i+1,restrt+1)];
      
		gam = sqrt(Hii*Hii + Hii1*Hii1);
            
		if (fabs(gam-ZERO) <= EPSILON)
		    gam = EPSMAC;
		
		c[i] = Hii  / gam;
		s[i] = Hii1 / gam;
		rs[i+1] = -s[i] * rs[i];
		rs[i]   =  c[i] * rs[i];
      
		HH[IDX2C(i,i,restrt+1)] = c[i]*Hii + s[i]*Hii1;
		ro = fabs(rs[i+1]);

		if (PRINT2 && is_inner){
		    MAGMA_PRINTF("     %s_GPU_GMRES_INNER_ITER %5lld     %8.2E\n", algoname, (long long) iters, ro);
        }
        else if (PRINT2 && ~is_inner){
		    MAGMA_PRINTF("     %s_GPU_GMRES_ITER %5lld  inner_iter %5lld   %8.2E\n", algoname, (long long) iters, (long long) inner_iter, ro);
        }


		/* test convergence */
		if (i+1 >=restrt || ro <=eps1 || iters >= maxiter)
		    in_flag = FALSE;
	
	    } /* end of inner loop */
              
	    rs[i] = rs[i]/HH[IDX2C(i,i,restrt+1)];
	    for (ii=2; ii<=i+1; ii++)
		{
		    k  = i-ii+1;
		    k1 = k+1;
		    t  = rs[k];
		    for (j=k1; j<=i; j++)
			t = t - HH[IDX2C(j,k,restrt+1)]*rs[j];
	
		    rs[k] = t / HH[IDX2C(k,k,restrt+1)];
		}
        
	    for (j=0; j<=i; j++)
		magma_daxpy( n, rs[j], &d_w[j*n2], 1, dX, 1, queue );
    
	    /* test solution */
	    if ( ro<=eps1 || iters >= maxiter)
		out_flag = FALSE;
	} /* end of outer loop */

    *niters = tot_inner_iter > 0 ? tot_inner_iter : iters; 
    if(!is_inner)
    {    
        snprintf(algoname_outer, sizeof(algoname_outer),"%s_outer_niter",algoname);
        snprintf(algoname_inner, sizeof(algoname_inner),"%s_inner_niter",algoname);
        MAGMA_PRINTF("\n\ninfo===> %s %lld  %s is %d \n\n", algoname_outer, (long long) iters, algoname_inner, tot_inner_iter); 
    }
    /*------------------------------------------------------*
      |          Finalization:  Free Memory
      *------------------------------------------------------*/
    fflush( stdout );
    magma_free(d_VV);
    magma_free(d_w);
    magma_free_cpu(HH);
    magma_free_cpu(c);
    magma_free_cpu(s);
    magma_free_cpu(rs);

    return iters;
} /* end of fgmres */

