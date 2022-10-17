/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

*/

#include "magma_internal.h"

#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define ALIGNMENT 512

#define EPSILON   lapackf77_dlamch("Epsilon") //1.0e-18
#define EPSMAC    1.0e-14

//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)

extern "C" magma_int_t
magma_dfgmres_spd_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    double  *dA, magma_int_t ldda,
    float   *dL, magma_int_t lddl, float* dD,
    double  *dB, magma_int_t lddb,
    double  *dX, magma_int_t lddx,
    float   *dSX,
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t maxiter_inner, magma_int_t restrt_inner,
    double tol, double innertol,
	double *rnorm0, magma_int_t *niters, magma_int_t is_inner,
	magma_int_t is_preprocessed, float miu,
    magma_queue_t queue)
{
    magma_int_t tot_inner_iter=0;

    magma_int_t i,j,k,k1,ii,out_flag,in_flag, info;
    double gam, eps1;
    double *d_VV, *d_w;

    // constants
    double c_one     = MAGMA_D_ONE;
    double c_zero    = MAGMA_D_ZERO;
    double c_neg_one = MAGMA_D_NEG_ONE;

    assert(uplo == MagmaLower);

    /*------------------------------------------*
     |        Initialization                    |
     *------------------------------------------*/
    // ALIGNMENT OF DEVICE MEM
    magma_int_t m = ALIGNMENT/sizeof(double);
    magma_int_t n2 = (n+m-1)/m*m;

    /*-----------------------------*
     |         DEVICE MEM          |
     *-----------------------------*/
    // Allocate Device mem for matrix Vm
    magma_dmalloc( &d_VV, (restrt+1)*n2 );
    magma_dmalloc( &d_w, restrt*n2 );          // workspace

    /*-----------------------------*
      |           HOST MEM          |
      *-----------------------------*/
    /* Allocate Host mem for Hessenberg matrix H */
    double *HH;
    magma_dmalloc_cpu( &HH, restrt*(restrt+1));
    /* Givens rotation */
    double *c;
    double *s, *rs;
    magma_dmalloc_cpu(&c, restrt);
    magma_dmalloc_cpu(&s, restrt);
    magma_dmalloc_cpu(&rs, restrt+1);

    /*-----------------------------------*
      |          Iteration                 |
      *------------------------------------*/
    out_flag = true;
    magma_int_t iters = 0;
    double ro, t;
    double t1;

    /* outer loop */
    while (out_flag) {
	    magma_dsymv(uplo, n, c_neg_one, dA, ldda, dX, 1, c_zero, d_VV, 1, queue); // d_VV = -A*x
	    magma_daxpy(n, c_one, dB, 1, d_VV, 1, queue);    // d_VV = b- Ax

	    ro = magma_ddot(n, d_VV, 1, d_VV, 1, queue);
	    ro = sqrt(ro);    // norm of d_VV (residual vector)

	    if (fabs(ro) <= EPSILON) {
	        MAGMA_PRINTF("fabs(ro-MAGMA_D_ZERO) = %e\n", fabs(ro-MAGMA_D_ZERO));
		    out_flag = false;
		    break;
	    }

	    t = 1.0 / ro;
	    magma_dscal(n, t, d_VV, 1, queue);    // normalize d_VV

	    if (iters == 0) {
	        *rnorm0 = ro;
	        MAGMA_PRINTF("Initial Residual: %e\n", ro);

	        eps1 = tol*ro;
	        MAGMA_PRINTF("ro = %e, tol = %e, eps1 = %e\n", ro, tol, eps1);

	        // tol is usually passed as cte = Anrm * sqrt(n) * BDWTH, with BDWTH=1.0
            double Anrm = tol / (lapackf77_dlamch("Epsilon") * magma_dsqrt((double)n));
            MAGMA_PRINTF("Before inner loop: GPU_GMRES_ITER %5lld  inner_iter %5lld   ro = %8.2E    Anrm = %8.2e    residual = %8.2e\n",
            (long long) iters, (long long) 0, ro, Anrm, ro / (Anrm * n));
	    }

	    rs[0] = MAGMA_D_MAKE( ro, 0.f );
	    i = -1;
	    in_flag = true;

	    /* Inner loop */
	    while (in_flag) {
		    i++;
		    iters ++;
            magma_int_t  inner_iter=0;

            magmablas_dlag2s(n, nrhs, &d_VV[i*n2], n2, dSX, n, queue, &info);
            if(is_preprocessed > 0) {
                magmablas_sdimv_invert(n, MAGMA_S_ONE, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
            }
            magma_spotrs_gpu(uplo, n, nrhs, dL, lddl, dSX, n, &info);
            if(is_preprocessed > 0) {
                magmablas_sdimv_invert(n, miu, dD, 1, dSX, 1, MAGMA_S_ZERO,  dSX, 1, queue);
            }

            magmablas_slag2d(n, nrhs, dSX, n, &d_w[i*n2], n2, queue, &info);
			magma_dsymv( uplo, n, c_one, dA, ldda, &d_w[i*n2], 1, c_zero, &d_VV[(i+1)*n2], 1, queue);

		    for (j=0; j<=i; j++) {
		        HH[IDX2C(i,j,restrt+1)] = magma_ddot( n, &d_VV[j*n2], 1, &d_VV[(i+1)*n2], 1, queue);

		        magma_daxpy( n, -HH[IDX2C(i,j,restrt+1)],
		                     &d_VV[j*n2],     1,
			                 &d_VV[(i+1)*n2], 1, queue );
		    }

		    t = MAGMA_D_REAL( magma_ddot( n, &d_VV[(i+1)*n2], 1, &d_VV[(i+1)*n2], 1, queue ) );
		    t = sqrt(t);

		    HH[IDX2C(i,i+1,restrt+1)] = MAGMA_D_MAKE( t, 0.f );

		    if (fabs(t-MAGMA_D_ZERO) > EPSILON) {
		        t = 1.0 / t;
		        magma_dscal( n, MAGMA_D_MAKE( t, 0.f ), &d_VV[(i+1)*n2], 1, queue );
		    }

		    if (i !=0 ) {
		        for (k=1; k<=i; k++) {
			        k1 = k-1;
			        t1  = HH[IDX2C(i,k1,restrt+1)];

			        HH[IDX2C(i,k1,restrt+1)] =
			        MAGMA_D_MAKE(c[k1], 0.f ) * t1 + s[k1] * HH[IDX2C(i,k,restrt+1)];

			        HH[IDX2C(i,k, restrt+1)] =
			        -MAGMA_D_CONJ( s[k1] )    * t1 + MAGMA_D_MAKE( c[k1], 0.f ) * HH[IDX2C(i,k,restrt+1)];
		        }
		    }

		    double rot;
		    double Hii  = HH[IDX2C(i,i,restrt+1)];
		    double Hii1 = HH[IDX2C(i,i+1,restrt+1)];
		    lapackf77_dlartg(&Hii, &Hii1, &c[i], &s[i], &rot);

		    rs[i+1] = -MAGMA_D_CONJ( s[i] )     * rs[i];
		    rs[i]   =  MAGMA_D_MAKE( c[i], 0.f) * rs[i];

		    HH[IDX2C(i,i,restrt+1)] = rot;
		    ro = MAGMA_D_ABS(rs[i+1]);

		    if( true ) {
                // tol is usually passed as cte = Anrm * sqrt(n) * BDWTH, with BDWTH=1.0
                float Anrm = tol / (lapackf77_dlamch("Epsilon") * magma_dsqrt((double)n));
                MAGMA_PRINTF("     GPU_GMRES_ITER %5lld  inner_iter %5lld   ro = %8.2E    Anrm = %8.2e    residual = %8.2e\n",
                (long long) iters, (long long) inner_iter, ro, Anrm, ro / (Anrm * n));
            }

		    /* test convergence */
		    if (i+1 >=restrt || ro <= eps1 || iters >= maxiter) {
		        in_flag = false;
		    }
	    } /* end of inner loop */

	    rs[i] = rs[i]/HH[IDX2C(i,i,restrt+1)];
	    for (ii=2; ii<=i+1; ii++) {
		    k  = i-ii+1;
		    k1 = k+1;
		    t1  = rs[k];
		    for (j=k1; j<=i; j++)
			t1 = t1 - HH[IDX2C(j,k,restrt+1)]*rs[j];

		    rs[k] = t1 / HH[IDX2C(k,k,restrt+1)];
		}

	    for (j=0; j<=i; j++) {
	        magma_daxpy( n, rs[j], &d_w[j*n2], 1, dX, 1, queue );
	    }

	    /* test solution */
	    if ( ro<=eps1 || iters >= maxiter)
		out_flag = false;
	} /* end of outer loop */

    *niters = tot_inner_iter > 0 ? tot_inner_iter : iters;

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

