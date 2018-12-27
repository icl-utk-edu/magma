/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar 

*/
#include <cuda.h>
#include "magma_internal.h"
//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)
// Azzam: external functions definition for internal debugging purpose
extern "C" magma_int_t
magma_dgmres_cpu_destroy( double *gmdwork, double *gmhwork);

extern "C" magma_int_t
magma_dgmres_cpu_init(
    magma_int_t n, 
    double **gmdwork, 
    double **gmhwork, magma_int_t *lwork,
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess,
    double *cntl, double tol,
    magma_int_t *irc, magma_int_t *icntl);


extern "C" magma_int_t
magma_dgmres_cpu_solve(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, double *gmdwork,
    double *gmhwork, magma_int_t lwork,
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess,
    double *cntl, double tol, double inner_tol,
    magma_int_t *irc, magma_int_t *icntl,
    magma_refinement_t facto_type, 
    magma_refinement_t solver_type,
    char *algoname,
    magma_queue_t queue);



extern "C" magma_int_t
magma_dgmres_cpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, 
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess, 
    double tol, double innertol,
    magma_refinement_t facto_type, 
    magma_refinement_t solver_type,
    char *algoname,
    magma_queue_t queue);


extern "C" magma_int_t
magma_dgmres_inner_cpu_destroy( double *gmdwork, double *gmhwork);

extern "C" magma_int_t
magma_dgmres_inner_cpu_init(
    magma_int_t n, 
    double **gmdwork, 
    double **gmhwork, magma_int_t *lwork,
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess,
    double *cntl, double innertol,
    magma_int_t *irc, magma_int_t *icntl);

extern "C" magma_int_t
magma_dgmres_inner_cpu_solve(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, double *gmdwork,
    double *gmhwork, magma_int_t lwork,
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess,
    double *cntl, double inner_tol,
    magma_int_t *irc, magma_int_t *icntl,
    magma_refinement_t solver_type,
    magma_queue_t queue);

extern "C" magma_int_t
magma_dgmres_inner_cpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dLU_sprec, magma_int_t lddlusp,
    magmaDouble_ptr dLU_dprec, magma_int_t lddludp,
    magmaInt_ptr ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX, 
    magma_int_t maxiter, magma_int_t restrt,
    magma_int_t userinitguess, double innertol,
    magma_refinement_t solver_type,
    magma_queue_t queue);





/***************************************************************************/
/***************************************************************************/
/***************************************************************************//**
    Purpose
    -------
    DSGESV or DHGESV expert interface.
    It computes the solution to a real system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
    the accomodate the Single Precision DSGESV and the Half precision dhgesv API.
    precision and iterative refinement solver are specified by facto_type, solver_type.
    For other API parameter please refer to the corresponding dsgesv or dhgesv.

    @param[in]
    facto_type    magma_refinement_t
                  Specify the mixed precision factorization algorithm.
                  Magma_PREC_SS for FP32 
                  Magma_PREC_SHT for FP16 Tensor Cores
                  More details will be released soon.

    @param[in]
    solver_type   magma_refinement_t
                  Specify the iterative refinement technique to be used.
                  classical IR or GMRES etc.
                  More details will be released soon.

    More details can be found in 
    Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham. 2018. 
    Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision 
    iterative refinement solvers. In Proceedings of the International Conference for 
    High Performance Computing, Networking, Storage, and Analysis (SC '18). 
    IEEE Press, Piscataway, NJ, USA, Article 47, 11 pages.

    @ingroup magma_gesv
*******************************************************************************/
extern "C" magma_int_t
magma_dxgesv_gmres_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_refinement_t facto_type, 
    magma_refinement_t solver_type,
    magma_int_t *iter,
    magma_int_t *info,
    real_Double_t *facto_time)
{
    #define dB(i,j)     (dB + (i) + (j)*lddb)
    #define dX(i,j)     (dX + (i) + (j)*lddx)
    #define dR(i,j)     (dR + (i) + (j)*lddr)
    // Constants
    const double BWDMAX    = 1.0;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double c_one     = MAGMA_D_ONE;
    const magma_int_t ione = 1;
    
    // Local variables
    magma_int_t lddasp, lddadp, lddr;
    magmaDouble_ptr dA_dprec=NULL, dR=NULL, dz=NULL;
    magmaFloat_ptr dA_sprec=NULL, dSX=NULL;
    double Xnrmv, Rnrmv;
    double          Anrm, Xnrm, Rnrm, cte, eps, work[1];
    magma_int_t     i, j, iiter, inner_iter=0, tot_inner_iter=0;
    float fp32_Anrm, fp32_cte, fp32_eps;
    magma_mp_type_t facto_base, facto_gem, facto_tc;
    real_Double_t start_time=0.0;
    
    #ifdef CHECKFOR_NAN_INF
    magma_int_t c_gpu_nan=-1, c_gpu_inf=-1;
    #endif

    //#define USE_GMRES_CPU
    #define USE_GMRES_GPU

    #if defined (USE_GMRES_GPU)
    double rnorm0;
    int niters, lddlud = ldda;
    double *d_LUd;
    #endif

    //#if defined (USE_GMRES_CPU)
    double *gmdwork, *gmhwork;
    magma_int_t irc[7], icntl[8]; 
    magma_int_t lwork;
    double cntl[5];
    double gmtol, innertol = 1e-4;
    const magma_int_t outer_iter_min = 10, inner_itermax = 50; //min(2100, n+100);
    magma_int_t ITERMAX, solver_outer_itermax;
    //#endif
    char algoname[40], solvername[20], factoname[20];
    char algoname_outer[80], algoname_inner[80];

    ITERMAX = min(n, outer_iter_min * inner_itermax);
    /* Check arguments */
    *iter = 0;
    *info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans )
        *info = -1;
    if ( n < 0 )
        *info = -2;
    else if ( nrhs < 0 )
        *info = -3;
    else if ( ldda < max(1,n))
        *info = -5;
    else if ( lddb < max(1,n))
        *info = -9;
    else if ( lddx < max(1,n))
        *info = -11;
    // check cuda version and if device has tensor cores in case tensor cores was requested to be used.
#if CUDA_VERSION < 9000
    if (facto_type != Magma_PREC_SS){
        *info = -14;
    }
#else
    magma_int_t arch = magma_getdevice_arch();
    if(arch < 530 && facto_type != Magma_PREC_SS){
        *info = -14;
    }
#endif    

    switch (facto_type) {
        case Magma_PREC_HS :
            snprintf(factoname, sizeof(factoname),"HS");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_HGEMM;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_HD :
            snprintf(factoname, sizeof(factoname),"HD");
            facto_base = Magma_MP_BASE_XHD;
            facto_gem  = Magma_MP_HGEMM;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_SS :
            snprintf(factoname, sizeof(factoname),"SS");
            facto_base = Magma_MP_BASE_SS;
            facto_gem  = Magma_MP_SGEMM;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_SHT :
            snprintf(factoname, sizeof(factoname),"SHT");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_GEMEX_I16_O32_C32;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XHS_H :
            snprintf(factoname, sizeof(factoname),"XHS_H");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_HGEMM;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_XHS_HTC :
            snprintf(factoname, sizeof(factoname),"XHS_HTC");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_HGEMM;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XHS_161616 :
            snprintf(factoname, sizeof(factoname),"XHS_666");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_GEMEX_I16_O16_C16;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_XHS_161616TC :
            snprintf(factoname, sizeof(factoname),"XHS_666TC");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_GEMEX_I16_O16_C16;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XHS_161632TC :
            snprintf(factoname, sizeof(factoname),"XHS_662TC");
            facto_base = Magma_MP_BASE_XHS;
            facto_gem  = Magma_MP_GEMEX_I16_O16_C32;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XSH_S :
            snprintf(factoname, sizeof(factoname),"XSH_S");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_SGEMM;
            facto_tc   = Magma_MP_ENABLE_DFLT_MATH;
            break;
        case Magma_PREC_XSH_STC :
            snprintf(factoname, sizeof(factoname),"XSH_STC");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_SGEMM;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XSH_163232TC :
            snprintf(factoname, sizeof(factoname),"XSH_622TC");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_GEMEX_I16_O32_C32;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        case Magma_PREC_XSH_323232TC :
            snprintf(factoname, sizeof(factoname),"XSH_222TC");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_GEMEX_I32_O32_C32;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            break;
        default:
            snprintf(factoname, sizeof(factoname),"unknown_facto");
            facto_base = Magma_MP_BASE_XSH;
            facto_gem  = Magma_MP_GEMEX_I16_O32_C32;
            facto_tc   = Magma_MP_ENABLE_TC_MATH;
            *info = -14;
    }
    switch (solver_type) {
        case Magma_REFINE_IRSTRS :
            snprintf(solvername, sizeof(solvername),"IRSTRS");
            solver_outer_itermax = ITERMAX;
            break;
        case Magma_REFINE_IRDTRS :
            snprintf(solvername, sizeof(solvername),"IRDTRS");
            solver_outer_itermax = ITERMAX;
            break;
        case Magma_REFINE_IRGMSTRS :
            snprintf(solvername, sizeof(solvername),"IRGMSTRS");
            solver_outer_itermax = max( outer_iter_min, ITERMAX/inner_itermax );
            break;
        case Magma_REFINE_IRGMDTRS :
            snprintf(solvername, sizeof(solvername),"IRGMDTRS");
            solver_outer_itermax = max( outer_iter_min, ITERMAX/inner_itermax );
            break;
        case Magma_REFINE_GMSTRS :
            snprintf(solvername, sizeof(solvername),"GMSTRS");
            solver_outer_itermax = ITERMAX;
            break;
        case Magma_REFINE_GMDTRS :
            snprintf(solvername, sizeof(solvername),"GMDTRS");
            solver_outer_itermax = ITERMAX;
            break;
        case Magma_REFINE_GMGMSTRS :
            snprintf(solvername, sizeof(solvername),"GMGMSTRS");
            solver_outer_itermax = max( outer_iter_min, ITERMAX/inner_itermax );
            break;
        case Magma_REFINE_GMGMDTRS :
            snprintf(solvername, sizeof(solvername),"GMGMDTRS");
            solver_outer_itermax = max( outer_iter_min, ITERMAX/inner_itermax );
            break;
        default: 
            snprintf(solvername, sizeof(solvername),"unknown_solver");
            solver_outer_itermax = ITERMAX;
            *info = -15;
    }
    //snprintf(algoname, sizeof(algoname),"%s_%s", factoname, solvername);
    snprintf(algoname, sizeof(algoname),"%lld_%s_%s", (long long) n, factoname, solvername);

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    if ( n == 0 || nrhs == 0 )
        return *info;


    lddasp = magma_roundup(n,32);
    lddadp = magma_roundup(n,32);
    lddr   = magma_roundup(n,32);

    
//    dA_sprec = dworks;
//    dSX = dA_sprec + lddasp*n;
    dSX = dworks;
    dR  = dworkd;
    
    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    //==============================
    // Factorize A
    //==============================
    if( facto_type == Magma_PREC_HD)
    {
        //allocate dA_dprec and use it. keep dA for residual computation
        if (MAGMA_SUCCESS != magma_dmalloc( &dA_dprec, lddadp*n )) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        magmablas_dlacpy( MagmaFull, n, n, dA, ldda, dA_dprec, lddadp, queue );
        // factor dA_dprec in double/half precision
        return MAGMA_ERR_NOT_SUPPORTED; //Azzam code not commited yet
        //magma_hdgetrf_gpu( n, n, dA_dprec, lddadp, ipiv, info );
    }
    else{
        //allocate dA_sprec and use it. keep dA for residual computation
        if (MAGMA_SUCCESS != magma_smalloc( &dA_sprec, lddasp*n )) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        magmablas_dlag2s( n, n, dA, ldda, dA_sprec, lddasp, queue, info );
        if (*info != 0) {
            *iter = -2;
            goto fallback;
        }

/*
        Azzam equilibrate or scale 
        magma_int_t ILO, IHI;
        float *hA_sprec, *SCALE;
        magma_smalloc_cpu( &hA_sprec,    ldda* n   );
        magma_smalloc_cpu( &SCALE,     n    );

        magma_sgetmatrix( n, n, dA_sprec, lddasp, hA_sprec, ldda, queue );
        lapackf77_sgebal("S", &n, hA_sprec, &ldda, &ILO, &IHI, SCALE, info );
        magma_ssetmatrix( n, n, hA_sprec, ldda, dA_sprec, lddasp, queue );
            printf("voici ILO %d IHI %d SCALE %e info %d\n",ILO, IHI, SCALE[0],*info);
*/


        start_time = magma_wtime();
        // factor dA_sprec in single precision
        if( facto_base == Magma_MP_BASE_SS)
        {
            magma_sgetrf_gpu( n, n, dA_sprec, lddasp, ipiv, info );
        }
        else if( facto_base == Magma_MP_BASE_XHS )
        {
            magma_xhsgetrf_gpu( n, n, dA_sprec, lddasp, ipiv, info, facto_tc, facto_gem);
        }
        else if( facto_base == Magma_MP_BASE_XSH )
        {
            magma_xshgetrf_gpu( n, n, dA_sprec, lddasp, ipiv, info, facto_tc, facto_gem);
        }
        *facto_time = magma_wtime() - start_time;
        
        if (*info != 0) {
            *iter = -3;
            MAGMA_PRINTF("dsgesv_gmres magma_xgetrf_gpu error info %lld fallback to FP64\n", (long long) *info);
            goto fallback;
        }
        // Azzam:insert the option if(solver_type == Magma_REFSLV_DTRS && facto_type != Magma_PREC_HD) here
    }

    if( facto_type != Magma_PREC_HD && 
            (   solver_type == Magma_REFINE_IRDTRS || solver_type == Magma_REFINE_IRGMDTRS
             || solver_type == Magma_REFINE_GMDTRS || solver_type == Magma_REFINE_GMGMDTRS) )
    {
        if (MAGMA_SUCCESS != magma_dmalloc( &dA_dprec, lddadp*n )) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        magmablas_slag2d( n, n, dA_sprec, lddasp, dA_dprec, lddadp, queue, info );
    }
    if( ( facto_type == Magma_PREC_HD )  
           && (   solver_type == Magma_REFINE_IRSTRS || solver_type == Magma_REFINE_IRGMSTRS
             || solver_type == Magma_REFINE_GMSTRS || solver_type == Magma_REFINE_GMGMSTRS) )
    {
        if (MAGMA_SUCCESS != magma_smalloc( &dA_sprec, lddasp*n )) {
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        magmablas_dlag2s( n, n, dA_dprec, lddadp, dA_sprec, lddasp, queue, info );
    }

    //==============================


    //==============================
    // Generate parallel pivots
    //==============================
    {
        magma_int_t *newipiv;
        magma_imalloc_cpu( &newipiv, n );
        if ( newipiv == NULL ) {
            *iter = -4;
            goto fallback;
        }
        magma_swp2pswp( trans, n, ipiv, newipiv );
        magma_isetvector( n, newipiv, 1, dipiv, 1, queue );
        magma_free_cpu( newipiv );
    }
    //==============================

    
#if 1
        eps  = lapackf77_dlamch("Epsilon");
        Anrm = magmablas_dlange( MagmaInfNorm, n, n, dA, ldda, (double*)dworkd, n*nrhs, queue );
        cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;
        gmtol = cte;
#else
        fp32_eps  = lapackf77_slamch("Epsilon");
        fp32_Anrm = magmablas_slange( MagmaInfNorm, n, n, dA_sprec, lddasp, (float*)dworkd, n*nrhs, queue );
        fp32_cte  = fp32_Anrm * fp32_eps * magma_ssqrt( (double) n ) ;
        gmtol = double(fp32_cte);
        cte   = double(fp32_cte);
#endif
        
    //==============================
    // Creating initial guess for FGMRes 
    // or the initial solution for IR
    //==============================
    // solve dA_sprec*dSX = dB in single precision
    // converts dB to dSX and applies pivots, solves, then converts result back to dX
    if( solver_type == Magma_REFINE_IRSTRS || solver_type == Magma_REFINE_IRGMSTRS)
    {
        magma_dsgetrs_gpu( trans, n, nrhs, dA_sprec, lddasp, dipiv, dB, lddb, dX, lddx, dSX, info );
    }
    else if( solver_type == Magma_REFINE_IRDTRS || solver_type == Magma_REFINE_IRGMDTRS)
    {
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );
        magma_dgetrs_gpu( trans, n, nrhs, dA_dprec, lddadp, ipiv, dX, lddx, info );
    }
    //==============================

//#define ajeter
#ifdef ajeter
    if(    solver_type == Magma_REFINE_GMSTRS || solver_type == Magma_REFINE_GMDTRS 
        || solver_type == Magma_REFINE_GMGMSTRS || solver_type == Magma_REFINE_GMGMDTRS    )
    {

        if( solver_type == Magma_REFINE_GMSTRS || solver_type == Magma_REFINE_GMGMSTRS)
        {
            magma_dsgetrs_gpu( trans, n, nrhs, dA_sprec, lddasp, dipiv, dB, lddb, dX, lddx, dSX, info );
        }
        else if( solver_type == Magma_REFINE_GMDTRS || solver_type == Magma_REFINE_GMGMDTRS)
        {
            magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );
            magma_dgetrs_gpu( trans, n, nrhs, dA_dprec, lddadp, ipiv, dX, lddx, info );
        }

        eps  = lapackf77_dlamch("Epsilon");
        Anrm = magmablas_dlange( MagmaInfNorm, n, n, dA, ldda, (double*)dworkd, n*nrhs, queue );
        cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;

        // residual dR = dB - dA*dX in double precision
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dR, lddr, queue );
            magma_dgemv( trans, n, n,
                    c_neg_one, dA, ldda,
                    dX, 1,
                    c_one,     dR, 1, queue );

        // TODO: use MAGMA_D_ABS( dX(i,j) ) instead of dlange?
        for( j=0; j < nrhs; j++ ) {
            i = magma_idamax( n, dX(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
            Xnrm = lapackf77_dlange( "F", &ione, &ione, &Xnrmv, &ione, work );

            i = magma_idamax( n, dR(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );

            Rnrm = MAGMA_D_ABS(Rnrmv);
            //Rnrm = lapackf77_dlange( "F", &ione, &ione, &Rnrmv, &ione, work );
            MAGMA_PRINTF("\n    CPU_GMRES_ITER   %lld  %10.5e  %10.5e  \n", (long long) 0, Rnrm, Rnrm );
        }
    }
#endif

    //==============================
    //   FGMRes 
    //==============================
    if(   solver_type == Magma_REFINE_GMSTRS   || solver_type == Magma_REFINE_GMDTRS
       || solver_type == Magma_REFINE_GMGMSTRS || solver_type == Magma_REFINE_GMGMDTRS )
    {
        #if defined (USE_GMRES_GPU)
            // FGMRES  GPU need that X is initialized to "0"
            magmablas_dlaset(MagmaFull, n, nrhs, 0.0, 0.0, dX, lddx, queue); 
            magma_dfgmres_plu_gpu(trans, n, nrhs, dA, ldda, 
                    dA_sprec, lddasp, dA_dprec, lddadp,
                    ipiv, dipiv, dB, lddb, dX, lddx, dSX, 
                    solver_outer_itermax, solver_outer_itermax, 
                    inner_itermax, inner_itermax,
                    0, gmtol, innertol, &rnorm0, iter,
                    solver_type, algoname, 0, queue);
        #elif defined (USE_GMRES_CPU)
            #if 0
            magma_dgmres_cpu_init(n, &gmdwork, &gmhwork, &lwork, 
                    solver_outer_itermax, solver_outer_itermax, 0, cntl, gmtol,
                    irc, icntl);
           
            *iter = magma_dgmres_cpu_solve(trans, n, nrhs, dA, ldda, 
                    dA_sprec, lddasp, dA_dprec, lddadp,
                    ipiv, dipiv, dB, lddb, dX, lddx, dSX, 
                    gmdwork, gmhwork, lwork,
                    solver_outer_itermax, solver_outer_itermax, 0, cntl, gmtol, innertol,
                    irc, icntl, facto_type, solver_type, algoname, queue);
           
            magma_dgmres_cpu_destroy(gmdwork, gmhwork);
            #else
            *iter = magma_dgmres_cpu(trans, n, nrhs, dA, ldda, 
                    dA_sprec, lddasp, dA_dprec, lddadp,
                    ipiv, dipiv, dB, lddb, dX, lddx, dSX, 
                    solver_outer_itermax, solver_outer_itermax, 0, gmtol, innertol,
                    facto_type, solver_type, algoname, queue);
            #endif
        #endif
        goto cleanup;
    }
    //==============================
    // ITERATIVE REFINEMENT using 
    // IR or FGMRES to solve Ax = b 
    //==============================
    else if (   solver_type == Magma_REFINE_IRSTRS   || solver_type == Magma_REFINE_IRDTRS
       || solver_type == Magma_REFINE_IRGMSTRS || solver_type == Magma_REFINE_IRGMDTRS )
    {
        //eps  = lapackf77_dlamch("Epsilon");
        //Anrm = magmablas_dlange( MagmaInfNorm, n, n, dA, ldda, (double*)dworkd, n*nrhs, queue );
        //cte  = Anrm * eps * magma_dsqrt( (double) n ) * BWDMAX;

        #ifdef CHECKFOR_NAN_INF
        magma_dnan_inf_gpu( MagmaFull, n, 1, dX, n, &c_gpu_nan, &c_gpu_inf, queue );
        printf("from inside dsgesv @ iter %2d here is c_gpu_nan %d c_gpu_inf %d\n",0,
                (int)c_gpu_nan, (int)c_gpu_inf);
        #endif

        // residual dR = dB - dA*dX in double precision
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dR, lddr, queue );
        if ( nrhs == 1 ) {
            magma_dgemv( trans, n, n,
                    c_neg_one, dA, ldda,
                    dX, 1,
                    c_one,     dR, 1, queue );
        }
        else {
            magma_dgemm( trans, MagmaNoTrans, n, nrhs, n,
                    c_neg_one, dA, ldda,
                    dX, lddx,
                    c_one,     dR, lddr, queue );
        }

        // TODO: use MAGMA_D_ABS( dX(i,j) ) instead of dlange?
        for( j=0; j < nrhs; j++ ) {
            i = magma_idamax( n, dX(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
            Xnrm = lapackf77_dlange( "F", &ione, &ione, &Xnrmv, &ione, work );

            i = magma_idamax( n, dR(0,j), 1, queue ) - 1;
            magma_dgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );

            Rnrm = MAGMA_D_ABS(Rnrmv);
            //Rnrm = lapackf77_dlange( "F", &ione, &ione, &Rnrmv, &ione, work );
            MAGMA_PRINTF("%s_IR_residual @ iter %lld using inner_iter %lld is Rnrm %10.5e    Xnrm*cte % 10.5e  Xnrm %10.5e   cte %10.5e  \n", 
                    algoname, (long long) 1, (long long) 0, Rnrm, Xnrm*cte, Xnrm, cte );

            if ( Rnrm >  Xnrm*cte ) {
                goto refinement;
            }
        }

        *iter = 0;
        goto cleanup;
        //return *info;

refinement:
        if ( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS)
        {
            #if defined (USE_GMRES_CPU)
            magma_dgmres_inner_cpu_init(n, &gmdwork, &gmhwork, &lwork, 
                    inner_itermax, inner_itermax, 0, cntl, innertol,
                    irc, icntl);
            #else
            if (MAGMA_SUCCESS != magma_dmalloc( &dz, nrhs*lddr )) {
                return MAGMA_ERR_DEVICE_ALLOC;
            }
            #endif
        }
        *info = 0;
        for( iiter=1; iiter <= solver_outer_itermax; ) {
            // convert residual dR to single precision dSX
            // solve dA_sprec*dSX = R in single precision
            // convert result back to double precision dR
            // it's okay that dR is used for both dB input and dX output.
            if( solver_type == Magma_REFINE_IRSTRS )
            {
                magma_dsgetrs_gpu( trans, n, nrhs, dA_sprec, lddasp, dipiv, dR, lddr, dR, lddr, dSX, info );
            }
            else if( solver_type == Magma_REFINE_IRDTRS )
            {
                magma_dgetrs_gpu( trans, n, nrhs, dA_dprec, lddadp, ipiv, dR, lddr, info );
            }
            else if( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS)
            {
                inner_iter = 0;       
                #if defined (USE_GMRES_GPU)
                magmablas_dlaset(MagmaFull, n, nrhs, 0.0, 0.0, dz, lddr, queue); 
                //inner_iter =
                    magma_dfgmres_plu_gpu(trans, n, nrhs, dA, ldda, 
                        dA_sprec, lddasp, dA_dprec, lddadp,
                        ipiv, dipiv, dR, lddr, dz, lddr, dSX, 
                        inner_itermax, inner_itermax, inner_itermax, inner_itermax,
                        0, innertol, innertol, &rnorm0, &inner_iter,
                        solver_type, algoname, 1, queue);
                magmablas_dlacpy( MagmaFull, n, nrhs, dz, lddr, dR, lddr, queue );
                #elif defined (USE_GMRES_CPU)
                inner_iter = magma_dgmres_inner_cpu_solve(trans, n, nrhs, dA, ldda, 
                        dA_sprec, lddasp, dA_dprec, lddadp,
                        ipiv, dipiv, dR, lddr, dR, lddr, dSX, 
                        gmdwork, gmhwork, lwork,
                        inner_itermax, inner_itermax, 0, cntl, innertol,
                        irc, icntl, solver_type, queue);
                #endif
                tot_inner_iter += inner_iter;
            }

#ifdef CHECKFOR_NAN_INF
            magma_dnan_inf_gpu( MagmaFull, n, 1, dX, n, &c_gpu_nan, &c_gpu_inf, queue );
            printf("from inside dsgesv @ iter %2d here is c_gpu_nan %d c_gpu_inf %d\n",(int)iiter,(int)c_gpu_nan, (int)c_gpu_inf);
#endif
            if (*info != 0) {
                *iter = -5;
                goto cleanup;
                goto fallback;
            }

            // Add correction and setup residual
            // dX += dR  --and--
            // dR = dB
            // This saves going through dR a second time (if done with one more kernel).
            // -- not really: first time is read, second time is write.
            for( j=0; j < nrhs; j++ ) {
                magmablas_daxpycp( n, dR(0,j), dX(0,j), dB(0,j), queue );
            }

            // residual dR = dB - dA*dX in double precision
            if ( nrhs == 1 ) {
                magma_dgemv( trans, n, n,
                        c_neg_one, dA, ldda,
                        dX, 1,
                        c_one,     dR, 1, queue );
            }
            else {
                magma_dgemm( trans, MagmaNoTrans, n, nrhs, n,
                        c_neg_one, dA, ldda,
                        dX, lddx,
                        c_one,     dR, lddr, queue );
            }

            // TODO: use MAGMA_D_ABS( dX(i,j) ) instead of dlange?
            /*  Check whether the nrhs normwise backward errors satisfy the
             *  stopping criterion. If yes, set ITER=IITER > 0 and return. */
            for( j=0; j < nrhs; j++ ) {
                i = magma_idamax( n, dX(0,j), 1, queue ) - 1;
                magma_dgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1, queue );
                Xnrm = lapackf77_dlange( "F", &ione, &ione, &Xnrmv, &ione, work );

                i = magma_idamax( n, dR(0,j), 1, queue ) - 1;
                magma_dgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1, queue );
                //Rnrm = lapackf77_dlange( "F", &ione, &ione, &Rnrmv, &ione, work );
                Rnrm = MAGMA_D_ABS(Rnrmv);

                MAGMA_PRINTF("%s_IR_residual @ iter %lld using inner_iter %lld is Rnrm %10.5e    Xnrm*cte % 10.5e  Xnrm %10.5e   cte %10.5e  \n", algoname, (long long) iiter+1, (long long) inner_iter, Rnrm, Xnrm*cte, Xnrm, cte );
                if ( Rnrm >  Xnrm*cte ) {
                    goto L20;
                }
            }

            /*  If we are here, the nrhs normwise backward errors satisfy
             *  the stopping criterion, we are good to exit. */
            if ( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS)
            {
                #if defined (USE_GMRES_CPU)
                magma_dgmres_inner_cpu_destroy(gmdwork, gmhwork);
                #else
                magma_free( dz );
                #endif
            }

            snprintf(algoname_outer, sizeof(algoname_outer),"%s_outer_niter",algoname);
            snprintf(algoname_inner, sizeof(algoname_inner),"%s_inner_niter",algoname);
            MAGMA_PRINTF("\n\ninfo===> %s %lld  %s is %lld \n\n", algoname_outer, (long long) iiter+1, algoname_inner, (long long) tot_inner_iter); 
            
            //*iter = iiter;
            *iter = ( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS) ? tot_inner_iter : iiter; 
            goto cleanup;
            //return *info;
L20:
            iiter++;
        }

        if ( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS)
        {
            #if defined (USE_GMRES_CPU)
            magma_dgmres_inner_cpu_destroy(gmdwork, gmhwork);
            #else
            magma_free( dz );
            #endif
        }
        /* If we are at this place of the code, this is because we have
         * performed ITER=solver_outer_itermax iterations and never satisified the
         * stopping criterion. Set up the ITER flag accordingly and follow
         * up on double precision routine. */
        *iter = ( solver_type == Magma_REFINE_IRGMSTRS   || solver_type == Magma_REFINE_IRGMDTRS) ? -tot_inner_iter : -iiter+1;
        snprintf(algoname_outer, sizeof(algoname_outer),"%s_outer_niter",algoname);
        snprintf(algoname_inner, sizeof(algoname_inner),"%s_inner_niter",algoname);
        MAGMA_PRINTF("\n\ninfo===> %s %lld  %s is %lld \n\n", algoname_outer, (long long) iiter, algoname_inner, (long long) tot_inner_iter); 
    }

fallback:
    /* Single-precision iterative refinement failed to converge to a
     * satisfactory solution, so we resort to double precision. */
    //printf("FALLBACK DISABLED FOR TESTING\n");
    //goto cleanup;
    magma_dgetrf_gpu( n, n, dA, ldda, ipiv, info );
    if (*info == 0) {
        magmablas_dlacpy( MagmaFull, n, nrhs, dB, lddb, dX, lddx, queue );
        magma_dgetrs_gpu( trans, n, nrhs, dA, ldda, ipiv, dX, lddx, info );
    }

    MAGMA_UNUSED( fp32_Anrm );
    MAGMA_UNUSED( fp32_cte );
    MAGMA_UNUSED( fp32_eps );
    MAGMA_UNUSED( niters );
    MAGMA_UNUSED( lddlud );
    MAGMA_UNUSED( d_LUd );
    MAGMA_UNUSED( gmdwork );
    MAGMA_UNUSED( gmhwork );
    MAGMA_UNUSED( irc );
    MAGMA_UNUSED( icntl );
    MAGMA_UNUSED( lwork );
    MAGMA_UNUSED( cntl );

cleanup:
    magma_queue_destroy( queue );
    magma_free( dA_dprec );
    magma_free( dA_sprec );

    return *info;
}


/***************************************************************************//**
    Purpose
    -------
    DSGESV computes the solution to a real system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.

    DSGESV first attempts to factorize the matrix in real SINGLE PRECISION
    and use this factorization within an iterative refinement procedure 
    to produce a solution with real DOUBLE PRECISION norm-wise backward error 
    quality (see below). If the approach fails the method switches to a
    real DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio real SINGLE PRECISION performance over real DOUBLE PRECISION
    performance is too small. A reasonable strategy should take the 
    number of right-hand sides and the size of the matrix into account. 
    This might be done with a call to ILAENV in the future. Up to now, we 
    always try iterative refinement.
    
    The iterative refinement process is stopped if
        ITER > solver_outer_itermax
    or for all the RHS we have:
        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
    where
        o ITER is the number of the current iteration in the iterative
          refinement process
        o RNRM is the infinity-norm of the residual
        o XNRM is the infinity-norm of the solution
        o ANRM is the infinity-operator-norm of the matrix A
        o EPS is the machine epsilon returned by DLAMCH('Epsilon')
    The value solver_outer_itermax and BWDMAX are fixed to 30 and 1.0D+00 respectively.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (ldda,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (info.EQ.0 and ITER.GE.0, see description below), A is
            unchanged. If double precision factorization has been used
            (info.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  ldda >= max(1,N).

    @param[out]
    ipiv    INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).
            Corresponds either to the single precision factorization
            (if info.EQ.0 and ITER.GE.0) or the double precision
            factorization (if info.EQ.0 and ITER.LT.0).

    @param[out]
    dipiv   INTEGER array on the GPU, dimension (N)
            The pivot indices; for 1 <= i <= N, after permuting, row i of the
            matrix was moved to row dIPIV(i).
            Note this is different than IPIV, where interchanges
            are applied one-after-another.

    @param[in]
    dB      DOUBLE PRECISION array on the GPU, dimension (lddb,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  lddb >= max(1,N).

    @param[out]
    dX      DOUBLE PRECISION array on the GPU, dimension (lddx,NRHS)
            If info = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  lddx >= max(1,N).

    @param
    dworkd  (workspace) DOUBLE PRECISION array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    @param
    dworks  (workspace) SINGLE PRECISION array on the GPU, dimension (N*(N+NRHS))
            This array is used to store the real single precision matrix
            and the right-hand sides or solutions in single precision.

    @param[out]
    iter    INTEGER
      -     < 0: iterative refinement has failed, double precision
                 factorization has been performed
        +        -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
        +        -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
        +        -3 : failure of SGETRF
        +        -31: stop the iterative refinement after the 30th iteration
      -     > 0: iterative refinement has been successfully used.
                 Returns the number of iterations
 
    @param[out]
    info   INTEGER
      -     = 0:  successful exit
      -     < 0:  if info = -i, the i-th argument had an illegal value
      -     > 0:  if info = i, U(i,i) computed in DOUBLE PRECISION is
                  exactly zero.  The factorization has been completed,
                  but the factor U is exactly singular, so the solution
                  could not be computed.

    @ingroup magma_gesv
*******************************************************************************/
magma_int_t
magma_dsgesv_iteref_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_int_t *iter,
    magma_int_t *info)
{
    real_Double_t facto_time;
    magma_dxgesv_gmres_gpu(
            trans, n, nrhs,
            dA, ldda, 
            ipiv, dipiv, 
            dB, lddb,
            dX, lddx,
            dworkd, dworks,
            Magma_PREC_SS,
            Magma_REFINE_GMSTRS,            
            iter, info, &facto_time);
    return *info;
}

/***************************************************************************//**
    Purpose
    -------
    DHGESV computes the solution to a real system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.

    DHGESV first attempts to factorize the matrix in real HALF PRECISION
    and use this factorization within an iterative refinement procedure 
    to produce a solution with real DOUBLE PRECISION norm-wise backward error 
    quality (see below). If the approach fails the method switches to a
    real DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio real HALF PRECISION performance over real DOUBLE PRECISION
    performance is too small. A reasonable strategy should take the 
    number of right-hand sides and the size of the matrix into account. 
    This might be done with a call to ILAENV in the future. Up to now, we 
    always try iterative refinement.
    
    The iterative refinement process is stopped if
        ITER > solver_outer_itermax
    or for all the RHS we have:
        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
    where
        o ITER is the number of the current iteration in the iterative
          refinement process
        o RNRM is the infinity-norm of the residual
        o XNRM is the infinity-norm of the solution
        o ANRM is the infinity-operator-norm of the matrix A
        o EPS is the machine epsilon returned by DLAMCH('Epsilon')
    The value solver_outer_itermax and BWDMAX are fixed to 30 and 1.0D+00 respectively.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A    * X = B  (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (ldda,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (info.EQ.0 and ITER.GE.0, see description below), A is
            unchanged. If double precision factorization has been used
            (info.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  ldda >= max(1,N).

    @param[out]
    ipiv    INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).
            Corresponds either to the single precision factorization
            (if info.EQ.0 and ITER.GE.0) or the double precision
            factorization (if info.EQ.0 and ITER.LT.0).

    @param[out]
    dipiv   INTEGER array on the GPU, dimension (N)
            The pivot indices; for 1 <= i <= N, after permuting, row i of the
            matrix was moved to row dIPIV(i).
            Note this is different than IPIV, where interchanges
            are applied one-after-another.

    @param[in]
    dB      DOUBLE PRECISION array on the GPU, dimension (lddb,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  lddb >= max(1,N).

    @param[out]
    dX      DOUBLE PRECISION array on the GPU, dimension (lddx,NRHS)
            If info = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  lddx >= max(1,N).

    @param
    dworkd  (workspace) DOUBLE PRECISION array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    @param
    dworks  (workspace) SINGLE PRECISION array on the GPU, dimension (N*(N+NRHS))
            This array is used to store the real single precision matrix
            and the right-hand sides or solutions in single precision.

    @param[out]
    iter    INTEGER
      -     < 0: iterative refinement has failed, double precision
                 factorization has been performed
        +        -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
        +        -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
        +        -3 : failure of SGETRF
        +        -31: stop the iterative refinement after the 30th iteration
      -     > 0: iterative refinement has been successfully used.
                 Returns the number of iterations
 
    @param[out]
    info   INTEGER
      -     = 0:  successful exit
      -     < 0:  if info = -i, the i-th argument had an illegal value
      -     > 0:  if info = i, U(i,i) computed in DOUBLE PRECISION is
                  exactly zero.  The factorization has been completed,
                  but the factor U is exactly singular, so the solution
                  could not be computed.

    @ingroup magma_gesv
*******************************************************************************/
magma_int_t
magma_dhgesv_iteref_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd, magmaFloat_ptr dworks,
    magma_int_t *iter,
    magma_int_t *info)
{
#if CUDA_VERSION >= 9000
    magma_int_t arch = magma_getdevice_arch();
    if(arch < 530){
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    real_Double_t facto_time;
    magma_dxgesv_gmres_gpu(
            trans, n, nrhs,
            dA, ldda, 
            ipiv, dipiv, 
            dB, lddb,
            dX, lddx,
            dworkd, dworks,
            Magma_PREC_SHT,
            Magma_REFINE_GMSTRS,            
            iter, info, &facto_time);
    return *info;
#else
    return MAGMA_ERR_NOT_SUPPORTED;
#endif
}
