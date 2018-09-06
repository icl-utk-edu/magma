/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

/**
    Purpose
    -------

    Prints information about a previously called solver.

    Arguments
    ---------

    @param[in]
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsolverinfo(
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    if( solver_par->verbose > 0 ){
        magma_int_t k = solver_par->verbose;
        printf("%%=================================================================================%%\n");
        switch( solver_par->solver ) {
            case Magma_CG:
                printf("%%   CG performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PCG:
                printf("%%   CG performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BICG:
            case Magma_BICGMERGE:
            case Magma_PBICG:
            case Magma_PBICGMERGE:
                printf("%%   BiCG performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_CGMERGE:
                printf("%%   CG (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BICGSTAB:
                printf("%%   BiCGSTAB performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PBICGSTAB:
                printf("%%   BiCGSTAB performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BICGSTABMERGE:
                printf("%%   BiCGSTAB (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BICGSTABMERGE2:
                printf("%%   BiCGSTAB (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_GMRES:
                printf("%%   GMRES(%lld) performance analysis every %lld iterations\n",
                        (long long) solver_par->restart, (long long) k );
                break;
            case Magma_PGMRES:
                printf("%%   GMRES(%lld) performance analysis every %lld iterations\n",
                        (long long) solver_par->restart, (long long) k );
                break;
            case Magma_IDR:
            case Magma_IDRMERGE:
                printf("%%   IDR(%lld) performance analysis every %lld iterations\n",
                        (long long) solver_par->restart, (long long) k );
                break;
            case Magma_PIDR:
            case Magma_PIDRMERGE:
                printf("%%   IDR(%lld) performance analysis every %lld iterations\n",
                        (long long) solver_par->restart, (long long) k );
                break;
            case Magma_ITERREF:
                printf("%%   Iterative refinement performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_JACOBI:
                printf("%%  Jacobi performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_CGS:
                printf("%%  CGS performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PCGS:
                printf("%%  CGS performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_CGSMERGE:
                printf("%%  CGS (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PCGSMERGE:
                printf("%%  CGS (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_TFQMR:
                printf("%%  TFQMR performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PCGMERGE:
                printf("%%  PCG performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PTFQMR:
                printf("%%  TFQMR performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_TFQMRMERGE:
                printf("%%  TFQMR (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PTFQMRMERGE:
                printf("%%  PTFQMR (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_QMR:
                printf("%%  QMR performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_LSQR:
                printf("%%  LSQR performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PQMR:
                printf("%%  PQMR performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_QMRMERGE:
                printf("%%  QMR (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PQMRMERGE:
                printf("%%  PQMR (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BOMBARD:
                printf("%%  BOMBARD performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BOMBARDMERGE:
                printf("%%  BOMBARD (merged) performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_BAITER:
            case Magma_BAITERO:
                printf("%%  BAITER performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            case Magma_PARDISO:
                printf("%%  PARDISO performance analysis every %lld iterations\n",
                        (long long) k );
                break;
            default:
                printf("%%  Detailed performance analysis not supported.\n");
                break;
        }
        
        switch( precond_par->solver ) {
            case Magma_CG:
                printf("%%   Preconditioner used: CG.\n");
                break;
            case Magma_BICGMERGE:
            case Magma_BICG:
                printf("%%   Preconditioner used: BiCG.\n");
                break;
            case Magma_BICGSTAB:
                printf("%%   Preconditioner used: BiCGSTAB.\n");
                break;
            case Magma_GMRES:
                printf("%%   Preconditioner used: GMRES.\n");
                break;
            case Magma_CGS:
                printf("%%   Preconditioner used: CGS.\n");
                break;
            case Magma_BOMBARD:
                printf("%%   Preconditioner used: BOMBARD.\n");
                break;
            case Magma_TFQMR:
                printf("%%   Preconditioner used: TFQMR.\n");
                break;
            case Magma_QMR:
                printf("%%   Preconditioner used: QMR.\n");
                break;
            case Magma_JACOBI:
                printf("%%   Preconditioner used: Jacobi.\n");
                break;
            case Magma_IDR:
            case Magma_IDRMERGE:
                printf("%%   Preconditioner used: Jacobi.\n");
                break;
            case Magma_BAITER:
            case Magma_BAITERO:
                printf("%%   Preconditioner used: Block-asynchronous iteration.\n");
                break;
            case Magma_ILU:
                printf("%%   Preconditioner used: ILU(%lld).\n",
                        (long long) precond_par->levels );
                break;
            case Magma_PARILU:
                printf("%%   Preconditioner used: ParILU(%lld).\n",
                        (long long) precond_par->levels );
                break;
            case Magma_ICC:
                printf("%%   Preconditioner used: IC(%lld).\n",
                        (long long) precond_par->levels );
                break;
            case Magma_PARIC:
                printf("%%   Preconditioner used: ParIC(%lld).\n",
                        (long long) precond_par->levels );
                break;
            case Magma_ISAI:
                printf("%%   Preconditioner used: ParILU-SPAI.\n" );
                break;
            default:
                break;
        }
        
        printf("%%=================================================================================%%\n");
        switch( solver_par->solver ) {
            case Magma_CG:
            case Magma_PCG:
            case Magma_CGMERGE:
            case Magma_BICGSTAB:
            case Magma_PBICGSTAB:
            case Magma_BICGSTABMERGE:
            case Magma_BICGSTABMERGE2:
            case Magma_GMRES:
            case Magma_PGMRES:
            case Magma_IDR:
            case Magma_IDRMERGE:
            case Magma_PIDR:
            case Magma_PIDRMERGE:
            case Magma_CGS:
            case Magma_BICG:
            case Magma_BICGMERGE:
            case Magma_PBICG:
            case Magma_PBICGMERGE:
            case Magma_PCGS:
            case Magma_PCGMERGE:
            case Magma_CGSMERGE:
            case Magma_PCGSMERGE:
            case Magma_QMR:
            case Magma_QMRMERGE:
            case Magma_LSQR:
            case Magma_PQMR:
            case Magma_PQMRMERGE:
            case Magma_TFQMR:
            case Magma_PTFQMR:
            case Magma_TFQMRMERGE:
            case Magma_PTFQMRMERGE:
            case Magma_ITERREF:
            case Magma_BOMBARD:
            case Magma_BOMBARDMERGE:
            case Magma_JACOBI:
            case Magma_BAITER:
            case Magma_BAITERO:
                printf("%%   iter   ||   residual-nrm2    ||   runtime    ||   SpMV-count*  ||   info\n");
                printf("%%=================================================================================%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf(" %8lld       %e          %f         %8lld          %3lld\n",
                           (long long)(j*k),
                           solver_par->res_vec[j],
                           solver_par->timing[j],
                           (long long) (magma_ceildiv(solver_par->spmv_count, solver_par->numiter)*(j*k)),
                           (long long) solver_par->info );
                }
                printf("%%=================================================================================%%\n");
                break;
            default:
                printf("%%=================================================================================%%\n");
                break;
        }
    }
    else{
        printf("%%   iter   ||   residual-nrm2    ||   runtime    ||   SpMV-count   ||   info\n");
        printf("%%=================================================================================%%\n");
        printf(" %8lld       %e          %f         %8lld          %3lld\n",
               (long long) solver_par->numiter,
               solver_par->iter_res,
               solver_par->runtime,
               (long long) solver_par->spmv_count,
               (long long) solver_par->info );
        printf("%%=================================================================================%%\n");
    }
                
    printf("\n%%=================================================================================%%\n");
    switch( solver_par->solver ) {
        case Magma_CG:
            printf("%% CG solver summary:\n");
            break;
        case Magma_PCG:
        case Magma_PCGMERGE:
            printf("%% PCG solver summary:\n");
            break;
        case Magma_CGMERGE:
            printf("%% CG solver summary:\n");
            break;
        case Magma_BICGSTAB:
            printf("%% BiCGSTAB solver summary:\n");
            break;
        case Magma_PBICGSTAB:
            printf("%% PBiCGSTAB solver summary:\n");
            break;
        case Magma_BICGSTABMERGE:
            printf("%% BiCGSTAB solver summary:\n");
            break;
        case Magma_BICGSTABMERGE2:
            printf("%% BiCGSTAB solver summary:\n");
            break;
        case Magma_BICG:
        case Magma_BICGMERGE:
            printf("%% BiCG solver summary:\n");
            break;
        case Magma_PBICG:
        case Magma_PBICGMERGE:
            printf("%% BiCG solver summary:\n");
            break;
        case Magma_GMRES:
            printf("%% GMRES(%lld) solver summary:\n",
                    (long long) solver_par->restart );
            break;
        case Magma_PGMRES:
            printf("%% PGMRES(%lld) solver summary:\n",
                    (long long) solver_par->restart );
            break;
        case Magma_IDR:
        case Magma_IDRMERGE:
            printf("%% IDR(%lld) solver summary:\n",
                    (long long) solver_par->restart );
            break;
        case Magma_PIDR:
        case Magma_PIDRMERGE:
            printf("%% PIDR(%lld) solver summary:\n",
                    (long long) solver_par->restart );
            break;
        case Magma_CGS:
        case Magma_CGSMERGE:
            printf("%% CGS solver summary:\n");
            break;
        case Magma_PCGS:
        case Magma_PCGSMERGE:
            printf("%% PCGS solver summary:\n");
            break;
        case Magma_TFQMR:
        case Magma_TFQMRMERGE:
            printf("%% TFQMR solver summary:\n");
            break;
        case Magma_PTFQMR:
        case Magma_PTFQMRMERGE:
            printf("%% PTFQMR solver summary:\n");
            break;
        case Magma_QMR:
        case Magma_QMRMERGE:
            printf("%% QMR solver summary:\n");
            break;
        case Magma_LSQR:
            printf("%% LSQR solver summary:\n");
            break;
        case Magma_PQMR:
        case Magma_PQMRMERGE:
            printf("%% PQMR solver summary:\n");
            break;
        case Magma_ITERREF:
            printf("%% Iterative refinement solver summary:\n");
            break;
        case Magma_JACOBI:
            printf("%% Jacobi solver summary:\n");
            break;
        case Magma_BAITER:
        case Magma_BAITERO:
            printf("%% Block-asynchronous iteration solver summary:\n");
            break;
        case Magma_LOBPCG:
            printf("%% LOBPCG iteration solver summary:\n");
            break;
        case Magma_BOMBARD:
        case Magma_BOMBARDMERGE:
            printf("%% multi-solver iteration summary:\n");
            break;
        case Magma_PARDISO:
            printf("%% PARDISO solver summary:\n");
            break;
        default:
            printf("%%   Solver info not supported.\n");
            goto cleanup;
    }
    printf("%%    initial residual: %e\n", solver_par->init_res );
    printf("%%    preconditioner setup: %.4f sec\n", precond_par->setuptime );
    printf("%%    iterations: %4lld\n", (long long) solver_par->numiter );
    printf("%%    SpMV-count: %4lld\n", (long long) solver_par->spmv_count );
    printf("%%    exact final residual: %e\n"
           "%%    runtime: %.4f sec\n",
            solver_par->final_res, solver_par->runtime);
    printf("%%    preconditioner runtime: %.4f sec\n", precond_par->runtime );
cleanup:
    printf("%%=================================================================================%%\n");
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Frees any memory assocoiated with the verbose mode of solver_par. The
    other values are set to default.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsolverinfo_free(
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    solver_par->init_res = 0.0;
    solver_par->iter_res = 0.0;
    solver_par->final_res = 0.0;

    if ( solver_par->res_vec != NULL ) {
        magma_free_cpu( solver_par->res_vec );
        solver_par->res_vec = NULL;
    }
    if ( solver_par->timing != NULL ) {
        magma_free_cpu( solver_par->timing );
        solver_par->timing = NULL;
    }
    if ( solver_par->eigenvectors != NULL ) {
        magma_free( solver_par->eigenvectors );
        solver_par->eigenvectors = NULL;
    }
    if ( solver_par->eigenvalues != NULL ) {
        magma_free_cpu( solver_par->eigenvalues );
        solver_par->eigenvalues = NULL;
    }
    
    magma_zprecondfree( precond_par, queue );
    
    return MAGMA_SUCCESS;
}

/**
    Purpose
    -------

    Initializes all solver and preconditioner parameters.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zsolverinfo_init(
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    solver_par->runtime         = 0.;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    precond_par->numiter = 0;
    precond_par->spmv_count = 0;
    precond_par->runtime       = 0.;
    precond_par->setuptime  = 0.;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;
    solver_par->eigenvectors = NULL;
    solver_par->eigenvalues = NULL;

    if( solver_par->maxiter == 0 )
        solver_par->maxiter = 1000;
    if( solver_par->version == 0 )
        solver_par->version = 0;
    if( solver_par->restart == 0 )
        solver_par->restart = 30;
    if( solver_par->solver == 0 )
        solver_par->solver = Magma_CG;

    if ( solver_par->verbose > 0 ) {
        CHECK( magma_malloc_cpu( (void **)&solver_par->res_vec, sizeof(real_Double_t)
                * ( (solver_par->maxiter)/(solver_par->verbose)+1) ));
        CHECK( magma_malloc_cpu( (void **)&solver_par->timing, sizeof(real_Double_t)
                *( (solver_par->maxiter)/(solver_par->verbose)+1) ));
    } else {
        solver_par->res_vec = NULL;
        solver_par->timing = NULL;
    }

    precond_par->d.val = NULL;
    precond_par->d2.val = NULL;
    precond_par->work1.val = NULL;
    precond_par->work2.val = NULL;

    precond_par->M.val = NULL;
    precond_par->M.col = NULL;
    precond_par->M.row = NULL;
    precond_par->M.blockinfo = NULL;

    precond_par->L.val = NULL;
    precond_par->L.col = NULL;
    precond_par->L.row = NULL;
    precond_par->L.blockinfo = NULL;

    precond_par->U.val = NULL;
    precond_par->U.col = NULL;
    precond_par->U.row = NULL;
    precond_par->U.blockinfo = NULL;
    
    precond_par->LT.val = NULL;
    precond_par->LT.col = NULL;
    precond_par->LT.row = NULL;
    precond_par->LT.blockinfo = NULL;

    precond_par->UT.val = NULL;
    precond_par->UT.col = NULL;
    precond_par->UT.row = NULL;
    precond_par->UT.blockinfo = NULL;

    precond_par->LD.val = NULL;
    precond_par->LD.col = NULL;
    precond_par->LD.row = NULL;
    precond_par->LD.blockinfo = NULL;

    precond_par->UD.val = NULL;
    precond_par->UD.col = NULL;
    precond_par->UD.row = NULL;
    precond_par->UD.blockinfo = NULL;
    
    precond_par->LDT.val = NULL;
    precond_par->LDT.col = NULL;
    precond_par->LDT.row = NULL;
    precond_par->LDT.blockinfo = NULL;

    precond_par->UDT.val = NULL;
    precond_par->UDT.col = NULL;
    precond_par->UDT.row = NULL;
    precond_par->UDT.blockinfo = NULL;
    
    precond_par->cuinfoL = NULL;
    precond_par->cuinfoU = NULL;
    precond_par->cuinfoLT = NULL;
    precond_par->cuinfoUT = NULL;
    
    precond_par->L_dgraphindegree = NULL;
    precond_par->U_dgraphindegree = NULL;
    precond_par->L_dgraphindegree_bak = NULL;
    precond_par->U_dgraphindegree_bak = NULL;

cleanup:
    if( info != 0 ){
        magma_free( solver_par->timing );
        magma_free( solver_par->res_vec );
    }
    return info;
}


/**
    Purpose
    -------

    Initializes space for eigensolvers.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_z_solver_par*
                structure containing all solver information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zeigensolverinfo_init(
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex *initial_guess=NULL;
    solver_par->eigenvectors = NULL;
    solver_par->eigenvalues = NULL;
    if ( solver_par->solver == Magma_LOBPCG ) {
        if( solver_par->num_eigenvalues==0 ){
            solver_par->num_eigenvalues = 32;
        }
        CHECK( magma_dmalloc_cpu( &solver_par->eigenvalues ,
                                3*solver_par->num_eigenvalues ));
        // setup initial guess EV using lapack
        // then copy to GPU
        magma_int_t ev = solver_par->num_eigenvalues * solver_par->ev_length;

        CHECK( magma_zmalloc_cpu( &initial_guess, ev ));
        CHECK( magma_zmalloc( &solver_par->eigenvectors, ev ));
        magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;
        lapackf77_zlarnv( &ione, ISEED, &ev, initial_guess );

        magma_zsetmatrix( solver_par->ev_length, solver_par->num_eigenvalues,
            initial_guess, solver_par->ev_length, solver_par->eigenvectors,
                                                    solver_par->ev_length, queue );
    } else {
        solver_par->eigenvectors = NULL;
        solver_par->eigenvalues = NULL;
    }

cleanup:
    if( info != 0 ){
        magma_free( solver_par->eigenvectors );
        magma_free( solver_par->eigenvalues );
    }
    magma_free_cpu( initial_guess );
    return info;
}


/**
    Checks whether a solver is among the list of Krylov solvers.
    The result is passed in info:
    1   yes - Krylov solver
    0   no - no Krylov solver of the list.
    
    @param[in]
    solver      magma_solver_type
    
    @return error
    
    @ingroup magma_util
    ********************************************************************/
extern "C" magma_int_t
magma_zKrylov_check( magma_solver_type solver )
{
    switch( solver ) {
        case  Magma_CG:
        case  Magma_CGMERGE:
        case  Magma_BICGSTAB:
        case  Magma_BICGSTABMERGE:
        case  Magma_BICGSTABMERGE2:
        case  Magma_GMRES:
        case  Magma_IDR:
        case  Magma_IDRMERGE:
        case  Magma_CGS:
        case  Magma_CGSMERGE:
        case  Magma_BICG:
        case  Magma_BICGMERGE:
        case  Magma_QMR:
        case  Magma_QMRMERGE:
        case  Magma_LSQR:
        case  Magma_TFQMR:
        case  Magma_TFQMRMERGE:
            return MAGMA_SUCCESS;                                
            break;
        
        default:
            return MAGMA_ERR_UNKNOWN;
            break;
    }
}
