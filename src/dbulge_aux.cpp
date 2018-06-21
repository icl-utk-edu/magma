/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @precisions normal d -> s
 *
 */

#include "magma_internal.h"
#include "magma_timer.h"
#include "magma_dbulgeinc.h"

//////////////////////////////////////////////////////////////
//          DSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_dstedc_withZ(magma_vec_t JOBZ, magma_int_t N, double *D, double * E, double *Z, magma_int_t LDZ)
{
    double *WORK;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK;
    magma_int_t INFO;
    
    // use log() as log2() is not defined everywhere (e.g., Windows)
    const double log_2 = 0.6931471805599453;
    if (JOBZ == MagmaVec) {
        LWORK  = 1 + 3*N + 3*N*((magma_int_t)(log( (double)N )/log_2) + 1) + 4*N*N + 256*N;
        LIWORK = 6 + 6*N + 6*N*((magma_int_t)(log( (double)N )/log_2) + 1) + 256*N;
    } else if (JOBZ == MagmaIVec) {
        LWORK  = 2*N*N + 256*N + 1;
        LIWORK = 256*N;
    } else if (JOBZ == MagmaNoVec) {
        LWORK  = 256*N + 1;
        LIWORK = 256*N;
    } else {
        printf("ERROR JOBZ %c\n", JOBZ);
        return MAGMA_ERR_ILLEGAL_VALUE; 
    }
    
    magma_dmalloc_cpu( &WORK,  LWORK  );
    magma_imalloc_cpu( &IWORK, LIWORK );
    
    lapackf77_dstedc( lapack_vec_const(JOBZ), &N, D, E, Z, &LDZ, WORK, &LWORK, IWORK, &LIWORK, &INFO);
    
    if (INFO != 0) {
        printf("=================================================\n");
        printf("DSTEDC ERROR OCCURED. HERE IS INFO %lld\n ", (long long) INFO );
        printf("=================================================\n");
        //assert(INFO == 0);
    }
    
    magma_free_cpu( IWORK );
    magma_free_cpu( WORK  );
}
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//          DSTEDX          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_dstedx_withZ(magma_int_t N, magma_int_t NE, double *D, double * E, double *Z, magma_int_t LDZ)
{
    double *WORK;
    double *dwork;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK;
    magma_int_t INFO;
    
    LWORK  = N*N+4*N+1;
    LIWORK = 3 + 5*N;
    
    magma_dmalloc_cpu( &WORK,  LWORK  );
    magma_imalloc_cpu( &IWORK, LIWORK );
    
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, 3*N*(N/2 + 1) )) {
        printf("=================================================\n");
        printf("DSTEDC ERROR OCCURED IN CUDAMALLOC\n");
        printf("=================================================\n");
        return;
    }
    printf("using magma_dstedx\n");

    magma_timer_t time=0;
    timer_start( time );

    //magma_range_t job = MagmaRangeI;
    //if (NE == N)
    //    job = MagmaRangeAll;
    
    magma_dstedx(MagmaRangeI, N, 0., 0., 1, NE, D, E, Z, LDZ, WORK, LWORK, IWORK, LIWORK, dwork, &INFO);
    
    if (INFO != 0) {
        printf("=================================================\n");
        printf("DSTEDC ERROR OCCURED. HERE IS INFO %lld\n ", (long long) INFO );
        printf("=================================================\n");
        //assert(INFO == 0);
    }

    timer_stop( time );
    timer_printf( "time dstedx = %6.2f\n", time );

    magma_free( dwork );
    magma_free_cpu( IWORK );
    magma_free_cpu( WORK  );
}
//////////////////////////////////////////////////////////////
