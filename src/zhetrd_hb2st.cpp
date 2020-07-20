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
#include "magma_internal.h"
#include "magma_bulge.h"
#include "magma_zbulge.h"

#ifndef MAGMA_NOAFFINITY
#include "affinity.h"
#endif

#define COMPLEX

static void *magma_zhetrd_hb2st_parallel_section(void *arg);

static void magma_ztile_bulge_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, magma_int_t wantz, 
    volatile magma_int_t *prog, pthread_barrier_t* myptbarrier);

static void magma_ztile_bulge_computeT_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t n, magma_int_t nb, magma_int_t Vblksiz);


/******************************************************************************/
typedef struct magma_zbulge_data_s {
    magma_int_t threads_num;
    magma_int_t n;
    magma_int_t nb;
    magma_int_t nbtiles;
    magma_int_t grsiz;
    magma_int_t Vblksiz;
    magma_int_t wantz;
    magmaDoubleComplex* A;
    magma_int_t lda;
    magmaDoubleComplex* V;
    magma_int_t ldv;
    magmaDoubleComplex* TAU;
    magmaDoubleComplex* T;
    magma_int_t ldt;
    volatile magma_int_t *prog;
    pthread_barrier_t myptbarrier;
} magma_zbulge_data;


/******************************************************************************/
void magma_zbulge_data_init(
    magma_zbulge_data *zbulge_data_S,
    magma_int_t threads_num, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, magma_int_t wantz,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    volatile magma_int_t* prog)
{
    zbulge_data_S->threads_num = threads_num;
    zbulge_data_S->n = n;
    zbulge_data_S->nb = nb;
    zbulge_data_S->nbtiles = nbtiles;
    zbulge_data_S->grsiz = grsiz;
    zbulge_data_S->Vblksiz = Vblksiz;
    zbulge_data_S->wantz = wantz;
    zbulge_data_S->A = A;
    zbulge_data_S->lda = lda;
    zbulge_data_S->V = V;
    zbulge_data_S->ldv= ldv;
    zbulge_data_S->TAU = TAU;
    zbulge_data_S->T = T;
    zbulge_data_S->ldt = ldt;
    zbulge_data_S->prog = prog;

    pthread_barrier_init(&(zbulge_data_S->myptbarrier), NULL, (unsigned) zbulge_data_S->threads_num);
}


/******************************************************************************/
void magma_zbulge_data_destroy(magma_zbulge_data *zbulge_data_S)
{
    pthread_barrier_destroy(&(zbulge_data_S->myptbarrier));
}


/******************************************************************************/
typedef struct magma_zbulge_id_data_s {
    magma_int_t id;
    magma_zbulge_data* data;
} magma_zbulge_id_data;


/******************************************************************************/
void magma_zbulge_id_data_init(magma_zbulge_id_data *id_data, magma_int_t id, magma_zbulge_data* data)
{
    id_data->id = id;
    id_data->data = data;
}


/***************************************************************************//**
    Purpose
    -------
    Reduces Hermitian band matrix to real symmetric tridiagonal.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangles of A is stored;
      -     = MagmaLower:  Lower triangles of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nb      INTEGER
            The order of the band matrix A.  n >= nb >= 0.

    @param[in]
    Vblksiz INTEGER
            The size of the block of householder vectors applied at once.

    @param[in]
    A       (workspace) COMPLEX_16 array, dimension (lda, n)
            On entry the band matrix stored in the following way:

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  lda >= 2*nb.

    @param[out]
    d       DOUBLE array, dimension (n)
            The diagonal elements of the tridiagonal matrix T:
            D(i) = A(i,i).

    @param[out]
    e       DOUBLE array, dimension (n-1)
            The off-diagonal elements of the tridiagonal matrix T:
            E(i) = A(i,i+1) if UPLO = MagmaUpper, E(i) = A(i+1,i) if UPLO = MagmaLower.

    @param[out]
    V       COMPLEX_16 array, dimension (BLKCNT, LDV, VBLKSIZ)
            On exit it contains the blocks of householder reflectors
            BLKCNT is the number of block and it is returned by the funtion MAGMA_BULGE_GET_BLKCNT.

    @param[in]
    ldv     INTEGER
            The leading dimension of V.
            LDV > nb + VBLKSIZ + 1

    @param[out]
    TAU     COMPLEX_16 dimension(BLKCNT, VBLKSIZ)
            ???

    @param[in]
    wantz   INTEGER
            if COMPT = 0 T is not computed
            if COMPT = 1 T is computed

    @param[out]
    T       COMPLEX_16 dimension(LDT *)
            if COMPT = 1 on exit contains the matrices T needed for Q2
            if COMPT = 0 T is not referenced

    @param[in]
    ldt     INTEGER
            The leading dimension of T.
            LDT > Vblksiz

    @ingroup magma_hetrd_hb2st
*******************************************************************************/
extern "C" magma_int_t
magma_zhetrd_hb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    magmaDoubleComplex *A, magma_int_t lda, double *d, double *e,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
    magma_int_t wantz, magmaDoubleComplex *T, magma_int_t ldt)
{
    #ifdef ENABLE_TIMER
    real_Double_t timeblg=0.0;
    #endif

    magma_int_t parallel_threads = magma_get_parallel_numthreads();
    magma_int_t mklth   = magma_get_lapack_numthreads();
    magma_int_t ompth   = magma_get_omp_numthreads();

    //magma_set_omp_numthreads(1);
    //magma_set_lapack_numthreads(1);

    magma_int_t blkcnt, sizTAU2, sizT2, sizV2;
    magma_zbulge_getstg2size(n, nb, wantz, 
                          Vblksiz, ldv, ldt, &blkcnt, 
                          &sizTAU2, &sizT2, &sizV2);
    memset(T,   0, sizT2*sizeof(magmaDoubleComplex));
    memset(TAU, 0, sizTAU2*sizeof(magmaDoubleComplex));
    memset(V,   0, sizV2*sizeof(magmaDoubleComplex));

    magma_int_t INgrsiz=1;
    magma_int_t nbtiles = magma_ceildiv(n, nb);
    volatile magma_int_t* prog;
    magma_malloc_cpu((void**) &prog, (2*nbtiles+parallel_threads+10)*sizeof(magma_int_t));
    memset((void *) prog, 0, (2*nbtiles+parallel_threads+10)*sizeof(magma_int_t));

    magma_zbulge_id_data* arg;
    magma_malloc_cpu((void**) &arg, parallel_threads*sizeof(magma_zbulge_id_data));

    pthread_t* thread_id;
    magma_malloc_cpu((void**) &thread_id, parallel_threads*sizeof(pthread_t));
    pthread_attr_t thread_attr;

    magma_zbulge_data data_bulge;
    magma_zbulge_data_init(&data_bulge, parallel_threads, n, nb, nbtiles, INgrsiz, Vblksiz, wantz,
                                 A, lda, V, ldv, TAU, T, ldt, prog);

    // Set one thread per core
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_setconcurrency( (unsigned)parallel_threads );

    //timing
    #ifdef ENABLE_TIMER
    timeblg = magma_wtime();
    #endif

    // Launch threads
    for (magma_int_t thread = 1; thread < parallel_threads; thread++) {
        magma_zbulge_id_data_init(&(arg[thread]), thread, &data_bulge);
        pthread_create(&thread_id[thread], &thread_attr, magma_zhetrd_hb2st_parallel_section, &arg[thread]);
    }
    magma_zbulge_id_data_init(&(arg[0]), 0, &data_bulge);
    magma_zhetrd_hb2st_parallel_section(&arg[0]);

    // Wait for completion
    for (magma_int_t thread = 1; thread < parallel_threads; thread++) {
        void *exitcodep;
        pthread_join(thread_id[thread], &exitcodep);
    }

    // timing
    #ifdef ENABLE_TIMER
    timeblg = magma_wtime()-timeblg;
    printf("  time BULGE+T = %f\n", timeblg);
    #endif

    magma_free_cpu(thread_id);
    magma_free_cpu(arg);
    magma_free_cpu((void *) prog);
    magma_zbulge_data_destroy(&data_bulge);

    magma_set_omp_numthreads(ompth);
    magma_set_lapack_numthreads(mklth);
    /*================================================
     *  store resulting diag and lower diag d and e
     *  note that d and e are always real
     *================================================*/

    /* Make diagonal and superdiagonal elements real,
     * storing them in d and e
     */
    /* In complex case, the off diagonal element are
     * not necessary real. we have to make off-diagonal
     * elements real and copy them to e.
     * When using HouseHolder elimination,
     * the ZLARFG give us a real as output so, all the
     * diagonal/off-diagonal element except the last one are already
     * real and thus we need only to take the abs of the last
     * one.
     *  */

#ifdef COMPLEX
    if (uplo == MagmaLower) {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = MAGMA_Z_REAL( A[i*lda  ] );
            e[i] = MAGMA_Z_REAL( A[i*lda+1] );
        }
        d[n-1] = MAGMA_Z_REAL(A[(n-1)*lda]);
    } else { /* MagmaUpper not tested yet */
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = MAGMA_Z_REAL( A[i*lda+nb]   );
            e[i] = MAGMA_Z_REAL( A[i*lda+nb-1] );
        }
        d[n-1] = MAGMA_Z_REAL(A[(n-1)*lda+nb]);
    } /* end MagmaUpper */
#else
    if ( uplo == MagmaLower ) {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = A[i*lda];   // diag
            e[i] = A[i*lda+1]; // lower diag
        }
        d[n-1] = A[(n-1)*lda];
    } else {
        for (magma_int_t i=0; i < n-1; i++) {
            d[i] = A[i*lda+nb];   // diag
            e[i] = A[i*lda+nb-1]; // lower diag
        }
        d[n-1] = A[(n-1)*lda+nb];
    }
#endif
    return MAGMA_SUCCESS;
}


/******************************************************************************/
static void *magma_zhetrd_hb2st_parallel_section(void *arg)
{
    magma_int_t my_core_id  = ((magma_zbulge_id_data*)arg) -> id;
    magma_zbulge_data* data = ((magma_zbulge_id_data*)arg) -> data;

    magma_int_t allcores_num   = data -> threads_num;
    magma_int_t n              = data -> n;
    magma_int_t nb             = data -> nb;
    magma_int_t nbtiles        = data -> nbtiles;
    magma_int_t grsiz          = data -> grsiz;
    magma_int_t Vblksiz        = data -> Vblksiz;
    magma_int_t wantz          = data -> wantz;
    magmaDoubleComplex *A      = data -> A;
    magma_int_t lda            = data -> lda;
    magmaDoubleComplex *V      = data -> V;
    magma_int_t ldv            = data -> ldv;
    magmaDoubleComplex *TAU    = data -> TAU;
    magmaDoubleComplex *T      = data -> T;
    magma_int_t ldt            = data -> ldt;
    volatile magma_int_t* prog = data -> prog;

    pthread_barrier_t* myptbarrier = &(data -> myptbarrier);

    //magma_int_t sys_corenbr    = 1;

    #ifdef ENABLE_TIMER
    real_Double_t timeB=0.0, timeT=0.0;
    #endif

    // with MKL and when using omp_set_num_threads instead of mkl_set_num_threads
    // it need that all threads setting it to 1.
    magma_set_omp_numthreads(1);
    //magma_set_lapack_numthreads(1);

/*
#ifndef MAGMA_NOAFFINITY
    // bind threads 
    cpu_set_t set;
    // bind threads 
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set);
#endif
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(1);

*/

#ifndef MAGMA_NOAFFINITY
//#define PRINTAFFINITY
#ifdef PRINTAFFINITY
    affinity_set print_set;
    print_set.print_affinity(my_core_id, "starting affinity");
#endif
    affinity_set original_set;
    affinity_set new_set(my_core_id);
    magma_int_t check  = 0;
    magma_int_t check2 = 0;
    // bind threads
    check = original_set.get_affinity();
    if (check == 0) {
        check2 = new_set.set_affinity();
        if (check2 != 0)
            printf("Error in sched_setaffinity (single cpu)\n");
    }
    else {
        printf("Error in sched_getaffinity\n");
    }
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "set affinity");
#endif
#endif

    /* compute the Q1 overlapped with the bulge chasing+T.
    * if all_cores_num=1 it call Q1 on GPU and then bulgechasing.
    * otherwise the first thread run Q1 on GPU and
    * the other threads run the bulgechasing.
    * */
    //=========================
    //    bulge chasing
    //=========================
    #ifdef ENABLE_TIMER
    if (my_core_id == 0)
        timeB = magma_wtime();
    #endif

    magma_ztile_bulge_parallel(my_core_id, allcores_num, A, lda, V, ldv, TAU, n, nb, nbtiles, grsiz, Vblksiz, wantz, prog, myptbarrier);
    if (allcores_num > 1) pthread_barrier_wait(myptbarrier);

    #ifdef ENABLE_TIMER
    if (my_core_id == 0) {
        timeB = magma_wtime()-timeB;
        printf("  Finish BULGE   timing= %f\n", timeB);
    }
    #endif

    //=========================
    // compute the T's to be used when applying Q2
    //=========================
    if ( wantz > 0 ) {
        #ifdef ENABLE_TIMER
        if (my_core_id == 0)
            timeT = magma_wtime();
        #endif
       
        magma_ztile_bulge_computeT_parallel(my_core_id, allcores_num, V, ldv, TAU, T, ldt, n, nb, Vblksiz);
        if (allcores_num > 1) pthread_barrier_wait(myptbarrier);
       
        #ifdef ENABLE_TIMER
        if (my_core_id == 0) {
            timeT = magma_wtime()-timeT;
            printf("  Finish T's     timing= %f\n", timeT);
        }
        #endif
    }

#ifndef MAGMA_NOAFFINITY
    // unbind threads
    if (check == 0) {
        check2 = original_set.set_affinity();
        if (check2 != 0)
            printf("Error in sched_setaffinity (restore cpu list)\n");
    }
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "restored_affinity");
#endif
#endif

    return 0;
}


/******************************************************************************/
#define expertmyss_cond_wait(m, n, val) \
do { \
    while (prog[(m)] != (val)) { \
        magma_yield(); \
    } \
    for (kk=0; kk < 100; kk++) { \
        __asm__ volatile ("nop;" :::); \
    } \
} while(0)

// __asm__ volatile ("" ::: "memory") is a compiler fence (but not a hardware memory fence),
// to prohibit load/store reordering across that point.
// However, inline assemble (asm) isn't available on Microsoft VS x64.
#define expertmyss_cond_set(m, n, val) \
do { \
    prog[(m)*(64/sizeof(magma_int_t))] = (val); \
    __asm__ volatile ("" ::: "memory"); \
} while(0)

#define expertmyss_init(m, n, init_val) \
do { \
    if (my_core_id == 0) { \
        magma_malloc_cpu((void**) &prog, ( (m)*(64/sizeof(int)) ) * sizeof(magma_int_t)); \
        memset((magma_int_t*)prog, 0, (m)*(64/sizeof(magma_int_t))); \
    } \
    pthread_barrier_init(&myptbarrier, NULL, cores_num); \
    pthread_barrier_wait(&myptbarrier); \
    barrier(my_core_id, cores_num); \
} while(0)

#define expertmyss_finalize() \
do { \
    pthread_barrier_wait(&myptbarrier); \
    pthread_barrier_destroy(&myptbarrier); \
    barrier(my_core_id, cores_num); \
    if (my_core_id == 0) { \
        magma_free_cpu((void *) prog); \
    } \
} while(0)



// see also __asm__ volatile ("" ::: "memory") in expertmyss_cond_set above
#define myss_cond_set(m, n, val) \
do { \
    prog[(m)] = (val); \
} while(0)

#define myss_cond_wait(m, n, val) \
do { \
    while (prog[(m)] != (val)) \
    { \
        magma_yield(); \
    } \
} while(0)

#define myss_init(m, n, init_val) \
do { \
    if (my_core_id == 0) { \
        magma_malloc_cpu((void**) &prog,  (m) * sizeof(magma_int_t)); \
        memset((magma_int_t*)prog, 0, (m)); \
    } \
    pthread_barrier_wait(myptbarrier); \
} while(0)

#define myss_finalize() \
do { \
    pthread_barrier_wait(myptbarrier); \
    if (my_core_id == 0) { \
        magma_free_cpu((void *) prog); \
    } \
} while(0)


/******************************************************************************/
static void magma_ztile_bulge_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU, magma_int_t n, magma_int_t nb, magma_int_t nbtiles,
    magma_int_t grsiz, magma_int_t Vblksiz, magma_int_t wantz, 
    volatile magma_int_t *prog, pthread_barrier_t* myptbarrier)
{
    magma_int_t sweepid, myid, shift, stt, st, ed, stind, edind;
    magma_int_t blklastind, colpt;
    magma_int_t stepercol;
    magma_int_t i, j, m, k;
    magma_int_t thgrsiz, thgrnb, thgrid, thed;
    magma_int_t coreid;
    magma_int_t colblktile, maxrequiredcores, colpercore, allcoresnb;
    magmaDoubleComplex *work;

    if (n <= 0)
        return;
    if (grsiz <= 0)
        return;

    //printf("=================> my core id %lld of %lld\n", (long long) my_core_id, (long long) cores_num );

    /* As I store V in the V vector there are overlap between
     * tasks so shift is now 4 where group need to be always
     * multiple of 2, because as example if grs=1 task 2 from
     * sweep 2 can run with task 6 sweep 1., but task 2 sweep 2
     * will overwrite the V of tasks 5 sweep 1 which are used by
     * task 6, so keep in mind that group need to be multiple of 2,
     * and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
     * However, when storing V in A, shift could be back to 3.
     * */

    magma_zmalloc_cpu(&work, nb);
    /* Some tunning for the bulge chasing code
     * see technical report for details */
    /* grsiz   = 2; */
    if ( wantz == 0 ) {
        shift = 3;
    } else {
        shift = 3; /* it was 5 before see above for explanation*/
    }

    if ( grsiz == 1 )
        colblktile = 1;
    else
        colblktile = grsiz/2;

    maxrequiredcores = max( nbtiles/colblktile, 1 );
    colpercore = colblktile*nb;
    allcoresnb = min( cores_num, maxrequiredcores );
    thgrsiz = n;
    #if defined (ENABLE_DEBUG)
    if (my_core_id == 0) {
        if (cores_num > maxrequiredcores)
        {
            printf("==================================================================================\n");
            printf("  WARNING only %3lld threads are required to run this test optimizing cache reuse\n", (long long) maxrequiredcores );
            printf("==================================================================================\n");
        }
        printf("  SS_COND Static bulgechasing version v9_9col threads  %4lld   threads_used  %4lld   n %5lld      nb %5lld    grs %4lld thgrsiz %4lld  wantz %4lld\n",
               (long long) cores_num, (long long) allcoresnb, (long long) n,
               (long long) nb, (long long) grsiz, (long long) thgrsiz, (long long) wantz );
    }
    #endif

    /* Initialize static scheduler progress table */
    //myss_init(2*nbtiles+shift+cores_num+10, 1, 0); // already initialized at top level
    /* main bulge chasing code */
    i = shift/grsiz;
    stepercol =  i*grsiz == shift ? i:i+1;
    i       = (n-1)/thgrsiz;
    thgrnb  = i*thgrsiz == (n-1) ? i:i+1;
    for (thgrid = 1; thgrid <= thgrnb; thgrid++) {
        stt  = (thgrid-1)*thgrsiz+1;
        thed = min( (stt + thgrsiz -1), (n-1));
        for (i = stt; i <= n-1; i++) {
            ed = min(i,thed);
            if (stt > ed) break;
            for (m = 1; m <= stepercol; m++) {
                st = stt;
                for (sweepid = st; sweepid <= ed; sweepid++)
                {
                    for (k = 1; k <= grsiz; k++) {
                        myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;
                        if (myid%2 == 0) {
                            colpt      = (myid/2)*nb+1+sweepid-1;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            blklastind = colpt;
                        } else {
                            colpt      = ((myid+1)/2)*nb + 1 +sweepid -1;
                            stind      = colpt-nb+1;
                            edind      = min(colpt,n);
                            if ( (stind >= edind-1) && (edind == n) )
                                blklastind=n;
                            else
                                blklastind=0;
                        }
                        coreid = (stind/colpercore)%allcoresnb;

                        if (my_core_id == coreid) {
                            if (myid == 1) {
                                myss_cond_wait(myid+shift-1, 0, sweepid-1);
                                magma_zhbtype1cb(n, nb, A, lda, V, ldv, TAU, stind-1, edind-1, sweepid-1, Vblksiz, wantz, work);
                                myss_cond_set(myid, 0, sweepid);

                                if (blklastind >= (n-1)) {
                                    for (j = 1; j <= shift; j++)
                                        myss_cond_set(myid+j, 0, sweepid);
                                }
                            } else {
                                myss_cond_wait(myid-1,       0, sweepid);
                                myss_cond_wait(myid+shift-1, 0, sweepid-1);
                                if (myid%2 == 0) {
                                    magma_zhbtype2cb(n, nb, A, lda, V, ldv, TAU, stind-1, edind-1, sweepid-1, Vblksiz, wantz, work);
                                } else {
                                    magma_zhbtype3cb(n, nb, A, lda, V, ldv, TAU, stind-1, edind-1, sweepid-1, Vblksiz, wantz, work);
                                }
                                myss_cond_set(myid, 0, sweepid);
                                if (blklastind >= (n-1)) {
                                    for (j = 1; j <= shift+allcoresnb; j++)
                                        myss_cond_set(myid+j, 0, sweepid);
                                }
                            } /* END if myid == 1 */
                        } /* END if my_core_id == coreid */

                        if (blklastind >= (n-1)) {
                            stt++;
                            break;
                        }
                    } /* END for k=1:grsiz */
                } /* END for sweepid=st:ed */
            } /* END for m=1:stepercol */
        } /* END for i=1:n-1 */
    } /* END for thgrid=1:thgrnb */

    /* finalize static sched */
    //myss_finalize(); // initialized at top level so freed there

    magma_free_cpu(work);
} // END FUNCTION


/******************************************************************************/
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)   &(T[(m)])
static void magma_ztile_bulge_computeT_parallel(
    magma_int_t my_core_id, magma_int_t cores_num,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t n, magma_int_t nb, magma_int_t Vblksiz)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t Vm, Vn, mt, nt;
    magma_int_t myrow, mycol, blkj, blki, firstrow;
    magma_int_t blkid, vpos, taupos, tpos;
    magma_int_t blkpercore, myid;

    if (n <= 0)
        return;

    magma_int_t blkcnt = magma_bulge_get_blkcnt(n, nb, Vblksiz);
    blkpercore = blkcnt/cores_num;
    blkpercore = (blkpercore == 0 ? 1 : blkpercore);
    //magma_int_t nbGblk  = magma_ceildiv(n-1, Vblksiz);

    #ifdef ENABLE_DEBUG
    if (my_core_id == 0)
        printf("  COMPUTE T parallel threads %lld with  n %lld   nb %lld   Vblksiz %lld\n", (long long) cores_num, (long long) n, (long long) nb, (long long) Vblksiz );
    #endif

    /*========================================
     * compute the T's in parallel.
     * The Ts are independent so each core pick
     * a T and compute it. The loop is based on
     * the version 113 of the applyQ
     * which go over the losange block_column
     * by block column. but it is not important
     * here the order because Ts are independent.
     * ======================================== */
    nt  = magma_ceildiv((n-1), Vblksiz);
    for (blkj=nt-1; blkj >= 0; blkj--) {
        /* the index of the first row on the top of block (blkj) */
        firstrow = blkj * Vblksiz + 1;
        /*find the number of tile for this block */
        if ( blkj == nt-1 )
            mt = magma_ceildiv( n -  firstrow,    nb);
        else
            mt = magma_ceildiv( n - (firstrow+1), nb);
        /*loop over the tiles find the size of the Vs and apply it */
        for (blki=mt; blki > 0; blki--) {
            /*calculate the size of each losange of Vs= (Vm,Vn)*/
            myrow     = firstrow + (mt-blki)*nb;
            mycol     = blkj*Vblksiz;
            Vm = min( nb+Vblksiz-1, n-myrow);
            if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                Vn = min (Vblksiz, Vm);
            } else {
                Vn = min (Vblksiz, Vm-1);
            }
            /*calculate the pointer to the Vs and the Ts.
             * Note that Vs and Ts have special storage done
             * by the bulgechasing function*/
            magma_bulge_findVTAUTpos(n, nb, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &taupos, &tpos, &blkid);
            myid = blkid/blkpercore;
            if ( my_core_id == (myid%cores_num) ) {
                if ( ( Vm > 0 ) && ( Vn > 0 ) ) {
                    lapackf77_zlarft( "F", "C", &Vm, &Vn, V(vpos), &ldv, TAU(taupos), T(tpos), &ldt);
                }
            }
        }
    }
}

#undef V
#undef TAU
#undef T
