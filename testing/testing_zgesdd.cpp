/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <ctype.h>  // tolower
#include <string>
#include <algorithm>  // find

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

#define max3( x, y, z ) max( max( (x), (y) ), (z) )

extern const char* zgesdd_path;


// ----------------------------------------
// returns true if vec contains value.
bool contains( std::vector< magma_int_t > vec, magma_int_t value )
{
    return (std::find( vec.begin(), vec.end(), value ) != vec.end());
}


// ----------------------------------------
// shorten lwork equation: change mn, mx to m or n, based on tall (M >= N)
std::string abbreviate_formula( const char* src, bool tall )
{
    std::string dst;
    while( *src != '\0' ) {
        if ( strncmp( src, "mx", 2 ) == 0 ) {
            dst += (tall ? 'm' : 'n');  // max(M,N)
            src++;
        }
        else if ( strncmp( src, "mn", 2 ) == 0 ) {
            dst += (tall ? 'n' : 'm');  // min(M,N)
            src++;
        }
        else {
            dst += *src;
        }
        src++;
    }
    return dst;
}


// ----------------------------------------
// assign formula's value to x, formula's string representation to str.
#define assign( x_, formula_ )      \
    do {                            \
        (x_).value   = (formula_);  \
        (x_).formula = abbreviate_formula( #formula_, M >= N );  \
    } while(0)


// ----------------------------------------
// small class to keep track of lwork value & equation string together
class lwork_formula_t {
public:
    lwork_formula_t(): value( 0 ) {}
    
    magma_int_t value;
    std::string formula;
    std::string pre;
    std::string post;
};


// ----------------------------------------
// Choose formula for lwork based on svd_work (query, min, opt, max, etc.)
// Also highlight errors such as lwork < min.
void choose_lwork(
    const magma_opts& opts,
    magma_svd_work_t svd_work,
    magma_vec_t jobz,
    magma_int_t M,
    magma_int_t N,
    magma_int_t query_magma,
    magma_int_t query_lapack,
    lwork_formula_t& lwork_magma,
    lwork_formula_t& lwork_lapack,
    std::string& work_str,
    std::string& path_str )
{
    lwork_magma .value = -1;
    lwork_lapack.value = -1;
    lwork_formula_t lwork_doc, lwork_doc_old,
                    lwork_min, lwork_min_old,
                    lwork_opt, lwork_opt_old, lwork_max;
    magma_int_t nb = magma_get_dgesvd_nb( M, N );
    magma_int_t mx = max( M, N );
    magma_int_t mn = min( M, N );
    magma_int_t path = 0;
    #ifdef COMPLEX
        /* =====================================================================
           lwork formulas for zgesdd (Complex)
           =================================================================== */
        magma_int_t mnthr = (magma_int_t) (mn * 17. / 9.);
        if (mx >= mnthr) {
            if (jobz == MagmaNoVec) {
                path = 1;
                assign( lwork_opt, 2*mn + 2*mn*nb );
                assign( lwork_min, 3*mn );
            }
            else if (jobz == MagmaOverwriteVec) {
                path = 2;
                assign( lwork_opt, 2*mn*mn       + 2*mn + 2*mn*nb );
                assign( lwork_max, mx*mn + mn*mn + 2*mn + 2*mn*nb );  // marginally faster?
                assign( lwork_min, 2*mn*mn + 3*mn );
            }
            else if (jobz == MagmaSomeVec) {
                path = 3;
                assign( lwork_opt, mn*mn + 2*mn + 2*mn*nb );
                assign( lwork_min, mn*mn + 3*mn );
            }
            else if (jobz == MagmaAllVec) {
                path = 4;
                assign( lwork_opt, mn*mn + max( 2*mn + 2*mn*nb, mn + mx*nb ) );
                assign( lwork_min,      mn*mn + max( 3*mn, mx + mn ) );  // fixed min
                assign( lwork_min_old,  mn*mn + 2*mn + mx );             // LAPACK's over-estimate
            }
        }
        else {
            path = 5;  // or 6
            if (jobz == MagmaNoVec) {
                assign( lwork_opt, 2*mn + (mx + mn)*nb );
                assign( lwork_min, 2*mn + mx );
            }
            else if (jobz == MagmaOverwriteVec) {
                assign( lwork_opt, mn*mn + 2*mn + (mx + mn)*nb );  // slower algorithm
                assign( lwork_max, mx*mn + 2*mn + (mx + mn)*nb );  // faster algorithm
                assign( lwork_min, mn*mn + 2*mn + mx );
            }
            else if (jobz == MagmaSomeVec) {
                assign( lwork_opt, 2*mn + (mx + mn)*nb );
                assign( lwork_min, 2*mn + mx );
            }
            else if (jobz == MagmaAllVec) {
                assign( lwork_opt, 2*mn + (mx + mn)*nb );
                assign( lwork_min, 2*mn + mx );
            }
        }
        
        // lwork per LAPACK's documentation (which over-estimates some cases)
        if (jobz == MagmaNoVec) {
            assign( lwork_doc, 2*mn + mx );
        }
        else if (jobz == MagmaOverwriteVec) {
            assign( lwork_doc, 2*mn*mn + 2*mn + mx );
        }
        else if (jobz == MagmaSomeVec) {
            assign( lwork_doc_old, mn*mn + 2*mn + mx );
            assign( lwork_doc,     mn*mn + 3*mn );
        }
        else if (jobz == MagmaAllVec) {
            // tight bound is mn*mn + max( 3*mn, mn + mx ),
            // but that breaks backwards compatability with LAPACK <= 3.6.
            assign( lwork_doc, mn*mn + 2*mn + mx );
        }
    #else
        /* =====================================================================
           lwork formulas for dgesdd (Real)
           =================================================================== */
        magma_int_t mnthr = (magma_int_t) (mn * 11. / 6.);
        if (mx >= mnthr) {
            // for mx >> mn
            if (jobz == MagmaNoVec) {
                path = 1;
                assign( lwork_opt, 3*mn + 2*mn*nb );
                assign( lwork_min, 8*mn );  // LAPACK's over-estimate
            }
            else if (jobz == MagmaOverwriteVec) {
                path = 2;
                assign( lwork_min, 2*mn*mn       + 3*mn +      3*mn*mn + 4*mn );
                assign( lwork_opt, 2*mn*mn       + 3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb ) );
                assign( lwork_max, mx*mn + mn*mn + 3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb ) );  // marginally faster?
            }
            else if (jobz == MagmaSomeVec) {
                path = 3;
                assign( lwork_min, mn*mn + 3*mn +      3*mn*mn + 4*mn );
                assign( lwork_opt, mn*mn + 3*mn + max( 3*mn*mn + 4*mn, 2*mn*nb ) );
            }
            else if (jobz == MagmaAllVec) {
                path = 4;
                if ( M >= N ) {
                    assign( lwork_min_old, mn*mn + 2*mn + mx + 3*mn*mn + 4*mn );  // LAPACK's over-estimate
                }
                else {
                    assign( lwork_min_old, mn*mn + 3*mn + 3*mn*mn + 4*mn );       // LAPACK's under-estimate
                }
                assign( lwork_min,      mn*mn + max( 3*mn*mn + 7*mn, mn + mx ) );  // fixed min
                assign( lwork_opt,      mn*mn + max3( 3*mn*mn + 7*mn, 3*mn + 2*mn*nb, mn + mx*nb ) );
                
                lwork_opt_old.value = max( lwork_opt.value, lwork_min_old.value );     // LAPACK's over-estimate affects optimal, too.
                lwork_opt_old.formula = "max( opt, min_old )";
            }
        }
        else {
            // mx >= mn
            path = 5;
            if (jobz == MagmaNoVec) {
                assign( lwork_min, 3*mn + max( 7*mn, mx ) );            // LAPACK's over-estimate
                assign( lwork_opt, 3*mn + max( 7*mn, (mx + mn)*nb ) );  // LAPACK's over-estimate
            }
            else if (jobz == MagmaOverwriteVec) {
                assign( lwork_min, mn*mn + 3*mn + max( 3*mn*mn + 4*mn, mx ) );
                assign( lwork_opt, mn*mn + 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb ) );  // slower algorithm
                assign( lwork_max, mx*mn + 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb ) );  // faster algorithm
            }
            else if (jobz == MagmaSomeVec) {
                assign( lwork_min, 3*mn + max( 3*mn*mn + 4*mn, mx ) );
                assign( lwork_opt, 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb ) );
            }
            else if (jobz == MagmaAllVec) {
                assign( lwork_min, 3*mn + max( 3*mn*mn + 4*mn, mx ) );
                assign( lwork_opt, 3*mn + max( 3*mn*mn + 4*mn, (mx + mn)*nb ) );
                if ( M >= N ) {
                    // bug in LAPACK, it gets only BDSPAC
                    assign( lwork_opt_old, 3*mn*mn + 7*mn );
                }
            }
        }
        
        // lwork per LAPACK's documentation (which over-estimates some cases, and under-estimates Path 4)
        if (jobz == MagmaNoVec) {
            assign( lwork_doc, 3*mn + max( 7*mn, mx ) );
        }
        else if (jobz == MagmaOverwriteVec) {
            assign( lwork_doc, 3*mn + max( 5*mn*mn + 4*mn, mx ) );
        }
        else if (jobz == MagmaSomeVec) {
            assign( lwork_doc, 4*mn*mn + 7*mn );
        }
        else if (jobz == MagmaAllVec) {
            // for Path 4, old formula fails if mx > 3*mn*mn + 6*mn
            assign( lwork_doc_old, 4*mn*mn + 7*mn );       // LAPACK's old formula
            // tight bound is 3*mn + max( 4*mn*mn + 4*mn, mn*mn + mx ),
            // but that breaks backwards compatability with LAPACK <= 3.6.
            assign( lwork_doc,     4*mn*mn + 6*mn + mx );  // fixed formula
        }
    #endif
    
    char tmp[80];
    snprintf( tmp, sizeof(tmp), "%lld%s%c", (long long) path,
              (M >= N ? "" : "t"),
              tolower( lapacke_vec_const(jobz) ) );
    path_str = tmp;
    
    /* =====================================================================
       Select between min, optimal, etc. lwork size
       =================================================================== */
    lwork_magma = lwork_opt;  // MAGMA requires optimal; overridden below by query, min-1, max
    switch( svd_work ) {
        case MagmaSVD_query:
            lwork_lapack.value = query_lapack;
            lwork_magma.value  = query_magma;
            lwork_lapack.formula = "query";
            lwork_magma .formula = "query";
            work_str = "query";
            break;
        
        case MagmaSVD_min:
        case MagmaSVD_min_1:
            lwork_lapack = lwork_min;
            work_str = "min";
            if ( svd_work == MagmaSVD_min_1 ) {
                lwork_lapack.value -= 1;
                lwork_magma.value  -= 1;
                lwork_lapack.formula += " - 1";
                lwork_magma .formula += " - 1";
                work_str = "min-1";
            }
            break;
        
        case MagmaSVD_min_old:
        case MagmaSVD_min_old_1:
            lwork_lapack = (lwork_min_old.value ? lwork_min_old : lwork_min);
            work_str = "min_old";
            if ( svd_work == MagmaSVD_min_old_1 ) {
                lwork_lapack.value -= 1;
                lwork_magma.value  -= 1;
                lwork_lapack.formula += " - 1";
                lwork_magma .formula += " - 1";
                work_str = "min_old-1";
            }
            break;
        
        case MagmaSVD_opt:
            lwork_lapack = lwork_opt;
            work_str = "opt";
            break;
        
        case MagmaSVD_opt_old:
            lwork_lapack = (lwork_opt_old.value ? lwork_opt_old : lwork_opt);
            work_str = "opt_old";
            break;
        
        case MagmaSVD_max:
            lwork_lapack = (lwork_max.value ? lwork_max : lwork_opt);
            lwork_magma  = (lwork_max.value ? lwork_max : lwork_opt);
            work_str = "max";
            break;
        
        case MagmaSVD_doc:
            lwork_lapack = lwork_doc;
            work_str = "doc";
            break;
        
        case MagmaSVD_doc_old:
            lwork_lapack = (lwork_doc_old.value ? lwork_doc_old : lwork_doc);
            work_str = "doc_old";
            break;
        
        default:
            fprintf( stderr, "Unsupported svd-work %d\n", svd_work );
            exit(1);
    }
    
    /* =====================================================================
       Determine lwork errors
       =================================================================== */
    // Attribute codes:
    // 00=normal 01=bold 04=underscore 05=blink 07=reverse 08=concealed
    //
    // Text color codes:
    // 30=black 31=red 32=green 33=yellow 34=blue 35=magenta 36=cyan 37=white
    const char esc[] = { 0x1b, '[', '\0' };
    const std::string ansi_esc          = esc;
    const std::string ansi_bold_red     = ansi_esc + "01;31m";
    const std::string ansi_bold_magenta = ansi_esc + "01;35m";
    const std::string ansi_norm         = ansi_esc + "0m";
    
    std::string error_pre = ansi_bold_red,     error_post = " !" + ansi_norm;  // current error
    std::string  back_pre = ansi_bold_magenta, back_post  = " ?" + ansi_norm;  // backwards compatability issue
    
    if ( lwork_lapack.value < lwork_min.value ) {
        // current lapack lwork error: require lwork >= min
        lwork_lapack.pre  = error_pre;
        lwork_lapack.post = error_post;
    }
    else if ( lwork_min_old.value && lwork_lapack.value < lwork_min_old.value ) {
        // backwards compatability issue: require lwork >= min_old, for LAPACK <= 3.6
        lwork_lapack.pre  = back_pre;
        lwork_lapack.post = back_post;
    }
    if ( lwork_magma.value < lwork_opt.value ) {
        // current magma lwork error: require lwork >= optimal
        lwork_magma.pre  = error_pre;
        lwork_magma.post = error_post;
    }
}
                
                
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesdd (SVD with Divide & Conquer)
      Please keep code in testing_zgesdd.cpp and testing_zgesvd.cpp similar.
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // Constants
    const magma_int_t ione     = 1;
    const magma_int_t ineg_one = -1;
    const double d_neg_one = -1;
    const double nan = MAGMA_D_NAN;
    
    // Local variables
    real_Double_t   gpu_time=0, cpu_time=0;
    magmaDoubleComplex *hA, *hR, *U, *Umalloc, *VT, *VTmalloc, *hwork;
    magmaDoubleComplex dummy[1], unused[1];
    double *S, *Sref, work[1], runused[1];
    #ifdef COMPLEX
    lwork_formula_t lrwork;
    double *rwork;
    #endif
    magma_int_t *iwork, iunused[1];
    magma_int_t M, N, N_U, M_VT, lda, ldu, ldv, min_mn, info;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    std::string work_str = "unknown";
    if ( opts.svd_work[0] == MagmaSVD_all ) {
        opts.svd_work.clear();
        opts.svd_work.push_back( MagmaSVD_min   );
        opts.svd_work.push_back( MagmaSVD_doc   );
        opts.svd_work.push_back( MagmaSVD_opt   );
        opts.svd_work.push_back( MagmaSVD_max   );
        opts.svd_work.push_back( MagmaSVD_query );
    }
    if ( opts.svd_work.size() > 1 ) {
        // set verbose = 1, at least, if doing multiple svd work sizes,
        // to print what size is tested
        opts.verbose = max( opts.verbose, 1 );
    }
    
    printf( "%% jobz     M     N   CPU time (sec)   GPU time (sec)" );
    if ( opts.magma ) {
        printf( "   |S-Sref|   |A-USV^H|  |I-UU^H|/M  |I-VV^H|/N" );
    }
    if ( opts.lapack && opts.check == 2 ) {
        printf( "   lapack |A-USV^H|  |I-UU^H|/M  |I-VV^H|/N" );
    }
    printf( "   S sorted" );
    if ( opts.verbose ) {
        // zgesdd_path
        printf( "       path [-path-]  lwork         magma      lapack   formula" );
        #ifdef COMPLEX
        printf( "      lrwork   formula" );
        #endif
    }
    printf( "\n" );
    
    printf( "%%==============================================================" );
    if ( opts.magma ) {
        printf( "===============================================" );
    }
    if ( opts.lapack && opts.check == 2 ) {
        printf( "===========================================" );
    }
    if ( opts.verbose ) {
        printf( "=======================================================" );
    }
    printf( "\n" );
    
    std::vector< magma_int_t > prev_magma_lwork;
    std::vector< magma_int_t > prev_lapack_lwork;
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( auto jobz = opts.jobu.begin(); jobz != opts.jobu.end(); ++jobz ) {
        for( auto svd_work = opts.svd_work.begin(); svd_work != opts.svd_work.end(); ++svd_work ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            N_U  = (*jobz == MagmaAllVec ? M : min_mn);
            M_VT = (*jobz == MagmaAllVec ? N : min_mn);
            lda = M;
            ldu = M;
            ldv = M_VT;
            
            /* =====================================================================
               query for workspace size
               =================================================================== */
            magma_int_t query_magma, query_lapack;
            magma_zgesdd( *jobz, M, N,
                          unused, lda, runused,
                          unused, ldu,
                          unused, ldv,
                          dummy, ineg_one,
                          #ifdef COMPLEX
                          runused,
                          #endif
                          iunused, &info );
            assert( info == 0 );
            query_magma = (magma_int_t) MAGMA_Z_REAL( dummy[0] );
            
            lapackf77_zgesdd( lapack_vec_const(*jobz), &M, &N,
                              unused, &lda, runused,
                              unused, &ldu,
                              unused, &ldv,
                              dummy, &ineg_one,
                              #ifdef COMPLEX
                              runused,
                              #endif
                              iunused, &info );
            assert( info == 0 );
            query_lapack = (magma_int_t) MAGMA_Z_REAL( dummy[0] );
            
            // Choose lwork size based on --svd-work option.
            // We recommend using the above query for lwork rather than
            // the formulas; we use formulas to verify the code in all cases.
            // lwork_formula_t is a special class, just for the tester, that
            // saves the lwork value together with a string describing its formula.
            lwork_formula_t lwork_magma, lwork_lapack;
            std::string path_str;
            choose_lwork( opts, *svd_work, *jobz, M, N, query_magma, query_lapack,
                          lwork_magma, lwork_lapack, work_str, path_str );
            
            // LAPACK and MAGMA may return different sizes;
            // since we call both, allocate max.
            magma_int_t lwork = max( lwork_magma.value, lwork_lapack.value );
            
            // skip lwork sizes we've already done for this m, n, job
            if ( ( ! opts.magma  || contains( prev_magma_lwork,  lwork_magma.value  )) &&
                 ( ! opts.lapack || contains( prev_lapack_lwork, lwork_lapack.value )) ) {
                // %78s without zgesdd_path
                printf( "   %c   %5lld %5lld   skipping repeated lwork %87s %-9s %9lld   %9lld   %s\n",
                        lapacke_vec_const(*jobz),
                        (long long) M, (long long) N,
                        "", work_str.c_str(),
                        (long long) lwork_magma.value,
                        (long long) lwork_lapack.value,
                        lwork_lapack.formula.c_str() );
                break;
            }
            prev_magma_lwork.push_back( lwork_magma.value );
            prev_lapack_lwork.push_back( lwork_lapack.value );
            
            // real workspace
            #ifdef COMPLEX
            magma_int_t mx = max( M, N );
            magma_int_t mn = min( M, N );
            magma_int_t mnthr = (magma_int_t) (mn * 17. / 9.);
            if (*jobz == MagmaNoVec) {
                // LAPACK 3.6 changed documentation from 5*mn to 7*mn.
                // This is due to dbdsdc incorrectly claiming it needs 4*n,
                // when actually it needed 6*n in some cases.
                // dbdsdc is fixed in LAPACK >= 3.7.0.
                //assign( lrwork, 5*mn );
                assign( lrwork, 7*mn );
            }
            else if (mx >= mnthr) {
                // LAPACK doesn't document the mx >> mn case,
                // but it doesn't need nearly as much as next case.
                assign( lrwork, 5*mn*mn + 5*mn );
            }
            else {
                // LAPACK's docs have 5*mn*mn + 7*mn for 1st term here,
                // but it doesn't need that much
                assign( lrwork, max( 5*mn*mn + 5*mn,
                                     2*mx*mn + 2*mn*mn + mn ) );
            }
            #endif
            
            /* =====================================================================
               Allocate memory
               =================================================================== */
            TESTING_CHECK( magma_zmalloc_cpu( &hA,    lda*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &S,     min_mn ));
            TESTING_CHECK( magma_dmalloc_cpu( &Sref,  min_mn ));
            TESTING_CHECK( magma_imalloc_cpu( &iwork, 8*min_mn ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &hR,    lda*N ));
            TESTING_CHECK( magma_zmalloc_pinned( &hwork, lwork ));
            
            // U and VT either overwrite hR, or are allocated as Umalloc, VTmalloc
            if ( *jobz == MagmaOverwriteVec && M >= N ) {
                U   = hR;
                ldu = lda;
                Umalloc = NULL;
            }
            else {
                TESTING_CHECK( magma_zmalloc_cpu( &Umalloc, ldu*N_U )); // M x M (jobz=A) or M x min(M,N)
                U = Umalloc;
            }
            if ( *jobz == MagmaOverwriteVec && M < N ) {
                VT  = hR;
                ldv = lda;
                VTmalloc = NULL;
            }
            else {
                TESTING_CHECK( magma_zmalloc_cpu( &VTmalloc, ldv*N )); // N x N (jobz=A) or min(M,N) x N
                VT = VTmalloc;
            }
            
            #ifdef COMPLEX
            TESTING_CHECK( magma_dmalloc_cpu( &rwork, lrwork.value ));
            #endif
            
            // map gesdd jobz to gesvd jobu/jobv for check
            magma_vec_t jobu = *jobz;
            magma_vec_t jobv = *jobz;
            if ( *jobz == MagmaOverwriteVec ) {
                if ( M >= N ) {
                    jobv = MagmaSomeVec;  // jobu is Overwrite
                }
                else {
                    jobu = MagmaSomeVec;  // jobv is Overwrite
                }
            }
            
            // force check to fail if gesdd returns info error
            double result[5]        = { nan, nan, nan, nan, nan };
            double result_lapack[5] = { nan, nan, nan, nan, nan };
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, M, N, hA, lda, Sref );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, hA, &lda, hR, &lda );
            
            // ----------
            if (opts.verbose) {
                printf( "A = " );  magma_zprint( M, N, hA, lda );
                printf( "S = " );  magma_dprint( 1, min_mn, Sref, 1 );
            }
            
            if ( opts.magma ) {
                /* ====================================================================
                   Performs operation using MAGMA
                   =================================================================== */
                magma_flush_cache( opts.cache );
                gpu_time = magma_wtime();
                magma_zgesdd( *jobz, M, N,
                              hR, lda, S, U, ldu, VT, ldv, hwork, lwork_magma.value,
                              #ifdef COMPLEX
                              rwork,
                              #endif
                              iwork, &info );
                gpu_time = magma_wtime() - gpu_time;
                
                const char *func = "magma_zgesdd";
                if ( *svd_work == MagmaSVD_min_1 || *svd_work == MagmaSVD_min_old_1 ) {
                    if (info == -12) {
                        printf( "ok: with lwork = min-1 = %lld, %s returned expected info = %lld\n",
                                (long long) lwork_magma.value, func, (long long) info );
                    }
                    else {
                        printf( "failed: with lwork = min-1 = %lld, %s returned unexpected info = %lld; expected info = -12\n",
                                (long long) lwork_magma.value, func, (long long) info );
                        status += 1;
                    }
                }
                else if (info != 0) {
                    printf( "%s returned error %lld: %s.\n",
                            func, (long long) info, magma_strerror( info ));
                    status += 1;
                }
                
                /* ====================================================================
                   Check the results
                   =================================================================== */
                if ( info == 0 ) {
                    check_zgesvd( opts.check, jobu, jobv, M, N, hA, lda, S, U, ldu, VT, ldv, result );
                }
                if (opts.verbose) {
                    printf( "Smagma = " );  magma_dprint( 1, min_mn, S, 1 );
                }
            }
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                lapackf77_zlacpy( MagmaFullStr, &M, &N, hA, &lda, hR, &lda );
                magma_flush_cache( opts.cache );
                cpu_time = magma_wtime();
                lapackf77_zgesdd( lapack_vec_const(*jobz), &M, &N,
                                  hR, &lda, Sref, U, &ldu, VT, &ldv, hwork, &lwork_lapack.value,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  iwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                
                const char *func = "lapackf77_zgesdd";
                if ( *svd_work == MagmaSVD_min_1 || *svd_work == MagmaSVD_min_old_1 ) {
                    if (info == -12) {
                        printf( "ok: with lwork = min-1 = %lld, %s returned expected info = %lld\n",
                                (long long) lwork_magma.value, func, (long long) info );
                    }
                    else {
                        printf( "failed: with lwork = min-1 = %lld, %s returned unexpected info = %lld; expected info = -12\n",
                                (long long) lwork_magma.value, func, (long long) info );
                        status += 1;
                    }
                }
                else if (info != 0) {
                    printf( "%s returned error %lld: %s.\n",
                            func, (long long) info, magma_strerror( info ));
                    status += 1;
                }
                
                /* =====================================================================
                   Check LAPACK's result
                   =================================================================== */
                if ( info == 0 && opts.check == 2 ) {
                    check_zgesvd( opts.check, jobu, jobv, M, N, hA, lda, Sref, U, ldu, VT, ldv, result_lapack );
                }
                if (opts.verbose) {
                    printf( "Slapack = " );  magma_dprint( 1, min_mn, Sref, 1 );
                }
                
                /* =====================================================================
                   Check MAGMA's singular values compared to LAPACK
                   =================================================================== */
                if ( opts.magma ) {
                    blasf77_daxpy( &min_mn, &d_neg_one, S, &ione, Sref, &ione );
                    result[4]  = lapackf77_dlange( "f", &min_mn, &ione, Sref, &min_mn, work );
                    result[4] /= lapackf77_dlange( "f", &min_mn, &ione, S,    &min_mn, work );
                }
                printf( "   %c   %5lld %5lld   %9.4f        %9.4f     ",
                        lapacke_vec_const(*jobz),
                        (long long) M, (long long) N,
                        cpu_time, gpu_time );
            }
            else {
                result[4] = -1;  // indicates S - Sref not checked
                printf( "   %c   %5lld %5lld      ---           %9.4f     ",
                        lapacke_vec_const(*jobz),
                        (long long) M, (long long) N,
                        gpu_time );
            }
            
            /* =====================================================================
               Print error checks
               =================================================================== */
            bool okay   = true;
            bool sorted = true;
            if ( opts.magma ) {
                if ( result[4] < 0. ) { printf(  "     ---   " ); } else { printf(  "   %8.2e", result[4]); }  // S - Sref
                if ( result[0] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[0]); }  // A - USV'
                if ( result[1] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[1]); }  // I - UU'
                if ( result[2] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[2]); }  // I - VV'
                okay = okay && (result[0] < tol) && (result[1] < tol)
                            && (result[2] < tol) && (result[3] == 0.)
                            && (result[4] < tol);
                sorted = sorted && (result[3] == 0.);
            }
            if ( opts.lapack && opts.check == 2 ) {
                printf( "       " );
                if ( result_lapack[0] < 0. ) { printf("      ---   "); } else { printf("    %8.2e", result_lapack[0]); }
                if ( result_lapack[1] < 0. ) { printf("      ---   "); } else { printf("    %8.2e", result_lapack[1]); }
                if ( result_lapack[2] < 0. ) { printf("      ---   "); } else { printf("    %8.2e", result_lapack[2]); }
                okay = okay && (result_lapack[0] < tol) && (result_lapack[1] < tol)
                            && (result_lapack[2] < tol) && (result_lapack[3] == 0.);
                sorted = sorted && (result_lapack[3] == 0.);
            }
            status += ! okay;
            printf( "   %-3s   %-6s", (sorted ? "yes" : "no"), (okay ? "ok" : "failed") );
            
            /* =====================================================================
               Print lwork sizes
               =================================================================== */
            if ( opts.verbose ) {
                printf( "   %-4s %-8s   %-9s %s%9lld%2s %s%9lld%2s %s",
                        path_str.c_str(),
                        zgesdd_path,
                        work_str.c_str(),
                        lwork_magma.pre.c_str(),  (long long) lwork_magma.value,  lwork_magma.post.c_str(),
                        lwork_lapack.pre.c_str(), (long long) lwork_lapack.value, lwork_lapack.post.c_str(),
                        lwork_lapack.formula.c_str() );
                #ifdef COMPLEX
                printf( "  %9lld   %s", (long long) lrwork.value, lrwork.formula.c_str() );
                #endif
            }
            printf( "\n" );
            
            magma_free_cpu( hA );
            magma_free_cpu( S  );
            magma_free_cpu( Sref );
            magma_free_cpu( iwork );
            
            magma_free_pinned( hR    );
            magma_free_pinned( hwork );
            
            magma_free_cpu( VTmalloc );
            magma_free_cpu( Umalloc  );
            
            #ifdef COMPLEX
            magma_free_cpu( rwork );
            #endif
            
            fflush( stdout );
        }}  // iter, svd_work
        if ( opts.niter > 1 || opts.svd_work.size() > 1 ) {
            printf( "\n" );
        }
        prev_magma_lwork.clear();
        prev_lapack_lwork.clear();
      }  // job
      if ( opts.jobu.size() > 1 ) {
          printf( "%%----------\n" );
      }
    }
    
    if ( opts.verbose ) {
        printf( "\n"
                "%% Error codes:  !  error:  lwork < min. For (min-1), this ought to appear.\n"
                "%%               ?  compatability issue:  lwork < min_old, will fail for lapack <= 3.6.\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
