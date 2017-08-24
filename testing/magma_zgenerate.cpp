/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Mark Gates

       Test matrix generation.
*/

#include <exception>
#include <string>
#include <vector>
#include <limits>

#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"

#include "magma_matrix.hpp"
#include "testings.h"

/******************************************************************************/
// constants

enum class MatrixType {
    uniform01 = 1,  // maps to larnv idist
    uniform11 = 2,  // maps to larnv idist
    normal    = 3,  // maps to larnv idist
    zero,
    identity,
    jordan,
    diag,
    svd,
    poev,
    syev,
    geev,
    geevx,
};

enum class Dist {
    uniform01 = 1,  // maps to larnv idist
    uniform11 = 2,  // maps to larnv idist
    normal    = 3,  // maps to larnv idist
    arith,
    geo,
    cluster,
    cluster2,
    rarith,
    rgeo,
    rcluster,
    rcluster2,
    logrand,
    specified,
};

/******************************************************************************/
// random number in (0, max_]
inline double rand( double max_ )
{
    return max_ * rand() / double(RAND_MAX);
}

/******************************************************************************/
// true if str begins with prefix
inline bool begins( std::string const &str, std::string const &prefix )
{
    return (str.compare( 0, prefix.size(), prefix) == 0);
}

/******************************************************************************/
// true if str contains pattern
inline bool contains( std::string const &str, std::string const &pattern )
{
    return (str.find( pattern ) != std::string::npos);
}

/******************************************************************************/
void magma_generate_sigma(
    Dist dist, bool rand_sign, double cond, double sigma_max, magma_int_t iseed[4],
    Vector<double>& sigma,
    Matrix<magmaDoubleComplex>& A )
{
    // constants
    const magma_int_t ione = 1;
    const magma_int_t idist_uniform01 = 1;
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;

    // locals
    magma_int_t minmn = min( A.m, A.n );
    assert( minmn == sigma.n );

    switch (dist) {
        case Dist::arith:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - i / double(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::rarith:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - (minmn - 1 - i) / double(minmn - 1) * (1 - 1/cond);
            }
            break;

        case Dist::geo:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -i / double(minmn - 1) );
            }
            break;

        case Dist::rgeo:
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -(minmn - 1 - i) / double(minmn - 1) );
            }
            break;

        case Dist::cluster:
            sigma[0] = 1;
            for (magma_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case Dist::rcluster:
            for (magma_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case Dist::cluster2:
            for (magma_int_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case Dist::rcluster2:
            sigma[0] = 1/cond;
            for (magma_int_t i = 1; i < minmn; ++i) {
                sigma[i] = 1;
            }
            break;

        case Dist::logrand: {
            double range = log( 1/cond );
            lapackf77_dlarnv( &idist_uniform01, iseed, &sigma.n, sigma(0) );
            for (magma_int_t i = 0; i < minmn; ++i) {
                sigma[i] = exp( sigma[i] * range );
            }
            break;
        }

        case Dist::normal:
        case Dist::uniform11:
        case Dist::uniform01: {
            // randn, randu, or rand already specifies sign; don't change it
            rand_sign = false;
            magma_int_t idist = (magma_int_t) dist;
            lapackf77_dlarnv( &idist, iseed, &sigma.n, sigma(0) );
            break;
        }

        case Dist::specified:
            // user-specified sigma values; don't modify
            sigma_max = 1;
            rand_sign = false;
            break;
    }

    if (sigma_max != 1) {
        blasf77_dscal( &sigma.n, &sigma_max, sigma(0), &ione );
    }

    if (rand_sign) {
        // apply random signs
        for (magma_int_t i = 0; i < minmn; ++i) {
            if (rand() > RAND_MAX/2) {
                sigma[i] = -sigma[i];
            }
        }
    }

    // copy sigma => A
    lapackf77_zlaset( "general", &A.m, &A.n, &c_zero, &c_zero, A(0,0), &A.ld );
    for (magma_int_t i = 0; i < minmn; ++i) {
        *A(i,i) = MAGMA_Z_MAKE( sigma[i], 0 );
    }
}

/******************************************************************************/
void magma_generate_svd(
    Dist dist, double cond, double condD, double sigma_max,
    magma_int_t iseed[4],
    Vector<double>& sigma,
    Matrix<magmaDoubleComplex>& A )
{
    // constants
    const magma_int_t ione = 1;
    const magma_int_t idist_normal = 3;

    // locals
    magmaDoubleComplex tmp;
    magma_int_t m = A.m;
    magma_int_t n = A.n;
    magma_int_t maxmn = max( m, n );
    magma_int_t minmn = min( m, n );
    magma_int_t sizeU;
    magma_int_t info = 0;
    Matrix<magmaDoubleComplex> U( maxmn, minmn );
    Vector<magmaDoubleComplex> tau( minmn );

    // query for workspace
    magma_int_t lwork = -1;
    lapackf77_zunmqr( "Left", "NoTrans", &A.m, &A.n, &minmn,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      &tmp, &lwork, &info );
    assert( info == 0 );
    lwork = magma_int_t( real( tmp ));
    magma_int_t lwork2 = -1;
    lapackf77_zunmqr( "Right", Magma_ConjTransStr, &A.m, &A.n, &minmn,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      &tmp, &lwork2, &info );
    assert( info == 0 );
    lwork2 = magma_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<magmaDoubleComplex> work( lwork );

    // ----------
    magma_generate_sigma( dist, false, cond, sigma_max, iseed, sigma, A );

    // random U, m-by-minmn
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapackf77_zlarnv( &idist_normal, iseed, &sizeU, U(0,0) );
    for (magma_int_t j = 0; j < minmn; ++j) {
        magma_int_t mj = m - j;
        lapackf77_zlarfg( &mj, U(j,j), U(j+1,j), &ione, tau(j) );
    }

    // A = U*A
    lapackf77_zunmqr( "Left", "NoTrans", &A.m, &A.n, &minmn,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      work(0), &lwork, &info );
    assert( info == 0 );

    // random V, n-by-minmn (stored column-wise in U)
    lapackf77_zlarnv( &idist_normal, iseed, &sizeU, U(0,0) );
    for (magma_int_t j = 0; j < minmn; ++j) {
        magma_int_t nj = n - j;
        lapackf77_zlarfg( &nj, U(j,j), U(j+1,j), &ione, tau(j) );
    }

    // A = A*V^H
    lapackf77_zunmqr( "Right", Magma_ConjTransStr, &A.m, &A.n, &minmn,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      work(0), &lwork, &info );
    assert( info == 0 );
}

/******************************************************************************/
void magma_generate_syev(
    Dist dist, bool rand_sign, double cond, double condD, double sigma_max,
    magma_int_t iseed[4],
    Vector<double>& sigma,
    Matrix<magmaDoubleComplex>& A )
{
    // check inputs
    assert( A.m == A.n );

    // constants
    const magma_int_t ione = 1;
    const magma_int_t idist_normal = 3;

    // locals
    magmaDoubleComplex tmp;
    magma_int_t n = A.n;
    magma_int_t sizeU;
    magma_int_t info = 0;
    Matrix<magmaDoubleComplex> U( n, n );
    Vector<magmaDoubleComplex> tau( n );

    // query for workspace
    magma_int_t lwork = -1;
    lapackf77_zunmqr( "Left", "NoTrans", &n, &n, &n,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      &tmp, &lwork, &info );
    assert( info == 0 );
    lwork = magma_int_t( real( tmp ));
    magma_int_t lwork2 = -1;
    lapackf77_zunmqr( "Right", Magma_ConjTransStr, &n, &n, &n,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      &tmp, &lwork2, &info );
    assert( info == 0 );
    lwork2 = magma_int_t( real( tmp ));
    lwork = max( lwork, lwork2 );
    Vector<magmaDoubleComplex> work( lwork );

    // ----------
    magma_generate_sigma( dist, rand_sign, cond, sigma_max, iseed, sigma, A );

    // random U, n-by-n
    // just make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    sizeU = U.size();
    lapackf77_zlarnv( &idist_normal, iseed, &sizeU, U(0,0) );
    for (magma_int_t j = 0; j < n; ++j) {
        magma_int_t nj = n - j;
        lapackf77_zlarfg( &nj, U(j,j), U(j+1,j), &ione, tau(j) );
    }

    // A = U*A
    lapackf77_zunmqr( "Left", "NoTrans", &n, &n, &n,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      work(0), &lwork, &info );
    assert( info == 0 );

    // A = A*U^H
    lapackf77_zunmqr( "Right", Magma_ConjTransStr, &n, &n, &n,
                      U(0,0), &U.ld, tau(0), A(0,0), &A.ld,
                      work(0), &lwork, &info );
    assert( info == 0 );

    // make diagonal real
    // usually LAPACK ignores imaginary part anyway, but Matlab doesn't
    for (int i = 0; i < n; ++i) {
        *A(i,i) = MAGMA_Z_MAKE( real( *A(i,i) ), 0 );
    }
}

/******************************************************************************/
void magma_generate_geev(
    Dist dist, double cond, double condD, double sigma_max,
    magma_int_t iseed[4],
    Vector<double>& sigma,
    Matrix<magmaDoubleComplex>& A )
{
    throw std::exception();  // not implemented
}

/******************************************************************************/
void magma_generate_geevx(
    Dist dist, double cond, double condD, double sigma_max,
    magma_int_t iseed[4],
    Vector<double>& sigma,
    Matrix<magmaDoubleComplex>& A )
{
    throw std::exception();  // not implemented
}

/***************************************************************************//**
    Purpose
    -------
    Generate an m-by-n test matrix A.
    Does not use LAPACK's libtmg.

    Arguments
    ---------
    @param[in]
    opts    MAGMA options. Uses matrix, cond, condD; see further details.

    @param[in]
    iseed   Integer array, dimension (4)
            Random number generator seed.
            Elements in [0, 4095], last element odd.

    @param[in]
    m       Integer
            The number of rows of the matrix A.  m >= 0.

    @param[in]
    n       Integer
            The number of columns of the matrix A.  n >= 0.

    @param[in,out]
    sigma   Double precision array, dimension (min(m,n))
            For matrix with "_specified", on input contains user-specified
            singular or eigenvalues.
            On output, contains singular or eigenvalues, if known,
            else set to NaN. sigma is not necesarily sorted.

    @param[out]
    A       COMPLEX_16 array, dimension (lda, n).
            On output, the m-by-n test matrix A.

    @param[in]
    lda     Integer
            The leading dimension of the array A.

    Further Details
    ---------------
    The --matrix command line option specifies the matrix name according to the
    table below. Where indicated, names take an optional distribution suffix (#)
    and an optional scaling suffix (*). The default distribution is rand.
    Examples: rand, rand_small, svd_arith, syev_geo_small.

    The --cond and --condD command line options specify condition numbers as
    described below. Default cond = 1/eps (eps = 2.2e-16 for double), condD = 1.

    Sigma is a diagonal matrix with entries sigma_i for i = 1, ..., n;
    Lambda is a diagonal matrix with entries lambda_i = sigma_i with random sign;
    U and V are random orthogonal matrices from the Haar distribution
    (See: Stewart, The efficient generation of random orthogonal matrices
     with an application to condition estimators, 1980);
    X is a random matrix.

    See LAPACK Working Note (LAWN) 41:
    Table  5 (Test matrices for the nonsymmetric eigenvalue problem)
    Table 10 (Test matrices for the symmetric eigenvalue problem)
    Table 11 (Test matrices for the singular value decomposition)

    Matrix      Description
    zero
    identity
    jordan      ones on diagonal and first subdiagonal

    rand*       matrix entries random uniform on (0, 1)
    randu*      matrix entries random uniform on (-1, 1)
    randn*      matrix entries random normal with mean 0, sigma 1

    diag#*      A = Sigma
    svd#*       A = U Sigma V^H
    poev#*      A = V Sigma V^H  (eigenvalues positive)
    syev#*      A = V Lambda V^H (eigenvalues mixed signs)
    geev#*      A = V T V^H, Schur-form T                       [not yet implemented]
    geevx#*     A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]

    # optional distribution suffix
    _rand       sigma_i random uniform on (0, 1) [default]
    _randu      sigma_i random uniform on (-1, 1)
    _randn      sigma_i random normal with mean 0, std 1
                Note for _randu and _randn, Sigma contains negative values.
                _rand* do not use cond, so the condition number is arbitrary.

    _logrand    log(sigma_i) uniform on (log(1/cond), log(1))
    _arith      sigma_i = 1 - (i - 1)/(n - 1)*(1 - 1/cond); sigma_{i+1} - sigma_i is constant
    _geo        sigma_i = (cond)^{ -(i-1)/(n-1) };          sigma_{i+1} / sigma_i is constant
    _cluster    sigma = [ 1, 1/cond, ..., 1/cond ]; 1 unit value, n-1 small values
    _cluster2   sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values, 1 small value
    _rarith     _arith,    reversed order
    _rgeo       _geo,      reversed order
    _rcluster   _cluster,  reversed order
    _rcluster2  _cluster2, reversed order
    _specified  user specified sigma on input

    * optional scaling & modifier suffix
    _ufl        scale near underflow         = 1e-308 for double
    _ofl        scale near overflow          = 2e+308 for double
    _small      scale near sqrt( underflow ) = 1e-154 for double
    _large      scale near sqrt( overflow  ) = 6e+153 for double
    _dominant   diagonally dominant: set A_ii = ± max( sum_j |A_ij|, sum_j |A_ji| )
                Note _dominant changes the singular or eigenvalues.

    [below not yet implemented]
    If condD != 1, then:
    For SVD, A = (U Sigma V^H) K D, where
    K is diagonal such that columns of (U Sigma V^H K) have unit norm,
    and D has log-random entries in ( log(1/condD), log(1) ).

    For syev, A0 = U Lambda U^H, A = D K A0 K D, where
    K is diagonal such that K A0 K) has unit diagonal,
    and D as above.

    Note using condD changes the singular or eigenvalues.
    See: Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.

    @ingroup testing
*******************************************************************************/
void magma_generate_matrix(
    magma_opts& opts,
    magma_int_t iseed[4],
    magma_int_t m, magma_int_t n,
    double* sigma_ptr,
    magmaDoubleComplex* A_ptr, magma_int_t lda )
{
    // constants
    const magma_int_t ione = 1;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double d_zero = MAGMA_D_ZERO;
    const double d_one  = MAGMA_D_ONE;
    const double ufl = lapackf77_dlamch( "safe min" );  //  1e-38 or  2e-308
    const double ofl = 1 / ufl;                         //   8e37 or   4e307
    const double eps = lapackf77_dlamch( "precision" ); // 1.2e-7 or 2.2e-16
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    // locals
    std::string name = opts.matrix;
    double cond  = opts.cond;
    if (cond == 0) {
        cond = 1/eps;
    }
    double condD = opts.cond;
    double sigma_max = 1;
    magma_int_t minmn = min( m, n );

    // vector & matrix wrappers
    Vector<double> sigma( sigma_ptr, minmn );
    Matrix<magmaDoubleComplex> A( A_ptr, m, n, lda );

    // ----------
    // set sigma to unknown (nan)
    lapackf77_dlaset( "general", &sigma.n, &ione, &nan, &nan, sigma(0), &sigma.n );

    // ----- decode matrix type
    MatrixType type = MatrixType::identity;
    if      (name == "zero")          { type = MatrixType::zero;      }
    else if (name == "identity")      { type = MatrixType::identity;  }
    else if (name == "jordan")        { type = MatrixType::jordan;    }
    else if (begins( name, "randn" )) { type = MatrixType::normal;    }
    else if (begins( name, "randu" )) { type = MatrixType::uniform11; }
    else if (begins( name, "rand"  )) { type = MatrixType::uniform01; }
    else if (begins( name, "diag"  )) { type = MatrixType::diag;      }
    else if (begins( name, "svd"   )) { type = MatrixType::svd;       }
    else if (begins( name, "poev"  )) { type = MatrixType::poev;      }
    else if (begins( name, "syev"  )) { type = MatrixType::syev;      }
    else if (begins( name, "geevx" )) { type = MatrixType::geevx;     }
    else if (begins( name, "geev"  )) { type = MatrixType::geev;      }
    else {
        fprintf( stderr, "Unrecognized matrix '%s'\n", name.c_str() );
        throw std::exception();
    }

    if (m != n &&
        (type == MatrixType::jordan ||
         type == MatrixType::poev   ||
         type == MatrixType::syev   ||
         type == MatrixType::geev   ||
         type == MatrixType::geevx))
    {
        fprintf( stderr, "Eigenvalue matrix requires m == n.\n" );
        throw std::exception();
    }

    // ----- decode distribution
    Dist dist = Dist::uniform01;
    if      (contains( name, "_randn"     )) { dist = Dist::normal;    }
    else if (contains( name, "_randu"     )) { dist = Dist::uniform11; }
    else if (contains( name, "_rand"      )) { dist = Dist::uniform01; } // after randn, randu
    else if (contains( name, "_logrand"   )) { dist = Dist::logrand;   }
    else if (contains( name, "_arith"     )) { dist = Dist::arith;     }
    else if (contains( name, "_geo"       )) { dist = Dist::geo;       }
    else if (contains( name, "_cluster2"  )) { dist = Dist::cluster2;  }
    else if (contains( name, "_cluster"   )) { dist = Dist::cluster;   } // after cluster2
    else if (contains( name, "_rarith"    )) { dist = Dist::rarith;    }
    else if (contains( name, "_rgeo"      )) { dist = Dist::rgeo;      }
    else if (contains( name, "_rcluster2" )) { dist = Dist::rcluster2; }
    else if (contains( name, "_rcluster"  )) { dist = Dist::rcluster;  } // after rcluster2
    else if (contains( name, "_specified" )) { dist = Dist::specified; }

    // ----- decode scaling
    if      (contains( name, "_small"  )) { sigma_max = sqrt( ufl ); }
    else if (contains( name, "_large"  )) { sigma_max = sqrt( ofl ); }
    else if (contains( name, "_ufl"    )) { sigma_max = ufl; }
    else if (contains( name, "_ofl"    )) { sigma_max = ofl; }

    // ----- generate matrix
    switch (type) {
        case MatrixType::zero:
            lapackf77_zlaset( "general", &A.m, &A.n, &c_zero, &c_zero, A(0,0), &A.ld );
            lapackf77_dlaset( "general", &sigma.n, &ione, &d_zero, &d_zero, sigma(0), &sigma.n );
            break;

        case MatrixType::identity:
            lapackf77_zlaset( "general", &A.m, &A.n, &c_zero, &c_one, A(0,0), &A.ld );
            lapackf77_dlaset( "general", &sigma.n, &ione, &d_one, &d_one, sigma(0), &sigma.n );
            break;

        case MatrixType::jordan: {
            magma_int_t n1 = A.n - 1;
            lapackf77_zlaset( "upper", &A.n, &A.n, &c_zero, &c_one, A(0,0), &A.ld );  // ones on diagonal
            lapackf77_zlaset( "lower", &n1,  &n1,  &c_zero, &c_one, A(1,0), &A.ld );  // ones on sub-diagonal
            break;
        }

        case MatrixType::uniform01:
        case MatrixType::uniform11:
        case MatrixType::normal: {
            magma_int_t idist = (magma_int_t) type;
            magma_int_t sizeA = A.ld * A.n;
            lapackf77_zlarnv( &idist, iseed, &sizeA, A(0,0) );
            if (sigma_max != 1) {
                magmaDoubleComplex scale = MAGMA_Z_MAKE( sigma_max, 0 );
                blasf77_zscal( &sizeA, &scale, A(0,0), &ione );
            }
            break;
        }

        case MatrixType::diag:
            magma_generate_sigma( dist, false, cond, sigma_max, iseed, sigma, A );
            break;

        case MatrixType::svd:
            magma_generate_svd( dist, cond, condD, sigma_max, iseed, sigma, A );
            break;

        case MatrixType::poev:
            magma_generate_syev( dist, false, cond, condD, sigma_max, iseed, sigma, A );
            break;

        case MatrixType::syev:
            magma_generate_syev( dist, true, cond, condD, sigma_max, iseed, sigma, A );
            break;

        case MatrixType::geev:
            magma_generate_geev( dist, cond, condD, sigma_max, iseed, sigma, A );
            break;

        case MatrixType::geevx:
            magma_generate_geevx( dist, cond, condD, sigma_max, iseed, sigma, A );
            break;
    }

    if (contains( name, "_dominant" )) {
        // make diagonally dominant; strict unless diagonal has zeros
        for (int i = 0; i < minmn; ++i) {
            double sum = max( magma_cblas_dzasum( m, A(0,i), ione ),    // i-th col
                              magma_cblas_dzasum( n, A(i,0), A.ld ) );  // i-th row
            sum = copysign( sum, real( *A(i,i) ) );
            *A(i,i) = MAGMA_Z_MAKE( sum, 0 );
        }
        // reset sigma to unknown (nan)
        lapackf77_dlaset( "general", &sigma.n, &ione, &nan, &nan, sigma(0), &sigma.n );
    }
}
