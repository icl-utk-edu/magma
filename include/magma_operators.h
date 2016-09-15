/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mathieu Faverge
       @author Mark Gates
*/

#ifndef MAGMA_OPERATORS_H
#define MAGMA_OPERATORS_H

#ifdef __cplusplus

// __host__ and __device__ are defined in CUDA headers.
#include "magma_types.h"

#ifdef HAVE_clBLAS
#define __host__
#define __device__
#endif

/// @addtogroup magma_complex
/// In C++, including magma_operators.h defines the usual unary and binary
/// operators for complex numbers: +, +=, -, -=, *, *=, /, /=, ==, !=.
/// Additionally, real(), imag(), conj(), fabs(), and abs1() are defined
/// to apply to both complex and real numbers.
///
/// In C, there are equivalent macros:
/// MAGMA_Z_{MAKE, REAL, IMAG, ADD, SUB, MUL, DIV, ABS, ABS1, CONJ} for double-complex,
/// MAGMA_C_{...} for float-complex,
/// MAGMA_D_{...} for double,
/// MAGMA_S_{...} for float.
///
/// Just the double-complex versions are documented here.


// =============================================================================
// names to match C++ std complex functions

/// @return real component of complex number x; x for real number.
/// @ingroup magma_complex
__host__ __device__ static inline double real(const magmaDoubleComplex &x) { return MAGMA_Z_REAL(x); }
__host__ __device__ static inline float  real(const magmaFloatComplex  &x) { return MAGMA_C_REAL(x); }
__host__ __device__ static inline double real(const double             &x) { return x; }
__host__ __device__ static inline float  real(const float              &x) { return x; }

/// @return imaginary component of complex number x; 0 for real number.
/// @ingroup magma_complex
__host__ __device__ static inline double imag(const magmaDoubleComplex &x) { return MAGMA_Z_IMAG(x); }
__host__ __device__ static inline float  imag(const magmaFloatComplex  &x) { return MAGMA_C_IMAG(x); }
__host__ __device__ static inline double imag(const double             &x) { return 0.; }
__host__ __device__ static inline float  imag(const float              &x) { return 0.f; }

/// @return conjugate of complex number x; x for real number.
/// @ingroup magma_complex
__host__ __device__ static inline magmaDoubleComplex conj(const magmaDoubleComplex &x) { return MAGMA_Z_CONJ(x); }
__host__ __device__ static inline magmaFloatComplex  conj(const magmaFloatComplex  &x) { return MAGMA_C_CONJ(x); }
__host__ __device__ static inline double             conj(const double             &x) { return x; }
__host__ __device__ static inline float              conj(const float              &x) { return x; }

/// @return 2-norm absolute value of complex number x: sqrt( real(x)^2 + imag(x)^2 ).
///         math.h or cmath provide fabs for real numbers.
/// @ingroup magma_complex
__host__ __device__ static inline double fabs(const magmaDoubleComplex &x) { return MAGMA_Z_ABS(x); }
__host__ __device__ static inline float  fabs(const magmaFloatComplex  &x) { return MAGMA_C_ABS(x); }
//__host__ __device__ static inline float  fabs(const float              &x) { return MAGMA_S_ABS(x); }  // conflicts with std::fabs in .cu files
// already have fabs( double ) in math.h

/// @return 1-norm absolute value of complex nmuber x: | real(x) | + | imag(x) |.
/// @ingroup magma_complex
__host__ __device__ static inline double abs1(const magmaDoubleComplex &x) { return MAGMA_Z_ABS1(x); }
__host__ __device__ static inline float  abs1(const magmaFloatComplex  &x) { return MAGMA_C_ABS1(x); }
__host__ __device__ static inline double abs1(const double             &x) { return MAGMA_D_ABS1(x); }
__host__ __device__ static inline float  abs1(const float              &x) { return MAGMA_S_ABS1(x); }


// =============================================================================
// magmaDoubleComplex

// ---------- negate
__host__ __device__ static inline magmaDoubleComplex
operator - (const magmaDoubleComplex &a)
{
    return MAGMA_Z_MAKE( -real(a),
                         -imag(a) );
}


// ---------- add
__host__ __device__ static inline magmaDoubleComplex
operator + (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE( real(a) + real(b),
                         imag(a) + imag(b) );
}

__host__ __device__ static inline magmaDoubleComplex
operator + (const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE( real(a) + s,
                         imag(a) );
}

__host__ __device__ static inline magmaDoubleComplex
operator + (const double s, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE( s + real(b),
                             imag(b) );
}

__host__ __device__ static inline magmaDoubleComplex&
operator += (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a = MAGMA_Z_MAKE( real(a) + real(b),
                      imag(a) + imag(b) );
    return a;
}

__host__ __device__ static inline magmaDoubleComplex&
operator += (magmaDoubleComplex &a, const double s)
{
    a = MAGMA_Z_MAKE( real(a) + s,
                      imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__ static inline magmaDoubleComplex
operator - (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE( real(a) - real(b),
                         imag(a) - imag(b) );
}

__host__ __device__ static inline magmaDoubleComplex
operator - (const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE( real(a) - s,
                         imag(a) );
}

__host__ __device__ static inline magmaDoubleComplex
operator - (const double s, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE( s - real(b),
                           - imag(b) );
}

__host__ __device__ static inline magmaDoubleComplex&
operator -= (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a = MAGMA_Z_MAKE( real(a) - real(b),
                      imag(a) - imag(b) );
    return a;
}

__host__ __device__ static inline magmaDoubleComplex&
operator -= (magmaDoubleComplex &a, const double s)
{
    a = MAGMA_Z_MAKE( real(a) - s,
                      imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__ static inline magmaDoubleComplex
operator * (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE( real(a)*real(b) - imag(a)*imag(b),
                         imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__ static inline magmaDoubleComplex
operator * (const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline magmaDoubleComplex
operator * (const double s, const magmaDoubleComplex a)
{
    return MAGMA_Z_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline magmaDoubleComplex&
operator *= (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a = MAGMA_Z_MAKE( real(a)*real(b) - imag(a)*imag(b),
                      imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__ static inline magmaDoubleComplex&
operator *= (magmaDoubleComplex &a, const double s)
{
    a = MAGMA_Z_MAKE( real(a)*s,
                      imag(a)*s );
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__ static inline magmaDoubleComplex
operator / (const magmaDoubleComplex x, const magmaDoubleComplex y)
{
    double a = real(x);
    double b = imag(x);
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return MAGMA_Z_MAKE( p, q );
}

__host__ __device__ static inline magmaDoubleComplex
operator / (const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE( real(a)/s,
                         imag(a)/s );
}

__host__ __device__ static inline magmaDoubleComplex
operator / (const double a, const magmaDoubleComplex y)
{
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return MAGMA_Z_MAKE( p, q );
}

__host__ __device__ static inline magmaDoubleComplex&
operator /= (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__ static inline magmaDoubleComplex&
operator /= (magmaDoubleComplex &a, const double s)
{
    a = MAGMA_Z_MAKE( real(a)/s,
                      imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__ static inline bool
operator == (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__ static inline bool
operator == (const magmaDoubleComplex a, const double s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__ static inline bool
operator == (const double s, const magmaDoubleComplex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__ static inline bool
operator != (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return ! (a == b);
}

__host__ __device__ static inline bool
operator != (const magmaDoubleComplex a, const double s)
{
    return ! (a == s);
}

__host__ __device__ static inline bool
operator != (const double s, const magmaDoubleComplex a)
{
    return ! (a == s);
}


// =============================================================================
// magmaFloatComplex

// ---------- negate
__host__ __device__ static inline magmaFloatComplex
operator - (const magmaFloatComplex &a)
{
    return MAGMA_C_MAKE( -real(a),
                         -imag(a) );
}


// ---------- add
__host__ __device__ static inline magmaFloatComplex
operator + (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE( real(a) + real(b),
                         imag(a) + imag(b) );
}

__host__ __device__ static inline magmaFloatComplex
operator + (const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE( real(a) + s,
                         imag(a) );
}

__host__ __device__ static inline magmaFloatComplex
operator + (const float s, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE( s + real(b),
                             imag(b) );
}

__host__ __device__ static inline magmaFloatComplex&
operator += (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a = MAGMA_C_MAKE( real(a) + real(b),
                      imag(a) + imag(b) );
    return a;
}

__host__ __device__ static inline magmaFloatComplex&
operator += (magmaFloatComplex &a, const float s)
{
    a = MAGMA_C_MAKE( real(a) + s,
                      imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__ static inline magmaFloatComplex
operator - (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE( real(a) - real(b),
                         imag(a) - imag(b) );
}

__host__ __device__ static inline magmaFloatComplex
operator - (const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE( real(a) - s,
                         imag(a) );
}

__host__ __device__ static inline magmaFloatComplex
operator - (const float s, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE( s - real(b),
                           - imag(b) );
}

__host__ __device__ static inline magmaFloatComplex&
operator -= (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a = MAGMA_C_MAKE( real(a) - real(b),
                      imag(a) - imag(b) );
    return a;
}

__host__ __device__ static inline magmaFloatComplex&
operator -= (magmaFloatComplex &a, const float s)
{
    a = MAGMA_C_MAKE( real(a) - s,
                      imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__ static inline magmaFloatComplex
operator * (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE( real(a)*real(b) - imag(a)*imag(b),
                         imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__ static inline magmaFloatComplex
operator * (const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline magmaFloatComplex
operator * (const float s, const magmaFloatComplex a)
{
    return MAGMA_C_MAKE( real(a)*s,
                         imag(a)*s );
}

__host__ __device__ static inline magmaFloatComplex&
operator *= (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a = MAGMA_C_MAKE( real(a)*real(b) - imag(a)*imag(b),
                      imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__ static inline magmaFloatComplex&
operator *= (magmaFloatComplex &a, const float s)
{
    a = MAGMA_C_MAKE( real(a)*s,
                      imag(a)*s );
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__ static inline magmaFloatComplex
operator / (const magmaFloatComplex x, const magmaFloatComplex y)
{
    float a = real(x);
    float b = imag(x);
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return MAGMA_C_MAKE( p, q );
}

__host__ __device__ static inline magmaFloatComplex
operator / (const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE( real(a)/s,
                         imag(a)/s );
}

__host__ __device__ static inline magmaFloatComplex
operator / (const float a, const magmaFloatComplex y)
{
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if ( fabs( d ) < fabs( c ) ) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return MAGMA_C_MAKE( p, q );
}

__host__ __device__ static inline magmaFloatComplex&
operator /= (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__ static inline magmaFloatComplex&
operator /= (magmaFloatComplex &a, const float s)
{
    a = MAGMA_C_MAKE( real(a)/s,
                      imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__ static inline bool
operator == (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__ static inline bool
operator == (const magmaFloatComplex a, const float s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__ static inline bool
operator == (const float s, const magmaFloatComplex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__ static inline bool
operator != (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return ! (a == b);
}

__host__ __device__ static inline bool
operator != (const magmaFloatComplex a, const float s)
{
    return ! (a == s);
}

__host__ __device__ static inline bool
operator != (const float s, const magmaFloatComplex a)
{
    return ! (a == s);
}

#ifdef HAVE_clBLAS
#undef __host__
#undef __device__
#endif

#endif /* __cplusplus */

#endif /* MAGMA_OPERATORS_H */
