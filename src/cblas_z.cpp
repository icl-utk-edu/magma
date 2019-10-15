/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
       @precisions normal z -> s d c

    Wrappers around a few CBLAS functions.
    
    Primarily, we use the standard Fortran BLAS interface in MAGMA. However,
    functions that return a value (as opposed to subroutines that are void)
    are not portable, as they depend on how Fortran returns values. The routines
    here provide a portable interface. These are not identical to CBLAS, in
    particular, [cz]dot[uc] return complex numbers (as in Fortran BLAS) rather
    than return values via an argument.
    
    Only these BLAS-1 functions are provided:
    
    magma_cblas_dzasum / dasum
    magma_cblas_dznrm2 / dnrm2
    magma_cblas_zdotc  / ddot
    magma_cblas_zdotu  / ddot

*/
#include "magma_internal.h"

#define COMPLEX

/***************************************************************************//**
    @return Sum of absolute values of vector x;
            \f$ \sum_i | real(x_i) | + | imag(x_i) | \f$.

    To avoid dependence on CBLAS and incompatability issues between BLAS
    libraries, MAGMA uses its own implementation, following BLAS reference.
    
    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    x       COMPLEX_16 array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @ingroup magma_asum
*******************************************************************************/
extern "C"
double magma_cblas_dzasum(
    magma_int_t n,
    const magmaDoubleComplex *x, magma_int_t incx )
{
    if ( n <= 0 || incx <= 0 ) {
        return 0;
    }
    double result = 0;
    if ( incx == 1 ) {
        for( magma_int_t i=0; i < n; ++i ) {
            result += MAGMA_Z_ABS1( x[i] );
        }
    }
    else {
        magma_int_t nincx = n*incx;
        for( magma_int_t i=0; i < nincx; i += incx ) {
            result += MAGMA_Z_ABS1( x[i] );
        }
    }
    return result;
}


static inline double sqr( double x ) { return x*x; }

/***************************************************************************//**
    Returns 2-norm of vector x. Avoids unnecesary over/underflow.

    To avoid dependence on CBLAS and incompatability issues between BLAS
    libraries, MAGMA uses its own implementation, following BLAS reference.
    
    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    x       COMPLEX_16 array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @ingroup magma_nrm2
*******************************************************************************/
extern "C"
double magma_cblas_dznrm2(
    magma_int_t n,
    const magmaDoubleComplex *x, magma_int_t incx )
{
    if (n <= 0 || incx <= 0) {
        return 0;
    }
    else {
        double scale = 0;
        double ssq   = 1;
        // the following loop is equivalent to this call to the lapack
        // auxiliary routine:
        // call zlassq( n, x, incx, scale, ssq )
        for( magma_int_t ix=0; ix < 1 + (n-1)*incx; ix += incx ) {
            if ( real( x[ix] ) != 0 ) {
                double temp = fabs( real( x[ix] ));
                if (scale < temp) {
                    ssq = 1 + ssq * sqr(scale/temp);
                    scale = temp;
                }
                else {
                    ssq += sqr(temp/scale);
                }
            }
            #ifdef COMPLEX
            if ( imag( x[ix] ) != 0 ) {
                double temp = fabs( imag( x[ix] ));
                if (scale < temp) {
                    ssq = 1 + ssq * sqr(scale/temp);
                    scale = temp;
                }
                else {
                    ssq += sqr(temp/scale);
                }
            }
            #endif
        }
        return scale*magma_dsqrt(ssq);
    }
}


#ifdef COMPLEX
/***************************************************************************//**
    Returns dot product of vectors x and y; \f$ x^H y \f$.

    To avoid dependence on CBLAS and incompatability issues between BLAS
    libraries, MAGMA uses its own implementation, following BLAS reference.
    
    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    x       COMPLEX_16 array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @param[in]
    y       COMPLEX_16 array on CPU host.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy > 0.

    @ingroup magma__dot
*******************************************************************************/
extern "C"
magmaDoubleComplex magma_cblas_zdotc(
    magma_int_t n,
    const magmaDoubleComplex *x, magma_int_t incx,
    const magmaDoubleComplex *y, magma_int_t incy )
{
    // after too many issues with MKL and other BLAS, just write our own dot product!
    magmaDoubleComplex value = MAGMA_Z_ZERO;
    magma_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value = value + conj( x[i] ) * y[i];
        }
    }
    else {
        magma_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx; }
        if ( incy < 0 ) { iy = (-n + 1)*incy; }
        for( i=0; i < n; ++i ) {
            value = value + conj( x[ix] ) * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
}
#endif  // COMPLEX


/***************************************************************************//**
    @return dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    To avoid dependence on CBLAS and incompatability issues between BLAS
    libraries, MAGMA uses its own implementation, following BLAS reference.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    x       COMPLEX_16 array on CPU host.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of x. incx > 0.

    @param[in]
    y       COMPLEX_16 array on CPU host.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy > 0.

    @ingroup magma__dot
*******************************************************************************/
extern "C"
magmaDoubleComplex magma_cblas_zdotu(
    magma_int_t n,
    const magmaDoubleComplex *x, magma_int_t incx,
    const magmaDoubleComplex *y, magma_int_t incy )
{
    // after too many issues with MKL and other BLAS, just write our own dot product!
    magmaDoubleComplex value = MAGMA_Z_ZERO;
    magma_int_t i;
    if ( incx == 1 && incy == 1 ) {
        for( i=0; i < n; ++i ) {
            value = value + x[i] * y[i];
        }
    }
    else {
        magma_int_t ix=0, iy=0;
        if ( incx < 0 ) { ix = (-n + 1)*incx; }
        if ( incy < 0 ) { iy = (-n + 1)*incy; }
        for( i=0; i < n; ++i ) {
            value = value + x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return value;
}

#undef COMPLEX
