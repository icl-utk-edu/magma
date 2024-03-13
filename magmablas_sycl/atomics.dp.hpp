#ifndef ATOMICS_CUH
#define ATOMICS_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
/******************************************************************************/
// Atomic adds 
/******************************************************************************/
static __inline__ float 
magmablas_satomic_add(float* address, float val)
{
    return dpct::atomic_fetch_add<float,
                                  sycl::access::address_space::generic_space>(
        address, val);
}

/******************************************************************************/
static __inline__ double 
magmablas_datomic_add(double* address, double val)
{
    return dpct::atomic_fetch_add<double,
                                  sycl::access::address_space::generic_space>(
        address, val);
}

/******************************************************************************/
static __inline__ magmaFloatComplex 
magmablas_catomic_add(magmaFloatComplex* address, magmaFloatComplex val)
{
    float re = magmablas_satomic_add(&(reinterpret_cast<float(&)[2]>(*address)[0]), val.real());
    float im = magmablas_satomic_add(&(reinterpret_cast<float(&)[2]>(*address)[1]), val.imag());
    return MAGMA_C_MAKE(re, im);
}

/******************************************************************************/
static __inline__ magmaDoubleComplex 
magmablas_zatomic_add(magmaDoubleComplex* address, magmaDoubleComplex val)
{
    double re = magmablas_datomic_add(&(reinterpret_cast<double(&)[2]>(*address)[0]), val.real());
    double im = magmablas_datomic_add(&(reinterpret_cast<double(&)[2]>(*address)[1]), val.imag());
    return MAGMA_Z_MAKE(re, im);
}

/******************************************************************************/
// Atomic exchanges 
/******************************************************************************/
static __inline__ int 
magmablas_iatomic_exchange(int* address, int val)
{
    return dpct::atomic_exchange<int,
                                 sycl::access::address_space::generic_space>(
        address, val);
}

/******************************************************************************/
static __inline__ unsigned long long int 
magmablas_iatomic_exchange(unsigned long long int* address, unsigned long long int val)
{
    return dpct::atomic_exchange<unsigned long long,
                                 sycl::access::address_space::generic_space>(
        address, val);
}

/******************************************************************************/
static __inline__ long long int 
magmablas_iatomic_exchange(long long int* address, long long int val)
{
    // a cast should be safe, because the function just exchanges the 64bit value (no arithmetics)
    return (long long int)dpct::atomic_exchange<
        unsigned long long, sycl::access::address_space::generic_space>(
        (unsigned long long *)address, (unsigned long long)val);
}

/******************************************************************************/
#endif // ATOMICS_CUH
