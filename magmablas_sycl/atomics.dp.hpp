#ifndef ATOMICS_CUH
#define ATOMICS_CUH

#include <CL/sycl.hpp>
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

// NOTE: HIP doesn't define anything specific for double atomics, but Im assuming int64 atomics are valid.
// SEE HERE: https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#hip_arch-defines
#if (DPCT_COMPATIBILITY_TEMP < 600) ||                                         \
    !(__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__) // atomic add for double precision
                                             // is natively supported on sm_60
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = dpct::atomic_compare_exchange_strong<
            unsigned long long, sycl::access::address_space::generic_space>(
            address_as_ull, assumed,
            (unsigned long long)(sycl::bit_cast<long long>(
                val + sycl::bit_cast<double, long long>(assumed))));
    } while (assumed != old);
    return sycl::bit_cast<double, long long>(old);
#else
    return atomicAdd(address, val);
#endif
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
