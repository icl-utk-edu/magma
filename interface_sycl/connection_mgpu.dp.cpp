/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "magma_internal.h"

extern "C" {
// TODO: how this works (or doesn't) with SYCL
#if defined(MAGMA_HAVE_SYCL) 
magma_int_t magma_buildconnection_mgpu(
    magma_int_t gnode[MagmaMaxGPUs + 2][MagmaMaxGPUs + 2], magma_int_t *ncmplx,
    magma_int_t ngpu) try {
    magma_int_t *deviceid = NULL;
    magma_imalloc_cpu( &deviceid, ngpu );
    memset( deviceid, 0, ngpu*sizeof(magma_int_t) );

    ncmplx[0] = 0;

    int samecomplex = -1;
    int err;
    dpct::device_info prop;

    magma_int_t cmplxnb = 0;
    magma_int_t cmplxid = 0;
    magma_int_t lcgpunb = 0;
    for( magma_int_t d = 0; d < ngpu; ++d ) {
        // check for unified memory & enable peer memory access between all GPUs.
        magma_setdevice( d );
        dpct::dev_mgr::instance().get_device(int(d)).get_device_info(prop);

        #ifdef MAGMA_HAVE_CUDA
        if ( ! prop.unifiedAddressing ) {
        #elif defined(MAGMA_HAVE_HIP)
        // assume it does, HIP does not have support for checking this
        if ( ! true ) {
        #endif
	#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
            printf( "device %lld doesn't support unified addressing\n", (long long) d );
            magma_free_cpu( deviceid );
            return -1;
        }
        #endif
        // add this device to the list if not added yet.
        // not added yet meaning belong to a new complex
        if (deviceid[d] == 0) {
            cmplxnb = cmplxnb + 1;
            cmplxid = cmplxnb - 1;
            gnode[cmplxid][MagmaMaxGPUs] = 1;
            lcgpunb = gnode[cmplxid][MagmaMaxGPUs]-1;
            gnode[cmplxid][lcgpunb] = d;
            deviceid[d] = -1;
        }
        //printf("device %lld:\n", (long long) d );

        for( magma_int_t d2 = d+1; d2 < ngpu; ++d2 ) {
            // check for unified memory & enable peer memory access between all GPUs.
            magma_setdevice( d2 );
            dpct::dev_mgr::instance().get_device(int(d2)).get_device_info(prop);
            #ifdef MAGMA_HAVE_CUDA
            if ( ! prop.unifiedAddressing ) {
            #elif defined(MAGMA_HAVE_HIP)
            // assume it does, HIP does not have support for checking this
            if ( ! true ) {
            #endif
	    #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
                printf( "device %lld doesn't support unified addressing\n", (long long) d2 );
                magma_free_cpu( deviceid );
                return -1;
            }
            #endif 

            /*
            DPCT1031:16: DPC++ currently does not support memory access across
            peer devices. The output parameter(s) are set to 0.
            */
            /* TODO err = */ *&samecomplex = 0;

            //printf(" device %lld and device %lld have samecomplex = %lld\n",
            //       (long long) d, (long long) d2, (long long) samecomplex );
            if (samecomplex == 1) {
                // d and d2 are on the same complex so add them, note that d is already added
                // so just enable the peer Access for d and enable+add d2.
                // FOR d:
                magma_setdevice( d );
                /*
                DPCT1027:17: The call to cudaDeviceEnablePeerAccess was replaced
                with 0 because DPC++ currently does not support memory access
                across peer devices.
                */
                err = 0;
                //printf("enabling devide %lld ==> %lld  error %lld\n",
                //       (long long) d, (long long) d2, (long long) err );
                if (err != 0 && err != 704) {
                    printf( "device %lld cudaDeviceEnablePeerAccess error %lld\n",
                            (long long) d2, (long long) err );
                    magma_free_cpu( deviceid );
                    return -2;
                }

                // FOR d2:
                magma_setdevice( d2 );
                /*
                DPCT1027:18: The call to cudaDeviceEnablePeerAccess was replaced
                with 0 because DPC++ currently does not support memory access
                across peer devices.
                */
                err = 0;
                //printf("enabling devide %lld ==> %lld  error %lld\n",
                //       (long long) d2, (long long) d, (long long) err );
                if ((err == 0) || (err == 704)) {
                    if (deviceid[d2] == 0) {
                        //printf("adding device %lld\n", (long long) d2 );
                        gnode[cmplxid][MagmaMaxGPUs] = gnode[cmplxid][MagmaMaxGPUs]+1;
                        lcgpunb                      = gnode[cmplxid][MagmaMaxGPUs]-1;
                        gnode[cmplxid][lcgpunb] = d2;
                        deviceid[d2] = -1;
                    }
                } else {
                    printf( "device %lld cudaDeviceEnablePeerAccess error %lld\n",
                            (long long) d, (long long) err );
                    magma_free_cpu( deviceid );
                    return -2;
                }
            }
        }
    }

    ncmplx[0] = cmplxnb;
    magma_free_cpu( deviceid );
    return cmplxnb;
#else 
    // Err: CUDA only
    return -1;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} /* extern "C" */


