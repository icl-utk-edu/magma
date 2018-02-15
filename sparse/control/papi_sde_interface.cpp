#include <stdio.h>
#include <stddef.h>
#include "papi_sde_interface.h"

#pragma weak papi_sde_init
#pragma weak papi_sde_register_counter
#pragma weak papi_sde_describe_counter

extern "C" papi_handle_t 
__attribute__((weak)) 
papi_sde_init(const char *name_of_library, int event_count)
{
    printf("weak papi_sde_init called from %s\n", __FILE__);
    void * ptr = NULL;
    return ptr;
}

extern "C" void 
__attribute__((weak)) 
papi_sde_register_counter(papi_handle_t handle, const char *event_name, int type, int mode, void *counter)
{
    printf("weak papi_sde_register_counter called from %s\n", __FILE__);
}

extern "C" void 
__attribute__((weak)) 
papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description)
{
    printf("weak papi_sde_describe_counter called from %s\n", __FILE__);
}
