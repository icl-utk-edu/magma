#ifndef PAPI_SDE_INTERFACE_H
#define PAPI_SDE_INTERFACE_H

#include "papi_sde.h"

// Type of SDE handle.
typedef void* papi_handle_t;

// Interface to papi SDE functions.
extern "C" papi_handle_t papi_sde_init(const char *name_of_library, int event_count);
extern "C" void papi_sde_register_counter(papi_handle_t handle, const char *event_name, int type, int mode, void *counter);
extern "C" void papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description );

// Required for papi_native_avail utility to work.
extern "C" void* papi_sde_hook_list_events( 
    void*   (*sym_init)( const char *, int ),
    void   (*sym_reg)( void *, const char *, int, int, void * ),
    void   (*sym_desc)( void *, const char *, const char * ) );

#endif
