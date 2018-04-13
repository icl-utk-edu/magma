#ifndef PAPI_SDE_INTERFACE_H
#define PAPI_SDE_INTERFACE_H

#include <stdint.h>

#define PAPI_SDE_RO       0x00
#define PAPI_SDE_RW       0x01
#define PAPI_SDE_DELTA    0x00
#define PAPI_SDE_INSTANT  0x10

#define PAPI_SDE_long_long 0x0
#define PAPI_SDE_int       0x1
#define PAPI_SDE_double    0x2
#define PAPI_SDE_float     0x3

#define PAPI_SDE_SUM       0x0
#define PAPI_SDE_MAX       0x1
#define PAPI_SDE_MIN       0x2

typedef long long int (*papi_sde_fptr_t)( void * );
typedef void * papi_handle_t;

typedef struct papi_sde_fptr_struct_s {
    papi_handle_t (*init)(const char *lib_name);
    int (*register_counter)( void *handle, const char *event_name, int mode, int type, void *counter );
    int (*register_fp_counter)( void *handle, const char *event_name, int mode, int type, papi_sde_fptr_t fp_counter, void *param );
    int (*unregister_counter)( void *handle, const char *event_name);
    int (*describe_counter)( void *handle, const char *event_name, const char *event_description );
    int (*add_counter_to_group)( void *handle, const char *event_name, const char *group_name, uint32_t group_flags );
}papi_sde_fptr_struct_t;

extern "C" papi_handle_t papi_sde_init(const char *name_of_library);
extern "C" int papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter);
extern "C" int papi_sde_register_fp_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t func_ptr, void *param);
extern "C" int papi_sde_unregister_counter( void *handle, const char *event_name);
extern "C" int papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description );
extern "C" int papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags);

extern "C" papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);

#define POPULATE_SDE_FPTR_STRUCT( _A_ ) do{\
    _A_.init = papi_sde_init;\
    _A_.register_counter = papi_sde_register_counter;\
    _A_.register_fp_counter = papi_sde_register_fp_counter;\
    _A_.unregister_counter = papi_sde_unregister_counter;\
    _A_.describe_counter = papi_sde_describe_counter;\
    _A_.add_counter_to_group = papi_sde_add_counter_to_group;\
}while(0)

#endif
