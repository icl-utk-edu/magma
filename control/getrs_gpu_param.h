#ifndef GETRS_GPU_PARAM_H
#define GETRS_GPU_PARAM_H

////////////////////////////////////////////////////////////////////////////////
// Tuning values for magma_zgetrs_get_laswp_gpu_threshold (see src/zgetrs_gpu.cpp)
#ifdef MAGMA_HAVE_CUDA
#define ZGETRS_GPU_LASWP_THRESHOLD    (50)
#define CGETRS_GPU_LASWP_THRESHOLD    (50)
#define DGETRS_GPU_LASWP_THRESHOLD    (100)
#define SGETRS_GPU_LASWP_THRESHOLD    (200)

#elif defined(MAGMA_HAVE_HIP)
// the gpu laswp kernel is much slower on AMD GPUs
// compared to nvidia
// disable it for now by using very large threshold
#define ZGETRS_GPU_LASWP_THRESHOLD    (1000000)
#define CGETRS_GPU_LASWP_THRESHOLD    (1000000)
#define DGETRS_GPU_LASWP_THRESHOLD    (1000000)
#define SGETRS_GPU_LASWP_THRESHOLD    (1000000)

#endif

#endif // GETRS_GPU_PARAM_H




