#include "locality_aware.h"

#if defined(GPU)
#include "heterogeneous/gpu_utils.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alloc(void** pointer, const int bytes)
{
    if (bytes == 0)
    {
        *pointer = nullptr;
    }
    else
    {
        *pointer = new char[bytes];
    }

    return MPI_SUCCESS;
}

#if defined(GPU)
int MPIL_GPU_Alloc(void** pointer, const int bytes)
{
    int gpu_error;
    if (bytes == 0)
    {
        *pointer = nullptr;
    }
    else
    {
        gpu_error = gpuMalloc((void**)pointer, bytes);
        gpu_check(gpu_error);
    }
    gpu_error = gpuDeviceSynchronize();
    gpu_check(gpu_error);

    return MPI_SUCCESS;
}
#endif

#ifdef __cplusplus
}
#endif
