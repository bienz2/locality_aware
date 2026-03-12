#include "collective/alltoall_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_alltoall_init.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Alltoall_init(const void* sendbuf,
                  const int sendcount,
                  MPI_Datatype sendtype,
                  void* recvbuf,
                  const int recvcount,
                  MPI_Datatype recvtype,
                  MPIL_Comm* mpi_comm,
                  MPIL_Info* mpil_info,
                  MPIL_Request** req_ptr)
{
    alltoall_init_ftn method;

    switch (mpil_alltoall_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLTOALL_GPU_PAIRWISE:
            method = gpu_aware_alltoall_pairwise_init;
            break;
        case ALLTOALL_GPU_NONBLOCKING:
            method = gpu_aware_alltoall_nonblocking_init;
            break;
#endif
        case ALLTOALL_CTC_PAIRWISE:
            method = copy_to_cpu_alltoall_pairwise_init;
            break;
        case ALLTOALL_CTC_NONBLOCKING:
            method = copy_to_cpu_alltoall_nonblocking_init;
            break;
#endif
        case ALLTOALL_PAIRWISE:
            method = alltoall_pairwise_init;
            break;
        case ALLTOALL_NONBLOCKING:
            method = alltoall_nonblocking_init;
            break;
        case ALLTOALL_RMA:
            method = alltoall_rma_init;
            break;
        default:
            method = alltoall_pairwise_init;
            break;
    }

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm,
            mpil_info, req_ptr);
}

#ifdef __cplusplus
}
#endif
