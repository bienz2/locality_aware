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
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_alltoall_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLTOALL_GPU_PAIRWISE:
            method = alltoall_pairwise_init;
            gpu_aware = true;
            break;
        case ALLTOALL_GPU_NONBLOCKING:
            method = alltoall_nonblocking_init;
            gpu_aware = true;
            break;
#endif
        case ALLTOALL_CTC_PAIRWISE:
            method = alltoall_pairwise_init;
            copy_to_cpu = true;
            break;
        case ALLTOALL_CTC_NONBLOCKING:
            method = alltoall_nonblocking_init;
            copy_to_cpu = true;
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

#if defined(GPU)
    if (gpu_aware)
        return gpu_aware_collective_init(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, mpi_comm, mpil_info, req_ptr);
    else if (copy_to_cpu)
        return copy_to_cpu_alltoall_init(method, sendbuf, sendcount, sendtype,
                recvbuf, recvcount, recvtype, mpi_comm, mpil_info, req_ptr);
#endif

    return method(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, mpi_comm,
            mpil_info, req_ptr);
}

#ifdef __cplusplus
}
#endif
