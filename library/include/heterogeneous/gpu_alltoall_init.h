#ifndef MPI_ADVANCE_GPU_ALLTOALL_INIT_H
#define MPI_ADVANCE_GPU_ALLTOALL_INIT_H

#include "collective/alltoall_init.h"
#include "gpu_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A GPU-aware wrapper around provided ::alltoall_ftn
 * @details This function assumes that the provided ::alltoall_ftn is GPU aware, and
 * can handle GPU buffers. No extra behavior from the normal ::atlltoall_ftn. allocates
 * host memory for send and recv buffers. has same arguments as alltoall_helper function.
 * @returns The result from the ::alltoall_ftn.
 **/
int gpu_aware_alltoall_init(alltoall_init_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);

/** @brief A GPU buffer variant wrapper around provided ::alltoall_ftn
 * @details Unlike the GPU-aware variant, this version first allocates memory on the
 * hosts, copies the data from the CPU to the host, performs the requests all-to-all, then
 * copies the final result back to the GPU. All parameters have the same requirements as
 * the ::alltoall_ftn .
 * @returns The result from the ::alltoall_ftn.
 **/
int copy_to_cpu_alltoall_init(alltoall_init_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         MPIL_Info* info,
                         MPIL_Request** req_ptr);

/** @brief Calls ::gpu_aware_alltoall with ::pairwise_helper function**/
int gpu_aware_alltoall_pairwise_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);
/** @brief Calls ::gpu_aware_alltoall with ::nonblocking_helper function**/
int gpu_aware_alltoall_nonblocking_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);

int gpu_aware_alltoall_rma_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);
/** @brief Calls ::copy_to_cpu_alltoall with ::pairwise_helper function**/
int copy_to_cpu_alltoall_pairwise_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);
/** @brief Calls ::copy_to_cpu_alltoall with nonblocking_helper function**/
int copy_to_cpu_alltoall_nonblocking_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr);

#ifdef __cplusplus
}
#endif

#endif
