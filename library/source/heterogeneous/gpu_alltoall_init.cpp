#include "heterogeneous/gpu_alltoall_init.h"

#include "collective/alltoall_init.h"
#include "communicator/MPIL_Comm.hpp"

// ASSUMES 1 CPU CORE PER GPU (Standard for applications)
int gpu_aware_alltoall_init(alltoall_init_ftn f,
                       const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return f(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm,
            info, req_ptr);
}

int gpu_aware_alltoall_nonblocking_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return gpu_aware_alltoall_init(alltoall_nonblocking_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}

int gpu_aware_alltoall_pairwise_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return gpu_aware_alltoall_init(alltoall_pairwise_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}


int gpu_aware_alltoall_rma_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return gpu_aware_alltoal_initl(alltoall_rma_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}

int copy_to_cpu_alltoall(alltoall_init_ftn f,
                         const void* sendbuf,
                         const int sendcount,
                         MPI_Datatype sendtype,
                         void* recvbuf,
                         const int recvcount,
                         MPI_Datatype recvtype,
                         MPIL_Comm* comm,
                         MPIL_Info* info,
                         MPIL_Request** req_ptr)
{
    int ierr = 0;

    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);

    int total_bytes_s = sendcount * send_bytes * num_procs;
    int total_bytes_r = recvcount * recv_bytes * num_procs;

    void* cpu_sendbuf, *cpu_recvbuf;
    MPIL_Alloc(&cpu_sendbuf, total_bytes_s);
    MPIL_Alloc(&cpu_recvbuf, total_bytes_r);

    // Collective Among CPUs
    ierr += f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm,
            info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = total_bytes_s;
    request->size_recvs = total_bytes_r;

    return ierr;
}

int copy_to_cpu_alltoall_nonblocking_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return copy_to_cpu_alltoall_init(alltoall_nonblocking_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}

int copy_to_cpu_alltoall_pairwise_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return copy_to_cpu_alltoall_init(alltoall_pairwise_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}


int copy_to_cpu_alltoall_rma_init(const void* sendbuf,
                       const int sendcount,
                       MPI_Datatype sendtype,
                       void* recvbuf,
                       const int recvcount,
                       MPI_Datatype recvtype,
                       MPIL_Comm* comm,
                       MPIL_Info* info,
                       MPIL_Request** req_ptr)
{
    return copy_to_cpu_alltoal_initl(alltoall_rma_init,
                              sendbuf,
                              sendcount,
                              sendtype,
                              recvbuf,
                              recvcount,
                              recvtype,
                              comm,
                              info,
                              req_ptr);
}
