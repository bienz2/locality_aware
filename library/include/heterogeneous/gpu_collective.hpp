#include <mpi.h>
#include "communicator/MPIL_Comm.hpp"

template <typename Ftn, typename... Args>
int gpu_aware_collective(Ftn f, Args&&... args)
{
    return f(std::forward<Args>(args)...);
}

template <typename Ftn, typename... Args>
int gpu_aware_collective_init(Ftn f, Args&&... args)
{
    return f(std::forward<Args>(args)...);
}


template <typename Ftn, typename... Args>
int copy_to_cpu(const void* sendbuf,
        const int sendcount, MPI_Datatype sendtype,
        const int recvcount, MPI_Datatype recvtype,
        void** cpu_sendbuf_ptr, void** cpu_recvbuf_ptr)
{
    int ierr = 0;
    
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    // gpuMalloc is too expensive for single allreduce
    void* cpu_sendbuf = malloc(sendcount*send_size);
    void* cpu_recvbuf = malloc(recvcount*recv_size);

#if defined(APU)
    memcpy(cpu_sendbuf, sendbuf, sendcount*send_size);
#else
    gpuMemcpy(cpu_sendbuf, sendbuf, sendcount*send_size, gpuMemcpyDeviceToHost);
    gpuStreamSynchronize(0);
#endif 

    *cpu_sendbuf_ptr = cpu_sendbuf;
    *cpu_recvbuf_ptr = cpu_recvbuf;

    return MPI_SUCCESS;
}


template <typename Ftn, typename... Args>
int copy_to_gpu(const void* recvbuf,
        const int sendcount, MPI_Datatype sendtype,
        const int recvcount, MPI_Datatype recvtype,
        void* cpu_sendbuf, void* cpu_recvbuf)
{
    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

#if defined(APU)
    memcpy(recvbuf, cpu_recvbuf, recvcount*recv_size);
#else
    gpuMemcpy(recvbuf, cpu_recvbuf, recvcount*recv_size, gpuMemcpyHostToDevice);
    gpuStreamSynchronize(0);
#endif

    free(cpu_sendbuf);
    free(cpu_recvbuf);

    gpuDeviceSynchronize();

    return MPI_SUCCESS;
}


template <typename Ftn>
int copy_to_cpu_allreduce(Ftn f,
        const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, MPIL_Comm* comm)
{
    void *cpu_sendbuf, *cpu_recvbuf;

    copy_to_cpu(sendbuf, count, datatype, count, datatype, 
            &cpu_sendbuf, &cpu_recvbuf);

    int ierr = f(cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm);

    copy_to_gpu(recvbuf, count, datatype, count, datatype,
            cpu_sendbuf, cpu_recvbuf);

    return ierr;
}

template <typename Ftn>
int copy_to_cpu_allgather(Ftn f,
                const void* sendbuf,
                int sendcount,
                MPI_Datatype sendtype,
                void* recvbuf,
                int recvcount,
                MPI_Datatype recvtype,
                MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    void *cpu_sendbuf, *cpu_recvbuf;

    copy_to_cpu(sendbuf, sendcount, sendtype, 
            recvcount*num_procs, recvtype, 
            &cpu_sendbuf, &cpu_recvbuf);

    int ierr = f(cpu_sendbuf, sendcount, sendtype, 
            cpu_recvbuf, recvcount, recvtype, comm);

    copy_to_gpu(recvbuf, sendcount, sendtype, 
            recvcount*num_procs, recvtype,
            cpu_sendbuf, cpu_recvbuf);

    return ierr;
}

template <typename Ftn>
int copy_to_cpu_alltoall(Ftn f,
        const void* sendbuf, const int sendcount, MPI_Datatype sendtype,
        void* recvbuf, const int recvcount, MPI_Datatype recvtype,
        MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    void *cpu_sendbuf, *cpu_recvbuf;

    copy_to_cpu(sendbuf, sendcount * num_procs, sendtype, 
            recvcount * num_procs, sendtype, 
            &cpu_sendbuf, &cpu_recvbuf);

    int ierr = f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm);

    copy_to_gpu(recvbuf, sendcount * num_procs, sendtype, 
            recvcount * num_procs, sendtype, 
            cpu_sendbuf, cpu_recvbuf);

    return ierr;

}

template <typename Ftn>
int copy_to_cpu_alltoallv(Ftn f,
                      const void* sendbuf,
                      const int sendcounts[],
                      const int sdispls[],
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcounts[],
                      const int rdispls[],
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);
    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    void *cpu_sendbuf, *cpu_recvbuf;

    copy_to_cpu(sendbuf, sendcount, sendtype, 
            recvcount, sendtype, &cpu_sendbuf, &cpu_recvbuf);

    int ierr = f(cpu_sendbuf,
              sendcounts,
              sdispls,
              sendtype,
              cpu_recvbuf,
              recvcounts,
              rdispls,
              recvtype,
              comm);

    copy_to_gpu(recvbuf, sendcount, sendtype, 
            recvcount, recvtype, cpu_sendbuf, cpu_recvbuf);

    return ierr;

}




template <typename Ftn>
int copy_to_cpu_allreduce_init(Ftn f,
        const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, MPIL_Comm* comm, MPIL_Info* info, MPI_Request** req_ptr)
{
    void *cpu_sendbuf, *cpu_recvbuf;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    void *cpu_sendbuf, *cpu_recvbuf;

    MPIL_Alloc(&cpu_sendbuf, count * type_size);
    MPIL_Alloc(&cpu_recvbuf, count * type_size);

    int ierr = f(cpu_sendbuf, cpu_recvbuf, count, datatype, op, comm,
            info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;

    return ierr;
}

template <typename Ftn>
int copy_to_cpu_allgather_init(Ftn f,
                const void* sendbuf,
                int sendcount,
                MPI_Datatype sendtype,
                void* recvbuf,
                int recvcount,
                MPI_Datatype recvtype,
                MPIL_Comm* comm,
                MPIL_Info* info,
                MPIL_Request** req_ptr)
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    void *cpu_sendbuf, *cpu_recvbuf;

    MPIL_Alloc(&cpu_sendbuf, sendcount * send_size);
    MPIL_Alloc(&cpu_recvbuf, recvcount * num_procs * recv_size);

    int ierr = f(cpu_sendbuf, sendcount, sendtype, 
            cpu_recvbuf, recvcount, recvtype, comm);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = sendcount * send_size;
    request->size_recvs = recvcount * num_procs * recv_size;

    return ierr;
}

template <typename Ftn>
int copy_to_cpu_alltoall_init(Ftn f,
        const void* sendbuf, const int sendcount, MPI_Datatype sendtype,
        void* recvbuf, const int recvcount, MPI_Datatype recvtype,
        MPIL_Comm* comm,
                MPIL_Info* info,
                MPIL_Request** req_ptr))
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    void *cpu_sendbuf, *cpu_recvbuf;

    MPIL_Alloc(&cpu_sendbuf, sendcount * num_procs * send_size);
    MPIL_Alloc(&cpu_recvbuf, recvcount * num_procs * recv_size);

    int ierr = f(cpu_sendbuf, sendcount, sendtype, cpu_recvbuf, recvcount, recvtype, comm,
            info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = sendcount * num_procs * send_size;
    request->size_recvs = recvcount * num_procs * recv_size;

    return ierr;

}

template <typename Ftn>
int copy_to_cpu_alltoallv_init(Ftn f,
                const void* sendbuf,
                const int sendcounts[],
                const int sdispls[],
                MPI_Datatype sendtype,
                void* recvbuf,
                const int recvcounts[],
                const int rdispls[],
                MPI_Datatype recvtype,
                MPIL_Comm* comm,
                MPIL_Info* info,
                MPIL_Request** req_ptr))
{
    int num_procs;
    MPI_Comm_size(comm->global_comm, &num_procs);
    int sendcount = 0;
    int recvcount = 0;
    for (int i = 0; i < num_procs; i++)
    {
        sendcount += sendcounts[i];
        recvcount += recvcounts[i];
    }

    int send_size, recv_size;
    MPI_Type_size(sendtype, &send_size);
    MPI_Type_size(recvtype, &recv_size);

    void *cpu_sendbuf, *cpu_recvbuf;

    MPIL_Alloc(&cpu_sendbuf, sendcount * send_size);
    MPIL_Alloc(&cpu_recvbuf, recvcount * recv_size);


    copy_to_cpu_init(sendbuf, sendcount, sendtype, 
            recvcount, sendtype, &cpu_sendbuf, &cpu_recvbuf);

    int ierr = f(cpu_sendbuf,
              sendcounts,
              sdispls,
              sendtype,
              cpu_recvbuf,
              recvcounts,
              rdispls,
              recvtype,
              comm, info, req_ptr);

    MPIL_Request* request = *req_ptr;
    request->tmp_gpubuf = cpu_sendbuf;
    request->gpu_sendbuf = sendbuf;
    request->gpu_recvbuf = recvbuf;
    request->size_sends = sendcount * send_size;
    request->size_recvs = recvcount * recv_size;

    return ierr;

}

