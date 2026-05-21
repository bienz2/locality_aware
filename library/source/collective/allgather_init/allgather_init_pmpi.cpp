#include "collective/allgather.h"

#if defined(MPI4)
int allgather_pmpi_init(const void* sendbuf,
                   int sendcount,
                   MPI_Datatype sendtype,
                   void* recvbuf,
                   int recvcount,
                   MPI_Datatype recvtype,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    MPIL_Request* request;
    init_request(&request);
    allocate_requests(1, request);

    PMPI_Allgather_init(sendbuf, sendcount, sendtype, recvbuf,
            recvcount, recvtype, comm, info, request->requests);

    request->start_function = allgather_pmpi_start;
    request->wait_function = allgather_pmpi_wait;

    return MPI_SUCCESS;
}

int allgather_pmpi_start(MPIL_Request* request)
{
    if (request == NULL)
        return 0;

#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends);
#else
    gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->size_sends, 
            gpuMemcpyDeviceToHost, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif


    PMPI_Start(requests->request);

    return MPI_SUCCESS;
}

int allgather_pmpi_wait(MPIL_Request* request, MPI_Status* status)
{
    PMPI_Wait(requests->request, MPI_STATUS_IGNORE);

#if defined(GPU)
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->size_recvs);
#else
    gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->size_recvs, 
            gpuMemcpyHostToDevice, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif
   
    return MPI_SUCCESS;
}

#endif
