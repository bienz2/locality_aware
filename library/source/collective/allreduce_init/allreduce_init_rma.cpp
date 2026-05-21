#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_rma_init(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr)
{

    if (count == 0)
        return MPI_SUCCESS;

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);

    int type_size;
    MPI_Type_size(datatype, &type_size);

    MPIL_Request_win_init(request, recvbuf, count*type_size, 1, comm->global_comm);
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->count = count;
    request->datatype = datatype;
    request->op = op;
    MPI_Comm_dup(comm->global_comm, &(request->global_comm));

    request->start_function = allreduce_rma_start;
    request->wait_function  = allreduce_rma_wait;

    *req_ptr = request;    

    return MPI_SUCCESS;
}

int allreduce_rma_start(MPIL_Request* request)
{
    if (request == NULL)
        return 0;

    int type_size;
    MPI_Type_size(request->datatype, &type_size);
    memset(request->recvbuf, 0, request->count*type_size);

#if defined(GPU)
if (request->gpu_sendbuf)
{
#if defined(APU)
    memcpy(request->tmp_gpubuf, request->gpu_sendbuf, request->count*type_size);
#else
    gpuMemcpyAsync(request->tmp_gpubuf, request->gpu_sendbuf, request->count*type_size, 
            gpuMemcpyDeviceToHost, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    MPI_Win_fence(0, request->win);

    MPI_Accumulate(request->sendbuf, request->count, request->datatype, 
            0, 0, request->count, request->datatype, request->op, 
            request->win);


    return MPI_SUCCESS;
}

int allreduce_rma_wait(MPIL_Request* request, MPI_Status* status)   
{
    if (request == NULL)
        return 0;

    MPI_Win_fence(0, request->win);
    MPI_Bcast(request->recvbuf, request->count, request->datatype,
            0, request->global_comm);

#if defined(GPU)
int type_size;
MPI_Type_size(request->datatype, &type_size);
if (request->gpu_recvbuf)
{
#if defined(APU)
    memcpy(request->gpu_recvbuf, request->recvbuf, request->count*type_size);
#else
    gpuMemcpyAsync(request->gpu_recvbuf, request->recvbuf, request->count*type_size, 
            gpuMemcpyHostToDevice, 0);
    gpuStreamSynchronize(0);
#endif
}
#endif

    return MPI_SUCCESS;
}

