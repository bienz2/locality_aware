#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_rma_earlybird_init(const void* sendbuf,
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

    return allreduce_rma_earlybird_init_helper(sendbuf, recvbuf, count,
            datatype, op, comm, info, req_ptr, MPIL_Alloc, MPIL_Free);
        
}


int allreduce_rma_earlybird_init_helper(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn)
{
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPIL_Request* request;
    init_request(&request);

    int type_size;
    MPI_Type_size(datatype, &type_size);

    MPI_Alloc_mem(count*type_size, MPI_INFO_NULL, &(request->win_array));
    MPIL_Request_win_init(request, request->win_array, count*type_size, 1, comm->global_comm);
    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->n_puts = num_procs;
    request->count = count;
    request->datatype = datatype;
    request->op = op;

    request->start_function = allreduce_rma_earlybird_start;
    request->wait_function  = allreduce_rma_earlybird_wait;

    *req_ptr = request;    

    memset(request->win_array, 0, request->count*type_size);
    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, request->win);

    return MPI_SUCCESS;
}

int allreduce_rma_earlybird_start(MPIL_Request* request)
{
    if (request == NULL)
        return 0;

#if defined(GPU)
int type_size;
MPI_Type_size(request->datatype, &type_size);
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

    for (int i = 0; i < request->n_puts; i++)
        MPI_Accumulate(request->sendbuf, request->count, request->datatype, 
                i, 0, request->count, request->datatype, request->op, 
                request->win);



    return MPI_SUCCESS;
}

int allreduce_rma_earlybird_wait(MPIL_Request* request, MPI_Status* status)   
{
    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, request->win);

    int type_size;
    MPI_Type_size(request->datatype, &type_size);
    memcpy(request->recvbuf, request->win_array, request->count*type_size);
    memset(request->win_array, 0, request->count*type_size);

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, request->win);

#if defined(GPU)
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

