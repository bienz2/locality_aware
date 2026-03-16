#include "collective/allreduce_init.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

int allreduce_rma_hierarchical_init(const void* sendbuf,
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

    return allreduce_rma_hierarchical_init_helper(sendbuf, recvbuf, count,
            datatype, op, comm, info, req_ptr, MPIL_Alloc, MPIL_Free);
        
}

int allreduce_rma_multileader_init(const void* sendbuf,
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

    return allreduce_rma_multileader_init_helper(sendbuf, recvbuf, count,
            datatype, op, comm, info, req_ptr, MPIL_Alloc, MPIL_Free);
        
}


int allreduce_rma_hierarchical_init_helper(const void* sendbuf,
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

    if (comm->local_comm == MPI_COMM_NULL)
        MPIL_Comm_topo_init(comm);
    int tag;
    MPIL_Comm_tag(comm, &tag);

    return allreduce_rma_hierarchical_init_core(sendbuf, recvbuf, count,
            datatype, op, comm->group_comm, comm->local_comm, tag,
            info, req_ptr, alloc_ftn, free_ftn);
}

int allreduce_rma_multileader_init_helper(const void* sendbuf,
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

    if (comm->local_comm == MPI_COMM_NULL)
        MPIL_Comm_topo_init(comm);

    int local_rank, ppn;
    MPI_Comm_rank(comm->local_comm, &local_rank);
    MPI_Comm_size(comm->local_comm, &ppn);

    // Convert to leader_comm (4 leaders per node)
    int num_leaders = 4;
    if (ppn < num_leaders)
        num_leaders = ppn;
    if (comm->leader_comm != MPI_COMM_NULL)
    {
        int ppl;
        MPI_Comm_size(comm->leader_comm, &ppl);
        if (ppn / num_leaders != ppl)
            MPIL_Comm_leader_free(comm);
    }
    if (comm->leader_comm == MPI_COMM_NULL)
        MPIL_Comm_leader_init(comm, num_leaders);

    int tag;
    MPIL_Comm_tag(comm, &tag);

    return allreduce_rma_hierarchical_init_core(sendbuf, recvbuf, count,
            datatype, op, comm->leader_group_comm, comm->leader_comm, tag,
            info, req_ptr, alloc_ftn, free_ftn);
}

int allreduce_rma_hierarchical_init_core(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPI_Comm group_comm,
                                 MPI_Comm local_comm,
                                 int tag,
                                 MPIL_Info* info,
                                 MPIL_Request** req_ptr,
                                 MPIL_Alloc_ftn alloc_ftn,
                                 MPIL_Free_ftn free_ftn)
{

    int local_rank, ppn;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &ppn);

    MPIL_Request* request;
    init_request(&request);

    int type_size;
    MPI_Type_size(datatype, &type_size);

    int bytes = 0;
    if (local_rank == 0) bytes = ppn*count*type_size;
    MPI_Win_allocate_shared(bytes,
                type_size,
                MPI_INFO_NULL,
                local_comm,
                &(request->win_array),
                &(request->win));
    request->win_alloc = 1;

    request->sendbuf = sendbuf;
    request->recvbuf = recvbuf;
    request->n_puts = ppn;
    request->count = count;
    request->datatype = datatype;
    request->op = op;
    MPI_Comm_dup(local_comm, &(request->local_comm));

    request->start_function = allreduce_rma_hierarchical_start;
    request->wait_function  = allreduce_rma_hierarchical_wait;

    

    if (local_rank == 0)
        allreduce_recursive_doubling_init_core(MPI_IN_PLACE, recvbuf, count, datatype,
                op, group_comm, tag, info, &(request->local_L_request), alloc_ftn, free_ftn);

    *req_ptr = request;    

    return MPI_SUCCESS;
}

int allreduce_rma_hierarchical_start(MPIL_Request* request)
{
    if (request == NULL)
        return 0;
    
    int local_rank;
    MPI_Comm_rank(request->local_comm, &local_rank);

#if defined(GPU)
if (request->gpu_sendbuf)
{
    int type_size;
    MPI_Type_size(request->datatype, &type_size);
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

    MPI_Put(request->sendbuf, request->count, request->datatype, 
            0, local_rank * request->count, request->count, 
            request->datatype, request->win);

    return MPI_SUCCESS;
}

int allreduce_rma_hierarchical_wait(MPIL_Request* request, MPI_Status* status)   
{
    int local_rank, ppn;
    MPI_Comm_rank(request->local_comm, &local_rank);
    MPI_Comm_size(request->local_comm, &ppn);

    int type_size;
    MPI_Type_size(request->datatype, &type_size);

    MPI_Win_fence(0, request->win);

    if (local_rank == 0)
    {
        memcpy(request->recvbuf, request->win_array, request->count * type_size);
        for (int i = 1; i < ppn; i++)
            MPI_Reduce_local((char*)request->win_array + (i * request->count * type_size),
                    request->recvbuf, request->count, request->datatype, request->op);
    }

    // Start Recursive Doubling Allreduce among leaders
    if (request->local_L_request)
    {
        MPIL_Start(request->local_L_request);
        MPIL_Wait(request->local_L_request, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(request->recvbuf, request->count, request->datatype,
            0, request->local_comm);

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

