#include "collective/alltoall.h"
#include "locality_aware.h"

int alltoall_rma(const void* sendbuf,
                      const int sendcount,
                      MPI_Datatype sendtype,
                      void* recvbuf,
                      const int recvcount,
                      MPI_Datatype recvtype,
                      MPIL_Comm* comm)
{
    
    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    char* send_buffer = (char*)(sendbuf);
    char* recv_buffer = (char*)(recvbuf);

    int send_bytes, recv_bytes;
    MPI_Type_size(sendtype, &send_bytes);
    MPI_Type_size(recvtype, &recv_bytes);
    int bytes = num_procs * recvcount * recv_bytes;

    if (bytes == 0) 
        return MPI_SUCCESS;

    MPI_Win win;
    char* win_array;
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &(win_array));
    MPI_Win_create(win_array, bytes, 1, MPI_INFO_NULL, comm->global_comm,
            &(win));

    send_bytes *= sendcount;
    recv_bytes *= recvcount;

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, win);
    for (int i = 0; i < num_procs; i++)
    {
         MPI_Put(&(send_buffer[i*send_bytes]), send_bytes, MPI_CHAR,
                 i, rank*recv_bytes, recv_bytes, MPI_CHAR, win);
    }
    MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED, win);

    // Need to memcpy because win_array is created with window
    // TODO : could explore just attaching recv_buffer to existing dynamic window 
    //        with persistent collectives
    memcpy(recv_buffer, win_array, bytes);

    MPI_Win_free(&win);
    MPI_Free_mem(win_array);

    return MPI_SUCCESS;
}

