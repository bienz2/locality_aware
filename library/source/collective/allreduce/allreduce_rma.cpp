#include "collective/allreduce.h"
#include "locality_aware.h"
#include <string.h>
#include <math.h>

// Dumb approach BUT allows for optimal early bird contributions

int allreduce_rma(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 MPI_Datatype datatype,
                                 MPI_Op op,
                                 MPIL_Comm* comm)
{
    if (count == 0)
        return MPI_SUCCESS;

    int type_size;
    MPI_Type_size(datatype, &type_size);

    int rank, num_procs;
    MPI_Comm_rank(comm->global_comm, &rank);
    MPI_Comm_size(comm->global_comm, &num_procs);

    MPI_Win win;
    MPI_Win_create(recvbuf, type_size*count, 1, MPI_INFO_NULL, 
            comm->global_comm, &(win));

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOPRECEDE, win);

    for (int i = 0; i < num_procs; i++)
        MPI_Accumulate(sendbuf, count, datatype, i, 0, count, datatype, op, win);

    MPI_Win_fence(MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED, win);

    MPI_Win_free(&(win));

    return MPI_SUCCESS;
}
