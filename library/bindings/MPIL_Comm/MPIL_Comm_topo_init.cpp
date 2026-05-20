#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Comm_topo_init(MPIL_Comm* xcomm)
{
    // Already Initialized (includes MPIL_COMM_WORLD)
    if (xcomm->local_comm != MPI_COMM_NULL)
    {
        return MPI_SUCCESS;
    }

    if (xcomm->cached_comm != NULL && xcomm->cached_comm->local_comm != MPI_COMM_NULL)
    {
        xcomm->local_comm = xcomm->cached_comm->local_comm;
        xcomm->group_comm = xcomm->cached_comm->group_comm;
    }
    else initialize_topo_communicator(xcomm);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
