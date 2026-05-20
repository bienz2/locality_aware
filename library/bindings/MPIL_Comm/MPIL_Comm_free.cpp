#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_free(MPIL_Comm** xcomm_ptr)
{
    MPIL_Comm* xcomm = *xcomm_ptr;

    if (xcomm->neighbor_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&(xcomm->neighbor_comm));
        xcomm->neighbor_comm = MPI_COMM_NULL;
    }
    MPIL_Comm_leader_free(xcomm);

    if (!xcomm->cached) // only free cached comms in MPI_Finalize
    {
        MPIL_Comm_topo_free(xcomm);
        MPIL_Comm_device_free(xcomm);

        free(xcomm);
    }

    return MPI_SUCCESS;
}
