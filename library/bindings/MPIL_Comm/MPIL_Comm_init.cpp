#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

int MPIL_Comm_init(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    initialize_comm_object(xcomm_ptr, global_comm);

    MPIL_Comm* xcomm = *xcomm_ptr;
    if (xcomm->cached_comm != NULL)
    {
        if (xcomm->cached_comm->local_comm != MPI_COMM_NULL)
        {
            xcomm->local_comm = xcomm->cached_comm->local_comm;
            xcomm->group_comm = xcomm->cached_comm->group_comm;

            xcomm->global_rank_to_local = xcomm->cached_comm->global_rank_to_local;
            xcomm->global_rank_to_node = xcomm->cached_comm->global_rank_to_node;
            xcomm->ordered_global_ranks = xcomm->cached_comm->ordered_global_ranks;
            xcomm->ppn = xcomm->cached_comm->ppn;
            xcomm->num_nodes = xcomm->cached_comm->num_nodes;
            xcomm->rank_node = xcomm->cached_comm->rank_node;
        }
    }
    else
    { 
        if (COMM_CACHE_SIZE < MAX_COMM_CACHE)
        {
            COMM_CACHE[COMM_CACHE_SIZE++] = xcomm;
        }
        else
        {
            xcomm->cached = false;
        }
    }

    return MPI_SUCCESS;
}
