#include <stdlib.h>

#include "communicator/MPIL_Comm.hpp"

#ifdef NUMA_H
#include <sched.h>
#include <numa.h>
#endif

int initialize_comm_object(MPIL_Comm** xcomm_ptr, MPI_Comm global_comm)
{
    MPIL_Comm* xcomm   = (MPIL_Comm*)malloc(sizeof(MPIL_Comm));
    xcomm->global_comm = global_comm;

    xcomm->cached = true;
    xcomm->cached_comm = NULL;
    for (int i = 0; i < COMM_CACHE_SIZE; i++)
    {
        MPIL_Comm* comm = COMM_CACHE[i];
        int result;
        MPI_Comm_compare(comm->global_comm, xcomm->global_comm, &result);
        if (result == MPI_IDENT || result == MPI_CONGRUENT)
        {
            xcomm->cached = false;
            xcomm->cached_comm = comm;
            break;
        }
    }

    xcomm->local_comm = MPI_COMM_NULL;
    xcomm->group_comm = MPI_COMM_NULL;

    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &(xcomm->max_tag), &flag);
    xcomm->tag = 126 % xcomm->max_tag;

    xcomm->global_rank_to_local = NULL;
    xcomm->global_rank_to_node  = NULL;
    xcomm->ordered_global_ranks = NULL;
#ifdef GPU
    xcomm->gpus_per_node = 0;
#endif

    xcomm->leader_comm       = MPI_COMM_NULL;
    xcomm->leader_group_comm = MPI_COMM_NULL;
    xcomm->leader_local_comm = MPI_COMM_NULL;

    // Deprecate?  Should always be using Topo objects
    xcomm->neighbor_comm = MPI_COMM_NULL;

    *xcomm_ptr = xcomm;

    return MPI_SUCCESS;
}

int initialize_topo_communicator(MPIL_Comm* xcomm)
{
    // Check if local_comm was already created
    if (xcomm->local_comm != MPI_COMM_NULL)
    {

        return MPI_SUCCESS;
    }

    if (xcomm->cached_comm != NULL && xcomm->cached_comm->local_comm != MPI_COMM_NULL)
    {
        xcomm->local_comm = xcomm->cached_comm->local_comm;
        xcomm->group_comm = xcomm->cached_comm->group_comm;
        return MPI_SUCCESS;
    }

    int rank;
    MPI_Comm_rank(xcomm->global_comm, &rank);

#ifdef MPIL_TEST_PPN
    int color = rank / MPIL_TEST_PPN;
    MPI_Comm_split(xcomm->global_comm, color, rank, &(xcomm->local_comm));
#else
    MPI_Comm_split_type(xcomm->global_comm,
                        MPI_COMM_TYPE_SHARED,
                        rank,
                        MPI_INFO_NULL,
                        &(xcomm->local_comm));

#ifdef NUMA_H
    int numa_node = numa_node_of_cpu(sched_getcpu());
    MPI_Comm node_comm;
    MPI_Comm_dup(xcomm->local_comm, &node_comm);
    MPI_Comm_free(&(xcomm->local_comm));
    MPI_Comm_split(node_comm, numa_node, rank, &(xcomm->local_comm));
#endif

#endif

    int local_rank;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);

    // Split global comm into group (per local rank) communicators
    MPI_Comm_split(xcomm->global_comm, local_rank, rank, &(xcomm->group_comm));

    return MPI_SUCCESS;
}

int initialize_rank_mapping(MPIL_Comm* xcomm)
{
    // Check if already created
    if (xcomm->global_rank_to_local != NULL)
        return MPI_SUCCESS;


    if (xcomm->cached_comm != NULL && xcomm->cached_comm->global_rank_to_local != NULL)
    {
        xcomm->global_rank_to_local = xcomm->cached_comm->global_rank_to_local;
        xcomm->global_rank_to_node = xcomm->cached_comm->global_rank_to_node;
        xcomm->ordered_global_ranks = xcomm->cached_comm->ordered_global_ranks;
        xcomm->ppn = xcomm->cached_comm->ppn;
        xcomm->num_nodes = xcomm->cached_comm->num_nodes;
        xcomm->rank_node = xcomm->cached_comm->rank_node;
        return MPI_SUCCESS;
    }


    int rank = -1, num_procs = -1;
    MPI_Comm_rank(xcomm->global_comm, &rank);
    MPI_Comm_size(xcomm->global_comm, &num_procs);
    int local_rank, ppn;
    MPI_Comm_rank(xcomm->local_comm, &local_rank);
    MPI_Comm_size(xcomm->local_comm, &ppn);
    int local_node;
    MPI_Comm_rank(xcomm->group_comm, &local_node);

    // Gather arrays for get_node, get_local, and get_global methods
    // These arrays allow for these methods to work with any ordering
    // No longer relying on SMP ordering of processes to nodes!
    // Does rely on constant ppn

    if (xcomm->global_rank_to_local == NULL)
    {
        xcomm->global_rank_to_local = (int*)malloc(num_procs * sizeof(int));
    }

    if (xcomm->global_rank_to_node == NULL)
    {
        xcomm->global_rank_to_node = (int*)malloc(num_procs * sizeof(int));
    }

    MPI_Allgather(&local_rank,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_local,
                  1,
                  MPI_INT,
                  xcomm->global_comm);
    MPI_Allgather(&local_node,
                  1,
                  MPI_INT,
                  xcomm->global_rank_to_node,
                  1,
                  MPI_INT,
                  xcomm->global_comm);

    if (xcomm->ordered_global_ranks == NULL)
    {
        xcomm->ordered_global_ranks = (int*)malloc(num_procs * sizeof(int));
    }

    for (int i = 0; i < num_procs; i++)
    {
        int local                                       = xcomm->global_rank_to_local[i];
        int node                                        = xcomm->global_rank_to_node[i];
        xcomm->ordered_global_ranks[node * ppn + local] = i;
    }

    // Set xcomm variables
    MPI_Comm_size(xcomm->local_comm, &(xcomm->ppn));
    xcomm->num_nodes = ((num_procs - 1) / xcomm->ppn) + 1;
    xcomm->rank_node = get_node(xcomm, rank);

    if (xcomm->cached_comm != NULL)
    {
        xcomm->cached_comm->global_rank_to_local = xcomm->global_rank_to_local;
        xcomm->cached_comm->global_rank_to_node = xcomm->global_rank_to_node;
        xcomm->cached_comm->ordered_global_ranks = xcomm->ordered_global_ranks;
        xcomm->cached_comm->ppn = xcomm->ppn;
        xcomm->cached_comm->num_nodes = xcomm->num_nodes;
        xcomm->cached_comm->rank_node = xcomm->rank_node;
    }

    return MPI_SUCCESS;
}
