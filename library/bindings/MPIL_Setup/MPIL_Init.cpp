#include "communicator/MPIL_Comm.hpp"
#include "communicator/global_comms.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

MPIL_Comm* MPIL_COMM_WORLD;

MPIL_Comm* COMM_CACHE[MAX_COMM_CACHE]; 
int COMM_CACHE_SIZE = 0;

int MPIL_Init(MPI_Comm world)
{
    if (MPI_COMM_NULL == world)
    {
        world = MPI_COMM_WORLD;
    }

    /* Duplicate World Communicator */
    MPI_Comm_dup(world, &Communicator::WORLD_COMM);

    /* Create MPIL_COMM_WORLD */
    MPIL_Comm_init(&MPIL_COMM_WORLD, Communicator::WORLD_COMM);
    MPIL_Comm_topo_init(MPIL_COMM_WORLD);

    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
