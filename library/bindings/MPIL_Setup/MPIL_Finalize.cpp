#include "communicator/global_comms.hpp"
#include "communicator/MPIL_Comm.hpp"
#include "locality_aware.h"

#ifdef __cplusplus
extern "C" {
#endif

int MPIL_Finalize()
{
    // MPIL_COMM_WORLD is in COMM_CACHE
    //MPIL_Comm_free(&MPIL_COMM_WORLD);

    for (int i = 0; i < COMM_CACHE_SIZE; i++)
    {
        MPIL_Comm* comm = COMM_CACHE[i];
        comm->cached = false;
        MPIL_Comm_free(&comm);
    }
    
    return MPI_SUCCESS;
}

#ifdef __cplusplus
}
#endif
