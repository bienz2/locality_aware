#include "collective/allreduce_init.h"
#include "locality_aware.h"
#ifdef GPU
#include "heterogeneous/gpu_collective.h"
#endif

int MPIL_Allreduce_init(const void* sendbuf,
                   void* recvbuf, 
                   int count,
                   MPI_Datatype datatype,
                   MPI_Op op,
                   MPIL_Comm* comm,
                   MPIL_Info* info,
                   MPIL_Request** req_ptr)
{
    allreduce_init_ftn method;
    bool gpu_aware = false;
    bool copy_to_cpu = false;

    switch (mpil_allreduce_init_implementation)
    {
#if defined(GPU) 
#if defined(GPU_AWARE)
        case ALLREDUCE_GPU_RECURSIVE_DOUBLING:
            method = allreduce_recursive_doubling_init;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_LOC:
            method = allreduce_dissemination_loc_init;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_ML:
            method = allreduce_dissemination_ml_init;
            gpu_aware = true;
            break;
        case ALLREDUCE_GPU_DISSEMINATION_RADIX:
            method = allreduce_dissemination_radix_init;
            gpu_aware = true;
            break;
#if defined(MPI4)
        case ALLREDUCE_GPU_PMPI:
            method = allreduce_pmpi_init;
            gpu_aware = true;
            break;
#endif
#endif
        case ALLREDUCE_CTC_RECURSIVE_DOUBLING:
            method = allreduce_recursive_doubling_init;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_LOC:
            method = allreduce_dissemination_loc_init;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_ML:
            method = allreduce_dissemination_ml_init;
            copy_to_cpu = true;
            break;
        case ALLREDUCE_CTC_DISSEMINATION_RADIX:
            method = allreduce_dissemination_radix_init;
            copy_to_cpu = true;
            break;
#if defined(MPI4)
        case ALLREDUCE_CTC_PMPI:
            method = allreduce_pmpi_init;
            copy_to_cpu = true;
            break;
#endif
#endif
        case ALLREDUCE_RECURSIVE_DOUBLING:
            method = allreduce_recursive_doubling_init;
            break;
        case ALLREDUCE_DISSEMINATION_LOC:
            method = allreduce_dissemination_loc_init;
            break;
        case ALLREDUCE_DISSEMINATION_ML:
            method = allreduce_dissemination_ml_init;
            break;
        case ALLREDUCE_DISSEMINATION_RADIX:
            method = allreduce_dissemination_radix_init;
            break;
#if defined(MPI4)
        case ALLREDUCE_PMPI:
            method = allreduce_pmpi_init;
            break;
#endif
        case ALLREDUCE_RMA:
            method = allreduce_rma_init;
            break;
        case ALLREDUCE_RMA_HIERARCHICAL:
            method = allreduce_rma_hierarchical_init;
            break;
        case ALLREDUCE_RMA_MULTILEADER:
            method = allreduce_rma_multileader_init;
            break;
        case ALLREDUCE_RMA_EARLYBIRD:
            method = allreduce_rma_earlybird_init;
            break;
        case ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD:
            method = allreduce_rma_hierarchical_earlybird_init;
            break;
        case ALLREDUCE_RMA_MULTILEADER_EARLYBIRD:
            method = allreduce_rma_multileader_earlybird_init;
            break;
        default:
            method = allreduce_recursive_doubling_init;
            break;
    } 

#if defined(GPU) 
    if (gpu_aware)
        return gpu_aware_collective_init(method, sendbuf, recvbuf, count, 
                datatype, op, comm, info, req_ptr);
    else if (copy_to_cpu)
        return copy_to_cpu_allreduce_init(method, sendbuf, recvbuf, count,
                datatype, op, comm, info, req_ptr);
#endif

    return method(sendbuf, recvbuf, count, datatype, op, comm, info, req_ptr);
}

