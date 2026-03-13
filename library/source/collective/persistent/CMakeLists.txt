target_sources(locality_aware PRIVATE 
    allreduce/allreduce_init_recursive_doubling.cpp
    allreduce/allreduce_init_dissemination_loc.cpp
    allreduce/allreduce_init_dissemination_radix.cpp
    allreduce/allreduce_init_rma.cpp
    allreduce/allreduce_init_rma_hierarchical.cpp
    allreduce/allreduce_init_rma_earlybird.cpp
    allreduce/allreduce_init_rma_hierarchical_earlybird.cpp
    alltoall/alltoall_init_standard.cpp
    alltoall/alltoall_init_rma.cpp
)
