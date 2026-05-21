target_sources(locality_aware PRIVATE 
    allreduce_init_rma.cpp
    allreduce_init_rma_hierarchical.cpp
    allreduce_init_rma_hierarchical_earlybird.cpp
    allreduce_init_rma_earlybird.cpp
    allreduce_init_recursive_doubling.cpp
    allreduce_init_dissemination_loc.cpp
    allreduce_init_dissemination_radix.cpp
)
