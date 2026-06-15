## Grab all .cpp files in test directory
set(CPP_SOURCES
    test_allgather.cpp
    test_allgather_init.cpp
    test_allreduce.cpp
    test_allreduce_init.cpp
    test_alltoall.cpp
    test_alltoall_c.c
    test_alltoall_init.cpp
    test_alltoall_enum.cpp
    test_alltoallv.cpp
    test_alltoallv_init.cpp
    test_alltoallv_enum.cpp
    test_neighbor_alltoallv_init.cpp
    test_neighbor_reorder.cpp
    test_neighbor_topo_alltoallv_init.cpp
    test_suitesparse_alltoall_crs.cpp
    test_suitesparse_alltoallv.cpp
    test_suitesparse_alltoallv_crs.cpp
    test_suitesparse_neighbor_alltoallv.cpp
    test_suitesparse_neighbor_alltoallv_enum.cpp
    test_suitesparse_neighbor_alltoallv_init_bidirectional.cpp
    test_suitesparse_neighbor_alltoallv_init.cpp
    test_suitesparse_neighbor_reorder.cpp
    test_suitesparse_neighbor_topo_alltoallv_init.cpp
)

foreach(src ${CPP_SOURCES})
	make_executable(${src} TRUE)
endforeach()

if(USE_GPU)
    set(GPU_SOURCES
        test_gpu_allreduce.cpp
        test_gpu_allreduce_init.cpp
        test_gpu_allgather.cpp
        test_gpu_allgather_init.cpp
        test_gpu_alltoall.cpp
        test_gpu_alltoall_init.cpp
        test_gpu_alltoallv.cpp
        test_gpu_alltoallv_init.cpp
        test_gpu_suitesparse_neighbor_alltoallv.cpp
        test_gpu_suitesparse_neighbor_alltoallv_init.cpp
)
endif()
set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE ${locality_aware_LANG})

## Go through each GPU file
foreach(src ${GPU_SOURCES})
    make_executable(${src} TRUE)
endforeach()

