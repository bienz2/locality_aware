## Grab all .cpp files in test directory
file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.c")

foreach(src ${CPP_SOURCES})
    get_filename_component(exec_name ${src} NAME_WE)
	make_executable(${src} TRUE)
    target_compile_definitions(${exec_name} PRIVATE MPIL_TEST_PPN=4)
endforeach()

if(USE_GPU)
	add_subdirectory(gpu_tests)
endif()
