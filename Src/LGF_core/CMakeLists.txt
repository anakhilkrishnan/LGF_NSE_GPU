# Define the library and its source files
add_library(LGF_core_lib STATIC
    LGFKernels.H
    directSumKernel.H
    DirectSumLGF.H
    directSumLGF.cpp
)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(LGF_core_lib)
endif()

# Tell CMake where the headers from local sources and submodules are
target_include_directories(LGF_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link to AMReX so the core knows about MultiFabs and MPI
target_link_libraries(LGF_core_lib PUBLIC AMReX::amrex)