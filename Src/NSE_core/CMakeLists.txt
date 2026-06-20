# Define the library and its source files
add_library(NSE_core_lib STATIC
                flowField.cpp
                projectionWorkspace.cpp
                FlowField.H
                ProjectionWorkspace.H
                RKCoefficients.H
                SpatialDiscretization.H)
                
# setup_target_for_cuda_compilation(NSE_core_lib)

# Tell CMake where the headers (like LGFCore.H) are
target_include_directories(NSE_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link to AMReX so the core knows about MultiFabs and MPI
target_link_libraries(NSE_core_lib PUBLIC LGF_core_lib AMReX::amrex)