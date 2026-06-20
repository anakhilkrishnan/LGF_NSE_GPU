# add files from BBFMM2d submodule
file(GLOB BBFMM2D_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/BBFMM2D/src/*.cpp")
file(GLOB BBFMM2D_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/BBFMM2D/header/*.hpp" )

# Define the library and its source files
add_library(LGF_core_lib STATIC
    addEverySource.cpp
    consolidateMultiFab.cpp
    tagSource.cpp
    solveBBFMM.cpp
    LGFCore.H
    LGFKernel.hpp
    ${BBFMM2D_SOURCES}
    ${BBFMM2D_HEADERS}
)

# setup_target_for_cuda_compilation(LGF_core_lib)

# Tell CMake where the headers from local sources and submodules are
target_include_directories(LGF_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/BBFMM2D/header)

# Link to AMReX so the core knows about MultiFabs and MPI
target_link_libraries(LGF_core_lib PUBLIC AMReX::amrex Eigen3::Eigen)