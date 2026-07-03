cmake_minimum_required(VERSION 3.16)

# typical values (included for ease of building)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
set(AMReX_SPACEDIM "2" CACHE STRING "Spatial Dimension")
set(AMReX_MPI YES CACHE BOOL "Enable MPI")
set(AMReX_TINY_PROFILE YES CACHE BOOL "Enable Tiny Profiling")
set(AMReX_MEM_PROFILE YES CACHE BOOL "Enable Memory Profiling")
# enable OpenMP by passing a flag when needed

project(LGF_NSE LANGUAGES C CXX)

if(AMREX_GPU_BACKED STREQUAL "CUDA")
  enable_language(CUDA)
  # always specify CMAKE_CUDA_ARCHITECTURES at compile time
endif()

if(NOT DEFINED AMReX_ROOT)
  message("-- Download and configure AMReX from GitHub")

  # Download AMReX from GitHub
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)

  FetchContent_Declare(
    amrex_code
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git/
    GIT_TAG        origin/development
    )

  FetchContent_MakeAvailable(amrex_code)


else()

  # Add AMReX
  message("-- Searching for AMReX install directory at ${AMReX_ROOT}")
  find_package(AMReX REQUIRED PATHS ${AMReX_ROOT}/lib/cmake/AMReX/AMReXConfig.cmake)

endif()

# adding additional repositories
add_subdirectory(Src/LGF_core)
add_subdirectory(Src/NSE_core)
add_subdirectory(Src/NSE_run) 


# make Exec directories in case they're not present
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test/Results")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test/Logs")

