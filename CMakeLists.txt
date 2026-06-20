cmake_minimum_required(VERSION 3.16)

# project(LGF_NSE LANGUAGES C CXX CUDA)

# # set of default values for a typical gpu run
# set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
# set(AMReX_SPACEDIM "2" CACHE STRING "Spatial Dimension")
# set(AMReX_MPI YES CACHE BOOL "Enable MPI")
# set(AMReX_GPU_BACKEND "CUDA" CACHE STRING "GPU Backend")
# set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "GPU Architecture")
# set(AMReX_TINY_PROFILE YES CACHE BOOL "Enable Tiny Profiling")
# set(AMReX_MEM_PROFILE YES CACHE BOOL "Enable Memory Profiling")

project(LGF_NSE LANGUAGES C CXX)

# set of default values for typical cpu run
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
set(AMReX_SPACEDIM "2" CACHE STRING "Spatial Dimension")
set(AMReX_MPI YES CACHE BOOL "Enable MPI")
set(AMReX_OMP OFF CACHE BOOL "Enable OpenMP")
set(AMReX_TINY_PROFILE YES CACHE BOOL "Enable Tiny Profiling")
set(AMReX_MEM_PROFILE YES CACHE BOOL "Enable Memory Profiling")

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
add_subdirectory(Ext/LGF_Accelerate)
add_subdirectory(Src/NSE_core)
add_subdirectory(Src/NSE_run) 


# make Exec directories in case they're not present
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test/Results")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test/Logs")

