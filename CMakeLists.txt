cmake_minimum_required(VERSION 3.24)
project(LGF_NSE LANGUAGES C CXX)


# forcing c++ standards for my code and AMReX
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# force user to choose 2 main flags (GPU BACKEND and SPACEDIM), with additional
# architecture specification

# GPU BACKEND
set(_backends NONE CUDA)
if(NOT DEFINED AMReX_GPU_BACKEND)
  message(FATAL_ERROR "Set -DAMReX_GPU_BACKEND=<NONE|CUDA>")
endif()
if(NOT AMReX_GPU_BACKEND IN_LIST _backends)
  message(FATAL_ERROR "AMReX_GPU_BACKEND='${AMReX_GPU_BACKEND}' must be one of: ${_backends}")
endif()

# SPACEDIM
if(NOT DEFINED AMReX_SPACEDIM)
  message(FATAL_ERROR "Set -DAMReX_SPACEDIM=<1|2|3>")
endif()
if(NOT AMReX_SPACEDIM MATCHES "^[123]$")
  message(FATAL_ERROR "AMReX_SPACEDIM='${AMReX_SPACEDIM}' must be 1, 2 or 3")
endif()

# ARCHITECTURE and enable CUDA language
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    message(FATAL_ERROR "Set -DCMAKE_CUDA_ARCHITECTURES=<e.g. 80 or 80;86 (use double quotes for multiple)>")
  endif()
  enable_language(CUDA)
endif()


# force regularly used flags
set(AMReX_MPI ON CACHE BOOL "Enable MPI" FORCE)
set(AMReX_FFT ON CACHE BOOL "Enable FFT" FORCE)
set(AMReX_TINY_PROFILE ON CACHE BOOL "Enable Tiny Profiling" FORCE)
set(AMReX_MEM_PROFILE ON CACHE BOOL "Enable Memory Profiling" FORCE)


# flags that can be overriden as needed
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
set(AMReX_OMP OFF CACHE BOOL "Enable OpenMP")


# Download AMReX from GitHub
message("-- Download and configure AMReX from GitHub")
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  amrex_code
  GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git/
  GIT_TAG        origin/development
  )

FetchContent_MakeAvailable(amrex_code)

# convenient for building anywhere, not just in source level dirs
cmake_path(GET CMAKE_BINARY_DIR PARENT_PATH BUILD_PARENT_DIR)


# adding additional repositories
add_subdirectory(Src/LGF_core)
add_subdirectory(Src/NSE_core)
add_subdirectory(Src/NSE_run) 


# make Exec directories in case they're not present
file(MAKE_DIRECTORY "${BUILD_PARENT_DIR}/NSE_test/Results")
file(MAKE_DIRECTORY "${BUILD_PARENT_DIR}/NSE_test/Logs")

