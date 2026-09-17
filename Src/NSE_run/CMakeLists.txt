add_executable(nserun
                main.cpp
                initializeFlow.cpp
                ioManager.cpp
                MyFunctions.H
                IOConfig.H
                IOManager.H)

# Use AMReX macro for cuda compilation using CMake                
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(nserun)
endif()

if(AMReX_OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(nserun PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(nserun PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link to the NSE core libraries
target_link_libraries(nserun PRIVATE NSE_core_lib)

set_target_properties(nserun PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${BUILD_PARENT_DIR}/Exec/NSE_test"
)

option(ENABLE_DEBUG_CHECKS "Cross-rank BoxArray/DistributionMapping consistency asserts" OFF)

if (ENABLE_DEBUG_CHECKS)
    target_compile_definitions(nserun PRIVATE ENABLE_DEBUG_CHECKS)
endif()

# Update the input file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/inputs"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/inputs"
    COMMENT "Syncing inputs file to Exec/NSE_test..."
)

# Update the jobscript file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/debug.sh"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/debug.sh"
    COMMENT "Syncing jobscript to Exec/NSE_test..."
)

# Update the jobscript file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/gpu_single.sh"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/gpu_single.sh"
)

# Update the jobscript file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/gpu_multi.sh"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/gpu_multi.sh"
    COMMENT "Syncing GPU jobscripts to Exec/NSE_test..."
)

# Update the KE postprocessing file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/postProcessKE.py"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/Results/postProcessKE.py"
    COMMENT "Syncing KE postprocessing file to Exec/NSE_test..."
)

# Update the case description file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/description.txt"
    "${BUILD_PARENT_DIR}/Exec/NSE_test/description.txt"
    COMMENT "Syncing case description file to Exec/NSE_test..."
)
