add_executable(nserun
                main.cpp
                initializeFlow.cpp
                ioManager.cpp
                MyFunctions.H
                SimConfig.H
                IOManager.H)

# uncomment below line for gpu compilation
# setup_target_for_cuda_compilation(nserun)

find_package(OpenMP REQUIRED)
target_link_libraries(nserun PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(nserun PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link to the NSE core libraries
target_link_libraries(nserun PRIVATE NSE_core_lib)

set_target_properties(nserun PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Exec/NSE_test"
)

# Update the input file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/inputs"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/inputs"
    COMMENT "Syncing inputs file to Exec/NSE_test..."
)

# Update the jobscript file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/debug.sh"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/debug.sh"
    COMMENT "Syncing jobscript to Exec/NSE_test..."
)

# Update the KE postprocessing file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/postProcessKE.py"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/Results/postProcessKE.py"
    COMMENT "Syncing KE postprocessing file to Exec/NSE_test..."
)

# Update the case description file
add_custom_command(TARGET nserun POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/Utils/description.txt"
    "${CMAKE_SOURCE_DIR}/Exec/NSE_test/description.txt"
    COMMENT "Syncing case description file to Exec/NSE_test..."
)
