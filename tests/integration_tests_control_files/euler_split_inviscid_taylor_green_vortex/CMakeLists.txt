set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_euler_split_inviscid_taylor_green_vortex.prm 3D_euler_split_inviscid_taylor_green_vortex.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_EULER_SPLIT_TAYLOR_GREEN                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_euler_split_inviscid_taylor_green_vortex.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)
set_tests_labels(MPI_3D_EULER_SPLIT_TAYLOR_GREEN    EULER_SPLIT_INVISCID_TAYLOR_GREEN_VORTEX
                                                    3D
                                                    PARALLEL
                                                    EULER
                                                    RUNGE-KUTTA
                                                    STRONG-SPLIT
                                                    UNCOLLOCATED
                                                    LONG
                                                    INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_euler_split_inviscid_taylor_green_vortex_curv.prm 3D_euler_split_inviscid_taylor_green_vortex_curv.prm COPYONLY)                                                                                                                                            
add_test(                                                                                                                                                                                                                                 
 NAME MPI_3D_EULER_SPLIT_TAYLOR_GREEN_CURV                                                                                                                                                                                    
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_euler_split_inviscid_taylor_green_vortex_curv.prm                                                                                                                                                                                                                                  
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}                                                                                                                                                                                                      
)                                                                                                                       
set_tests_labels(MPI_3D_EULER_SPLIT_TAYLOR_GREEN_CURV   EULER_SPLIT_INVISCID_TAYLOR_GREEN_VORTEX
                                                        3D
                                                        PARALLEL
                                                        EULER
                                                        RUNGE-KUTTA
                                                        STRONG-SPLIT
                                                        UNCOLLOCATED
                                                        CURVILINEAR
                                                        LONG
                                                        INTEGRATION_TEST)