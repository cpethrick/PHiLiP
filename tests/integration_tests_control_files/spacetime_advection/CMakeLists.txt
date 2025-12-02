# TO DO BEFORE MERGING: add comments + labels

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1Dp1_linear_advection.prm 1Dp1_linear_advection.prm COPYONLY)
add_test(
    NAME serial_1Dp1_ADVECTION
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/1Dp1_linear_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1Dp1_linear_advection.prm 1Dp1_linear_advection.prm COPYONLY)
add_test(
    NAME MPI_1Dp1_ADVECTION_OOA
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/1Dp1_linear_advection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1Dp1_linear_advection_c_ramp.prm 1Dp1_linear_advection_c_ramp.prm COPYONLY)
add_test(
    NAME MPI_1Dp1_ADVECTION_C_RAMP
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/1Dp1_linear_advection_c_ramp.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# -------------------
# This param file is consistent with 1Dp1_linear_advection_c_ramp.prm
# It is commented from ctest because the advection speed and 
# +0.01 offset need to be manually changed in physics.
# -------------------
#set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
#configure_file(1D_consistent_c_ramp.prm 1D_consistent_c_ramp.prm COPYONLY)
#add_test(
#    NAME serial_1D_ADVECTION_C_RAMP
#COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_consistent_c_ramp.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2Dp1_linear_advection_flowsolver.prm 2Dp1_linear_advection_flowsolver.prm COPYONLY)
add_test(
    NAME serial_2Dp1_ADVECTION
COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/2Dp1_linear_advection_flowsolver.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2Dp1_linear_advection_OOA.prm 2Dp1_linear_advection_OOA.prm COPYONLY)
add_test(
    NAME MPI_2Dp1_ADVECTION_OOA
    COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/2Dp1_linear_advection_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# set_tests_labels(MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG ADVECTION_EXPLICIT_PERIODIC
#                                                                 2D
#                                                                 PARALLEL
#                                                                 ADVECTION
#                                                                 RUNGE-KUTTA
#                                                                 STRONG
#                                                                 UNCOLLOCATED
#                                                                 MODERATE
#                                                                 INTEGRATION_TEST)
