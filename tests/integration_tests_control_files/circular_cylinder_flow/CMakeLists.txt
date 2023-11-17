set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# Circular Cylinder Test
# =======================================
# ----------------------------------------
# - Add description later.
# ----------------------------------------
configure_file(circular_cylinder_flow.prm circular_cylinder_flow.prm COPYONLY)
add_test(
    NAME MPI_2D_CIRCULAR_CYLINDER_FLOW
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/circular_cylinder_flow.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# ----------------------------------------
