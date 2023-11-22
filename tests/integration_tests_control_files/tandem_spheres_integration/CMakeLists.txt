set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
#  Tandem spheres test case
# =======================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- -- HiOCFD5 workshop, CS1 - Tandem Spheres Re=3900, lead by Ryan Glasby
# -- -- Using pointwise GMSH files provided by Steve Karman
# ----------------------------------------
configure_file(tandem_spheres.prm tandem_spheres.prm COPYONLY)
add_test(
    NAME MPI_3D_TANDEM_SPHERES
    COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/tandem_spheres.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# =======================================
#  Tandem spheres test case
# =======================================
# ----------------------------------------
# -- Reference for flow setup: 
# -- -- HiOCFD5 workshop, CS1 - Tandem Spheres Re=3900, lead by Ryan Glasby
# -- -- Using pointwise GMSH files provided by Steve Karman
# ----------------------------------------
configure_file(tandem_spheres.prm tandem_spheres.prm COPYONLY)
add_test(
    NAME SERIAL_3D_TANDEM_SPHERES
    COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/tandem_spheres.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
