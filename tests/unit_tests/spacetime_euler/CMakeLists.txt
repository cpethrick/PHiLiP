set(TEST_SRC
    euler_manufactured_solution_source.cpp
    )

# Output executable
string(CONCAT TEST_TARGET 1Dp1_euler_manufactured_solution_source)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=2)

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(2D ${TEST_TARGET})

# Library dependency
string(CONCAT PhysicsLib Physics_2D)
target_link_libraries(${TEST_TARGET} ${PhysicsLib})
# Setup target with deal.II
if (NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

add_test(
  NAME ${TEST_TARGET}
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set_tests_labels(${TEST_TARGET} EULER_UNIT_TEST
                                ${dim}D
                                SERIAL
                                MANUFACTURED_SOLUTION
                                QUICK
                                UNIT_TEST)

unset(TEST_TARGET)
unset(PhysicsLib)

