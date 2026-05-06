#include "spacetime_cartesian_problem.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_semiperiodic_cube.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// FLOW IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nspecies, int nstate>
SpacetimeCartesianProblem<dim,nspecies,nstate>::SpacetimeCartesianProblem(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim,nspecies,nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }


template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> SpacetimeCartesianProblem<dim,nspecies,nstate>::generate_grid() const
{
    if(this->all_param.flow_solver_param.use_gmsh_mesh) {
        this->pcout << "ERROR: gmsh mesh not configured for this flow case." << std::endl;
        std::abort();
    } else {
        this->pcout << "- Generating grid using dealii GridGenerator" << std::endl;
        
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
        );
        
        Grids::straight_semiperiodic_cube<dim, Triangulation>(grid, domain_left, domain_right,
                                                              number_of_cells_per_direction);
        return grid;
    }
}


template <int dim, int nspecies, int nstate>
void SpacetimeCartesianProblem<dim,nspecies,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Empty for now.
}

#if PHILIP_DIM>1
template class SpacetimeCartesianProblem <PHILIP_DIM,PHILIP_SPECIES,1>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

