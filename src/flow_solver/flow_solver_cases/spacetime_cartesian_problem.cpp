#include "spacetime_cartesian_problem.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_semiperiodic_cube.hpp"
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/grid_tools.h>
#include "dg/dg_base_state.hpp"

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// Flow in a spatially periodic cartesian grid.
// The grid is generated for dim = spatial_dim + temporal_dim
//=========================================================
template <int dim, int nstate>
SpacetimeCartesianProblem<dim, nstate>::SpacetimeCartesianProblem(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }

// Helper function to scale the width of the time-slab
dealii::Point<2> scale_timeslab(const double factor, const dealii::Point<2> &in)
{
    return dealii::Point<2,double>(in(0), in(1) * factor);
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> SpacetimeCartesianProblem<dim,nstate>::generate_grid() const
{
    if(this->all_param.flow_solver_param.use_gmsh_mesh) {
        if constexpr(dim==2){
            const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename + std::string(".msh");
            this->pcout << "- Generating grid using input mesh: " << mesh_filename << std::endl;
            std::shared_ptr <HighOrderGrid<dim, double>> cube_mesh = read_gmsh<dim, dim>(
                mesh_filename, 
                this->all_param.flow_solver_param.use_periodic_BC_in_x, 
                this->all_param.flow_solver_param.use_periodic_BC_in_y, 
                this->all_param.flow_solver_param.use_periodic_BC_in_z, 
                this->all_param.flow_solver_param.x_periodic_id_face_1, 
                this->all_param.flow_solver_param.x_periodic_id_face_2, 
                this->all_param.flow_solver_param.y_periodic_id_face_1, 
                this->all_param.flow_solver_param.y_periodic_id_face_2, 
                this->all_param.flow_solver_param.z_periodic_id_face_1, 
                this->all_param.flow_solver_param.z_periodic_id_face_2,
                this->all_param.flow_solver_param.mesh_reader_verbose_output,
                this->all_param.do_renumber_dofs);

            const double factor = 2.0 / cube_mesh->triangulation->n_cells();
            // See deal.ii tutorial steps 49 and 53 for details on transforming a mesh
            dealii::GridTools::transform(std::bind( scale_timeslab,
                        std::cref(factor),
                        std::placeholders::_1 ),
                    *(cube_mesh->triangulation));
            return cube_mesh->triangulation;
        }else{
            this->pcout << "ERROR: gmsh mesh not configured for this flow case." << std::endl;
            std::abort();
        }
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


template <int dim, int nstate>
void SpacetimeCartesianProblem<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Empty for now.
}

template <int dim, int nstate>
void SpacetimeCartesianProblem<dim, nstate>::modify_dg_object(std::shared_ptr <DGBase<dim, double>> dg) const
{
    // Dynamic cast to DGBaseState to gain access to dg_state->->conv_num_flux<> and dg_base_state->pde_physics<>
    std::shared_ptr <DGBaseState<dim,nstate,double>> dg_state = std::dynamic_pointer_cast<DGBaseState<dim,nstate, double>> (dg);

    // Go through all AD types & modify temporal advection direction
    dg_state->pde_physics_double->temporal_advection *= -1;
    dg_state->pde_physics_fad->temporal_advection *= -1;
    dg_state->pde_physics_rad->temporal_advection *= -1;
    dg_state->pde_physics_fad_fad->temporal_advection *= -1;
    dg_state->pde_physics_rad_fad->temporal_advection *= -1;

    dg_state->conv_num_flux_double->temporal_advection *= -1;
    dg_state->conv_num_flux_fad->temporal_advection *= -1;
    dg_state->conv_num_flux_rad->temporal_advection *= -1;
    dg_state->conv_num_flux_fad_fad->temporal_advection *= -1;
    dg_state->conv_num_flux_rad_fad->temporal_advection *= -1;

    if (dg->get_current_time() == 0.0){
        // No longer need to apply IC
        dg_state->pde_physics_double->apply_initial_condition=false;
        dg_state->pde_physics_fad->apply_initial_condition=false;
        dg_state->pde_physics_rad->apply_initial_condition=false;
        dg_state->pde_physics_fad_fad->apply_initial_condition=false;
        dg_state->pde_physics_rad_fad->apply_initial_condition=false;
    }

}

#if PHILIP_DIM>1
template class SpacetimeCartesianProblem <PHILIP_DIM,1>;
template class SpacetimeCartesianProblem <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

