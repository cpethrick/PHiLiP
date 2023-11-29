#include "periodic_cube_flow.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_periodic_cube.hpp"
#include "mesh/grids/biased_periodic_cube.hpp"
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/grid_out.h>

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// FLOW IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
PeriodicCubeFlow<dim, nstate>::PeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , number_of_cells_per_direction(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension)
        , domain_left(this->all_param.flow_solver_param.grid_left_bound)
        , domain_right(this->all_param.flow_solver_param.grid_right_bound)
        , domain_size(pow(this->domain_right - this->domain_left, dim))
{ }

template <int dim, int nstate>
std::shared_ptr<Triangulation> PeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    if(this->all_param.flow_solver_param.use_gmsh_mesh) {
        if constexpr(dim == 3) {
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

            return cube_mesh->triangulation;
        } 
        else {
            this->pcout << "ERROR: read_gmsh() has not been tested with periodic_cube_flow() for 1D and 2D. Aborting..." << std::endl;
            std::abort();
        }
    } else {
        using GridEnum = Parameters::FlowSolverParam::PeriodicCubeGridType;
        GridEnum grid_type = this->all_param.flow_solver_param.periodic_cube_grid_type;

        
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
        );
        
        if (grid_type == GridEnum::straight) {
            this->pcout << "- Generating straight periodic grid using dealii GridGenerator" << std::endl;
            Grids::straight_periodic_cube<dim, Triangulation>(grid, domain_left, domain_right,
                                                            number_of_cells_per_direction);
        } else if (grid_type == GridEnum::biased) {
            this->pcout << "- Generating biased periodic grid using dealii GridGenerator" << std::endl;
            Grids::biased_periodic_cube<dim, Triangulation>(grid, domain_left, domain_right,
                                                            number_of_cells_per_direction);
        } else {
            this->pcout << "Grid type not recognized. Aborting..." << std::endl;
            std::abort();
        }

        return grid;
    }
}

template <int dim, int nstate>
void PeriodicCubeFlow<dim,nstate>::display_grid_parameters() const
{
    const std::string grid_type_string = "straight_periodic_cube";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain left: " << this->domain_left << std::endl;
    this->pcout << "- - Domain right: " << this->domain_right << std::endl;
    this->pcout << "- - Number of cells in each direction: " << this->number_of_cells_per_direction << std::endl;
    if constexpr(dim==1) this->pcout << "- - Domain length: " << this->domain_size << std::endl;
    if constexpr(dim==2) this->pcout << "- - Domain area: " << this->domain_size << std::endl;
    if constexpr(dim==3) this->pcout << "- - Domain volume: " << this->domain_size << std::endl;

    std::string c_parameter_string;
    using FREnum = Parameters::AllParameters::Flux_Reconstruction;
    FREnum fr_type = this->all_param.flux_reconstruction_type;
    if (fr_type == FREnum::cDG)         c_parameter_string = "cDG";
    if (fr_type == FREnum::cSD)         c_parameter_string = "cSD";
    if (fr_type == FREnum::cHU)         c_parameter_string = "cHU";
    if (fr_type == FREnum::cNegative)   c_parameter_string = "cNegative";
    if (fr_type == FREnum::cNegative2)  c_parameter_string = "cNegative2";
    if (fr_type == FREnum::cPlus)       c_parameter_string = "cPlus";
    if (fr_type == FREnum::c10Thousand) c_parameter_string = "c10Thousand";
    if (fr_type == FREnum::cHULumped)   c_parameter_string = "cHULumped";

    this->pcout << "- Using c parameter: " << c_parameter_string << std::endl;
}

template <int dim, int nstate>
void PeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->display_grid_parameters();
}

#if PHILIP_DIM==1
template class PeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM>;
#else
template class PeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

