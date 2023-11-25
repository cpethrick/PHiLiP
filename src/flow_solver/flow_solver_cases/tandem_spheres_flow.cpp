#include "tandem_spheres_flow.h"
#include "mesh/gmsh_reader.hpp"
#include "mesh/grids/circular_cylinder.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
TandemSpheresFlow<dim, nstate>::TandemSpheresFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{
}

//TO DO
//  Add CFL-adaptive time stepping
//  Report integrated lift and drag

template <int dim, int nstate>
std::shared_ptr<Triangulation> TandemSpheresFlow<dim,nstate>::generate_grid() const
{
    this -> pcout << "Reading mesh." << std::endl;
    //if(this->all_param.flow_solver_param.use_gmsh_mesh) {
    if(false) {
        if constexpr(dim == 3 || dim==2) {
            const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename + std::string(".msh");
            this->pcout << "- Generating grid using input mesh: " << mesh_filename << std::endl;
            
            std::shared_ptr <HighOrderGrid<dim, double>> tandem_spheres_mesh = read_gmsh<dim, dim>(
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
                //this->all_param.flow_solver_param.grid_degree);

            // Assign boundaries
            // In the provided .msh files, the following boundary IDs are assigned
            // 2 "BackSphere"
            // 3 "FarField"
            // 4 "FrontSphere"
            // Need to assign spheres to 1001, wall
            // and FarField to 1004, farfield
            /*
            std::cout << "About to change BC settings." << std::endl; 
            std::cout << "active_cell_iterator : " << (tandem_spheres_mesh->triangulation)->begin_active() 
                      << " end : " << (tandem_spheres_mesh->triangulation)->end();
            for (typename dealii::parallel::distributed::Triangulation<2>::active_cell_iterator cell = (tandem_spheres_mesh->triangulation)->begin_active(); cell != (tandem_spheres_mesh->triangulation)->end(); ++cell) {
                std::cout << "Here" << std::endl;
                for (unsigned int face=0; face<dealii::GeometryInfo<2>::faces_per_cell; ++face) {
                    std::cout << "Here" << std::endl;
                    if (cell->face(face)->at_boundary()) {
                        unsigned int current_id = cell->face(face)->boundary_id();
                        if (current_id == 3) {
                            std::cout << "Setting farfield BC" << std::endl;
                            cell->face(face)->set_boundary_id (1004); // farfield
                        } else if (current_id == 2 || current_id == 4) {
                            std::cout << "Setting wall BC" << std::endl;
                            cell->face(face)->set_boundary_id (1001); // wall bc
                        } else {
                            this ->pcout << "Error: unrecognized boundary ID! Aborting..." << std::endl;
                            std::abort();
                        }
                    }
                }
            }
            */

            return tandem_spheres_mesh->triangulation;
            //
            //
        }
    } else {
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
#if PHILIP_DIM!=1
            this->mpi_communicator
#endif
        );
        Grids::circular_cylinder<dim>(*grid);
        return grid;
         
    }
}

template <int dim, int nstate>
void TandemSpheresFlow<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    if(this->all_param.flow_solver_param.use_gmsh_mesh) { 
        this->pcout << "Setting HO mesh using the gmsh reader." << std::endl;
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename + std::string(".msh");
        std::shared_ptr <HighOrderGrid<dim, double>> tandem_spheres_mesh = read_gmsh<dim, dim>(
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

        dg->set_high_order_grid(tandem_spheres_mesh);

        //dg->high_order_grid->refine_global();
    }
       // else do nothing; don't need to re-set HO grid if using dealii mesh generator.
}

template <int dim, int nstate>
void TandemSpheresFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}


#if PHILIP_DIM==3
    template class TandemSpheresFlow<PHILIP_DIM, PHILIP_DIM+2>;
#endif
#if PHILIP_DIM==2
    template class TandemSpheresFlow<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace
