#include "tandem_spheres_flow.h"
#include "mesh/gmsh_reader.hpp"

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
    if(this->all_param.flow_solver_param.use_gmsh_mesh) {
        if constexpr(dim == 3) {
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
        } 
    }
    else{
        this->pcout << "ERROR: Grid must be provided as a gmsh mesh."<< std::endl;
        std::abort();
        return nullptr;
    }
}

template <int dim, int nstate>
void TandemSpheresFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Do nothing for now.
}

#if PHILIP_DIM==3
    template class TandemSpheresFlow<PHILIP_DIM, PHILIP_DIM+2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace
