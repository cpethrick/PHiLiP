#include "spacetime_cartesian_problem.h"

#include <stdlib.h>
#include <iostream>
#include "mesh/grids/straight_semiperiodic_cube.hpp"
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/grid_tools.h>
#include "dg/dg_base_state.hpp"

#include "dg/dg_base.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/fe/mapping_fe_field.h>

namespace PHiLiP {

namespace FlowSolver {
//=========================================================
// Flow in a spatially periodic cartesian grid.
// The grid is generated for dim = spatial_dim + temporal_dim
//=========================================================
template <int dim, int nspecies, int nstate>
SpacetimeCartesianProblem<dim,nspecies,nstate>::SpacetimeCartesianProblem(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim,nspecies,nstate>(parameters_input)
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

template <int dim, int nspecies, int nstate>
std::shared_ptr<Triangulation> SpacetimeCartesianProblem<dim,nspecies,nstate>::generate_grid() const
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


template <int dim, int nspecies, int nstate>
void SpacetimeCartesianProblem<dim,nspecies,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Empty for now.
}

template <int dim, int nspecies, int nstate>
template<typename adtype>
void SpacetimeCartesianProblem<dim,nspecies,nstate>::get_surface_solution_for_BC(std::shared_ptr <DGBase<dim,nspecies,double>> dg,
std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim, nspecies, nstate, adtype>> pde_physics) const
{
    const double grid_height = 2.0/dg->triangulation->n_cells();

    //Get operators for cell loop
    const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, dg->max_degree, init_grid_degree); 
    OPERATOR::basis_functions<dim,2*dim> soln_basis_ext(1, dg->max_degree, init_grid_degree); 
    OPERATOR::basis_functions<dim,2*dim> flux_basis_int(1, dg->max_degree, init_grid_degree); 
    OPERATOR::basis_functions<dim,2*dim> flux_basis_ext(1, dg->max_degree, init_grid_degree); 
    OPERATOR::local_basis_stiffness<dim,2*dim> flux_basis_stiffness(1, dg->max_degree, init_grid_degree, true); 
    OPERATOR::vol_projection_operator<dim,2*dim> soln_basis_projection_oper(1, dg->max_degree, init_grid_degree); 
    OPERATOR::vol_projection_operator<dim,2*dim> soln_basis_projection_oper_ext(1, dg->max_degree, init_grid_degree); 
    OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, init_grid_degree, init_grid_degree);

    dg->reinit_operators_for_cell_residual_loop(
            dg->max_degree, dg->max_degree, init_grid_degree, 
            soln_basis, soln_basis_ext, 
            flux_basis_int, flux_basis_ext, 
            flux_basis_stiffness, 
            soln_basis_projection_oper, soln_basis_projection_oper_ext,
            mapping_basis);

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    for (auto soln_cell = dg->dof_handler.begin_active(); soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell) 
    {
        if (!soln_cell->is_locally_owned()) continue;

        // ############ Get local solution 

        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = dg->fe_collection[i_fele];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> soln_dofs_indices;
        soln_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices (soln_dofs_indices);
        const unsigned int poly_degree = i_fele;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate; 

        const unsigned int n_dofs_cell = n_dofs_curr_cell;
        std::vector<adtype> local_solution(n_dofs_cell);
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            // Extract local solution
            const double val = dg->solution(soln_dofs_indices[idof]);
            local_solution[idof] = val;
        }

        // ########### Reorder solution coeffs for strong DG coding.
        std::array<std::vector<adtype>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = local_solution[idof];
        }

        // ############### Face loop
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            // ######### Filter out correct face by direction of normal (LATER)
            // if normal NOT +1 in time CONTINUE

            const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
            const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
            const unsigned int n_grid_nodes  = n_metric_dofs / dim;
            const unsigned int grid_degree = dg->high_order_grid->fe_system.tensor_degree();
            //setup metric cell
            std::vector<dealii::types::global_dof_index> metric_dofs_indices(n_metric_dofs);
            metric_cell->get_dof_indices (metric_dofs_indices);
            // get mapping_support points
            std::array<std::vector<double>,dim> mapping_support_points;
            for(int idim=0; idim<dim; idim++){
                mapping_support_points[idim].resize(n_metric_dofs/dim);
            }
            const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
            for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
                const double val = (dg->high_order_grid->volume_nodes[metric_dofs_indices[idof]]);
                const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
                const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
                const unsigned int igrid_node = index_renumbering[ishape];
                mapping_support_points[istate][igrid_node] = val; 
            }
            //const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
            OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(1, poly_degree, grid_degree, false, true); //store_surf_flux_nodes = true
            metric_oper.build_facet_metric_operators(
                iface,
                dg->face_quadrature_collection[poly_degree].size(),
                n_grid_nodes,
                mapping_support_points,
                mapping_basis,
                dg->all_parameters->use_invariant_curl_form);
            dealii::Point<dim,adtype> surf_flux_node;
            const unsigned int n_face_quad_pts  = dg->face_quadrature_collection[poly_degree].size();
            bool on_outflow_face = true;
            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
                for(int idim=0; idim<dim; idim++){
                    surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
                    //std::cout << surf_flux_node[idim] << " ";
                }
                //std::cout << std::endl;
                if ((surf_flux_node[dim-1] == grid_height  && pde_physics->temporal_advection>0 )
                        ||( surf_flux_node[dim-1] == 0.0 && pde_physics->temporal_advection<0)) {
                    //std::cout << "On top face! " << std::endl;
                } else {
                    on_outflow_face = false;
                }
                    
            }
            if (!on_outflow_face) continue;


            std::vector<bool> face_orientation = {soln_cell->face_orientation(iface), soln_cell->face_rotation(iface), soln_cell->face_flip(iface)};

            // Find solution at surface and volume quad
            const unsigned int n_quad_pts_vol   = dg->volume_quadrature_collection[poly_degree].size();
            std::array<std::vector<adtype>,nstate> soln_at_vol_q;
            std::array<std::vector<adtype>,nstate> soln_at_surf_q;
            for(int istate=0; istate<nstate; ++istate){
                //allocate
                soln_at_vol_q[istate].resize(n_quad_pts_vol);
                //solve soln at volume cubature nodes
                soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_vol_q[istate],
                        soln_basis.oneD_vol_operator);

                /* I don't think I need this 
                //allocate
                soln_at_surf_q[istate].resize(n_face_quad_pts);
                //solve soln at facet cubature nodes
                soln_basis.matrix_vector_mult_surface_1D(face_orientation, 
                iface,
                soln_coeff[istate], soln_at_surf_q[istate],
                soln_basis.oneD_surf_operator,
                soln_basis.oneD_vol_operator);
                */

            }

            //Find entropy variables
            //
            // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
            std::array<std::vector<adtype>,nstate> entropy_var_vol;
            for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
                std::array<adtype,nstate> soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_vol_q[istate][iquad];
                }
                std::array<adtype,nstate> entropy_var;
                entropy_var = pde_physics->compute_entropy_variables(soln_state);
                for(int istate=0; istate<nstate; istate++){
                    if(iquad==0){
                        entropy_var_vol[istate].resize(n_quad_pts_vol);
                    }
                    entropy_var_vol[istate][iquad] = entropy_var[istate];
                }
            }

            //Entropy-project to surface
            //project it onto the solution basis functions and interpolate it
            std::array<std::vector<adtype>,nstate> projected_entropy_var_vol;
            std::array<std::vector<adtype>,nstate> projected_entropy_var_surf;
            for(int istate=0; istate<nstate; istate++){
                // allocate
                //projected_entropy_var_vol[istate].resize(n_quad_pts_vol);
                projected_entropy_var_surf[istate].resize(n_face_quad_pts);

                //interior
                std::vector<adtype> entropy_var_coeff(n_shape_fns);
                soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_vol[istate],
                        entropy_var_coeff,
                        soln_basis_projection_oper.oneD_vol_operator);
                //soln_basis.matrix_vector_mult_1D(entropy_var_coeff,
                //        projected_entropy_var_vol[istate],
                //        soln_basis.oneD_vol_operator);
                soln_basis.matrix_vector_mult_surface_1D(face_orientation, 
                        iface,
                        entropy_var_coeff, 
                        projected_entropy_var_surf[istate],
                        soln_basis.oneD_surf_operator,
                        soln_basis.oneD_vol_operator);
            }


            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
                std::array<adtype,nstate> entropy_var_face;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var_face[istate] = projected_entropy_var_surf[istate][iquad];
                    // std::cout << entropy_var_face[istate] << " at " <<  soln_cell->active_cell_index() << " " << iquad << " " << istate << " core" << (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) << std::endl;
                }
                // Get entropy-projected surface soln.
                 std::array<adtype,nstate> conservative_vars_quad = pde_physics->compute_conservative_variables_from_entropy_variables (entropy_var_face);    
                // STORE: 
                const int icell = soln_cell->active_cell_index();
                for (int istate = 0; istate<nstate; ++istate){
                    pde_physics->imposed_boundary[icell][iquad][istate] = conservative_vars_quad[istate];
                }
            }
        }
    }
}

template <int dim, int nspecies, int nstate>
void SpacetimeCartesianProblem<dim,nspecies,nstate>::modify_dg_object(std::shared_ptr <DGBase<dim,nspecies,double>> dg) const
{
    // Dynamic cast to DGBaseState to gain access to dg_state->->conv_num_flux<> and dg_base_state->pde_physics<>
    std::shared_ptr <DGBaseState<dim,nspecies,nstate,double>> dg_state = std::dynamic_pointer_cast<DGBaseState<dim,nspecies,nstate,double>> (dg);

    if (dg->get_current_time() > 0.0){
        // No longer need to apply IC
        dg_state->pde_physics_double->apply_initial_condition=false;
        dg_state->pde_physics_fad->apply_initial_condition=false;
        dg_state->pde_physics_rad->apply_initial_condition=false;
        dg_state->pde_physics_fad_fad->apply_initial_condition=false;
        dg_state->pde_physics_rad_fad->apply_initial_condition=false;


        // NOTE TO SELF: Will probably also need to re-assign solution gradient at the boundary for NS.
        dg_state->pde_physics_double->imposed_boundary.resize(dg->triangulation->n_active_cells());
        dg_state->pde_physics_fad->imposed_boundary.resize(dg->triangulation->n_active_cells());
        dg_state->pde_physics_rad->imposed_boundary.resize(dg->triangulation->n_active_cells());
        dg_state->pde_physics_fad_fad->imposed_boundary.resize(dg->triangulation->n_active_cells());
        dg_state->pde_physics_rad_fad->imposed_boundary.resize(dg->triangulation->n_active_cells());
        for (unsigned int icell = 0; icell < dg->triangulation->n_active_cells();++icell) {
            // if (!(dg->triangulation->get_cell(icell).is_locally_owned())) continue;

            // LATER: Only allocate memory when that cell is active.
            const int n_quad_face = dg->all_parameters->flow_solver_param.poly_degree + dg->all_parameters->overintegration + 1;
            dg_state->pde_physics_double->imposed_boundary[icell].resize(n_quad_face);
            dg_state->pde_physics_fad->imposed_boundary[icell].resize(n_quad_face);
            dg_state->pde_physics_rad->imposed_boundary[icell].resize(n_quad_face);
            dg_state->pde_physics_fad_fad->imposed_boundary[icell].resize(n_quad_face);
            dg_state->pde_physics_rad_fad->imposed_boundary[icell].resize(n_quad_face);
            for (int iquad = 0; iquad < n_quad_face; ++iquad) {
                dg_state->pde_physics_double->imposed_boundary[icell][iquad].resize(nstate);
                dg_state->pde_physics_fad->imposed_boundary[icell][iquad].resize(nstate);
                dg_state->pde_physics_rad->imposed_boundary[icell][iquad].resize(nstate);
                dg_state->pde_physics_fad_fad->imposed_boundary[icell][iquad].resize(nstate);
                dg_state->pde_physics_rad_fad->imposed_boundary[icell][iquad].resize(nstate);
            }

        }

        this->pcout << "About to find surface solutions from converged solution..." << std::endl;
        get_surface_solution_for_BC<double>(dg, dg_state->pde_physics_double);
        get_surface_solution_for_BC<FadType>(dg, dg_state->pde_physics_fad);
        get_surface_solution_for_BC<RadType>(dg, dg_state->pde_physics_rad);
        get_surface_solution_for_BC<FadFadType>(dg, dg_state->pde_physics_fad_fad);
        get_surface_solution_for_BC<RadFadType>(dg, dg_state->pde_physics_rad_fad);
        this->pcout << "Done!" << std::endl;
        //std::abort();

        

    }

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
}

#if PHILIP_DIM>1
template class SpacetimeCartesianProblem <PHILIP_DIM,PHILIP_SPECIES,1>;
template class SpacetimeCartesianProblem <PHILIP_DIM,PHILIP_SPECIES,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

