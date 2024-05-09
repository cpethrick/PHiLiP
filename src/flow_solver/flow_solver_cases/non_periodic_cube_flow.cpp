#include "non_periodic_cube_flow.h"
#include "mesh/grids/non_periodic_cube.h"
#include <deal.II/grid/grid_generator.h>
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : CubeFlow_UniformGrid<dim, nstate>(parameters_input)
    , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    //create the Physics object
    this->pde_physics = std::dynamic_pointer_cast<Physics::PhysicsBase<dim,nstate,double>>(
                Physics::PhysicsFactory<dim,nstate,double>::create_Physics(parameters_input));
    
   // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = Parameters::AllParameters::PartialDifferentialEquation::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );

    bool use_number_mesh_refinements = false;
    if(this->all_param.flow_solver_param.number_of_mesh_refinements>0)
        use_number_mesh_refinements = true;
    
    const unsigned int number_of_refinements = use_number_mesh_refinements ? this->all_param.flow_solver_param.number_of_mesh_refinements 
                                                                           : log2(this->all_param.flow_solver_param.number_of_grid_elements_per_dimension);

    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    int left_boundary_id = 9999;
    using flow_case_enum = Parameters::FlowSolverParam::FlowCaseType;
    flow_case_enum flow_case_type = this->all_param.flow_solver_param.flow_case_type;

    if (flow_case_type == flow_case_enum::sod_shock_tube
        || flow_case_type == flow_case_enum::leblanc_shock_tube) {
        left_boundary_id = 1001;
    } else if (flow_case_type == flow_case_enum::shu_osher_problem) {
        left_boundary_id = 1004;
    }


    Grids::non_periodic_cube<dim>(*grid, domain_left, domain_right, colorize, left_boundary_id);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
}

template<int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::check_positivity_density(DGBase<dim, double>& dg)
{
    //create 1D solution polynomial basis functions and corresponding projection operator
    //to interpolate the solution to the quadrature nodes, and to project it back to the
    //modal coefficients.
    const unsigned int init_grid_degree = dg.max_grid_degree;
    const unsigned int poly_degree = this->all_param.flow_solver_param.poly_degree;
    //Constructor for the operators
    OPERATOR::basis_functions<dim, 2 * dim, double> soln_basis(1, poly_degree, init_grid_degree);
    OPERATOR::vol_projection_operator<dim, 2 * dim, double> soln_basis_projection_oper(1, dg.max_degree, init_grid_degree);


    // Build the oneD operator to perform interpolation/projection
    soln_basis.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);
    soln_basis_projection_oper.build_1D_volume_operator(dg.oneD_fe_collection_1state[poly_degree], dg.oneD_quadrature_collection[poly_degree]);

    for (auto soln_cell = dg.dof_handler.begin_active(); soln_cell != dg.dof_handler.end(); ++soln_cell) {
        if (!soln_cell->is_locally_owned()) continue;


        std::vector<dealii::types::global_dof_index> current_dofs_indices;
        // Current reference element related to this physical cell
        const int i_fele = soln_cell->active_fe_index();
        const dealii::FESystem<dim, dim>& current_fe_ref = dg.fe_collection[i_fele];
        const int poly_degree = current_fe_ref.tensor_degree();

        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        // Obtain the mapping from local dof indices to global dof indices
        current_dofs_indices.resize(n_dofs_curr_cell);
        soln_cell->get_dof_indices(current_dofs_indices);

        // Extract the local solution dofs in the cell from the global solution dofs
        std::array<std::vector<double>, nstate> soln_coeff;
        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;

        for (unsigned int istate = 0; istate < nstate; ++istate) {
            soln_coeff[istate].resize(n_shape_fns);
        }

        // Allocate solution dofs and set local max and min
        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = dg.fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg.fe_collection[poly_degree].system_to_component_index(idof).second;
            soln_coeff[istate][ishape] = dg.solution[current_dofs_indices[idof]];
        }

        const unsigned int n_quad_pts = dg.volume_quadrature_collection[poly_degree].size();

        std::array<std::vector<double>, nstate> soln_at_q;

        // Interpolate solution dofs to quadrature pts.
        for (int istate = 0; istate < nstate; istate++) {
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate], soln_basis.oneD_vol_operator);
        }

        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad) {
            // Verify that positivity of density is preserved
            if (soln_at_q[0][iquad] < 0 || (isnan(soln_at_q[0][iquad])) ) {
                std::cout << "Error: Density is negative or NaN - Aborting... " << std::endl << std::flush;
                std::abort();
            }
        }
    }
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::update_numerical_entropy(
        const double FR_entropy_contribution_RRK_solver,
        const unsigned int current_iteration,
        const std::shared_ptr <DGBase<dim, double>> dg)
{

    const double current_numerical_entropy = this->compute_current_integrated_numerical_entropy(dg);

    if (current_iteration==0) {
        this->previous_numerical_entropy = current_numerical_entropy;
        this->initial_numerical_entropy_abs = abs(current_numerical_entropy);
    }

    const double current_numerical_entropy_change_FRcorrected = (current_numerical_entropy - this->previous_numerical_entropy + FR_entropy_contribution_RRK_solver)/this->initial_numerical_entropy_abs;
    this->previous_numerical_entropy = current_numerical_entropy;
    this->cumulative_numerical_entropy_change_FRcorrected+=current_numerical_entropy_change_FRcorrected;

}
template<int dim, int nstate>
double NonPeriodicCubeFlow<dim, nstate>::compute_current_integrated_numerical_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    const double poly_degree = this->all_param.flow_solver_param.poly_degree;

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;

    OPERATOR::vol_projection_operator<dim,2*dim,double> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    // Construct the basis functions and mapping shape functions.
    OPERATOR::basis_functions<dim,2*dim,double> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::mapping_shape_functions<dim,2*dim,double> mapping_basis(1, poly_degree, dg->max_grid_degree);
    mapping_basis.build_1D_shape_functions_at_grid_nodes(dg->high_order_grid->oneD_fe_system, dg->high_order_grid->oneD_grid_nodes);
    mapping_basis.build_1D_shape_functions_at_flux_nodes(dg->high_order_grid->oneD_fe_system, dg->oneD_quadrature_collection[poly_degree], dg->oneD_face_quadrature);

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);
    
    double integrand_numerical_entropy_function=0;
    double integral_numerical_entropy_function=0;
    const std::vector<double> &quad_weights = dg->volume_quadrature_collection[poly_degree].get_weights();

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    // Changed for loop to update metric_cell.
    for (auto cell = dg->dof_handler.begin_active(); cell!= dg->dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);
        
        // We first need to extract the mapping support points (grid nodes) from high_order_grid.
        const dealii::FESystem<dim> &fe_metric = dg->high_order_grid->fe_system;
        const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
        const unsigned int n_grid_nodes  = n_metric_dofs / dim;
        std::vector<dealii::types::global_dof_index> metric_dof_indices(n_metric_dofs);
        metric_cell->get_dof_indices (metric_dof_indices);
        std::array<std::vector<double>,dim> mapping_support_points;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points[idim].resize(n_grid_nodes);
        }
        // Get the mapping support points (physical grid nodes) from high_order_grid.
        // Store it in such a way we can use sum-factorization on it with the mapping basis functions.
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(dg->max_grid_degree);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const double val = (dg->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points[istate][igrid_node] = val; 
        }
        // Construct the metric operators.
        OPERATOR::metric_operators<double, dim, 2*dim> metric_oper(nstate, poly_degree, dg->max_grid_degree, false, false);
        // Build the metric terms to compute the gradient and volume node positions.
        // This functions will compute the determinant of the metric Jacobian and metric cofactor matrix. 
        // If flags store_vol_flux_nodes and store_surf_flux_nodes set as true it will also compute the physical quadrature positions.
        metric_oper.build_volume_metric_operators(
            n_quad_pts, n_grid_nodes,
            mapping_support_points,
            mapping_basis,
            dg->all_parameters->use_invariant_curl_form);

        // Fetch the modal soln coefficients
        // We immediately separate them by state as to be able to use sum-factorization
        // in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
        // mult would sum the states at the quadrature point.
        // That is why the basis functions are based off the 1state oneD fe_collection.
        std::array<std::vector<double>,nstate> soln_coeff;
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
            }
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }
        // Interpolate each state to the quadrature points using sum-factorization
        // with the basis functions in each reference direction.
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }

        // Loop over quadrature nodes, compute quantities to be integrated, and integrate them.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::array<double,nstate> soln_state;
            // Extract solution in a way that the physics ca n use them.
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
#if PHILIP_DIM==1
            integrand_numerical_entropy_function = this->navier_stokes_physics->compute_numerical_entropy_function(soln_state);
#endif
            integral_numerical_entropy_function += integrand_numerical_entropy_function * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    // update integrated quantities and return
    const double mpi_integrated_numerical_entropy = dealii::Utilities::MPI::sum(integral_numerical_entropy_function, this->mpi_communicator);

    return mpi_integrated_numerical_entropy;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver, 
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    //unpack current iteration and current time from ode solver
    const unsigned int current_iteration = ode_solver->current_iteration;
    const double current_time = ode_solver->current_time;

    this->check_positivity_density(*dg);
    if (this->mpi_rank == 0) {

        unsteady_data_table->add_value("iteration", current_iteration);
        // Add values to data table
        this->add_value_to_data_table(current_time, "time", unsteady_data_table);

        this->update_numerical_entropy(ode_solver->FR_entropy_contribution_RRK_solver,current_iteration, dg);
        this->add_value_to_data_table(this->cumulative_numerical_entropy_change_FRcorrected,"numerical_entropy_scaled_cumulative",unsteady_data_table);

        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    if (current_iteration % this->all_param.ode_solver_param.print_iteration_modulo == 0) {
        // Print to console
        this->pcout << "    Iter: " << current_iteration
            << "    Time: " << current_time
            << "    Num. Entropy: " << this->cumulative_numerical_entropy_change_FRcorrected;

        this->pcout << std::endl;
    }
}


#if PHILIP_DIM==2
    template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;
#else
    template class NonPeriodicCubeFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace
