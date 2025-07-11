#include "runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {
    template <int dim, int nstate, typename real>
real compute_pressure ( const std::array<real,nstate> &conservative_soln )
{
    const real density = conservative_soln[0];

    const real tot_energy  = conservative_soln[nstate-1];

    dealii::Tensor<1,dim,real> vel;
    for (int d=0; d<dim; ++d) { vel[d] = conservative_soln[1+d]/density; }

    real vel2 = 0.0;
    for (int d=0; d<dim; d++) {
        vel2 = vel2 + vel[d]*vel[d];
    }
    real pressure = 0.4*(tot_energy - 0.5*density*vel2);

    return pressure;
}
template <int dim, int nstate, typename real>
real compute_entropy (const real density, const real pressure)
{
    // copy density and pressure such that the check will not modify originals
    if (density>0 && pressure>0) {
        real entropy = pressure * pow(density, -1.4);
        entropy = log(entropy);
        return entropy;
    } else {
        return 1E16;
    }

}
template <int dim, int nstate, typename real>
real compute_numerical_entropy_function ( const std::array<real,nstate> &conservative_soln )
{
    const real pressure = compute_pressure<dim,nstate,real>(conservative_soln);
    const real density = conservative_soln[0];

    const real entropy = compute_entropy<dim,nstate,real>(density, pressure);

    const real numerical_entropy_function = - density * entropy;

    return numerical_entropy_function;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
double RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::compute_current_integrated_numerical_entropy(
        const std::shared_ptr <DGBase<dim, double,MeshType>> dg
        ) const
{
    const double poly_degree = dg->all_parameters->flow_solver_param.poly_degree;
    const int nstate = 5; // hard code for Euler/NS

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
            integrand_numerical_entropy_function = compute_numerical_entropy_function<dim,nstate,double>(soln_state);
            integral_numerical_entropy_function += integrand_numerical_entropy_function * quad_weights[iquad] * metric_oper.det_Jac_vol[iquad];
        }
    }
    // update integrated quantities and return
    const double mpi_integrated_numerical_entropy = dealii::Utilities::MPI::sum(integral_numerical_entropy_function, this->mpi_communicator);

    return mpi_integrated_numerical_entropy;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
RungeKuttaODESolver<dim,real,n_rk_stages, MeshType>::RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
        std::shared_ptr<RKTableauButcherBase<dim,real,MeshType>> rk_tableau_input,
        std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input)
        : RungeKuttaBase<dim,real,n_rk_stages,MeshType>(dg_input, RRK_object_input)
        , butcher_tableau(rk_tableau_input)
        , epsilon{1.0, 1.0, 1.0} 
        , atol(this->ode_param.atol)
        , rtol(this->ode_param.rtol)
        , beta1(this->ode_param.beta1)
        , beta2(this->ode_param.beta2)
        , beta3(this->ode_param.beta3)
{}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_solution (int istage, real dt, const bool pseudotime)
{
    this->rk_stage[istage]=0.0; //resets all entries to zero
    
    for (int j = 0; j < istage; ++j){
        if (this->butcher_tableau->get_a(istage,j) != 0){
            this->rk_stage[istage].add(this->butcher_tableau->get_a(istage,j), this->rk_stage[j]);
        }
    } //sum(a_ij *k_j), explicit part

    
    if(pseudotime) {
        const double CFL = dt;
        this->dg->time_scale_solution_update(this->rk_stage[istage], CFL);
    }else {
        this->rk_stage[istage]*=dt;
    }//dt * sum(a_ij * k_j)
    
    this->rk_stage[istage].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
    
    //implicit solve if there is a nonzero diagonal element
    if (!this->butcher_tableau_aii_is_zero[istage]){
        /* // AD version - keeping in comments as it may be useful for future testing
        // Solve (M/dt - dRdW) / a_ii * dw = R
        // w = w + dw
        // Note - need to have assembled residual using this->dg->assemble_residual(true);
        //        and have mass matrix assembled, and include linear_solver
        dealii::LinearAlgebra::distributed::Vector<double> temp_u(this->dg->solution.size());

        this->dg->system_matrix *= -1.0/butcher_tableau_a[istage][istage]; //system_matrix = -1/a_ii*dRdW
        this->dg->add_mass_matrices(1.0/butcher_tableau_a[istage][istage]/dt); //system_matrix = -1/a_ii*dRdW + M/dt/a_ii = A

        solve_linear ( //Solve Ax=b using Aztec00 gmres
                    this->dg->system_matrix, //A = -1/a_ii*dRdW + M/dt/a_ii
                    this->dg->right_hand_side, //b = R
                    temp_u, // result,  x = dw
                    this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

        this->rk_stage[istage].add(1.0, temp_u);
        */

        //JFNK version
        this->solver.solve(dt*this->butcher_tableau->get_a(istage,istage), this->rk_stage[istage]);
        this->rk_stage[istage] = this->solver.current_solution_estimate;

    } // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(istage) <implicit>
    
    // If using the entropy formulation of RRK, solutions must be stored.
    // Call store_stage_solutions before overwriting rk_stage with the derivative.
    this->relaxation_runge_kutta->store_stage_solutions(istage, this->rk_stage[istage]);

    this->dg->solution = this->rk_stage[istage];

}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::calculate_stage_derivative (int istage, real dt)
{
     //set the DG current time for unsteady source terms
    this->dg->set_current_time(this->current_time + this->butcher_tableau->get_c(istage)*dt);
    
    //solve the system's right hand side
    this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(istage)))

    if(this->all_parameters->use_inverse_mass_on_the_fly){
        this->dg->apply_inverse_global_mass_matrix(this->dg->right_hand_side, this->rk_stage[istage]); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    } else{
        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[istage], this->dg->right_hand_side); //rk_stage[istage] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
    }
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::sum_stages (real dt, const bool pseudotime)
{
    //assemble solution from stages
    for (int istage = 0; istage < n_rk_stages; ++istage){
        if (pseudotime){
            const double CFL = this->butcher_tableau->get_b(istage) * dt;
            this->dg->time_scale_solution_update(this->rk_stage[istage], CFL);
            this->solution_update.add(1.0, this->rk_stage[istage]);
        } else {
            this->solution_update.add(dt* this->butcher_tableau->get_b(istage),this->rk_stage[istage]);
        }
    }
}


template<int dim, typename real, int n_rk_stages, typename MeshType>
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::apply_limiter ()
{
    // Apply limiter at every RK stage
    if (this->limiter) {
        this->limiter->limit(this->dg->solution,
            this->dg->dof_handler,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->dg->high_order_grid->fe_system.tensor_degree(),
            this->dg->max_degree,
            this->dg->oneD_fe_collection_1state,
            this->dg->oneD_quadrature_collection);
    }
}

template<int dim, typename real, int n_rk_stages, typename MeshType>
real RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::adjust_time_step (real dt)
{

    // Calculates relaxation parameter and modify the time step size as dt*=relaxation_parameter.
    // if not using RRK, the relaxation parameter will be set to 1, such that dt is not modified.
    //this->relaxation_parameter_RRK_solver = this->relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);

    // hijack this function to instead return the entropy change estimate.
    const double entropy_change_est = this->relaxation_runge_kutta->update_relaxation_parameter(dt, this->dg, this->rk_stage, this->solution_update);
    //this->pcout << "Relaxation parameter from RRK: " << this->relaxation_parameter_RRK_solver << std::endl;
    //dt *= this->relaxation_parameter_RRK_solver;
    //this->modified_time_step = dt;

    this->dg->solution = this->solution_update; // at this point, the solution_update holds u_n
    double numerical_entropy_un = compute_current_integrated_numerical_entropy(this->dg);    

    if (this->current_time == 0.0){
        initial_entropy = numerical_entropy_un;
    }

    this->pcout << numerical_entropy_un << std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> u_np1_temp(this->solution_update);
    u_np1_temp = this->solution_update;
    for (int istage = 0; istage < n_rk_stages; ++istage){
        u_np1_temp.add(dt* this->butcher_tableau->get_b(istage),this->rk_stage[istage]);
    }
    this->dg->solution = u_np1_temp;
    double numerical_entropy_unp1 = compute_current_integrated_numerical_entropy(this->dg);
    //this->pcout << numerical_entropy_unp1 << std::endl;
    w = pow( (numerical_entropy_un-numerical_entropy_unp1+entropy_change_est) / (atol + rtol * std::max(std::abs(numerical_entropy_un),std::abs(numerical_entropy_unp1))), 2);
    //w = pow( (numerical_entropy_un-initial_entropy) / (atol + rtol * std::max(std::abs(numerical_entropy_un),std::abs(initial_entropy))), 2);
    //this->pcout << (numerical_entropy_un-numerical_entropy_unp1) << " " << 
    //                (numerical_entropy_un-numerical_entropy_unp1) / (atol + rtol * std::max(std::abs(numerical_entropy_un),std::abs(numerical_entropy_unp1)))
    //                << " " << (atol + rtol * std::max(std::abs(numerical_entropy_un),std::abs(numerical_entropy_unp1))) << " " <<  
    //    w << std::endl;
    w = pow(w,  0.5);
    epsilon[2] = epsilon[1];
    epsilon[1] = epsilon[0];
    epsilon[0] = 1.0 / w;
    double rk_order = this->ode_param.rk_order;
    double adjustment_factor = pow(epsilon[0], 1.0 * beta1/rk_order) * pow(epsilon[1], 1.0 * beta2/rk_order) * pow(epsilon[2], 1.0 * beta3/rk_order);
    //this->pcout << epsilon[0] << " " << epsilon[1] << " "  << epsilon[2] << std::endl;
    //this->pcout << "Adjustment factor from PID: " <<  adjustment_factor <<  " " << std::endl << std::endl;
    adjustment_factor = (adjustment_factor-1.0) /-266 + 1.0;
    //this->pcout << "Scale by 100: " << adjustment_factor << std::endl;

    if (isinf(adjustment_factor)){
        adjustment_factor = 1.0;
    } 

    dt *= adjustment_factor;
    this->modified_time_step = dt;
    return dt;
}

template <int dim, typename real, int n_rk_stages, typename MeshType> 
void RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>::allocate_runge_kutta_system ()
{

    this->butcher_tableau->set_tableau();
    
    this->butcher_tableau_aii_is_zero.resize(n_rk_stages);
    std::fill(this->butcher_tableau_aii_is_zero.begin(),
              this->butcher_tableau_aii_is_zero.end(),
              false); 
    for (int istage=0; istage<n_rk_stages; ++istage) {
        if (this->butcher_tableau->get_a(istage,istage)==0.0)     this->butcher_tableau_aii_is_zero[istage] = true;
    
    }
    if(this->all_parameters->use_inverse_mass_on_the_fly == false) {
        this->pcout << " evaluating inverse mass matrix..." << std::flush;
        this->dg->evaluate_mass_matrices(true); // creates and stores global inverse mass matrix
        //RRK needs both mass matrix and inverse mass matrix
        using ODEEnum = Parameters::ODESolverParam::ODESolverEnum;
        ODEEnum ode_type = this->ode_param.ode_solver_type;
        if (ode_type == ODEEnum::rrk_explicit_solver){
            this->dg->evaluate_mass_matrices(false); // creates and stores global mass matrix
        }
    }
}

template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RungeKuttaODESolver<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RungeKuttaODESolver<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
