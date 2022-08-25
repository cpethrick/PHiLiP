#include "two_derivative_rk_ode_solver.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
TwoDerivativeRKODESolver<dim,real,MeshType>::TwoDerivativeRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : RungeKuttaODESolver<dim,real,MeshType>(dg_input)
        , use_relaxation(false)
        , do_write_root_function(false)
{
    if (do_write_root_function){
        std::ofstream root_function_file("root_function_evolution.txt", std::ios::out);
        root_function_file << std::fixed;
        if (root_function_file.is_open()){
            root_function_file << "Time, ";
            for (int i = 0; i < 21; ++i){
                root_function_file << "R" << i;
                if (i != 20) root_function_file << ", ";
            }
            root_function_file << std::endl;
            root_function_file << "-1, "; //first row will have x coords
            std::array<double, 21> x_root_function;
            for (int i = 0; i < 21; ++i){
                x_root_function[i] = -0.5 + 0.1 * i;
                root_function_file << x_root_function[i];
                if (i != 20) root_function_file << ", ";
            }
            root_function_file << std::endl;
            root_function_file.close();
        } else{
            this->pcout << "Couldn't open file root_function_evolution.txt, aborting..." << std::endl;
            std::abort();
        }
    }
}

template <int dim, typename real, typename MeshType>
void TwoDerivativeRKODESolver<dim,real,MeshType>::step_in_time (real dt, const bool pseudotime)
{  
    
    if (pseudotime) {
        this->pcout << "This ODE solver does not implement pseudotime. Aborting..." << std::endl;
        std::abort();
    }

    this->solution_update = this->dg->solution; //storing u_n
    
    //calculating stages **Note that rk_stage[i] stores the RHS at a partial time-step (not solution u)
    for (int i = 0; i < this->number_of_stages; ++i){

        this->rk_stage[i]=0.0; //resets all entries to zero
        
        for (int j = 0; j < i; ++j){
            if (this->butcher_tableau_a[i][j] != 0){
                this->rk_stage[i].add(this->butcher_tableau_a[i][j]*dt, this->rk_stage[j]);
            }
            if (this->butcher_tableau_a_dot[i][j] != 0){
                this->rk_stage[i].add(this->butcher_tableau_a_dot[i][j]*dt*dt, this->rk_stage_2nd_deriv[j]);
            }
        } //sum(a_ij *k_j), explicit part

        //this->rk_stage[i]*=dt;
        this->rk_stage[i].add(1.0,this->solution_update); //u_n + dt * sum(a_ij * k_j)
       
        //implicit solve for diagonal element
        if (this->butcher_tableau_a[i][i] != 0 || this->butcher_tableau_a_dot[i][i] != 0){
            
            //this->dg->solution = this->rk_stage[i];
            //this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))
            //JFNK version
            this->solver.solve(dt, this->rk_stage[i], this->butcher_tableau_a[i][i], this->butcher_tableau_a_dot[i][i]);
            this->rk_stage[i] = this->solver.current_solution_estimate;
            // u_n + dt * sum(a_ij * k_j) <explicit> + dt * a_ii * u^(i) <implicit>
            
        }
        
        this->rk_stage_2nd_deriv[i] = this->solver.jacobian_vector_product.compute_second_derivative(this->rk_stage[i]);
        this->dg->solution = this->rk_stage[i];
        this->dg->assemble_residual(); //RHS : du/dt = RHS = F(u_n + dt* sum(a_ij*k_j) + dt * a_ii * u^(i)))
        this->dg->global_inverse_mass_matrix.vmult(this->rk_stage[i], this->dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))

    }

    modify_time_step(dt);

    //assemble solution from stages
    for (int i = 0; i < this->number_of_stages; ++i){
        this->solution_update.add(dt* this->butcher_tableau_b[i],this->rk_stage[i]);
        this->solution_update.add(dt*dt*this->butcher_tableau_b_dot[i], this->rk_stage_2nd_deriv[i]);
    }
    this->dg->solution = this->solution_update; // u_np1 = u_n + dt* sum(k_i * b_i)

    ++(this->current_iteration);
    this->current_time += dt;
}

template <int dim, typename real, typename MeshType>
void TwoDerivativeRKODESolver<dim,real,MeshType>::modify_time_step(real &dt)
{
    if (use_relaxation){
        real relaxation_parameter = compute_relaxation_parameter_explicit(dt);
        dt *= relaxation_parameter;
    } else (void) dt;
}

template <int dim, typename real, typename MeshType>
real TwoDerivativeRKODESolver<dim,real,MeshType>::compute_relaxation_parameter_explicit(real &dt) const
{
    return dt;
}

template <int dim, typename real, typename MeshType>
void TwoDerivativeRKODESolver<dim,real,MeshType>::write_root_function(real &dt) const
{
    std::ofstream root_function_file("root_function_evolution.txt", std::ios::out | std::ios::app);
    if (root_function_file.is_open()){
        
        root_function_file << this->current_time << ", ";
        root_function_file << std::fixed;
        root_function_file << std::setprecision(16);

        std::array<double, 21> x_root_function;
        std::array<double, 21> root_function_values;
        for (int i = 0; i < 21; ++i){
            x_root_function[i] = -0.5 + 0.1 * i;
            //Update root function calculation
            //root_function_values[i] = x_root_function[i] * x_root_function[i] * dt * dt * denominator - x_root_function[i] * dt * dt * numerator;
            root_function_file << root_function_values[i];
            if (i != 20) root_function_file << ", ";
        }
        
        root_function_file << std::endl;

        root_function_file.close();
    } else{
        this->pcout << "Couldn't open file root_function_evolution.txt, aborting..." << std::endl;
        std::abort();
    }

    (void) dt;
}

template <int dim, typename real, typename MeshType>
real TwoDerivativeRKODESolver<dim,real,MeshType>::compute_inner_product (
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
        const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
        ) const
{
    // Intention is to point to physics (mimic structure in flow_solver_cases/periodic_turbulence.cpp for converting to solution for general nodes) 
    // For now, only energy on collocated nodes is implemented.
    
    real inner_product = 0;
    for (unsigned int i = 0; i < this->dg->solution.size(); ++i) {
        inner_product += 1./(this->dg->global_inverse_mass_matrix.diag_element(i))
                         * stage_i[i] * stage_j[i];
    }
    return inner_product;
}

template <int dim, typename real, typename MeshType>
void TwoDerivativeRKODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating inverse mass matrix..." << std::endl;
    const bool do_inverse_mass_matrix = true;
    this->solution_update.reinit(this->dg->right_hand_side);
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    // Assigning butcher tableau
    if (this->rk_order == 1){
        this->pcout << "Initializing to order 1" << std::endl;
        this->number_of_stages = 1;
        this->butcher_tableau_a.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b.reinit(this->number_of_stages);
        this->butcher_tableau_a_dot.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b_dot.reinit(this->number_of_stages);
        // Implicit Euler method (note b = a)
        // Implemented for verification
        const double butcher_tableau_a_values[1] = {1.0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        this->butcher_tableau_b.fill(butcher_tableau_a_values);
        const double butcher_tableau_a_dot_values[1] = {0.0};
        this->butcher_tableau_a_dot.fill(butcher_tableau_a_dot_values);
        this->butcher_tableau_b_dot.fill(butcher_tableau_a_dot_values);
    }else if (this->rk_order == 2){
        this->pcout << "Initializing to order 2" << std::endl;
        this->number_of_stages = 1;
        this->butcher_tableau_a.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b.reinit(this->number_of_stages);
        this->butcher_tableau_a_dot.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b_dot.reinit(this->number_of_stages);
        // Implicit Taylor series method (note b = a)
        const double butcher_tableau_a_values[1] = {1.0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        this->butcher_tableau_b.fill(butcher_tableau_a_values);
        const double butcher_tableau_a_dot_values[1] = {-1.0/2.0};
        this->butcher_tableau_a_dot.fill(butcher_tableau_a_dot_values);
        this->butcher_tableau_b_dot.fill(butcher_tableau_a_dot_values);
    }else if (this->rk_order == 3){
        this->pcout << "Initializing to order 3" << std::endl;
        this->number_of_stages = 2;
        this->butcher_tableau_a.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b.reinit(this->number_of_stages);
        this->butcher_tableau_a_dot.reinit(this->number_of_stages,this->number_of_stages);
        this->butcher_tableau_b_dot.reinit(this->number_of_stages);
        // butcher tableau from Gottlieb paper 
        const double butcher_tableau_a_values[4] = {0,0,0,1.0};
        this->butcher_tableau_a.fill(butcher_tableau_a_values);
        const double butcher_tableau_b_values[2] = {0,1.0};
        this->butcher_tableau_b.fill(butcher_tableau_b_values);
        const double butcher_tableau_a_dot_values[4] = {-1.0/6.0, 0, -1.0/6.0, -1.0/3.0};
        this->butcher_tableau_a_dot.fill(butcher_tableau_a_dot_values);
        const double butcher_tableau_b_dot_values[2] = {-1.0/6.0, -1.0/3.0};
        this->butcher_tableau_b_dot.fill(butcher_tableau_b_dot_values);
    } else{
        this->pcout << "Invalid RK order" << std::endl;
        std::abort();
    }
    
    this->rk_stage.resize(this->number_of_stages);
    this->rk_stage_2nd_deriv.resize(this->number_of_stages);
    for (int i=0; i<this->number_of_stages; i++) {
        this->rk_stage[i].reinit(this->dg->solution);
        this->rk_stage_2nd_deriv[i].reinit(this->dg->solution);
    }

}


template class TwoDerivativeRKODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class TwoDerivativeRKODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
//currently only tested in 1D - commenting out higher dimensions
//#if PHILIP_DIM != 1
//template class TwoDerivativeRKODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
//#endif

} // ODESolver namespace
} // PHiLiP namespace
