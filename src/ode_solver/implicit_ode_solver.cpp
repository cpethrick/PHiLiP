#include "implicit_ode_solver.h"

//for jacobian free
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
ImplicitODESolver<dim,real,MeshType>::ImplicitODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
        : ODESolverBase<dim,real,MeshType>(dg_input)
        , solver(dg_input)
{
    // Jv = JacobianVectorProduct(this->dg);
}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::step_in_time (real dt, const bool/* pseudotime*/)
{
/*
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    this->current_time += dt;
    // Solve (M/dt - dRdW) dw = R
    // w = w + dw

    this->dg->system_matrix *= -1.0;

    if (pseudotime) {
        const double CFL = dt;
        this->dg->time_scaled_mass_matrices(CFL);
        this->dg->add_time_scaled_mass_matrices();
    } else {
        this->dg->add_mass_matrices(1.0/dt);
    }

    if ((this->ode_param.ode_output) == Parameters::OutputEnum::verbose &&
        (this->current_iteration%this->ode_param.print_iteration_modulo) == 0 ) {
        this->pcout << " Evaluating system update... " << std::endl;
    }

    solve_linear (
            this->dg->system_matrix,
            this->dg->right_hand_side,
            this->solution_update,
            this->ODESolverBase<dim,real,MeshType>::all_parameters->linear_solver_param);

    //linesearch();
    this->dg->solution.add(1.0, this->solution_update);
    //this->dg->assemble_residual ();

    this->update_norm = this->solution_update.l2_norm();
    ++(this->current_iteration);
*/

    /// JFNK Version
    // #include <deal.II/lac/solver_gmres.h>
    // #include <deal.II/lac/precondition.h>
    // System : M dw/dt = R ---> transform to dw/dt = IMM * R
    // Want to solve 0 = f(w) = dw/dt - IMM*R(w)
    // Implicit Euler:
    // 0 = (w-w_n)/dt - IMM*R(w) = f(w)
    // QUESTION: is it better to use mass matrix? Doesn't make a difference computationally, just changes the definition of f(w)
    // Therefore JFNK will be
    // J(w_k) * dw = -f(w)
    // where J(wk) is defined by a jacobian-vector product, Jv = 1/eps * (f(wk + eps*v)-f(wk))
    
/*    
    const int max_iter = 1000;
    const double gmres_tol = 1E-6;
    dealii::SolverControl solver_control(max_iter, 
                                         gmres_tol,
                                         false,     //log_history 
                                         true);     //log_result 
    //const dealii::AdditionalData additional_data(10);     //max_n_tmp_vectors
    dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>> solver(solver_control, 
            dealii::SolverGMRES<dealii::LinearAlgebra::distributed::Vector<double>>::AdditionalData(30)); //max_n_tmp_vectors = 10

    dealii::LinearAlgebra::distributed::Vector<double> dxk;
    dxk.reinit(this->dg->solution); //init to zero
    dealii::LinearAlgebra::distributed::Vector<double> soln_at_previous_step = this->dg->solution;
    const double newton_tol = 1E-7;
    double mag_dxk = 1;
    int k = 0;
    double epsilon_jacobian = 1.490116119384765625E-8; //sqrt(machine epsilon)
    this->solution_update = this->dg->solution; //initialize as previous solution
    Jv.reinit_for_next_timestep(dt, epsilon_jacobian, this->dg->solution);
    while ((mag_dxk > newton_tol)&&(k < 100)){
        
        Jv.reinit_for_next_Newton_iter(this->solution_update);

        dealii::LinearAlgebra::distributed::Vector<double> b;
        b.reinit(this->dg->solution);
        this->dg->solution = this->solution_update;
        this->dg->assemble_residual();
        this->dg->global_inverse_mass_matrix.vmult(this->dg->solution, this->dg->right_hand_side);
        b += this->solution_update;
        b -= soln_at_previous_step;
        b /= dt;
        b -= this->dg->solution;
        b *= -1.0;
        //b = -f(wk) = -((w_k - w_n)/dt - IMM * R(w_k));
        //
        solver.solve(Jv, dxk, b, dealii::PreconditionIdentity()); //GMRES solve of J(x_k)*dxk = f(x_k)
        
        this->solution_update += dxk; 
        //xk = xk + dxk;
        //k += 1;
        mag_dxk = dxk.l2_norm();
        k ++;


        this->pcout << "Newton iteration : " << k << " " << mag_dxk << std::endl;
    }
*/
    std::cout << "Calculating timestep number " << this->current_iteration << std::endl;
    solver.solve(dt, this->dg->solution);
    std::cout << "Finished calculating timestep number " << this->current_iteration << std::endl;
    this->dg->solution = solver.current_solution_estimate;

    this->current_time += dt;
    ++(this->current_iteration);

}

template <int dim, typename real, typename MeshType>
double ImplicitODESolver<dim,real,MeshType>::linesearch ()
{
    const auto old_solution = this->dg->solution;
    double step_length = 1.0;

    const double step_reduction = 0.5;
    const int maxline = 10;
    const double reduction_tolerance_1 = 1.0;
    const double reduction_tolerance_2 = 2.0;

    const double initial_residual = this->dg->get_residual_l2norm();

    this->dg->solution.add(step_length, this->solution_update);
    this->dg->assemble_residual ();
    double new_residual = this->dg->get_residual_l2norm();
    this->pcout << " Step length " << step_length << ". Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;

    int iline = 0;
    for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1; ++iline) {
        step_length = step_length * step_reduction;
        this->dg->solution = old_solution;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
    }
    if (iline == 0) this->CFL_factor *= 2.0;

    if (iline == maxline) {
        step_length = 1.0;
        this->pcout << " Line search failed. Will accept any valid residual less than " << reduction_tolerance_2 << " times the current " << initial_residual << "residual. " << std::endl;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }
    if (iline == maxline) {
        this->CFL_factor *= 0.5;
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        return 0.0;
    }

    if (iline == maxline) {
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_1 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
    }

    if (iline == maxline) {
        this->pcout << " Line search failed. Trying to step in the opposite direction. " << std::endl;
        step_length = -1.0;
        this->dg->solution.add(step_length, this->solution_update);
        this->dg->assemble_residual ();
        new_residual = this->dg->get_residual_l2norm();
        this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        for (iline = 0; iline < maxline && new_residual > initial_residual * reduction_tolerance_2 ; ++iline) {
            step_length = step_length * step_reduction;
            this->dg->solution = old_solution;
            this->dg->solution.add(step_length, this->solution_update);
            this->dg->assemble_residual ();
            new_residual = this->dg->get_residual_l2norm();
            this->pcout << " Step length " << step_length << " . Old residual: " << initial_residual << " New residual: " << new_residual << std::endl;
        }
        //std::abort();
    }
    if (iline == maxline) {
        this->pcout << " Reached maximum number of linesearches. Terminating... " << std::endl;
        this->pcout << " Resetting solution and reducing CFL_factor by : " << this->CFL_factor << std::endl;
        this->dg->solution = old_solution;
        this->CFL_factor *= 0.5;
    }

    return step_length;
}

template <int dim, typename real, typename MeshType>
void ImplicitODESolver<dim,real,MeshType>::allocate_ode_system ()
{
    this->pcout << "Allocating ODE system and evaluating mass matrix..." << std::endl;
    bool do_inverse_mass_matrix = false;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);
    do_inverse_mass_matrix = true;
    this->dg->evaluate_mass_matrices(do_inverse_mass_matrix);

    this->solution_update.reinit(this->dg->right_hand_side);
}

template class ImplicitODESolver<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ImplicitODESolver<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ImplicitODESolver<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // ODE namespace
} // PHiLiP namespace
