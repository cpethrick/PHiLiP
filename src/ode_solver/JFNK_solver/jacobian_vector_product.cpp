#include "jacobian_vector_product.h"

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JacobianVectorProduct<dim,real,MeshType>::JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
{
    two_derivative_flag = false; // Forward Euler step (or RK) by default
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> &current_solution_estimate_input)
{
    current_solution_estimate = current_solution_estimate_input; 
    current_solution_estimate_residual = compute_unsteady_residual(current_solution_estimate);
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_timestep(double dt_input,
                double epsilon_input,
                dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution_input)
{
    dt = dt_input;
    epsilon = epsilon_input;
    previous_step_solution = previous_step_solution_input;
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::compute_dg_residual(dealii::LinearAlgebra::distributed::Vector<double> &w) const
{
    dg->solution = w;
    dg->assemble_residual();
    
    dg->global_inverse_mass_matrix.vmult(dg->solution, dg->right_hand_side);//temp = IMM*RHS
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> &w, bool do_negate) const
{
    if (two_derivative_flag){
        return compute_two_derivative_unsteady_residual(w, do_negate);
    } else {
        compute_dg_residual(w);

        dg->solution*=-1;

        dg->solution.add(-1.0/dt, previous_step_solution);
        dg->solution.add(1.0/dt, w);

        if (do_negate) { 
            // this is included so that -R*(w) can be found with the same
            // function for the RHS of the Newton iterations 
            // and the Jacobian estimate
            // Recall  J(wk) * dwk = -R*(wk)
            dg->solution *= -1.0; 
        } 

        return dg->solution; // R* = (w-previous_step_solution)/dt - R(w)
    }
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_second_derivative(dealii::LinearAlgebra::distributed::Vector<double> &w) const
{
    dealii::LinearAlgebra::distributed::Vector<double> second_derivative(dg->solution);
    dealii::LinearAlgebra::distributed::Vector<double> first_derivative(dg->solution);

    compute_dg_residual(w);
    first_derivative = dg->solution;

    //using second_derivative as temp vector
    second_derivative = w;
    second_derivative.add(epsilon, first_derivative);
    compute_dg_residual(second_derivative);
    second_derivative = dg->solution;

    //Changing to centered difference
    first_derivative *= -1.0*epsilon;
    first_derivative.add(1.0, w);
    compute_dg_residual(first_derivative);
    first_derivative = this->dg->solution;

    second_derivative.add(-1.0, first_derivative);
    second_derivative *= 1.0/2.0/epsilon;
    //second_derivative *= 1.0/epsilon;

    return second_derivative; // d2d/dt2 = Rdot = R' * R = 1/eps * ( R(w + eps*R(w) ) - R(w))
/*
    //AD version
    compute_dg_residual(w);
    dealii::LinearAlgebra::distributed::Vector<double> first_derivative = this->dg->solution;
    bool do_compute_dRdW = true;
    this->dg->solution = w;
    this->dg->assemble_residual(do_compute_dRdW);
    dealii::LinearAlgebra::distributed::Vector<double> second_derivative(this->dg->solution);
    this->dg->system_matrix.vmult(second_derivative, first_derivative); //second = dRdW * R
    return second_derivative;
*/        
}

template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_two_derivative_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> &w, 
        bool do_negate) const
{
    dealii::LinearAlgebra::distributed::Vector<double> second_derivative = compute_second_derivative(w);
    compute_dg_residual(w);
    dealii::LinearAlgebra::distributed::Vector<double> first_derivative = dg->solution; 

    
    dealii::LinearAlgebra::distributed::Vector<double> unsteady_residual = w;

    unsteady_residual.add(-1.0, previous_step_solution);
    unsteady_residual *= 1.0/dt/dt;

    first_derivative *= -a/dt;
    second_derivative *= -a_dot;

    unsteady_residual.add(1.0, first_derivative);
    unsteady_residual.add(1.0, second_derivative);

    if (do_negate) { 
        // this is included so that -R*(w) can be found with the same
        // function for the RHS of the Newton iterations 
        // and the Jacobian estimate
        // Recall  J(wk) * dwk = -R*(wk)
        unsteady_residual *= -1.0; 
    } 

    return unsteady_residual;
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src) const
{
    dst = src;
    dst *= epsilon; 
    dst += current_solution_estimate;
    dst = compute_unsteady_residual(dst);
    dst -= current_solution_estimate_residual;
    dst *= 1.0/epsilon; // dst = 1/epsilon * (R*(current_soln_estimate + epsilon*src) - R*(curr_sol_est))

}

template class JacobianVectorProduct<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
