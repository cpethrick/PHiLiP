#include "jacobian_vector_product.h"

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JacobianVectorProduct<dim,real,MeshType>::JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
{
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate_input)
{
    current_solution_estimate = current_solution_estimate_input; 
    current_solution_estimate_residual = compute_unsteady_residual(current_solution_estimate);
}


template <int dim, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> w) const
{
    dg->solution = w;
    dg->assemble_residual();
    
    //TO DO: see if GMM * du/dt + RHS works
    dg->global_inverse_mass_matrix.vmult(dg->solution, dg->right_hand_side);//solution = IMM*RHS

    dg->solution *=-1;

    dg->solution.add(-1.0/dt, previous_step_solution);
    dg->solution.add(1.0/dt, w);

    return dg->solution; // R* = (w-previous_step_solution)/dt - IMM*RHS
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src) const
{
    //dst = compute_unsteady_residual(src*epsilon);
    dealii::LinearAlgebra::distributed::Vector<double> temp = src;
    temp *= epsilon;
    temp += current_solution_estimate;
    temp = compute_unsteady_residual(temp);
    temp -= current_solution_estimate_residual;
    temp *= 1/epsilon;
    dst = temp; // dst = 1/epsilon * (R*(current_soln_estimate + epsilon*src) - R*(curr_sol_est))

}

template class JacobianVectorProduct<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class JacobianVectorProduct<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


}
}
