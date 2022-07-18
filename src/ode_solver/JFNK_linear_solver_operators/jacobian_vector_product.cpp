#include "jacobian_vector_product.h"

namespace PHiLiP{
namespace ODE{

template <int dim, typename real, typename MeshType>
JacobianVectorProduct<dim,real,MeshType>::JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input
                          double dt_input,
                          double epsilon_input,
                          dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution_input);
    , dg(dg_input)
    , dt(dt_input)
    , epsilon(epsilon_input)
    , previous_step_solution(previous_step_solution_input)
{
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate_input);
{
    current_solution_estimate = current_solution_estimate_input; 
}


dealii::LinearAlgebra::distributed::Vector<double> JacobianVectorProduct<dim,real,MeshType>::compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> w)
{
    dg->solution = w;
    dg->assemble_residual();

    dg->global_inverse_mass_matrix.vmult(dg->solution, dg->right_hand_side);//solution = IMM*RHS

    return (w - current_solution_estimate)/dt + dg->solution; 
}

template <int dim, typename real, typename MeshType>
void JacobianVectorProduct<dim,real,MeshType>::vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src)
{
    dst = 1/epsilon * (compute_unsteady_residual(src* epsilon + current_solution_estimate)-compute_unsteady_residual(current_solution_estimate)); 
}
}
}
