#ifndef __JACOBIAN_VECTOR_PRODUCT__
#define __JACOBIAN_VECTOR_PRODUCT__

#include "dg/dg.h"

namespace PHiLiP {
namespace ODE{

//template things
//UPDATE WITH #IF STUFF
template <int dim, typename real, typename MeshType>
class JacobianVectorProduct{
public:

    JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input
                          double dt_input,
                          double epsilon_input,
                          dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution_input);

    ~JacobianVectorProduct() {};

    void reinit_for_next_timestep(dt_input,
                epsilon_input,
                dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution_input);

    void reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate_input);

    // Application of matrix to vector src. Write result into dst.
    void vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src) const;
protected:

    /// pointer to dg
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// timestep size for implicit Euler step
    double dt;
    
    /// small number for finite difference
    double epsilon;
    
    /// solution at previous timestep
    dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution;
    
    /// Unsteady residual = dw/dt - R
    dealii::LinearAlgebra::distributed::Vector<double> compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> solution);

    //wtonKrylovRHS RHS;
    
    /// current estimate for the solution
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate;



}
/*
//probably overkill to make another class here, especially with the confusing signs
class NewtonKrylovRHS{
public:
    
    vector evaluate(u, uk, dt); // R* = dw/dt-R

    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

}
*/

}
}
