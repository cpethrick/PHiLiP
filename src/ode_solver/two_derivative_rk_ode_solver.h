#ifndef __TWO_DERIVATIVE_RK_ODESOLVER__
#define __TWO_DERIVATIVE_RK_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "runge_kutta_ode_solver.h"

namespace PHiLiP {
namespace ODE {

/// Two-derivative Runge-Kutta ODE Solver derived from RungeKuttaODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class TwoDerivativeRKODESolver: public RungeKuttaODESolver <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    TwoDerivativeRKODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    /// Destructor
    ~TwoDerivativeRKODESolver() {};
    
    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime) override;

    /// Function to allocate the ODE system
    void allocate_ode_system () override;

    /// Relaxation Runge-Kutta parameter gamma^n
    /** See:  Ketcheson 2019, "Relaxation Runge--Kutta methods: Conservation and stability for inner-product norms"
     *       Ranocha 2020, "Relaxation Runge--Kutta Methods: Fully Discrete Explicit Entropy-Stable Schemes for the Compressible Euler and Navier--Stokes Equations"
     */
    real relaxation_parameter;

protected:
    
    //Temporary flags -- will remove these before merging
    const bool use_relaxation;
    const bool do_write_root_function;

    /// Compute relaxation parameter explicitly (i.e. if energy is the entropy variable)
    /// See Ketcheson 2019, Eq. 2.4
    real compute_relaxation_parameter_explicit(real &dt) const;

    void write_root_function(real &dt) const;

    /// Modify timestep based on relaxation
    void modify_time_step (real &dt) override;

    /// Compute inner product according to the nodes being used
    /** This is the same calculation as energy, but using the residual instead of solution
     */
    real compute_inner_product(
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_i,
            const dealii::LinearAlgebra::distributed::Vector<double> &stage_j
            ) const;

    /// Storage for the second derivative at each Runge-Kutta stage
    /** Note that rk_stage (in RungeKuttaODESolver) stores first derivative
     */
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage_2nd_deriv;

    /// Butcher tableau "a_dot"
    dealii::Table<2,double> butcher_tableau_a_dot;

    /// Butcher tableau "b_dot"
    dealii::Table<1,double> butcher_tableau_b_dot;

};

} // ODE namespace
} // PHiLiP namespace

#endif
