#ifndef __RUNGE_KUTTA_ODESOLVER__
#define __RUNGE_KUTTA_ODESOLVER__

#include "JFNK_solver/JFNK_solver.h"
#include "dg/dg_base.hpp"
#include "runge_kutta_base.h"
#include "runge_kutta_methods/rk_tableau_butcher_base.h"
#include "relaxation_runge_kutta/empty_RRK_base.h"

namespace PHiLiP {
namespace ODE {

/// Runge-Kutta ODE solver (explicit or implicit) derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, int n_rk_stages, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RungeKuttaODESolver: public RungeKuttaBase <dim, real, n_rk_stages, MeshType>
{
public:
    RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauButcherBase<dim,real,MeshType>> rk_tableau_input,
            std::shared_ptr<EmptyRRKBase<dim,real,MeshType>> RRK_object_input); ///< Constructor.

    void allocate_runge_kutta_system () override;

    void calculate_stage_solution (int i, real dt, const bool pseudotime) override;

    void calculate_stage_derivative (int i, real dt) override;

    void sum_stages (real dt, const bool pseudotime) override;

    void apply_limiter () override;

    real adjust_time_step (real dt) override;

protected:
    /// Stores Butcher tableau a and b, which specify the RK method
    std::shared_ptr<RKTableauButcherBase<dim,real,MeshType>> butcher_tableau;

    /// Storage for the weighted/relative error estimate
    real w;

        /// Storage for the error estimate at step n-1, n, and n+1
    double epsilon[3];

    /// Storage for the absolute tolerance
    const double atol;

    /// Storage for the relative tolerance
    const double rtol;
        /// Storage for the first beta controller value
    const double beta1;

    /// Storage for the second beta controller value
    const double beta2;

    /// Storage for the third beta controller value
    const double beta3;

    double initial_entropy;
public:

    double compute_current_integrated_numerical_entropy(
            const std::shared_ptr <DGBase<dim, double, MeshType>> dg
            ) const;
};

} // ODE namespace
} // PHiLiP namespace

#endif

