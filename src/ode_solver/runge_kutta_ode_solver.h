#ifndef __RUNGE_KUTTA_ODESOLVER__
#define __RUNGE_KUTTA_ODESOLVER__

#include "dg/dg.h"
#include "ode_solver_base.h"
#include "JFNK_solver/JFNK_solver.h"

namespace PHiLiP {
namespace ODE {

/// Explicit ODE solver derived from ODESolver.
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RungeKuttaODESolver: public ODESolverBase <dim, real, MeshType>
{
public:
    /// Default constructor that will set the constants.
    RungeKuttaODESolver(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input); ///< Constructor.

    /// Destructor
    ~RungeKuttaODESolver() {};

    /// Function to evaluate solution update
    void step_in_time(real dt, const bool pseudotime);

    /// Function to allocate the ODE system
    void allocate_ode_system ();

protected:
    /// Runge-Kutta order
    const int rk_order;

    /// Flag for implicit RK
    const bool implicit_flag;
    
    /// Solver for JFNK 
    //TO DO: check initialization (storage when not implicit )
    JFNKSolver<dim,real,MeshType> solver;

    /// Storage for the derivative at each Runge-Kutta stage
    std::vector<dealii::LinearAlgebra::distributed::Vector<double>> rk_stage;
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;

    /// Modify timestep
    virtual void modify_time_step(real &dt); 
};

} // ODE namespace
} // PHiLiP namespace

#endif