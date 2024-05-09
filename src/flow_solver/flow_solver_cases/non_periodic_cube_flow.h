#ifndef __NON_PERIODIC_CUBE_FLOW__
#define __NON_PERIODIC_CUBE_FLOW__

#include "flow_solver_case_base.h"
#include "cube_flow_uniform_grid.h"
#include "dg/dg_base.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class NonPeriodicCubeFlow : public CubeFlow_UniformGrid<dim, nstate>
{
#if PHILIP_DIM==1
     using Triangulation = dealii::Triangulation<PHILIP_DIM>;
 #else
     using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
 #endif

 public:
     explicit NonPeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
     
     std::shared_ptr<Triangulation> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;

 protected:
    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step_initial;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nstate>::update_maximum_local_wave_speed;

    /// Updates the maximum local wave speed
    void check_positivity_density(DGBase<dim, double>& dg);

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;
    
    /// Update numerical entropy variables
    void update_numerical_entropy(
            const double FR_entropy_contribution_RRK_solver,
            const unsigned int current_iteration,
            const std::shared_ptr <DGBase<dim, double>> dg);

    /// Calculate numerical entropy by matrix-vector product
    double compute_current_integrated_numerical_entropy(
            const std::shared_ptr <DGBase<dim, double>> dg) const;

    using FlowSolverCaseBase<dim,nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver, 
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table) override;
 
 private:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Numerical entropy at previous timestep
    double previous_numerical_entropy = 0;

    /// Cumulative change in numerical entropy
    double cumulative_numerical_entropy_change_FRcorrected = 0;

    /// Numerical entropy at initial time
    double initial_numerical_entropy_abs = 0;

    /// Pointer to Physics object for computing things on the fly
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;
    
    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
