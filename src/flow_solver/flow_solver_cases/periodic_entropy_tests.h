#ifndef __PERIODIC_ENTROPY_TESTS_H__
#define __PERIODIC_ENTROPY_TESTS_H__

#include "periodic_cube_flow.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class PeriodicEntropyTests : public PeriodicCubeFlow<dim,nstate>
{
public:

    /// Constructor.
    PeriodicEntropyTests(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~PeriodicEntropyTests() {};

    /// Calculate energy
    double compute_energy_collocated(const std::shared_ptr <DGBase<dim, double>> dg) const;
    
    /// Calculate entropy by matrix-vector products
    double compute_entropy(const std::shared_ptr <DGBase<dim, double>> dg) const;
protected:

    /// Function to compute the constant time step
    /** Calculates based on CFL for Euler, and from parameters otherwise */
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
    
    /// Filename for unsteady data
    std::string unsteady_data_table_filename_with_extension;
    
    /// Storing entropy at first step
    double initial_entropy;
    
};


} // FlowSolver namespace
} // PHiLiP namespace
#endif