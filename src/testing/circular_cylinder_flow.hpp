#ifndef __CIRCULAR_CYLINDER_FLOW_H__
#define __CIRCULAR_CYLINDER_FLOW_H__

#include "tests.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"

namespace PHiLiP {
namespace Tests {

/// Flow over a circular cylinder.
template <int dim, int nstate>
class CircularCylinderFlow : public TestsBase
{
public:
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     * */
    explicit CircularCylinderFlow(
        const PHiLiP::Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Ensure that the kinetic energy is bounded.
    /** If the kinetic energy increases about its initial value, then the test should fail.
     *  Ref: Gassner 2016.
     * */
    int run_test() const override;
};


} // Tests namespace
} // PHiLiP namespace
#endif
