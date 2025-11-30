#ifndef __EULER_SPACETIME__
#define __EULER_SPACETIME__

#include <deal.II/base/tensor.h>
#include "../euler.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Euler equations. Derived from PhysicsBase
/** Only 2D and 3D
 *  State variable and convective fluxes given by
 *
 *  \f[ 
 *  \mathbf{w} = 
 *  \begin{bmatrix} \rho \\ \rho v_1 \\ \rho v_2 \\ \rho v_3 \\ \rho E \end{bmatrix}
 *  , \qquad
 *  \mathbf{F}_{conv} = 
 *  \begin{bmatrix} 
 *      \mathbf{f}^x_{conv}, \mathbf{f}^y_{conv}, \mathbf{f}^z_{conv}
 *  \end{bmatrix}
 *  =
 *  \begin{bmatrix} 
 *  \begin{bmatrix} 
 *  \rho v_1 \\
 *  \rho v_1 v_1 + p \\
 *  \rho v_1 v_2     \\ 
 *  \rho v_1 v_3     \\
 *  v_1 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_2 \\
 *  \rho v_1 v_2     \\
 *  \rho v_2 v_2 + p \\ 
 *  \rho v_2 v_3     \\
 *  v_2 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_3 \\
 *  \rho v_1 v_3     \\
 *  \rho v_2 v_3     \\ 
 *  \rho v_3 v_3 + p \\
 *  v_3 (\rho e+p)
 *  \end{bmatrix}
 *  \end{bmatrix} \f]
 *  
 *  where, \f$ E \f$ is the specific total energy and \f$ e \f$ is the specific internal
 *  energy, related by
 *  \f[
 *      E = e + |V|^2 / 2
 *  \f] 
 *  For a calorically perfect gas
 *
 *  \f[
 *  p=(\gamma -1)(\rho e-\frac{1}{2}\rho \|\mathbf{v}\|)
 *  \f]
 *
 *  Dissipative flux \f$ \mathbf{F}_{diss} = \mathbf{0} \f$
 *
 *  Source term \f$ s(\mathbf{x}) \f$
 *
 *  Equation:
 *  \f[ \boldsymbol{\nabla} \cdot
 *         (  \mathbf{F}_{conv}( w ) 
 *          + \mathbf{F}_{diss}( w, \boldsymbol{\nabla}(w) )
 *      = s(\mathbf{x})
 *  \f]
 *
 *************************************************************************
 * TO DO: update the above part of this comment.
 * Note that for an nD+1 spacetime implementation, the state vector remains size $PHILIP_DIM+1$.
 * The last velocity (temporal dim) must always remain zero.
 * This choice is not memory-efficient, but allows the EulerSpacetime to use many functions
 * from Euler without re-writing.
 */
template <int dim, int nstate, typename real>
class EulerSpacetime : public Euler<dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    //using PhysicsBase<dim,nstate,real>::dissipative_flux;
    //using PhysicsBase<dim,nstate,real>::source_term;
public:
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    EulerSpacetime ( 
        const Parameters::AllParameters *const                    parameters_input,
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false):
        Euler<dim,nstate,real>(parameters_input,ref_length,gamma_gas,mach_inf,angle_of_attack,side_slip_angle,manufactured_solution_function,two_point_num_flux_type,has_nonzero_diffusion,has_nonzero_physical_source){
            this->velocities_inf=0;
            if(dim==1) {
                this->velocities_inf[0] = 1.0;
            } else if(dim==2) {
                this->velocities_inf[0] = 1.0;
                this->velocities_inf[1] = 0.0;
            } else if (dim==3) {
                this->velocities_inf[0] = cos(angle_of_attack);
                this->velocities_inf[1] = sin(angle_of_attack); // Maybe minus?? -- Clarify with Doug
                this->velocities_inf[2] = 0.0;
            }
            assert(std::abs(this->velocities_inf.norm() - 1.0) < 1e-14);
        };
    
    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const override;
    
    /// Convective flux Jacobian: \f$ \frac{\partial \mathbf{F}_{conv}}{\partial w} \cdot \mathbf{n} \f$
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const override;

    /// Convective flux contribution to the source term
    //std::array<real,nstate> convective_source_term (
    //    const dealii::Point<dim,real> &pos) const override;

    ///  Evaluates convective flux based on the chosen split form.
    /// Currently, none of these are implemented, so this function returns zero.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const override;

    /// Boundary condition handler
    /// Not all boundaries will have been modified.
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const override;

    /// Purely upwind boundary
    /// For temporal dimension surfaces only.
    /// Dirichlet on the t=0 surface, outflow on t=t_f surface.
    void boundary_purely_upwind(
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Get manufactured solution gradient
    //std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient(
    //    const dealii::Point<dim,real> &pos) const override;
    /////// Numerical fluxes here....:
    // Add all and describe that they aren't implemented.
    
    /// Ranocha pressure equilibrium preserving, entropy and energy conserving flux.
    /// Temporal dim uses the two-point flux described in Friedrich et al. 2019.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_ranocha (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const override;

    /// Entropy-stable matrix dissipation per Gassner, Winters, Hindenlang, Kopriva 2018 Appendix A
    /// Jump is calculated in entropy variables.
     std::array<real, nstate> dissipation_for_entropy_stable_numerical_flux(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;
};


} // Physics namespace
} // PHiLiP namespace

#endif
