#include "ADTypes.hpp"

#include "euler_spacetime.h"

namespace PHiLiP {
namespace Physics {


template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &conservative_soln) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const real density = conservative_soln[0];
    const real pressure = this->template compute_pressure<real>(conservative_soln);
    const dealii::Tensor<1,dim,real> vel = this->template compute_velocities<real>(conservative_soln);
    const real specific_total_energy = conservative_soln[nstate-1]/conservative_soln[0];
    const real specific_total_enthalpy = specific_total_energy + pressure/density;

    // spatial
    for (int flux_dim=0; flux_dim<dim-1; ++flux_dim) {
        // Density equation
        conv_flux[0][flux_dim] = conservative_soln[1+flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_flux[1+velocity_dim][flux_dim] = density*vel[flux_dim]*vel[velocity_dim];
        }
        conv_flux[1+flux_dim][flux_dim] += pressure; // Add diagonal of pressure
        // Energy equation
        conv_flux[nstate-1][flux_dim] = density*vel[flux_dim]*specific_total_enthalpy;
    }

    // temporal
    const real temporal_advection = 1.0; // unit by definition
    for (int istate = 0; istate < nstate; ++istate){
        conv_flux[istate][dim-1] += temporal_advection * conservative_soln[istate]; 
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> EulerSpacetime<dim,nstate,real>
::convective_source_term (
    const dealii::Point<dim,real> &/*pos*/) const
{
    this->pcout << "ERROR: convective_source_term not implemented! Aborting..." << std::endl;
    std::abort();
    //Note, I don't think this would be different than Euler base, but
    //that assumption hasn't been validated so best to abort.
    std::array<real,nstate> convective_source_term;

    return convective_source_term;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim,nstate,real>
::get_manufactured_solution_gradient (
    const dealii::Point<dim,real> &pos) const
{
    this->pcout << "ERROR: get_manufactured_solution_gradient not implemented! Aborting..." << std::endl;
    std::abort();
    //Note, I don't think this would be different than Euler base, but
    //that assumption hasn't been validated so best to abort.
    std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
    this->manufactured_solution_function->vector_gradient(pos,manufactured_solution_gradient_dealii);
    std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient;
    for (int d=0;d<dim;d++) { // Not sure whether this should loop over dim or dim-1
        for (int s=0; s<nstate; s++) {
            manufactured_solution_gradient[s][d] = manufactured_solution_gradient_dealii[s][d];
        }
    }
    return manufactured_solution_gradient;
}

template <int dim, int nstate, typename real>
void EulerSpacetime<dim,nstate,real>
::boundary_face_values (
   const int boundary_type,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    if (boundary_type == 1010) {
        // Manufactured solution boundary condition
        boundary_purely_upwind(pos, normal_int, soln_int, soln_grad_int, soln_bc, soln_grad_bc);
    } else {
        this->pcout << "Warning: Only pure upwind has been verified for EulerSpacetime!" << std::endl
              << "Proceed with caution!" << std::endl;
        return Euler<dim,nstate,real>::boundary_face_values (boundary_type,pos,normal_int,soln_int,soln_grad_int,soln_bc,soln_grad_bc);
    }
}


template <int dim, int nstate, typename real>
void EulerSpacetime<dim,nstate,real>::
boundary_purely_upwind(
    const dealii::Point<dim, real> &/*pos*/,
    const dealii::Tensor<1,dim,real> &/*normal_int*/,
    const std::array<real,nstate> &soln_int,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
    std::array<real,nstate> &soln_bc,
    std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{

    soln_bc = soln_int;
    // TO DO:
    // add check that normal is purely temporal
    // add 
    this->pcout << "Warning: haven't yet implemented boundary_purely_upwind! " << std::endl;

    // soln_grad_bc = 0
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim,nstate,real>::
convective_numerical_split_flux (
        const std::array<real,nstate> &/*conservative_soln1*/,
        const std::array<real,nstate> &/*conservative_soln2*/) const
{

    this->pcout << "ERROR: not yet implemented!!" << std::endl;
    std::abort();
    std::array<dealii::Tensor<1,dim,real>,nstate> nothing_tensor;
    return nothing_tensor;
}
#if PHILIP_DIM>1
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, double >;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, FadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, RadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, FadFadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, RadFadType>;
#endif
} // Physics namespace
} // PHiLiP namespace
