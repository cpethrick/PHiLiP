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
        for (int velocity_dim=0; velocity_dim<dim-1; ++velocity_dim){
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
dealii::Tensor<2,nstate,real> EulerSpacetime<dim,nstate,real>
::convective_flux_directional_jacobian (
    const std::array<real,nstate> &conservative_soln,
    const dealii::Tensor<1,dim,real> &normal) const
{
    // See Blazek (year?) Appendix A.9 p. 429-430
    // For Blazek (2001), see Appendix A.7 p. 419-420
    // Alternatively, see Masatsuka 2018 "I do like CFD", p.77, eq.(3.6.8)
    const dealii::Tensor<1,dim,real> vel = this-> template compute_velocities<real>(conservative_soln);
    real vel_normal = 0.0;
    for (int d=0;d<dim;d++) { vel_normal += vel[d] * normal[d]; }

    const real vel2 = this->template compute_velocity_squared<real>(vel);
    const real phi = 0.5*this->gamm1 * vel2;

    const real density = conservative_soln[0];
    const real tot_energy = conservative_soln[nstate-1];
    const real E = tot_energy / density;
    const real a1 = this->gam*E-phi;
    const real a2 = this->gam-1.0;
    const real a3 = this->gam-2.0;

    dealii::Tensor<2,nstate,real> jacobian;
    if (abs(abs(normal[dim-1])- 1) < 1E-13) {
        // Last flux is solution, so directional Jacobian is identity
        for (int istate = 0; istate < nstate; ++istate){
            if (istate != nstate-2){
                jacobian[istate][istate]=1.0;
            }
        }
    }
    else{
        // Density
        for (int d=0; d<dim-1; ++d) {
            jacobian[0][1+d] = normal[d];
        }
        // Momentum equations
        for (int row_dim=0; row_dim<dim-1; ++row_dim) {
            jacobian[1+row_dim][0] = normal[row_dim]*phi - vel[row_dim] * vel_normal;
            for (int col_dim=0; col_dim<dim-1; ++col_dim){
                if (row_dim == col_dim) {
                    jacobian[1+row_dim][1+col_dim] = vel_normal - a3*normal[row_dim]*vel[row_dim];
                } else {
                    jacobian[1+row_dim][1+col_dim] = normal[col_dim]*vel[row_dim] - a2*normal[row_dim]*vel[col_dim];
                }
            }
            jacobian[1+row_dim][nstate-1] = normal[row_dim]*a2;
        }

        //Energy
        jacobian[nstate-1][0] = vel_normal*(phi-a1);
        for (int d=0; d<dim; ++d){
            jacobian[nstate-1][1+d] = normal[d]*a1 - a2*vel[d]*vel_normal;
        }
        jacobian[nstate-1][nstate-1] = this->gam*vel_normal;
    }
    return jacobian;
}

// template <int dim, int nstate, typename real>
// std::array<real,nstate> EulerSpacetime<dim,nstate,real>
// ::convective_source_term (
//     const dealii::Point<dim,real> &/*pos*/) const
// {
// 
//     const real pi = atan(1) * 4;
//     //Note, I don't think this would be different than Euler base, but
//     //that assumption hasn't been validated so best to abort.
//     std::array<real,nstate> convective_source_term;
// 
//     convective_source_term[1] = pi * this->gamm1 * ( 7 + 4 * sin(2 * pi * ( pos[0]-pos[1])))*cos(2 * pi * ( pos[0]-pos[1]));
//     convective_source_term[nstate-1] = pi * this->gamm1 * ( 7 + 4 * sin(2 * pi * ( pos[0]-pos[1])))*cos(2 * pi * ( pos[0]-pos[1]));
// 
//     return convective_source_term;
// }

// template <int dim, int nstate, typename real>
// std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim,nstate,real>
// ::get_manufactured_solution_gradient (
//     const dealii::Point<dim,real> &pos) const
// {
//     this->pcout << "ERROR: get_manufactured_solution_gradient not implemented! Aborting..." << std::endl;
//     std::abort();
//     //Note, I don't think this would be different than Euler base, but
//     //that assumption hasn't been validated so best to abort.
//     std::vector<dealii::Tensor<1,dim,real>> manufactured_solution_gradient_dealii(nstate);
//     this->manufactured_solution_function->vector_gradient(pos,manufactured_solution_gradient_dealii);
//     std::array<dealii::Tensor<1,dim,real>,nstate> manufactured_solution_gradient;
//     for (int d=0;d<dim;d++) { // Not sure whether this should loop over dim or dim-1
//         for (int s=0; s<nstate; s++) {
//             manufactured_solution_gradient[s][d] = manufactured_solution_gradient_dealii[s][d];
//         }
//     }
//     return manufactured_solution_gradient;
// }

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
    const dealii::Point<dim, real> &pos,
    const dealii::Tensor<1,dim,real> &normal_int,
    const std::array<real,nstate> &soln_int,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
    std::array<real,nstate> &soln_bc,
    std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{

    if (abs(normal_int[dim-1]-1) < 1E-14) {
        // normal in temporal dimension = 1: this boundary will be pure convective outflow
        soln_bc = soln_int;
        soln_grad_bc = soln_grad_int;
    } else if (abs(normal_int[dim-1]+1) < 1E-14){
        // normal in temporal dimension = -1: this boundary will be pure upwinding
        // of a Dirichlet boundary

        /// Manufactured solution from Friedrich et al 2019 eqn 4.4
        /// Extended to 2D+1 with uniform 0 vel in y.
        const double pi = atan(1.0)*4;
        soln_bc[0] = 2 + sin(2 * pi  * (pos[0]));

        std::array<real,dim> soln_momentums;
        soln_momentums[0] = 2 + sin(2 * pi * (pos[0] ));
        if constexpr(dim==3) {
            soln_momentums [1] = 0.0;
        }
        // last dim: always zero because we store an additional unused state
        soln_momentums[dim-1] = 0.0;

        for (int idim=0; idim < dim; ++idim){
            soln_bc[idim+1] = soln_momentums[idim];
        }
        
        soln_bc[nstate-1] = pow(2 + sin(2 * pi * pos[0]),2);
        for (int istate = 0; istate < nstate;  ++istate){
            soln_grad_bc[istate] = 0;
        }

    } else {
        this->pcout << "Warning: attempting to use purely upwind boundary on a non-temporal boundary!" << std::endl;
        //Return internal state.
        soln_bc = soln_int;
        soln_grad_bc = soln_grad_int;
    }

}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim,nstate,real>::
convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const
{

    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    if(this->two_point_num_flux_type == two_point_num_flux_enum::Ra) {
        conv_num_split_flux = convective_numerical_split_flux_ranocha(conservative_soln1, conservative_soln2);
    }
    else {
        this->pcout << "ERROR: not yet implemented!!" << std::endl;
        std::abort();
        std::array<dealii::Tensor<1,dim,real>,nstate> nothing_tensor;
        conv_num_split_flux = nothing_tensor;
    }


    return conv_num_split_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> EulerSpacetime<dim, nstate, real>
::convective_numerical_split_flux_ranocha(const std::array<real,nstate> &conservative_soln1,
                                                const std::array<real,nstate> &conservative_soln2) const
{

    std::array<dealii::Tensor<1,dim,real>,nstate> conv_num_split_flux;
    const real rho_log = this->compute_ismail_roe_logarithmic_mean(conservative_soln1[0], conservative_soln2[0]);
    const real pressure1 = this->template compute_pressure<real>(conservative_soln1);
    const real pressure2 = this->template compute_pressure<real>(conservative_soln2);

    const real beta1 = conservative_soln1[0]/(pressure1);
    const real beta2 = conservative_soln2[0]/(pressure2);

    const real beta_log = this->compute_ismail_roe_logarithmic_mean(beta1, beta2);
    const dealii::Tensor<1,dim,real> vel1 = this->template compute_velocities<real>(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = this->template compute_velocities<real>(conservative_soln2);

    const real pressure_hat = 0.5*(pressure1+pressure2);

    dealii::Tensor<1,dim,real> vel_avg;
    real vel_square_avg = 0.0;
    for(int idim=0; idim<dim; idim++){
        vel_avg[idim] = 0.5*(vel1[idim]+vel2[idim]);
        vel_square_avg += (0.5 *(vel1[idim]+vel2[idim])) * (0.5 *(vel1[idim]+vel2[idim]));
    }

    real enthalpy_hat = 1.0/(beta_log*this->gamm1) + vel_square_avg + 2.0*pressure_hat/rho_log;

    real vel_square_avg_1122=0;
    for(int idim=0; idim<dim; idim++){
        vel_square_avg_1122 += (0.5*(vel1[idim]*vel1[idim] + vel2[idim]*vel2[idim]));
    }
    enthalpy_hat -= 0.5*(vel_square_avg_1122);

    /// Spatial part
    for(int flux_dim=0; flux_dim<dim - 1; flux_dim++){
        // Density equation
        conv_num_split_flux[0][flux_dim] = rho_log * vel_avg[flux_dim];
        // Momentum equation
        for (int velocity_dim=0; velocity_dim<dim; ++velocity_dim){
            conv_num_split_flux[1+velocity_dim][flux_dim] = rho_log*vel_avg[flux_dim]*vel_avg[velocity_dim];
        }
        conv_num_split_flux[1+flux_dim][flux_dim] += pressure_hat; // Add diagonal of pressure

        // Energy equation
        conv_num_split_flux[nstate-1][flux_dim] = rho_log * vel_avg[flux_dim] * enthalpy_hat;
        conv_num_split_flux[nstate-1][flux_dim] -= ( 0.5 *(pressure1*vel1[flux_dim] + pressure2*vel2[flux_dim]));
    }

    if (dim == 3) this->pcout << "WARNING: Not checked for 2D+1...." << std::endl;

    ///////// TO DO before merging, move this into another function and add other spatial fluxes.
    /// Temporal part
    // Density equation
    conv_num_split_flux[0][dim-1] = rho_log;
    for (int velocity_dim=0; velocity_dim<dim-1; ++velocity_dim) {
        conv_num_split_flux[1+velocity_dim][dim-1] = rho_log * vel_avg[0];
        //Unsure what velocity this would be in 2D+1
    }
    conv_num_split_flux[nstate-1][dim-1] = 0.5 * rho_log / (0.5*beta_log * this->gamm1) + rho_log * (vel_avg[0]*vel_avg[0] - 0.5 * vel_square_avg_1122);
    
   return conv_num_split_flux; 

}

template <int dim, int nstate, typename real>
std::array<real, nstate> EulerSpacetime<dim, nstate, real>
::dissipation_for_entropy_stable_numerical_flux(const std::array<real,nstate> &conservative_soln1,
                                                const std::array<real,nstate> &conservative_soln2) const
{
    //std::array<dealii::Tensor<1,dim,real>,nstate> dissipation;

    if (dim == 3) this->pcout << "WARNING: Not valid for 2D+1...." << std::endl;
    const dealii::Tensor<1,dim,real> vel1 = this->template compute_velocities<real>(conservative_soln1);
    const dealii::Tensor<1,dim,real> vel2 = this->template compute_velocities<real>(conservative_soln2);

    dealii::Tensor<1,dim,real> vel_avg;
    for(int idim=0; idim<dim; idim++){
        vel_avg[idim] = 0.5*(vel1[idim]+vel2[idim]);
    }
    const real rho_log = this->compute_ismail_roe_logarithmic_mean(conservative_soln1[0], conservative_soln2[0]);
    const real rho_avg = 0.5 *(conservative_soln1[0] + conservative_soln2[0]);

    const real vel_sq_bar = 2.0*vel_avg[0]*vel_avg[0] - 0.5 * (vel1[0]*vel1[0] + vel2[0]*vel2[0]);

    const real pressure1 = this->template compute_pressure<real>(conservative_soln1);
    const real pressure2 = this->template compute_pressure<real>(conservative_soln2);

    const real beta1 = 0.5*conservative_soln1[0]/(pressure1);
    const real beta2 = 0.5*conservative_soln2[0]/(pressure2);
    const real beta_log = this->compute_ismail_roe_logarithmic_mean(beta1, beta2);
    const real p_hat = 0.5 * rho_avg /( 0.5 * (beta1 + beta2));

    const real h_bar = this->gam/(2.0 * beta_log * this->gamm1) + 0.5 * vel_sq_bar;
    const real a_bar = sqrt(this->gam * p_hat / rho_log);

    dealii::Tensor<2,nstate,real> R_hat;
    for (int istate = 0; istate < nstate; ++istate){
        if (istate != nstate-2)  {
            R_hat[0][istate] = 1.0;
        }
    }
    const real v_avg = vel_avg[0]; //NOT VALID FOR 2D+1
    R_hat[1][0] = (v_avg - a_bar);
    R_hat[1][1] = v_avg;
    R_hat[1][3] = (v_avg + a_bar);

    R_hat[3][0] = (h_bar - v_avg*a_bar);
    R_hat[3][1] = 0.5*vel_sq_bar;
    R_hat[3][3] = (h_bar + v_avg * a_bar);

    //R_hat, lambda_hat, T_hat  verified against  julia code
/*
    std::cout << "R_hat " << std::endl;
    for (int istate = 0; istate < nstate; ++istate){
        for (int jstate = 0; jstate < nstate; ++jstate){
            std::cout << R_hat[istate][jstate] << " ";
        }
        std::cout << std::endl;
    }
*/
    std::array<real,nstate> Lambda_hat = {{  abs(v_avg - a_bar), abs(v_avg),0, abs(v_avg+a_bar) }};

    std::array<real,nstate> T_hat = {{rho_log/2.0/this->gam, rho_log * (this->gamm1)/this->gam, 0, rho_log/2.0/this->gam }};

    dealii::Tensor<2,nstate,real> temp1;
    for (int istate = 0; istate < nstate; ++istate) {
        for (int jstate = 0; jstate < nstate; ++jstate) {
            temp1[istate][jstate] = R_hat[jstate][istate];
        }
    }
    for (int istate = 0; istate < nstate; ++istate) {
        for (int jstate = 0; jstate < nstate; ++jstate) {
            temp1[istate][jstate] *= -0.5 * Lambda_hat[istate]*T_hat[istate];
        }
    }
    

    dealii::Tensor<2,nstate,real> dissipation_scaling_matrix;
    // matrix multiplication
    for (int istate = 0; istate < nstate; ++istate) {
        for (int jstate = 0; jstate < nstate; ++jstate) {
            for (int kstate = 0; kstate < nstate; ++kstate) {
                dissipation_scaling_matrix[istate][jstate] += R_hat[istate][kstate] * temp1[kstate][jstate];
            }
        }
    }

    std::array<real,nstate> entropy_var_2 = this->compute_entropy_variables(conservative_soln2); //ext
    std::array<real,nstate> entropy_var_1 =  this->compute_entropy_variables(conservative_soln1); // int
    std::array<real,nstate> dissipation_vector = {};
    //// NOTE: Justification for dividing by gamma -1? 
    for (int istate = 0; istate < nstate; ++istate) {
        for (int jstate = 0; jstate < nstate; ++jstate) {
            dissipation_vector[istate] += dissipation_scaling_matrix[istate][jstate] * (entropy_var_2[jstate] / this->gamm1-entropy_var_1[jstate]/this->gamm1);
        }
    }

    // verified against julia to here
    return dissipation_vector;
    
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
