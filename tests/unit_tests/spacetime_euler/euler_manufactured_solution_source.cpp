#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>


#include <deal.II/grid/grid_generator.h>

#include "parameters/parameters.h"
#include "physics/spacetime/euler_spacetime.h"
#include "physics/manufactured_solution.h"

const double TOLERANCE = 1E-5;

template <int dim, int nstate>
void print_flux(std::array<dealii::Tensor<1,dim,double>,nstate> flux) {
    for (int idim = 0; idim < dim; ++idim){
        std::cout << "idim = " << idim << std::endl;
        for (int istate = 0; istate < nstate; ++istate){
            std::cout << flux[istate][idim] << " ";
        }
        std::cout << std::endl;
    }
}

int main (int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    //const int spatial_dim = PHILIP_DIM-1;
    const int nstate = dim+2;

    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    const double a = 1.0 , b = 0.0, c = 1.4;
    //default parameters
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler); // default fills options
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    all_parameters.two_point_num_flux_type  = PHiLiP::Parameters::AllParameters::TwoPointNumericalFlux::Ra;
    //all_parameters.manufactured_convergence_study_param.manufactured_solution_param.manufactured_solution_type = PHiLiP::Parameters::ManufacturedSolutionParam::ManufacturedSolutionType::euler_spacetime;
    std::shared_ptr< PHiLiP::ManufacturedSolutionFunction<dim,double> > spacetime_manuf_soln =     std::make_shared<PHiLiP::ManufacturedSolutionEulerSpacetime<dim,double>>(nstate);
    PHiLiP::Physics::EulerSpacetime<dim, nstate, double> euler_physics = PHiLiP::Physics::EulerSpacetime<dim, nstate, double>(&all_parameters,a,c,a,b,b, spacetime_manuf_soln);

    std::array<double, nstate> soln_plus={{2.9510565162951536, 2.9510565162951536,0.0, 8.708734562368088}};
    std::array<double, nstate> soln_mins={{2.8090169943749475, 2.8090169943749475, 0.0, 7.890576474687264}};
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins;
    
    for (int i = 0; i < nstate; ++i){
        std::cout << soln_plus[i] << std::endl;
    }
    conv_flux_plus = euler_physics.convective_flux(soln_plus);
    std::cout << "Conv flux" << std::endl;
    print_flux<dim,nstate>(conv_flux_plus); // matches personal code

    std::array<dealii::Tensor<1,dim,double>,nstate> two_point_flux;
    two_point_flux = euler_physics.convective_numerical_split_flux_ranocha(soln_plus, soln_mins);
    std::cout << "Ra split flux" << std::endl;
    print_flux<dim,nstate>(two_point_flux); // matches julia.

    std::array<double, nstate> dissipation;
    dissipation = euler_physics.dissipation_for_entropy_stable_numerical_flux(soln_plus,soln_mins);
    for (int i = 0; i < nstate; ++i){
        std::cout << dissipation[i] << std::endl;
    }

    (void) soln_plus;
    (void) soln_mins;
    (void) conv_flux_plus;
    (void) conv_flux_mins;
    
    dealii::Tensor<1,dim,double> point_tensor;
    point_tensor[0] = 0.3;
    point_tensor[1] = 1.2;
    std::array<double,nstate> source = euler_physics.source_term(dealii::Point(point_tensor), soln_plus, 0.0);
    std::cout << "Source" << std::endl;
    for (int i = 0; i < nstate; ++i){
        std::cout << source[i] << std::endl;
    }



    MPI_Finalize();
    return 0;
}

