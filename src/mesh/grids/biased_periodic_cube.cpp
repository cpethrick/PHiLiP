#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <Sacado.hpp>
#include "biased_periodic_cube.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim, typename TriangulationType>
void biased_periodic_cube(
    std::shared_ptr<TriangulationType> &grid,
    const double domain_left,
    const double domain_right,
    const int number_of_cells_per_direction)
{
    // Get equivalent number of refinements
    const int number_of_refinements = log(number_of_cells_per_direction)/log(2);

    // Check that number_of_cells_per_direction is a power of 2 if number_of_refinements is non-zero
    if(number_of_refinements >= 0){
        int val_check = number_of_cells_per_direction;
        while(val_check > 1) {
            if(val_check % 2 == 0) val_check /= 2;
            else{
                std::cout << "ERROR: number_of_cells_per_direction is not a power of 2. " 
                          << "Current value is " << number_of_cells_per_direction << ". "
                          << "Change value of number_of_grid_elements_per_dimension in .prm file." << std::endl;
                std::abort();
            }
        }
    }

    const bool colorize = true;
    //dealii::GridGenerator::subdivided_hyper_rectangle (grid, n_subdivisions, p1, p2, colorize);

    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    std::vector<dealii::GridTools::PeriodicFacePair<typename TriangulationType::cell_iterator> > matched_pairs;
    if (dim>=1) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }
    if (dim>=2) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }
    if (dim>=3) {
        matched_pairs.clear();
        dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
        grid->add_periodicity(matched_pairs);
    }

    grid->refine_global(number_of_refinements);
    

    const BiasedManifold<dim,dim,dim> biased_manifold(number_of_cells_per_direction);

    dealii::GridTools::transform (
        [&biased_manifold](const dealii::Point<dim> &chart_point) {
          return biased_manifold.push_forward(chart_point);}, *grid);
    
/*    
    // Assign a manifold to have curved geometry
    unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
    grid.reset_all_manifolds();
    grid.set_all_manifold_ids(manifold_id);
    grid.set_manifold ( manifold_id, biased_manifold );
*/
}

template<int dim,int spacedim,int chartdim>
template<typename real>
dealii::Point<spacedim,real> BiasedManifold<dim,spacedim,chartdim>::mapping(const dealii::Point<chartdim,real> &chart_point) const 
{
    dealii::Point<spacedim,real> phys_point = chart_point;

/*    if constexpr (dim >= 2) {

        real x_perturbation = amplitude*dx[0];
        x_perturbation *= sin(n*pi*chart_point[1] / L0);

        phys_point[0] = chart_point[0];
        phys_point[0] += x_perturbation;

        real y_perturbation = amplitude*dx[1];
        y_perturbation *= sin(n*pi*chart_point[0] / L0);

        phys_point[1] = chart_point[1];
        phys_point[1] += y_perturbation;
    }
    if constexpr (dim >= 3) {
        phys_point[0] = chart_point[0];
        phys_point[0] += amplitude*sin(pi*chart_point[1]);
        phys_point[1] = chart_point[1];
        phys_point[1] += amplitude*sin(pi*chart_point[2]);
        phys_point[2] = chart_point[2];
        phys_point[2] += amplitude*sin(pi*chart_point[0]);

        real x_perturbation = amplitude*dx[0];
        x_perturbation *= sin(n*pi*chart_point[1] / L0) * sin (n*pi*chart_point[2] / L0);

        phys_point[0] = chart_point[0];
        phys_point[0] += x_perturbation;

        real y_perturbation = amplitude*dx[1];
        y_perturbation *= sin(n*pi*chart_point[0] / L0) * sin (n*pi*chart_point[2] / L0);

        phys_point[1] = chart_point[1];
        phys_point[1] += y_perturbation;

        real z_perturbation = amplitude*dx[2];
        z_perturbation *= sin(n*pi*chart_point[0] / L0) * sin (n*pi*chart_point[1] / L0);

        phys_point[2] = chart_point[2];
        phys_point[2] += z_perturbation;
    }
    //phys_point[sd] += chart_point[cd]+sin(pi*chart_point[cd]);
    //for (int sd=0; sd<spacedim; ++sd) {
    //    phys_point[sd] = 0.0;
    //    for (int cd=0; cd<chartdim; ++cd) {
    //        if (cd != sd) {
    //            phys_point[sd] += sin(pi*chart_point[cd]);
    //        }
    //    }
    //    phys_point[sd] *= 0.1;
    //}
*/
    // Bias coordinates in the x direction to be more dense close to the middle
    const double pt_new = pi/101;
    const double pt_old = pi/2;
    if (chart_point[0] < -pt_old)      phys_point[0] = (chart_point[0] + pi ) * (-1*pt_new + pi) / (-1*pt_old + pi) - pi;
    else if (chart_point[0] < pt_old)  phys_point[0] = (chart_point[0]) * (pt_new) / (pt_old);
    else if (chart_point[0] <= pi)     phys_point[0] = (chart_point[0] - pi) * (pt_new - pi) / (pt_old - pi) + pi;
    return phys_point;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<chartdim> BiasedManifold<dim,spacedim,chartdim>::pull_back(const dealii::Point<spacedim> &space_point) const {

    using FadType = Sacado::Fad::DFad<double>;
    dealii::Point<chartdim,FadType> chart_point_ad;
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d] = space_point[d];
    }
    for (int i=0; i<200; i++) {
        for (int d=0; d<chartdim; ++d) {
            chart_point_ad[d].diff(d,chartdim);
        }
        dealii::Point<spacedim,FadType> new_point = mapping<FadType>(chart_point_ad);

        dealii::Tensor<1,dim,double> fun;
        for (int d=0; d<chartdim; ++d) {
            fun[d] = new_point[d].val() - space_point[d];
        }
        double l2_norm = fun.norm();
        if(l2_norm < 1e-15) break;

        dealii::Tensor<2,dim,double> derivative;
        for (int sd=0; sd<spacedim; ++sd) {
            for (int cd=0; cd<chartdim; ++cd) {
                derivative[sd][cd] = new_point[sd].dx(cd);
            }
        }
        dealii::Tensor<2,dim,double> inv_jac = dealii::invert(derivative);
        dealii::Tensor<1,dim,double> dx = - inv_jac * fun;

        for (int d=0; d<chartdim; ++d) {
            chart_point_ad[d] = chart_point_ad[d].val() + dx[d];
        }
    }


    dealii::Point<dim,double> chart_point;
    for (int d=0; d<chartdim; ++d) {
        chart_point[d] = chart_point_ad[d].val();
    }
    dealii::Point<spacedim,double> new_point = mapping<double>(chart_point);
    dealii::Tensor<1,dim,double> fun = new_point - space_point;

    const double error = fun.norm();
    if (error > 1e-13) {
        std::cout << "Large error " << error << std::endl;
        std::cout << "Input space_point: " << space_point
                  << " Output space_point " << new_point << std::endl;
    }

    return chart_point;
}

template<int dim,int spacedim,int chartdim>
dealii::Point<spacedim> BiasedManifold<dim,spacedim,chartdim>::push_forward(const dealii::Point<chartdim> &chart_point) const 
{
    return mapping<double>(chart_point);
}

template<int dim,int spacedim,int chartdim>
dealii::DerivativeForm<1,chartdim,spacedim> BiasedManifold<dim,spacedim,chartdim>::push_forward_gradient(const dealii::Point<chartdim> &chart_point) const
{
    using FadType = Sacado::Fad::DFad<double>;
    dealii::Point<chartdim,FadType> chart_point_ad;
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d] = chart_point[d];
    }
    for (int d=0; d<chartdim; ++d) {
        chart_point_ad[d].diff(d,chartdim);
    }
    dealii::Point<spacedim,FadType> new_point = mapping<FadType>(chart_point_ad);

    dealii::DerivativeForm<1, chartdim, spacedim> dphys_dref;
    for (int sd=0; sd<spacedim; ++sd) {
        for (int cd=0; cd<chartdim; ++cd) {
            dphys_dref[sd][cd] = new_point[sd].dx(cd);
        }
    }
    return dphys_dref;
}

template<int dim,int spacedim,int chartdim>
std::unique_ptr<dealii::Manifold<dim,spacedim> > BiasedManifold<dim,spacedim,chartdim>::clone() const
{
    return std::make_unique<BiasedManifold<dim,spacedim,chartdim>>(n_subdivisions);
}

#if PHILIP_DIM==1
    template void biased_periodic_cube<PHILIP_DIM, dealii::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif
#if PHILIP_DIM!=1
    template void biased_periodic_cube<PHILIP_DIM, dealii::parallel::distributed::Triangulation<PHILIP_DIM>> (std::shared_ptr<dealii::parallel::distributed::Triangulation<PHILIP_DIM>> &grid, const double domain_left, const double domain_right, const int number_of_cells_per_direction);
#endif

} // namespace Grids
} // namespace PHiLiP


