#include <deal.II/grid/grid_out.h>
#include "mesh/grids/straight_periodic_cube.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerTaylorGreenScaling<dim, nstate>::EulerTaylorGreenScaling(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, -10.0, 10.0, 2);

    dealii::GridOut output_grid;
    std::ofstream output_file("circular_cylinder_grid.svg");
    output_grid.qrite_svg(&grid);



}
}
