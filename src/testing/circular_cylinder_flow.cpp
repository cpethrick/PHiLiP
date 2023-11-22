#include <deal.II/grid/grid_out.h>
//#include "mesh/grids/straight_periodic_cube.hpp"
#include <fstream>

#include "mesh/grids/circular_cylinder.hpp"

#include "circular_cylinder_flow.hpp"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
CircularCylinderFlow<dim, nstate>::CircularCylinderFlow(
    const PHiLiP::Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
      , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int CircularCylinderFlow<dim, nstate>::run_test() const
{
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
        MPI_COMM_WORLD,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));


    PHiLiP::Grids::circular_cylinder<dim>(*grid);

    std::ofstream output_vtu("circular_cylinder_grid.vtu");
    dealii::GridOut().write_vtu(*grid, output_vtu);
    
    std::ofstream output_svg("circular_cylinder_grid.svg");
    dealii::GridOut().write_svg(*grid, output_svg);

    std::cout << grid->n_locally_owned_active_cells() <<std::endl;
    
    this->pcout << grid->n_global_active_cells() <<std::endl;

    return 0;
}

#if PHILIP_DIM==2
    template class CircularCylinderFlow<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace
