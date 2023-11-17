#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>

#include "circular_cylinder.hpp"

namespace PHiLiP {
namespace Grids {

template<int dim>
void circular_cylinder(
    dealii::parallel::distributed::Triangulation<dim> &grid)
{
    //make triangulation in 2D and extrude if needed
    using Tria2D = dealii::parallel::distributed::Triangulation<2>;
    using Triangulation = dealii::parallel::distributed::Triangulation<2>;

    Tria2D boundary_triangulation(MPI_COMM_WORLD);
    dealii::GridGenerator::concentric_hyper_shells(boundary_triangulation,
                                                   dealii::Point<2>(), //origin
                                                   0.5, //inner radius
                                                   2.0, //outer radius
                                                   1, //n_shells
                                                   2.0, //skewness factor; bigger means more biased to inner radius
                                                   0, //default of 8 cells per shell
                                                   true); //colorize=true

    Tria2D nearfield_triangulation(MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_cube_with_cylindrical_hole(nearfield_triangulation,
                                                            2.0, //inner_radius
                                                            10.0, //outer_radius
                                                            0, //unused for 2D
                                                            0, //unused for 2D
                                                            true); //colorize=true
    
    dealii::GridGenerator::merge_triangulations(boundary_triangulation,//first grid to merge
            nearfield_triangulation, //second grid to merge
            grid, //grid to store
            1E-12, //tolerance for repeated vertices
            true); //copy boundary id

    Tria2D farfield_triangulation(MPI_COMM_WORLD);
    dealii::GridGenerator::subdivided_hyper_rectangle(farfield_triangulation,
            {1,2}, //number of cells in each direction
            dealii::Point<2>(10.0, -10.0), //point 1
            dealii::Point<2>(30.0,10.0), //point 2
            true); //colorize=true

    dealii::GridGenerator::merge_triangulations({&boundary_triangulation, &nearfield_triangulation, &farfield_triangulation},
            grid,//first grid to merge
            1E-12, //tolerance for repeated vertices
            false); //copy boundary id
   grid.refine_global(1);       

}

#if PHILIP_DIM==2
    template void circular_cylinder<PHILIP_DIM> (
        dealii::parallel::distributed::Triangulation<PHILIP_DIM> &grid);
#endif
}
}
