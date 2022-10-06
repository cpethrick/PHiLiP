#include <fstream>
#include "dg/dg_factory.hpp"
#include "euler_isentropic_vortex.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "physics/initial_conditions/initial_condition_function.h"
#include "mesh/grids/straight_periodic_cube.hpp"

#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/Dense>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerIsentropicVortex<dim, nstate>::EulerIsentropicVortex(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
{}


template<int dim, int nstate>
double EulerIsentropicVortex<dim, nstate>::compute_kinetic_energy(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    //returns the energy in the L2-norm (physically relevant)
    int overintegrate = 10 ;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
                    dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    double total_kinetic_energy = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        //Please see Eq. 3.21 in Gassner, Gregor J., Andrew R. Winters, and David A. Kopriva. "Split form nodal discontinuous Galerkin schemes with summation-by-parts property for the compressible Euler equations." Journal of Computational Physics 327 (2016): 39-66.
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
             const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const double density = soln_at_q[0];

            double quadrature_kinetic_energy = 0.0;
            for(int i=0;i<dim;i++){
                quadrature_kinetic_energy += 0.5*(soln_at_q[i+1]*soln_at_q[i+1])/density;
            }

            total_kinetic_energy += quadrature_kinetic_energy * fe_values_extra.JxW(iquad);
        }
    }
    return total_kinetic_energy;
}

template<int dim, int nstate>
double EulerIsentropicVortex<dim, nstate>::get_timestep(std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree, const double delta_x) const
{
    //get local CFL
    const unsigned int n_dofs_cell = nstate*pow(poly_degree+1,dim);
    const unsigned int n_quad_pts = pow(poly_degree+1,dim);
    std::vector<dealii::types::global_dof_index> dofs_indices1 (n_dofs_cell);

    double cfl_min = 1e100;
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, double > > pde_physics_double  = PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices (dofs_indices1);
        std::vector< std::array<double,nstate>> soln_at_q(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate=0; istate<nstate; istate++) {
                soln_at_q[iquad][istate]      = 0;
            }
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
          dealii::Point<dim> qpoint = dg->volume_quadrature_collection[poly_degree].point(iquad);
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                soln_at_q[iquad][istate] += dg->solution[dofs_indices1[idof]] * dg->fe_collection[poly_degree].shape_value_component(idof, qpoint, istate);
            }
        }

        std::vector< double > convective_eigenvalues(n_quad_pts);
        for (unsigned int isol = 0; isol < n_quad_pts; ++isol) {
            convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        }
        const double max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));
        double cfl = 0.05 * delta_x/max_eig;

       // double cfl = 0.000005 * delta_x/max_eig;
        if(cfl < cfl_min)
            cfl_min = cfl;
    }
    return cfl_min;
}

template <int dim, int nstate>
int EulerIsentropicVortex<dim, nstate>::run_test() const
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
        using Triangulation = dealii::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#else
        using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            MPI_COMM_WORLD,
            typename dealii::Triangulation<dim>::MeshSmoothing(
                dealii::Triangulation<dim>::smoothing_on_refinement |
                dealii::Triangulation<dim>::smoothing_on_coarsening));
#endif

    using real = double;

    PHiLiP::Parameters::AllParameters all_parameters_new = *all_parameters;  
    double left = -5.0;
    double right = 5.0;
//    const bool colorize = true;
    const int n_refinements = 8;
   // const int n_refinements = 32;
   // const int n_refinements = 16;
   // unsigned int poly_degree = 5;
    unsigned int poly_degree = 3;
    const unsigned int grid_degree = 1;

    // set the warped grid
   // PHiLiP::Grids::nonsymmetric_curved_grid<dim,Triangulation>(*grid, n_refinements);

    PHiLiP::Grids::straight_periodic_cube<dim,Triangulation>(grid, left, right, n_refinements);
//     dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);
//     std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<PHILIP_DIM>::cell_iterator> > matched_pairs;
//     dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
//     if constexpr (dim>1)
//        dealii::GridTools::collect_periodic_faces(*grid,2,3,1,matched_pairs);
//     if constexpr (dim>2)
//        dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs);
//     grid->add_periodicity(matched_pairs);
//     grid->refine_global(n_refinements);

    // Create DG
    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(&all_parameters_new, poly_degree, poly_degree, grid_degree, grid);
    dg->allocate_system ();

    std::cout << "Implement initial conditions" << std::endl;
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function = 
                InitialConditionFactory<dim,nstate,double>::create_InitialConditionFunction(&all_parameters_new);
    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

#if 0
    //Do eigenvalues

    std::cout<<"doing eig"<<std::endl;
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
    Eigen::MatrixXd dRdU(dg->solution.size(), dg->solution.size());
    std::cout<<" allocating col DRdU"<<std::endl;
    dealii::LinearAlgebra::distributed::Vector<double> col_dRdU(dg->right_hand_side.size());
//    dealii::LinearAlgebra::distributed::Vector<double> col_dRdU;
//    col_dRdU.reinit(dg->locally_owned_dofs, dg->ghost_dofs, MPI_COMM_WORLD);
    const double perturbation = 1e-8;
    std::cout<<"doing perturbations"<<std::endl;
    for(unsigned int eig_direction=0; eig_direction<dg->solution.size(); eig_direction++){
        double solution_init_value = dg->solution[eig_direction];
        dg->solution[eig_direction] += perturbation;
        dg->assemble_residual();

//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

//        if(all_parameters_new.use_inverse_mass_on_the_fly){
//            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        } else{
//            dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        }
        for(unsigned int i=0; i<dg->solution.size(); i++){
           // dRdU[i][eig_direction] = dg->right_hand_side[i];
           col_dRdU[i] = dg->right_hand_side[i];
            dRdU(i,eig_direction) = col_dRdU[i];
        }
        dg->solution[eig_direction] -= 2.0 * perturbation;
        dg->assemble_residual();
//        if(all_parameters_new.use_inverse_mass_on_the_fly){
//            dg->apply_inverse_global_mass_matrix(dg->right_hand_side, col_dRdU); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        } else{
//            dg->global_inverse_mass_matrix.vmult(col_dRdU, dg->right_hand_side); //rk_stage[i] = IMM*RHS = F(u_n + dt*sum(a_ij*k_j))
//        }
        for(unsigned int i=0; i<dg->solution.size(); i++){
           // dRdU[i][eig_direction] -= dg->right_hand_side[i];
           col_dRdU[i] = dg->right_hand_side[i];
            dRdU(i,eig_direction) -= col_dRdU[i];
            dRdU(i,eig_direction) /= (2.0 * perturbation);
        }
        dg->solution[eig_direction] = solution_init_value;//set back to the IC

//    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);
    }
    std::cout<<"got perts"<<std::endl;
//    Eigen::MatrixXd dRdU_mpi(dg->solution.size(), dg->solution.size());
//    for(unsigned int i=0; i<dg->solution.size(); i++){
//        for(unsigned int j=0; j<dg->solution.size(); j++){
//            dRdU_mpi(i,j) = dealii::Utilities::MPI::sum(dRdU(i,j), this->mpi_communicator);
//        }
//    }
    eigen_solver.compute(dRdU);
   // eigen_solver.compute(dRdU_mpi);
    std::cout<<"got eigs"<<std::endl;
    std::ofstream myfile2 ("computed_eigenvalues_euler_play_around.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile2<< std::fixed << std::setprecision(16) << eigen_solver.eigenvalues() << " "<< std::fixed<<std::endl;

    myfile2.close();

    std::ofstream myfile3 ("computed_eigenvectors_euler_play_around.gpl" , std::ios::trunc);
    std::cout << std::setprecision(16) << std::fixed;
    myfile3<< std::fixed << std::setprecision(16) << eigen_solver.eigenvectors()<< std::fixed << std::endl<<std::endl;
    myfile3.close();

    //end eigenvalues


    SetInitialCondition<dim,nstate,double>::set_initial_condition(initial_condition_function, dg, &all_parameters_new);

#endif

    const unsigned int n_global_active_cells2 = grid->n_global_active_cells();
    double delta_x = (right-left)/n_global_active_cells2/(poly_degree+1.0);
    pcout<<" delta x "<<delta_x<<std::endl;

    all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree,delta_x);
     
    std::cout << "creating ODE solver" << std::endl;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout << "ODE solver successfully created" << std::endl;
    double finalTime = 5.;
    // finalTime = 0.1;//to speed things up locally in tests, doesn't need full 14seconds to verify.
    double dt = all_parameters_new.ode_solver_param.initial_time_step;
    // double dt = all_parameters_new.ode_solver_param.initial_time_step / 10.0;
//    finalTime=dt;

    std::cout << " number dofs " << dg->dof_handler.n_dofs()<<std::endl;
    std::cout << "preparing to advance solution in time" << std::endl;

    // Currently the only way to calculate energy at each time-step is to advance solution by dt instead of finaltime
    // this causes some issues with outputs (only one file is output, which is overwritten at each time step)
    // also the ode solver output doesn't make sense (says "iteration 1 out of 1")
    // but it works. I'll keep it for now and need to modify the output functions later to account for this.
    double initialcond_energy = compute_kinetic_energy(dg, poly_degree);
    double initialcond_energy_mpi = (dealii::Utilities::MPI::sum(initialcond_energy, mpi_communicator));
    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy for initial condition " << initialcond_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;

    pcout << "Energy at time " << 0 << " is " << compute_kinetic_energy(dg, poly_degree) << std::endl;
    ode_solver->current_iteration = 0;
	ode_solver->advance_solution_time(dt/10.0);
	double initial_energy = compute_kinetic_energy(dg, poly_degree);
	double initial_energy_mpi = (dealii::Utilities::MPI::sum(initial_energy, mpi_communicator));

    std::cout << std::setprecision(16) << std::fixed;
    pcout << "Energy at one timestep is " << initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
    // std::ofstream myfile ("kinetic_energy_3D_TGV_cdg_curv_grid_4x4.gpl" , std::ios::trunc);
    std::ofstream myfile (all_parameters_new.energy_file + ".gpl"  , std::ios::trunc);

    for (int i = 0; i < std::ceil(finalTime/dt); ++ i) {
        ode_solver->advance_solution_time(dt);
       // ode_solver->step_in_time(dt,false);
      //  ode_solver->step_in_time(dt,true);
        // double current_energy = compute_kinetic_energy(dg,poly_degree) / initial_energy;
        double current_energy = compute_kinetic_energy(dg,poly_degree);
        double current_energy_mpi = (dealii::Utilities::MPI::sum(current_energy, mpi_communicator))/initial_energy_mpi;
        std::cout << std::setprecision(16) << std::fixed;
        // pcout << "Energy at time " << i * dt << " is " << current_energy << std::endl;
        pcout << "Energy at time " << ode_solver->current_time << " is " << current_energy_mpi << std::endl;
        pcout << "Actual Energy Divided by volume at time " << ode_solver->current_time << " is " << current_energy_mpi*initial_energy_mpi/(8*pow(dealii::numbers::PI,3)) << std::endl;
        // myfile << i * dt << " " << current_energy << std::endl;
        myfile << ode_solver->current_time << " " << current_energy_mpi << std::endl;
        // if (current_energy*initial_energy - initial_energy >= 1.00)
//        if (current_energy_mpi*initial_energy_mpi - initial_energy_mpi >= 1.00)
//        {
//          pcout << " Energy was not monotonically decreasing" << std::endl;
//          return 1;
//        }
        all_parameters_new.ode_solver_param.initial_time_step =  get_timestep(dg,poly_degree, delta_x);
        dt = all_parameters_new.ode_solver_param.initial_time_step;
       // ode_solver->current_iteration++;
//        if(i%10000==0){
//            const int file_number = i / all_parameters_new.ode_solver_param.output_solution_every_x_steps;
//            dg->output_results_vtk(file_number);
//        }
    }

    myfile.close();

    return 0;
}

template class EulerIsentropicVortex <PHILIP_DIM,PHILIP_DIM+2>;

} // Tests namespace
} // PHiLiP namespace
