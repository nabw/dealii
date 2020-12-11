/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *         Timo Heister, University of Goettingen, 2009, 2010
 */

// @sect3{Include files}
//
// Most of the include files we need for this program have already been
// discussed in previous programs. In particular, all of the following should
// already be familiar friends:
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// uncomment the following \#define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode

// This will either import PETSc or TrilinosWrappers into the namespace
// LA. Note that we are defining the macro USE_PETSC_LA so that we can detect
// if we are using PETSc (see solve() for an example where this is necessary)
namespace LA
{
  using namespace dealii::LinearAlgebraPETSc;
} // namespace LA

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <petscpc.h>

#include <fstream>
#include <iostream>

namespace BDDC
{
  using namespace dealii;

  // @sect3{The <code>LaplaceProblem</code> class template}

  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results() const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };

  // @sect3{The <code>LaplaceProblem</code> class implementation}

  // @sect4{Constructor}

  // Constructors and destructors are rather trivial. In addition to what we
  // do in step-6, we set the set of processors we want to work on to all
  // machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
  // the mesh remains smooth and free to refined islands, for example; and
  // initialize the <code>pcout</code> variable to only allow processor zero
  // to output anything. The final piece is to initialize a timer that we
  // use to determine how much compute time the different parts of the program
  // take:
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}

  // @sect4{LaplaceProblem::setup_system}

  // The following function is, arguably, the most interesting one in the
  // entire program since it goes to the heart of what distinguishes %parallel
  // step-40 from sequential step-6.
  //
  // At the top we do what we always do: tell the DoFHandler object to
  // distribute degrees of freedom. Since the triangulation we use here is
  // distributed, the DoFHandler object is smart enough to recognize that on
  // each processor it can only distribute degrees of freedom on cells it
  // owns; this is followed by an exchange step in which processors tell each
  // other about degrees of freedom on ghost cell. The result is a DoFHandler
  // that knows about the degrees of freedom on locally owned cells and ghost
  // cells (i.e. cells adjacent to locally owned cells) but nothing about
  // cells that are further away, consistent with the basic philosophy of
  // distributed computing that no processor can know everything.
  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    // system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
    //                      mpi_communicator);

    // First compute dofs from owned cells
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

    system_matrix.reinit_IS(locally_owned_dofs,
                            locally_active_dofs,
                            locally_owned_dofs,
                            locally_active_dofs,
                            dsp,
                            mpi_communicator);
  }

  // @sect4{LaplaceProblem::assemble_system}

  // The function that then assembles the linear system is comparatively
  // boring, being almost exactly what we've seen before. The points to watch
  // out for are:
  // - Assembly must only loop over locally owned cells. There
  //   are multiple ways to test that; for example, we could compare a cell's
  //   subdomain_id against information from the triangulation as in
  //   <code>cell->subdomain_id() ==
  //   triangulation.locally_owned_subdomain()</code>, or skip all cells for
  //   which the condition <code>cell->is_ghost() ||
  //   cell->is_artificial()</code> is true. The simplest way, however, is to
  //   simply ask the cell whether it is owned by the local processor.
  // - Copying local contributions into the global matrix must include
  //   distributing constraints and boundary values. In other words, we cannot
  //   (as we did in step-6) first copy every local contribution into the global
  //   matrix and only in a later step take care of hanging node constraints and
  //   boundary values. The reason is, as discussed in step-17, that the
  //   parallel vector classes do not provide access to arbitrary elements of
  //   the matrix once they have been assembled into it -- in parts because they
  //   may simply no longer reside on the current processor but have instead
  //   been shipped to a different machine.
  // - The way we compute the right hand side (given the
  //   formula stated in the introduction) may not be the most elegant but will
  //   do for a program whose focus lies somewhere entirely different.
  template <int dim>
  void LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    // COPY TEST, USE system_matrix AS MATIS ON ASSEMBLY
    // Mat &mat = system_matrix.petsc_matrix();
    // MatConvert(mat, MATIS, MAT_INPLACE_MATRIX, &mat);
    // pcout << "SETUP MATIS " << std::endl;
    // MatSetLocalToGlobalMapping(mat, l2gmap, l2gmap);
    // MatISSetPreallocation(mat, 100, NULL, 100, NULL);
    // MatSetUp(mat);
    // pcout << "END SETUP MATIS " << std::endl;

    // simo
    // MatZeroEntries(system_matrix_petsc);
    // VecSet(rhs_petsc, 0.0);

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

          fe_values.reinit(cell);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const double rhs_value =
                (fe_values.quadrature_point(q_point)[1] >
                     0.5 +
                       0.25 * std::sin(4.0 * numbers::PI *
                                       fe_values.quadrature_point(q_point)[0]) ?
                   1. :
                   -1.);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) *
                                            fe_values.shape_grad(j, q_point) *
                                            std::pow(cell->diameter(), 2) +
                                          fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point)) *
                                         fe_values.JxW(q_point);

                  cell_rhs(i) += rhs_value *                         //
                                 fe_values.shape_value(i, q_point) * //
                                 fe_values.JxW(q_point);
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);

          // simo
          // assemble res and jac petsc
          // for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          // lres_bddc[i] = cell_rhs[i];
          // for (unsigned int j = 0; j < dofs_per_cell; ++j)
          // ljac_bddc[i][j] = cell_matrix[i][j];
          // }
          // for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          //   int row = local_dof_indices[i];
          //   double value = cell_rhs[i];
          //   VecSetValues(rhs_petsc, 1, &row, &value, ADD_VALUES);
          //   for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          //     int col = local_dof_indices[j];
          //     double value = cell_matrix[i][j];
          //     MatSetValues(system_matrix_petsc, 1, &row, 1, &col, &value,
          //                  ADD_VALUES);
          //   }
          // }
        }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress "Compressing distributed objects"
    // for more information on what is compress() designed to do.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    // VecAssemblyBegin(rhs_petsc);
    // VecAssemblyEnd(rhs_petsc);

    // MatAssemblyBegin(system_matrix_petsc, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(system_matrix_petsc, MAT_FINAL_ASSEMBLY);

    // double mat_norm(0.0);
    // double mat_norm2(0.0);
    // Mat lA;

    // MatISGetLocalMat(system_matrix_petsc, &lA);
    // int ln, lm;
    // MatGetSize(lA, &lm, &ln);
    // pcout << "Sizes: " << ln << ", " << lm << std::endl;

    // // int idi, idj;
    // double val;
    // for (int i = 0; i < lm; ++i)
    //   for (int j = 0; j < ln; ++j) {
    //     MatGetValues(lA, 1, &i, 1, &j, &val);
    //     mat_norm += val * val;
    //   }
    // MatISRestoreLocalMat(system_matrix_petsc, &lA);
    // mat_norm = std::sqrt(mat_norm);

    // MatISGetLocalMat(mat, &lA);
    // for (int i = 0; i < lm; ++i)
    //   for (int j = 0; j < ln; ++j) {
    //     MatGetValues(lA, 1, &i, 1, &j, &val);
    //     mat_norm2 += val * val;
    //   }
    // MatISRestoreLocalMat(mat, &lA);
    // mat_norm2 = std::sqrt(mat_norm2);

    // std::cout << "NORMS TEST " << mat_norm2 << ", " << mat_norm << std::endl;

  } // namespace BDDC

  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    // cpu_t1 = MPI_Wtime();
    LA::SolverCG solver(solver_control, mpi_communicator);
    // LA::MPI::PreconditionAMG preconditioner;

    dealii::PETScWrappers::PreconditionBDDC preconditioner;

    // LA::MPI::PreconditionAMG::AdditionalData data;
    dealii::PETScWrappers::PreconditionBDDC::AdditionalData data;

    preconditioner.initialize(system_matrix, data);

    // solver.initialize(preconditioner);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    // constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;

    // double sol_norm;
    // VecNorm(solution_petsc, NORM_2, &sol_norm);
    // pcout << "Norms: deal.II=" << locally_relevant_solution.l2_norm()
    // << ", PETSc=" << sol_norm << std::endl;
    pcout << "Norms: deal.II=" << locally_relevant_solution.l2_norm()
          << std::endl;
  }

  // @sect4{LaplaceProblem::refine_grid}

  // The function that estimates the error and refines the grid is again
  // almost exactly like the one in step-6. The only difference is that the
  // function that flags cells to be refined is now in namespace
  // parallel::distributed::GridRefinement -- a namespace that has functions
  // that can communicate between all involved processors and determine global
  // thresholds to use in deciding which cells to refine and which to coarsen.
  //
  // Note that we didn't have to do anything special about the
  // KellyErrorEstimator class: we just give it a vector with as many elements
  // as the local triangulation has cells (locally owned cells, ghost cells,
  // and artificial ones), but it only fills those entries that correspond to
  // cells that are locally owned.
  template <int dim>
  void LaplaceProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      locally_relevant_solution,
      estimated_error_per_cell);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
  }

  // @sect4{LaplaceProblem::output_results}
  template <int dim>
  void LaplaceProblem<dim>::output_results() const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "u");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    // The next step is to write this data to disk. We write up to 8 VTU files
    // in parallel with the help of MPI-IO. Additionally a PVTU record is
    // generated, which groups the written VTU files.
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", 0, mpi_communicator, 2, 8);
  }

  // @sect4{LaplaceProblem::run}
  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    pcout << "Running"
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;



    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);


    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_system();
    solve();

    if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
      {
        TimerOutput::Scope t(computing_timer, "output");
        output_results();
      }

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }
} // namespace BDDC

// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like the other programs that use
// MPI, we have to initialize and finalize MPI, which is done using the helper
// object Utilities::MPI::MPI_InitFinalize. The constructor of that class also
// initializes libraries that depend on MPI, such as p4est, PETSc, SLEPc, and
// Zoltan (though the last two are not used in this tutorial). The order here
// is important: we cannot use any of these libraries until they are
// initialized, so it does not make sense to do anything before creating an
// instance of Utilities::MPI::MPI_InitFinalize.
//
// After the solver finishes, the LaplaceProblem destructor will run followed
// by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize(). This order is
// also important: Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() calls
// <code>PetscFinalize</code> (and finalization functions for other
// libraries), which will delete any in-use PETSc objects. This must be done
// after we destruct the Laplace solver to avoid double deletion
// errors. Fortunately, due to the order of destructor call rules of C++, we
// do not need to worry about any of this: everything happens in the correct
// order (i.e., the reverse of the order of construction). The last function
// called by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() is
// <code>MPI_Finalize</code>: i.e., once this object is destructed the program
// should exit since MPI will no longer be available.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace BDDC;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<3> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
