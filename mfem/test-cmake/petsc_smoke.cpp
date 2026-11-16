// Poisson on the unit square through MFEM's PETSc wrappers: the assembled
// HypreParMatrix as a PETSc AIJ matrix, solved by CG with GAMG (PETSc's own
// AMG), so the package's PETSc (with hypre, MPI) works from an MFEM consumer.
#include "mfem.hpp"
// mfem.hpp includes only petscconf.h; KSP/PC and the options API come from here.
#include <petscksp.h>
#include <cstring>
#include <iostream>

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::MFEMInitializePetsc(&argc, &argv, nullptr, nullptr);
   const int rank = mfem::Mpi::WorldRank();
   int status = 1;
   {
      PetscOptionsSetValue(nullptr, "-ksp_type", "cg");
      PetscOptionsSetValue(nullptr, "-pc_type", "gamg");

      mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
         16, 16, mfem::Element::QUADRILATERAL);
      mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
      serial.Clear();
      mfem::H1_FECollection fec(2, mesh.Dimension());
      mfem::ParFiniteElementSpace fes(&mesh, &fec);

      mfem::Array<int> ess_tdofs;
      fes.GetBoundaryTrueDofs(ess_tdofs);
      mfem::ConstantCoefficient one(1.0);
      mfem::ParLinearForm b(&fes);
      b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
      b.Assemble();
      mfem::ParGridFunction x(&fes);
      x = 0.0;
      mfem::ParBilinearForm a(&fes);
      a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
      a.Assemble();

      mfem::HypreParMatrix A;
      mfem::Vector B, X;
      a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

      mfem::PetscParMatrix pA(&A, mfem::Operator::PETSC_MATAIJ);
      mfem::PetscLinearSolver solver(MPI_COMM_WORLD);
      solver.SetOperator(pA);
      solver.SetRelTol(1e-10);
      solver.SetMaxIter(100);
      solver.Mult(B, X);

      PC pc;
      PCType pc_type;
      KSPGetPC(KSP(solver), &pc);
      PCGetType(pc, &pc_type);
      const bool gamg = std::strcmp(pc_type, "gamg") == 0;
      const bool converged = solver.GetConverged();
      if (rank == 0)
      {
         std::cout << "PETSc " << PETSC_VERSION_MAJOR << '.' << PETSC_VERSION_MINOR
                   << " CG + " << pc_type << " on " << mfem::Mpi::WorldSize()
                   << " ranks, " << fes.GlobalTrueVSize() << " dofs: "
                   << (converged ? "converged" : "NOT converged") << " in "
                   << solver.GetNumIterations() << " iterations" << std::endl;
      }
      status = (gamg && converged) ? 0 : 1;
   }
   mfem::MFEMFinalizePetsc();
   return status;
}
