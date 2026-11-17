// Poisson on the unit square through MFEM's hypre path: the assembled
// HypreParMatrix solved by HyprePCG with HypreBoomerAMG. Also checks the
// index widths MFEM was built with against the expected hypre: 32-bit
// HYPRE_BigInt (conda-forge's hypre) or 64-bit (hypre-dgns, mixed-int).
#include "mfem.hpp"
#include <iostream>

#ifdef EXPECT_HYPRE_MIXEDINT
#ifndef HYPRE_MIXEDINT
#error "expected a mixed-int hypre (HYPRE_MIXEDINT)"
#endif
static_assert(sizeof(HYPRE_BigInt) == 8, "mixed-int: 64-bit HYPRE_BigInt");
#else
#if defined(HYPRE_MIXEDINT) || defined(HYPRE_BIGINT)
#error "expected a 32-bit hypre"
#endif
static_assert(sizeof(HYPRE_BigInt) == 4, "32-bit HYPRE_BigInt");
#endif
static_assert(sizeof(HYPRE_Int) == 4, "32-bit HYPRE_Int");

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   const int rank = mfem::Mpi::WorldRank();

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

   mfem::HypreBoomerAMG amg(A);
   amg.SetPrintLevel(0);
   mfem::HyprePCG pcg(A);
   pcg.SetTol(1e-10);
   pcg.SetMaxIter(100);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(B, X);

   int iters = 0;
   pcg.GetNumIterations(iters);
   HYPRE_Real res = 1.0;
   HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(pcg, &res);
   const bool converged = res < 1e-9;
   if (rank == 0)
   {
      std::cout << "hypre " << HYPRE_RELEASE_VERSION << " (HYPRE_BigInt "
                << 8 * sizeof(HYPRE_BigInt) << "-bit, HYPRE_Int "
                << 8 * sizeof(HYPRE_Int) << "-bit): PCG + BoomerAMG on "
                << mfem::Mpi::WorldSize() << " ranks, " << fes.GlobalTrueVSize()
                << " dofs: " << (converged ? "converged" : "NOT converged")
                << " in " << iters << " iterations, relative residual " << res
                << std::endl;
   }
   return converged ? 0 : 1;
}
