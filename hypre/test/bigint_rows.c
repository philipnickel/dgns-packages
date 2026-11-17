/* A 1D Laplacian (Dirichlet ends) on 2 x 1000 rows whose global numbers
   straddle 2^31: rank 0 owns 2^31 - 1000 ... 2^31 - 1, rank 1 owns
   2^31 ... 2^31 + 999. Built through the IJ interface, solved by PCG with
   BoomerAMG. With a 32-bit HYPRE_BigInt the row numbers cannot be stored. */
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   HYPRE_Initialize();

   int status = 1;
   if (sizeof(HYPRE_BigInt) != 8 || sizeof(HYPRE_Int) != 4)
   {
      if (rank == 0)
      {
         printf("expected mixed-int: sizeof(HYPRE_BigInt) = %zu, sizeof(HYPRE_Int) = %zu\n",
                sizeof(HYPRE_BigInt), sizeof(HYPRE_Int));
      }
      HYPRE_Finalize();
      MPI_Finalize();
      return 1;
   }

   const HYPRE_BigInt n_local = 1000;
   const HYPRE_BigInt first = ((HYPRE_BigInt)1 << 31) - n_local;
   const HYPRE_BigInt last = first + n_local * size - 1;
   const HYPRE_BigInt ilower = first + n_local * rank;
   const HYPRE_BigInt iupper = ilower + n_local - 1;

   HYPRE_IJMatrix A;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(A);
   for (HYPRE_BigInt i = ilower; i <= iupper; i++)
   {
      HYPRE_BigInt cols[3];
      HYPRE_Real vals[3];
      HYPRE_Int ncols = 0;
      if (i > first) { cols[ncols] = i - 1; vals[ncols++] = -1.0; }
      cols[ncols] = i; vals[ncols++] = 2.0;
      if (i < last) { cols[ncols] = i + 1; vals[ncols++] = -1.0; }
      HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
   }
   HYPRE_IJMatrixAssemble(A);
   HYPRE_ParCSRMatrix parA;
   HYPRE_IJMatrixGetObject(A, (void **)&parA);

   HYPRE_IJVector b, x;
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorInitialize(x);
   for (HYPRE_BigInt i = ilower; i <= iupper; i++)
   {
      HYPRE_Real one = 1.0, zero = 0.0;
      HYPRE_IJVectorSetValues(b, 1, &i, &one);
      HYPRE_IJVectorSetValues(x, 1, &i, &zero);
   }
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorAssemble(x);
   HYPRE_ParVector parb, parx;
   HYPRE_IJVectorGetObject(b, (void **)&parb);
   HYPRE_IJVectorGetObject(x, (void **)&parx);

   HYPRE_BigInt global_rows, global_cols;
   HYPRE_ParCSRMatrixGetDims(parA, &global_rows, &global_cols);

   HYPRE_Solver amg, pcg;
   HYPRE_BoomerAMGCreate(&amg);
   HYPRE_BoomerAMGSetMaxIter(amg, 1);
   HYPRE_BoomerAMGSetTol(amg, 0.0);
   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg);
   HYPRE_PCGSetTol(pcg, 1e-10);
   HYPRE_PCGSetMaxIter(pcg, 200);
   HYPRE_PCGSetTwoNorm(pcg, 1);
   HYPRE_PCGSetPrecond(pcg, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                       (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);
   HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx);
   HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx);

   HYPRE_Int iters;
   HYPRE_Real res;
   HYPRE_PCGGetNumIterations(pcg, &iters);
   HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &res);
   const int converged = res < 1e-9;
   if (rank == 0)
   {
      printf("hypre mixed-int: rows %lld..%lld (%lld global), PCG + BoomerAMG "
             "on %d ranks: %s in %d iterations, relative residual %.3e\n",
             (long long)first, (long long)last, (long long)global_rows, size,
             converged ? "converged" : "NOT converged", (int)iters, res);
   }
   status = (converged && global_rows == n_local * size) ? 0 : 1;

   HYPRE_ParCSRPCGDestroy(pcg);
   HYPRE_BoomerAMGDestroy(amg);
   HYPRE_IJVectorDestroy(x);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(A);
   HYPRE_Finalize();
   MPI_Finalize();
   return status;
}
