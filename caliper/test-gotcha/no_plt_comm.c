/* MPI calls from a library built with -fno-plt (CMakeLists.txt). */
#include <mpi.h>

double no_plt_sum(double x)
{
   double s = 0.0;
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Allreduce(&x, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return s;
}
