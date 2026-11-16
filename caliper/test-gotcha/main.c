/* Calls MPI_Allreduce and MPI_Barrier only through libno_plt_comm. */
#include <caliper/cali.h>
#include <mpi.h>

double no_plt_sum(double x);

int main(int argc, char **argv)
{
   cali_init(); /* before MPI_Init, so the mpi service wraps it */
   MPI_Init(&argc, &argv);
   double s = 0.0;
   CALI_MARK_BEGIN("gotcha_no_plt");
   for (int i = 0; i < 10; ++i) { s += no_plt_sum(1.0); }
   CALI_MARK_END("gotcha_no_plt");
   MPI_Finalize();
   return s > 0.0 ? 0 : 1;
}
