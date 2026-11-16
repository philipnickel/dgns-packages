# Caliper's mpi service (GOTCHA on linux) must see MPI calls made from a
# library linked without a PLT, as conda-forge builds hypre and mfem-dgns
# (-fno-plt, -z now): every MPI call goes through a GLOB_DAT GOT entry and
# the library has no DT_PLTREL/DT_JMPREL (gotcha-no-plt-rela.patch).
cmake_minimum_required(VERSION 3.20)
project(caliper_gotcha_consumer LANGUAGES C)

find_package(MPI REQUIRED COMPONENTS C)
find_package(caliper REQUIRED CONFIG)

# The library under test: no PLT at all.
add_library(no_plt_comm SHARED no_plt_comm.c)
target_link_libraries(no_plt_comm PRIVATE MPI::MPI_C)
target_compile_options(no_plt_comm PRIVATE -fno-plt)
target_link_options(no_plt_comm PRIVATE -Wl,-z,now -Wl,-z,relro)

# The executable keeps a PLT (-fplt, as the solver builds), so Caliper sees
# its MPI_Init/MPI_Finalize; it makes no other MPI call itself.
add_executable(gotcha_no_plt main.c)
target_link_libraries(gotcha_no_plt PRIVATE no_plt_comm caliper MPI::MPI_C)
target_compile_options(gotcha_no_plt PRIVATE -fplt)
