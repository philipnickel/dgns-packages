cmake_minimum_required(VERSION 3.20)
project(hypre_dgns_test LANGUAGES C)

find_package(MPI REQUIRED COMPONENTS C)
find_package(HYPRE 3.2 REQUIRED CONFIG)

add_executable(bigint_rows bigint_rows.c)
target_link_libraries(bigint_rows PRIVATE HYPRE::HYPRE MPI::MPI_C)
