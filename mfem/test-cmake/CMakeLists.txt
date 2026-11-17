cmake_minimum_required(VERSION 3.20)
project(mfem_dgns_consumer LANGUAGES CXX)

find_package(MFEM 4.10 REQUIRED CONFIG)

# The hypre_int=mixed build (64-bit HYPRE_BigInt) has no PETSc.
option(EXPECT_HYPRE_MIXEDINT "hypre_smoke expects a mixed-int hypre" OFF)
set(smokes parmesh_smoke hdf5_smoke hypre_smoke)
if(MFEM_USE_PETSC)
  list(APPEND smokes petsc_smoke)
endif()

foreach(smoke ${smokes})
  add_executable(${smoke} ${smoke}.cpp)
  target_include_directories(${smoke} PRIVATE ${MFEM_INCLUDE_DIRS})
  target_link_libraries(${smoke} PRIVATE ${MFEM_LIBRARIES})
endforeach()
if(EXPECT_HYPRE_MIXEDINT)
  target_compile_definitions(hypre_smoke PRIVATE EXPECT_HYPRE_MIXEDINT)
endif()
