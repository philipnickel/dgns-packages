#!/usr/bin/env bash
# Build stock MFEM 4.10 (upstream tarball, no source patches) into $PREFIX
# against conda-forge's hypre, METIS, zlib, BLAS/LAPACK, parallel HDF5 and PETSc and
# this directory's sibling packages caliper and gslib. Feature set:
#   mfem@4.10.0 +mpi +metis +zlib +lapack +gslib +caliper +hdf5 ~mpfr
#               +petsc ~threadsafe ~libceed ~openmp cxxstd=17 (hypre, CPU only)
# The hypre_int=mixed variant (recipe.yaml) builds against hypre-dgns
# (64-bit HYPRE_BigInt) instead, without PETSc (~petsc).
set -euxo pipefail

# ---- hypre index width -----------------------------------------------------
# MFEM takes HYPRE_BigInt from hypre's HYPRE_config.h; nothing to configure
# beyond leaving out conda-forge's PETSc, which links the 32-bit hypre.
case "${MFEM_HYPRE_INT:-32}" in
  32)
    PETSC_ARGS=(-DMFEM_USE_PETSC=ON -DPETSC_DIR="$PREFIX" -DPETSC_ARCH=)
    ;;
  mixed)
    grep -q '^#define HYPRE_MIXEDINT 1' "$PREFIX/include/HYPRE_config.h"
    PETSC_ARGS=(-DMFEM_USE_PETSC=OFF)
    ;;
  *)
    echo "unknown MFEM_HYPRE_INT=${MFEM_HYPRE_INT}" >&2
    exit 1
    ;;
esac

# ---- Compiler flags --------------------------------------------------------
# Conda-forge's CXXFLAGS carry -O2, which comes after CMake's Release -O3 in
# config.mk; append -O3 so the partial-assembly kernels are optimized.
# The CPU baseline: on linux-64 the x86_64-microarch-level build dependency
# appends -march=x86-64-v3; on osx-arm64 Apple M1.
# Conda-forge's CXXFLAGS also carry -fvisibility-inlines-hidden, which hides
# the function-local static of every inline Kernels::Get() (the PA and
# quadrature-interpolator dispatch tables, fem/kernel_dispatch.hpp). A
# consumer's AddSpecialization calls then fill a private copy of the table
# that libmfem never reads. Export inline functions so the tables coalesce.
export CXXFLAGS="${CXXFLAGS:-} -fno-visibility-inlines-hidden -O3"
if [[ "${target_platform}" == osx-arm64 ]]; then
  export CXXFLAGS="${CXXFLAGS} -mcpu=apple-m1"
fi

# ---- MFEM ------------------------------------------------------------------
cmake -S . -B build -G Ninja \
  ${CMAKE_ARGS} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_SHARED_LIBS=ON \
  -DMPI_C_COMPILER="$PREFIX/bin/mpicc" \
  -DMPI_CXX_COMPILER="$PREFIX/bin/mpicxx" \
  -DMFEM_USE_MPI=ON \
  -DHYPRE_DIR="$PREFIX" \
  -DMFEM_USE_METIS=ON \
  -DMFEM_USE_METIS_5=ON \
  -DMETIS_DIR="$PREFIX" \
  -DMFEM_USE_ZLIB=ON \
  -DZLIB_DIR="$PREFIX" \
  -DMFEM_USE_LAPACK=ON \
  -DBLAS_LIBRARIES="$PREFIX/lib/libblas${SHLIB_EXT}" \
  -DLAPACK_LIBRARIES="$PREFIX/lib/liblapack${SHLIB_EXT}" \
  -DMFEM_USE_GSLIB=ON \
  -DGSLIB_DIR="$PREFIX" \
  -DMFEM_USE_CALIPER=ON \
  -DCALIPER_DIR="$PREFIX" \
  -DMFEM_USE_HDF5=ON \
  -DHDF5_DIR="$PREFIX" \
  "${PETSC_ARGS[@]}" \
  -DMFEM_USE_MPFR=OFF \
  -DMFEM_THREAD_SAFE=OFF \
  -DMFEM_USE_OPENMP=OFF \
  -DMFEM_USE_CEED=OFF \
  -DMFEM_USE_CUDA=OFF \
  -DMFEM_USE_EXCEPTIONS=OFF \
  -DMFEM_ENABLE_TESTING=OFF \
  -DMFEM_ENABLE_EXAMPLES=OFF \
  -DMFEM_ENABLE_MINIAPPS=OFF \
  || {
    # A failed configure: show FindPETSc's try_compile/try_run attempts.
    log=build/CMakeFiles/CMakeConfigureLog.yaml
    [[ -f "$log" ]] && grep -n -A60 'petsc_works' "$log" | tail -n 400
    exit 1
  }

cmake --build build --parallel "${CPU_COUNT}"
cmake --install build

# The build-environment compiler path recorded by MFEMConfig.cmake does not
# exist in a consumer environment; point it at the packaged MPI wrapper.
sed -i.bak -E \
  "s|^set\\(MFEM_CXX_COMPILER \".*\"\\)|set(MFEM_CXX_COMPILER \"$PREFIX/bin/mpicxx\")|" \
  "$PREFIX/lib/cmake/mfem/MFEMConfig.cmake"
rm -f "$PREFIX/lib/cmake/mfem/MFEMConfig.cmake.bak"
