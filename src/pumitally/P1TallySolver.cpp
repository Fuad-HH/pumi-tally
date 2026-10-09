/**
 * @brief P1TallySolver Implementations
 */

// PETSc version and Kokkos support are checked in CMake
#include <petscksp.h>

#include <Kokkos_Core.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_shape.hpp>

#include <cstdio>
#include <type_traits>

#include "P1TallySolver.h"

namespace pumitally {

// Omega_h arrays are handed to PETSc without a copy
static_assert(std::is_same_v<PetscScalar, Omega_h::Real>,
              "PETSc must be built with real double precision scalars");

namespace {
constexpr PetscReal kP1SolveRelTol = 1e-12;
constexpr PetscReal kP1SolveAbsTol = 1e-50;
constexpr int kTetVerts = 4;
constexpr int kTetBlock = kTetVerts * kTetVerts;

using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

// analytic consistent P1 tet mass matrix: M_e = V_e / 20 * (1 + delta_ij)
// repeated (row, col) pairs are summed by PETSc's COO assembly
// (a free function since CUDA extended lambdas cannot live in constructors)
void FillP1MassCoo(const Omega_h::LOs &elem2verts, const Omega_h::Reals &coords,
                   const Omega_h::LO nelems,
                   const Kokkos::View<PetscInt *, DeviceMemSpace> &coo_i,
                   const Kokkos::View<PetscInt *, DeviceMemSpace> &coo_j,
                   const Kokkos::View<PetscScalar *, DeviceMemSpace> &coo_v) {
  auto fill_coo = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    const auto elem_verts = Omega_h::gather_verts<kTetVerts>(elem2verts, elem_id);
    const auto elem_vert_coords =
        Omega_h::gather_vectors<kTetVerts, 3>(coords, elem_verts);
    const auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
    const Omega_h::Real volume =
        Kokkos::fabs(Omega_h::simplex_size_from_basis(b));

    for (int i = 0; i < kTetVerts; ++i) {
      for (int j = 0; j < kTetVerts; ++j) {
        const auto idx = static_cast<PetscCount>(elem_id) * kTetBlock +
                         i * kTetVerts + j;
        coo_i(idx) = static_cast<PetscInt>(elem_verts[i]);
        coo_j(idx) = static_cast<PetscInt>(elem_verts[j]);
        coo_v(idx) = volume / 20.0 * ((i == j) ? 2.0 : 1.0);
      }
    }
  };
  Omega_h::parallel_for(nelems, fill_coo, "fill P1 mass matrix COO");
}
} // namespace

P1TallySolver::P1TallySolver(Omega_h::Mesh &mesh)
    : num_vertices_(static_cast<PetscInt>(mesh.nverts())) {
  OMEGA_H_CHECK_PRINTF(mesh.dim() == 3,
                       "P1 tally only supports 3D tet meshes, got dim %d\n",
                       mesh.dim());

  const Omega_h::LO nelems = mesh.nelems();
  const PetscCount nnz = static_cast<PetscCount>(nelems) * kTetBlock;

  Kokkos::View<PetscInt *, DeviceMemSpace> coo_i("p1_mass_coo_i", nnz);
  Kokkos::View<PetscInt *, DeviceMemSpace> coo_j("p1_mass_coo_j", nnz);
  Kokkos::View<PetscScalar *, DeviceMemSpace> coo_v("p1_mass_coo_v", nnz);
  FillP1MassCoo(mesh.ask_elem_verts(), mesh.coords(), nelems, coo_i, coo_j,
                coo_v);
  Kokkos::fence();

  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateSeqAIJKokkos(PETSC_COMM_SELF, num_vertices_,
                                       num_vertices_, 0, nullptr,
                                       &mass_matrix_));
  // the sparsity pattern is set once, so a host copy of the indices is fine
  // TODO Resolve this host copy? Why required?
  const auto coo_i_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coo_i);
  const auto coo_j_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coo_j);
  PetscCallAbort(PETSC_COMM_SELF,
                 MatSetPreallocationCOO(mass_matrix_, nnz, coo_i_host.data(),
                                        coo_j_host.data()));
  // values are passed as a device pointer; PETSc detects the memory type
  PetscCallAbort(PETSC_COMM_SELF, MatSetValuesCOO(mass_matrix_, coo_v.data(),
                                                  INSERT_VALUES));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatSetOption(mass_matrix_, MAT_SPD, PETSC_TRUE));

  // M is SPD and well conditioned after Jacobi scaling: CG + Jacobi converges
  // in a small, mesh size independent number of iterations
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp_));
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPSetOperators(ksp_, mass_matrix_, mass_matrix_));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp_, KSPCG));
  PC pc = nullptr;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp_, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));
  PetscCallAbort(PETSC_COMM_SELF,
                 KSPSetTolerances(ksp_, kP1SolveRelTol, kP1SolveAbsTol,
                                  PETSC_CURRENT, PETSC_CURRENT));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetFromOptions(ksp_));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetUp(ksp_));

  printf("[INFO] P1 tally mass matrix assembled with %d vertices and %d "
         "elements\n",
         mesh.nverts(), nelems);
}

P1TallySolver::~P1TallySolver() {
  if (ksp_ != nullptr) {
    KSPDestroy(&ksp_);
  }
  if (mass_matrix_ != nullptr) {
    MatDestroy(&mass_matrix_);
  }
}

Omega_h::Reals P1TallySolver::Solve(const Omega_h::Reals &rhs) const {
  OMEGA_H_CHECK_PRINTF(rhs.size() == num_vertices_,
                       "P1 rhs size %d does not match %d mesh vertices\n",
                       rhs.size(), static_cast<int>(num_vertices_));

  const Omega_h::Write<Omega_h::Real> phi(rhs.size(), 0.0, "p1_phi");
  // the Omega_h kernels producing rhs must be done before PETSc reads it
  Kokkos::fence();

  // wrap the device arrays without copying them; PETSc does not free them
  Vec rhs_vec = nullptr;
  Vec phi_vec = nullptr;
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, num_vertices_,
                                             rhs.data(), &rhs_vec));
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, num_vertices_,
                                             phi.data(), &phi_vec));

  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp_, rhs_vec, phi_vec));

  KSPConvergedReason reason;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp_, &reason));
  if (reason < 0) {
    PetscInt iterations = 0;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp_, &iterations));
    fprintf(stderr,
            "[WARNING] P1 tally mass matrix solve did not converge (%s) after "
            "%d iterations\n",
            KSPConvergedReasons[reason], static_cast<int>(iterations));
  }

  // make sure the up to date solution is in phi's device memory
  const PetscScalar *phi_ptr = nullptr;
  PetscMemType mem_type;
  PetscCallAbort(PETSC_COMM_SELF,
                 VecGetArrayReadAndMemType(phi_vec, &phi_ptr, &mem_type));
  if (phi_ptr != phi.data()) {
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    const Kokkos::View<Omega_h::Real *, Kokkos::DefaultExecutionSpace, Unmanaged>
        phi_view(phi.data(), phi.size());
    if (PetscMemTypeHost(mem_type)) {
      const Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Unmanaged>
          src(phi_ptr, phi.size());
      Kokkos::deep_copy(phi_view, src);
    } else {
      const Kokkos::View<const PetscScalar *, Kokkos::DefaultExecutionSpace,
                         Unmanaged>
          src(phi_ptr, phi.size());
      Kokkos::deep_copy(phi_view, src);
    }
  }
  PetscCallAbort(PETSC_COMM_SELF,
                 VecRestoreArrayReadAndMemType(phi_vec, &phi_ptr));

  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&rhs_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&phi_vec));
  // PETSc may run on its own execution space instance
  Kokkos::fence();

  return {phi};
}

} // namespace pumitally
