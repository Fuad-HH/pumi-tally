//
// Consistent P1 mass matrix solver for the nodal (P1) flux tally
//

#ifndef PUMITALLY_P1TALLYSOLVER_H
#define PUMITALLY_P1TALLYSOLVER_H

#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
#include <petscksp.h>

namespace pumitally {

/**
 * @brief Cached Galerkin solver for `M * phi = b` on a tetrahedral mesh
 * @details
 * The consistent P1 mass matrix `M_ij = int(N_i N_j dV)` is assembled once at
 * construction from the analytic tetrahedral local matrix
 * `M_e = V_e / 20 * (1 + delta_ij)` into a sequential PETSc Kokkos AIJ matrix,
 * and a CG + Jacobi KSP is set up on it. Every @ref Solve then only calls
 * `KSPSolve` with the cached operator and preconditioner. `M^{-1}` is never
 * formed. The solver can be changed from the command line through
 * `KSPSetFromOptions` (e.g. `-ksp_type preonly -pc_type cholesky`).
 *
 * Everything lives on `PETSC_COMM_SELF`. Omega_h arrays are handed to PETSc
 * directly (no host copies), which requires PETSc to be built with Kokkos on
 * the same backend as Omega_h.
 *
 * Non-copyable since it owns PETSc handles.
 */
class P1TallySolver {
public:
  /**
   * Assemble the mass matrix and set up the KSP
   * @param mesh 3D tetrahedral Omega_h mesh the P1 field lives on
   */
  explicit P1TallySolver(Omega_h::Mesh &mesh);
  ~P1TallySolver();

  P1TallySolver(const P1TallySolver &) = delete;
  P1TallySolver &operator=(const P1TallySolver &) = delete;

  /**
   * Solve `M * phi = rhs`
   * @param rhs Assembled load vector, one value per mesh vertex
   * @return Nodal coefficients `phi`, one value per mesh vertex
   */
  Omega_h::Reals Solve(const Omega_h::Reals &rhs) const;

private:
  Mat mass_matrix_ = nullptr; //!< Consistent P1 mass matrix
  KSP ksp_ = nullptr;         //!< Cached solver/preconditioner
  PetscInt num_vertices_ = 0; //!< Number of DOFs (mesh vertices)
};

} // namespace pumitally

#endif // PUMITALLY_P1TALLYSOLVER_H
