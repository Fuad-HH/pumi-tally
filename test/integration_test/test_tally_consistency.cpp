//
// Created by Fuad Hasan on 6/30/26.
//
// Integration test: verify that element-tally and node-tally totals match.
//
// Mesh: test/assets/tet6-222.osh  (6-tet Omega_h mesh, 3D)
// Particles: 4, same initial positions and displacement as
//            test_new_api_integration, but moved in a single call
//            ("at once") rather than sequentially.
//
// Check:  sum(node_tally) == sum(element_tally)  (and neither is zero).
//

#include <Omega_h_library.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include <PumiTallyImpl.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kNumPtcls = 4;
constexpr int kNelemsExpected = 6; // tet6 mesh
constexpr int kMeshDim = 3;

#ifdef TEST_MESH_FILE
const std::string kMeshFilename = TEST_MESH_FILE;
#else
const std::string kMeshFilename = "test/assets/tet6-222.osh";
#endif

// ---------------------------------------------------------------------------
// Same initial positions as test_new_api_integration.
// ---------------------------------------------------------------------------
// clang-format off
std::vector<double> kInitPos = {
    0.60, 0.60, 0.60, // P0
    1.40, 0.60, 0.60, // P1
    0.60, 1.40, 0.60, // P2
    0.60, 0.60, 1.40  // P3
};
// clang-format on

// ---------------------------------------------------------------------------
// Destination after a single (+0.10, +0.10, +0.10) move — same as Step 4
// of test_new_api_integration, but performed "at once" with both element
// and node tallies registered.
// ---------------------------------------------------------------------------
// clang-format off
std::vector<double> kDest = {
    0.70, 0.70, 0.70, // P0
    1.50, 0.70, 0.70, // P1
    0.70, 1.50, 0.70, // P2
    0.70, 0.70, 1.50  // P3
};
// clang-format on

// ===========================================================================
// Test: single-move element vs. node tally consistency
// ===========================================================================
TEST_CASE("Element and node tally totals are equal after a single move",
          "[integration][tally_consistency]") {

  auto lib = Omega_h::Library{};
  int argc = 0;
  char **argv = nullptr;

  printf("\n===== Tally Consistency Test =====\n");

  auto pumi_tally = std::make_unique<pumitally::PumiTallyImpl>(
      kMeshFilename, kNumPtcls, argc, argv);

  // --- Sanity: mesh and particle counts ---
  REQUIRE(pumi_tally->full_mesh.nelems() == kNelemsExpected);
  REQUIRE(pumi_tally->full_mesh.dim() == kMeshDim);
  REQUIRE(pumi_tally->num_particles == kNumPtcls);
  REQUIRE(pumi_tally->pumipic_ptcls->nPtcls() == kNumPtcls);

  // --- Register both element and node tallies (single bin each) ---
  pumi_tally->AddElementTally({1});
  pumi_tally->AddNodeTally({1});

  // All particles in bin 0
  std::vector<unsigned int> bins(kNumPtcls, 0);
  pumi_tally->UpdateFilterBins(bins);

  // --- Load initial positions ---
  pumi_tally->CopyInitialPositionToBuffer(
      kInitPos.data(), static_cast<Omega_h::LO>(kInitPos.size()));

  // --- Move all particles at once: init → dest ---
  std::vector<int8_t> flying(kNumPtcls, 1);
  std::vector<double> weights(kNumPtcls, 1.0);

  pumi_tally->MoveToNextLocation(
      kInitPos.data(), kDest.data(), flying.data(), weights.data(),
      static_cast<int64_t>(kDest.size()));

  // --- Read back element and node tally totals ---
  const auto &h = pumi_tally->p_pumi_particle_at_elem_boundary_handler;

  auto elem_tallies_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          h->element_tallies);
  auto node_tallies_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                          h->node_tallies);

  double total_elem = 0.0;
  for (size_t i = 0; i < elem_tallies_host.size(); ++i)
    total_elem += elem_tallies_host(i);

  double total_node = 0.0;
  for (size_t i = 0; i < node_tallies_host.size(); ++i)
    total_node += node_tallies_host(i);

  printf("[INFO] Element tally total: %.10f\n", total_elem);
  printf("[INFO] Node    tally total: %.10f\n", total_node);
  printf("[INFO] Expected  ~%.6f  (4 × √(3×0.1²))\n",
         4.0 * std::sqrt(0.03));

  // --- Guard: totals must be non-zero ---
  REQUIRE(total_elem > 0.0);
  REQUIRE(total_node > 0.0);

  // --- Main check: element and node tally totals must match ---
  REQUIRE_THAT(total_elem,
               Catch::Matchers::WithinAbs(total_node, 1e-8));

  printf("[PASS] Tally consistency: element total == node total "
         "(%.10f)\n", total_elem);
  printf("===== Tally Consistency Test PASSED =====\n");
}
