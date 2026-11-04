/**
 * @brief  PumiTallyImpl Implementations
 */

#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <ParticleTracer.tpp>
#include <pumipic_adjacency.tpp>
#include <pumipic_library.hpp>
#include <pumipic_mesh.hpp>
#include <pumipic_ptcl_ops.hpp>

#include "P1TallySolver.h"
#include "PumiTallyImpl.h"

namespace pumitally {
std::unique_ptr<PPPS> CreateParticleDS(const Omega_h::Mesh &mesh,
                                       pumipic::lid_t num_ptcls);

void InitializeParticlesInElement0(Omega_h::Mesh &mesh, pumitally::PPPS *ptcls);

void TallyTimes::PrintTimes() const {
  printf("\n");
  printf("[TIME] Initialization time     : %f seconds\n", initialization_time);
  printf("[TIME] Total time to tally     : %f seconds\n", total_time_to_tally);
  printf("[TIME] VTK file write time     : %f seconds\n", vtk_file_write_time);
  printf("[TIME] Total PUMI-Tally time   : %f seconds\n",
         initialization_time + total_time_to_tally + vtk_file_write_time);
}

PumiTallyImpl::PumiTallyImpl(const std::string &mesh_filename,
                             const Omega_h::LO num_ptcls, int argc, char **argv,
                             const Estimator estimator)
    : num_particles(num_ptcls), estimator(estimator) {
  oh_mesh_filename = mesh_filename;

  position_dev_buffer = Omega_h::Write<Omega_h::Real>(num_particles * 3, 0.0,
                                                      "device_pos_buffer");
  flying_dev_buffer =
      Omega_h::Write<Omega_h::I8>(num_particles, 0, "device_in_adv_que");
  weights_dev_buffer =
      Omega_h::Write<Omega_h::Real>(num_particles, 0.0, "weights");

  // todo can track lengths be here?

  LoadMeshAndInitParticles(argc, argv);
  InitializeParticlesInElement0(*p_picparts->mesh(), pumipic_ptcls.get());

  p_particle_tracer = std::make_unique<
      ParticleTracer<PPParticle, pumitally::ParticleAtElemBoundary>>(
      *p_picparts, pumipic_ptcls.get(),
      *p_pumi_particle_at_elem_boundary_handler, 1e-8);
}

PumiTallyImpl::~PumiTallyImpl() {
  // PETSc objects held by the handler must go before PETSc is finalized, and
  // PETSc must be finalized before pumipic_lib finalizes MPI
  p_particle_tracer.reset();
  p_pumi_particle_at_elem_boundary_handler.reset();
  if (owns_petsc) {
    PetscFinalize();
  }
}

void PumiTallyImpl::CopyInitialPositionToBuffer(double *init_particle_positions,
                                                const Omega_h::LO size) {
  // copy to host buffer
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(init_particle_positions);
  MoveToInitialLocation();
  // TODO Get total is_initial_track particle weight
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::MoveToNextLocation(double *particle_origin,
                                       double *particle_destinations,
                                       int8_t *flying, double *weights,
                                       Omega_h::LO size) {

  // *************** Start Initial Move to Origin ************************** //
  assert(size == num_particles * 3);
  CopyLocationsToBuffer(particle_origin);

  // copy position buffer ps
  auto particle_orig = pumipic_ptcls->get<0>();
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();
  auto p_wgt = pumipic_ptcls->get<4>();

  // copy fly to device buffer
  CopyFlyingFlagToBuffer(flying);

  const Omega_h::LO pumi_ps_size = num_particles;
  const auto &device_pos_buffer_l = position_dev_buffer;
  const auto &device_in_adv_que_l = flying_dev_buffer;

  auto set_particle_dest_orig =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size) {
      // everyone is in flight for this is_initial_track search
      // in_flight(pid) = 1;
      in_flight(pid) = static_cast<unsigned char>(device_in_adv_que_l[pid]);

      if (in_flight(pid) == 1) {
        particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
        particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
        particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];
      } else {
        particle_dest(pid, 0) = particle_orig(pid, 0);
        particle_dest(pid, 1) = particle_orig(pid, 1);
        particle_dest(pid, 2) = particle_orig(pid, 2);
      }

      p_wgt(pid) = 0.0;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest_orig,
                        "set particle orig position as dest");

  bool migrate = iter_count % 100 == 0;
  SearchAndRebuild(/*is_initial_track*/ false, /*migrate*/ true);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif

  // ************** End Initial Move to Origin ****************************** //

  assert(size == num_particles * 3);

  // copy to device buffer
  CopyLocationsToBuffer(particle_destinations);
  CopyWeightsToBuffer(weights);

  Kokkos::fence();

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < num_particles) {
      if (in_flight(pid) == 1) {
        particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
        particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
        particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];
      } else {
        particle_dest(pid, 0) = particle_orig(pid, 0);
        particle_dest(pid, 1) = particle_orig(pid, 1);
        particle_dest(pid, 2) = particle_orig(pid, 2);
      }
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set particle position as dest");

  iter_count++;
  SearchAndRebuild(false, migrate);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::AccumulateBatchTally(
    const double normalization_factor) const {
  p_pumi_particle_at_elem_boundary_handler->AccumulateBatchFlux(
      normalization_factor);
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::DiscardBatchTally() const {
  p_pumi_particle_at_elem_boundary_handler->DiscardBatchFlux();
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::WriteTallyResults() {
  p_pumi_particle_at_elem_boundary_handler->FinalizeTallies(full_mesh,
                                                            "fluxresult");
#ifdef PUMI_MEASURE_TIME
  Kokkos::fence();
#endif
}

void PumiTallyImpl::CopyFlyingFlagToBuffer(int8_t *flying) const {
  // todo get the size too
  const Kokkos::View<Omega_h::I8 *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_host_view(flying, num_particles);
  const Kokkos::View<Omega_h::I8 *, PPExeSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      flying_device_view(flying_dev_buffer.data(), flying_dev_buffer.size());
  Kokkos::deep_copy(flying_device_view, flying_host_view);

  for (int64_t pid = 0; pid < num_particles; ++pid) {
    // reset flying flag to zero // TODO: why, specific reason
    flying[pid] = 0;
  }
}

void PumiTallyImpl::CopyWeightsToBuffer(double *weights) const {
  auto weights_l = weights_dev_buffer;
  Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_weights_view(weights, num_particles);
  Kokkos::View<Omega_h::Real *, PPExeSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      device_weights_view(weights_l.data(), weights_l.size());

  Kokkos::deep_copy(device_weights_view, host_weights_view);

  auto p_wgt = pumipic_ptcls->get<4>();
  auto copy_particle_weights =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    p_wgt(pid) = weights_l[pid];
  };
  pumipic::parallel_for(pumipic_ptcls.get(), copy_particle_weights,
                        "copy particle weights");
}

void PumiTallyImpl::MoveToInitialLocation() { // assign the location to ptcl
  // dest
  auto particle_dest = pumipic_ptcls->get<1>();
  auto in_flight = pumipic_ptcls->get<3>();

  const int64_t pumi_ps_size_l = num_particles;
  const auto &device_pos_buffer_l = position_dev_buffer;

  auto set_particle_dest =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0 && pid < pumi_ps_size_l) {
      particle_dest(pid, 0) = device_pos_buffer_l[pid * 3 + 0];
      particle_dest(pid, 1) = device_pos_buffer_l[pid * 3 + 1];
      particle_dest(pid, 2) = device_pos_buffer_l[pid * 3 + 2];

      // everyone is in flight for this is_initial_track search
      in_flight(pid) = 1;
    }
  };
  pumipic::parallel_for(pumipic_ptcls.get(), set_particle_dest,
                        "set is_initial_track position as dest");

  // *is_initial_track* build and search to find the is_initial_track elements
  // of the particles
  SearchAndRebuild(true, true);
  is_pumipic_initialized = true;
}

void PumiTallyImpl::CopyLocationsToBuffer(double *particle_positions) const {
  // fixme it should get size too to avoid memory error
  const Kokkos::View<Omega_h::Real *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      position_host_view(particle_positions,
                         static_cast<size_t>(num_particles * 3));

  const Kokkos::View<Omega_h::Real *, PPExeSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      position_device_view(position_dev_buffer.data(),
                           static_cast<size_t>(num_particles * 3));

  Kokkos::deep_copy(position_device_view, position_host_view);
}

void PumiTallyImpl::InitPUMILibrary(int &argc, char **&argv) {
  pumipic_lib = std::make_unique<pumipic::Library>(&argc, &argv);
  oh_lib = pumipic_lib->omega_h_lib();

  // MPI and Kokkos are up now; PETSc reuses both. argv is passed so that
  // -ksp_* / -pc_* options reach the P1 tally solver
  PetscBool petsc_initialized = PETSC_FALSE;
  PetscCallAbort(PETSC_COMM_SELF, PetscInitialized(&petsc_initialized));
  if (!petsc_initialized) {
    PetscCallAbort(PETSC_COMM_SELF,
                   PetscInitialize(&argc, &argv, nullptr, nullptr));
    owns_petsc = true;
  }
}

void UpdateCurrentElement(PPPS *ptcls,
                          const Omega_h::Write<Omega_h::LO> &elem_ids,
                          const Omega_h::Write<Omega_h::LO> &next_elems,
                          const Omega_h::Write<Omega_h::LO> &last_exit) {
  const auto in_flight = ptcls->get<3>();
  auto move_to_next = PS_LAMBDA(const int e, const int pid, const int mask) {
    // move to the next element only if the particle actually crossed a face in
    // this iteration (last_exit != -1) and is not leaving the domain. For
    // particles that reached their destination or were deleted (last_exit ==
    // -1) next_elems is stale; copying it would corrupt the element pumipic
    // found for them (e.g. by relocation) or resurrect a deleted particle.
    if (mask > 0 && in_flight(pid) && next_elems[pid] != -1 &&
        last_exit[pid] != -1) {
      elem_ids[pid] = next_elems[pid];
    }
  };
  pumipic::parallel_for(ptcls, move_to_next, "move to next element");
}

void ApplyVacuumBC(const Omega_h::Mesh &mesh, PPPS *ptcls,
                   const Omega_h::Write<Omega_h::LO> &elem_ids,
                   const Omega_h::Write<Omega_h::LO> &next_elems,
                   const Omega_h::Write<Omega_h::LO> &ptcl_done,
                   const Omega_h::Write<Omega_h::LO> &last_exit,
                   const Omega_h::Write<Omega_h::LO> &x_face,
                   const Omega_h::Write<Omega_h::Real> &inter_points) {

  // TODO: make this a member variable of the struct
  const auto particle_destination = ptcls->get<1>();
  auto check_exposed_edges =
      PS_LAMBDA(const int e, const int pid, const int mask) {
    if (mask > 0 && !ptcl_done[pid]) {
      const bool reached_destination = (last_exit[pid] == -1);
      const bool hit_boundary =
          ((next_elems[pid] == -1) && (elem_ids[pid] != -1));
      ptcl_done[pid] =
          (reached_destination || hit_boundary) ? 1 : ptcl_done[pid];

      if (hit_boundary) { // just reached the boundary
        x_face[pid] = last_exit[pid];
        // particle reaches the boundary
        particle_destination(pid, 0) = inter_points[pid * 3];
        particle_destination(pid, 1) = inter_points[pid * 3 + 1];
        particle_destination(pid, 2) = inter_points[pid * 3 + 2];
      }
    }
  };
  pumipic::parallel_for(ptcls, check_exposed_edges,
                        "apply vacumm boundary condition");
}

ParticleAtElemBoundary::ParticleAtElemBoundary(Omega_h::Mesh &mesh,
                                               const Omega_h::LO capacity,
                                               const Estimator estimator)
    : is_initial_track(true), estimator(estimator),
      flux(mesh.nelems(), 0.0, "flux"),
      flux_sum(mesh.nelems(), 0.0, "flux_sum"),
      flux_sum_sq(mesh.nelems(), 0.0, "flux_sum_sq"),
      prev_xpoint(capacity * 3, 0.0, "prev_xpoint"),
      elem2verts(mesh.ask_elem_verts()), coords(mesh.coords()),
      p1_rhs(mesh.nverts(), 0.0, "p1_rhs"),
      p1_flux_sum(mesh.nverts(), 0.0, "p1_flux_sum"),
      p1_flux_sum_sq(mesh.nverts(), 0.0, "p1_flux_sum_sq"),
      p1_solver(std::make_shared<P1TallySolver>(mesh)) {
  printf("[INFO] Particle handler at boundary with %d elements, %d vertices "
         "and %d x points size (3 * n_particles)\n",
         flux.size(), p1_rhs.size(), prev_xpoint.size());
}

void ParticleAtElemBoundary::operator()(
    const Omega_h::Mesh &mesh, pumitally::PPPS *ptcls,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &next_elems,
    const Omega_h::Write<Omega_h::LO> &inter_faces,
    const Omega_h::Write<Omega_h::LO> &last_exit,
    const Omega_h::Write<Omega_h::Real> &inter_points,
    const Omega_h::Write<Omega_h::LO> &ptcl_done,
    decltype(ptcls->get<0>())
        origin_segment, // NOLINT(performance-unnecessary-value-param)
    decltype(ptcls->get<1>()) dest_segment)
    const { // NOLINT(performance-unnecessary-value-param)
  if (!is_initial_track) {
    EvaluateFlux(ptcls, inter_points, elem_ids, last_exit, ptcl_done);
    UpdatePreviousXPoints(inter_points);
  }
  ApplyVacuumBC(mesh, ptcls, elem_ids, next_elems, ptcl_done, last_exit,
                inter_faces, inter_points);
  UpdateCurrentElement(ptcls, elem_ids, next_elems, last_exit);
}

void ParticleAtElemBoundary::MarkAsInitial(const bool is_initial) {
  is_initial_track = is_initial;
}

void ParticleAtElemBoundary::UpdatePreviousXPoints(
    const Omega_h::Write<Omega_h::Real> &xpoints) const {
  OMEGA_H_CHECK_PRINTF(xpoints.size() <= prev_xpoint.size() &&
                           prev_xpoint.size() != 0,
                       "xpoints size %d is greater than prev_xpoint size %d\n",
                       xpoints.size(), prev_xpoint.size());
  const auto &prev_xpoint_l = prev_xpoint;
  auto update = OMEGA_H_LAMBDA(const Omega_h::LO i) {
    prev_xpoint_l[i] = xpoints[i];
  };
  Omega_h::parallel_for(xpoints.size(), update, "update previous xpoints");
}

void ParticleAtElemBoundary::UpdatePreviousXPoints(PPPS *ptcls) const {
  // todo add checks of size
  const auto prev_xpoints_l = prev_xpoint;
  OMEGA_H_CHECK_PRINTF(
      ptcls->capacity() * 3 == prev_xpoints_l.size(),
      "Error: prev_xpoints_s are not size properly capacity %d size %d\n",
      ptcls->capacity(), prev_xpoints_l.size());
  auto xpoints = ptcls->get<0>();
  auto update = PS_LAMBDA(const auto &e, const auto &pid, const auto &mask) {
    prev_xpoints_l[pid * 3 + 0] = xpoints(pid, 0);
    prev_xpoints_l[pid * 3 + 1] = xpoints(pid, 1);
    prev_xpoints_l[pid * 3 + 2] = xpoints(pid, 2);
  };
  pumipic::parallel_for(ptcls, update,
                        "update previous xpoints from origin points");
}

void ParticleAtElemBoundary::EvaluateFlux(
    PPPS *ptcls, const Omega_h::Write<Omega_h::Real> &xpoints,
    const Omega_h::Write<Omega_h::LO> &elem_ids,
    const Omega_h::Write<Omega_h::LO> &last_exit,
    const Omega_h::Write<Omega_h::LO> &ptcl_done) const {
  const bool collision = estimator == Estimator::Collision;
  const auto prev_xpoint_l = prev_xpoint;
  const auto flux_l = flux;
  const auto p1_rhs_l = p1_rhs;
  const auto elem2verts_l = elem2verts;
  const auto coords_l = coords;
  const auto in_flight = ptcls->get<3>();
  const auto p_wgt = ptcls->get<4>();
  const auto &xpoints_l = xpoints; // todo shouldn't need it, so remove

  auto evaluate_flux =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    // the collision estimator only scores where the particle reaches its
    // destination (last_exit == -1), i.e. the collision site
    if ((mask > 0) && (in_flight(pid) == 1) && !ptcl_done[pid] &&
        (!collision || last_exit[pid] == -1)) {
      const Omega_h::Vector<3> dest = {xpoints_l[pid * 3 + 0],
                                       xpoints_l[pid * 3 + 1],
                                       xpoints_l[pid * 3 + 2]};
      const Omega_h::Vector<3> orig = {prev_xpoint_l[pid * 3 + 0],
                                       prev_xpoint_l[pid * 3 + 1],
                                       prev_xpoint_l[pid * 3 + 2]};

      // TODO: Get total number of particles and divide here
      const Omega_h::Real segment_length = Omega_h::norm(dest - orig);

      // collision: the weight is already w / Sigma_t
      const Omega_h::Real contribution =
          collision ? p_wgt(pid) : segment_length * p_wgt(pid);
      Kokkos::atomic_add(&flux_l[elem_ids[pid]], contribution);

      // P1: int_segment(N_v) dl = L * N_v(midpoint) since N_v is linear;
      // the collision estimator evaluates N_v at the collision site
      const auto elem_verts = Omega_h::gather_verts<4>(elem2verts_l, elem_ids[pid]);
      const auto elem_vert_coords =
          Omega_h::gather_vectors<4, 3>(coords_l, elem_verts);
      const Omega_h::Vector<3> score_point =
          collision ? dest : 0.5 * (orig + dest);
      const auto bcc =
          Omega_h::barycentric_from_global<3, 3>(score_point, elem_vert_coords);
      for (int i = 0; i < 4; ++i) {
        Kokkos::atomic_add(&p1_rhs_l[elem_verts[i]], contribution * bcc[i]);
      }
    }
  };
  pumipic::parallel_for(ptcls, evaluate_flux, "flux evaluation loop");
}

void ParticleAtElemBoundary::AccumulateBatchFlux(
    const Omega_h::Real normalization_factor) {
  const auto flux_l = flux;
  const auto flux_sum_l = flux_sum;
  const auto flux_sum_sq_l = flux_sum_sq;

  auto accumulate = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    const Omega_h::Real value = flux_l[elem_id] * normalization_factor;
    flux_sum_l[elem_id] += value;
    flux_sum_sq_l[elem_id] += value * value;
    flux_l[elem_id] = 0.0;
  };
  Omega_h::parallel_for(flux.size(), accumulate, "accumulate batch flux");

  // P1: phi = M^{-1} (normalization_factor * rhs), statistics on phi
  const auto p1_rhs_l = p1_rhs;
  auto scale_rhs = OMEGA_H_LAMBDA(const Omega_h::LO vert_id) {
    p1_rhs_l[vert_id] *= normalization_factor;
  };
  Omega_h::parallel_for(p1_rhs.size(), scale_rhs, "scale batch P1 rhs");

  const Omega_h::Reals phi = p1_solver->Solve(Omega_h::Reals(p1_rhs));

  const auto p1_flux_sum_l = p1_flux_sum;
  const auto p1_flux_sum_sq_l = p1_flux_sum_sq;
  auto accumulate_p1 = OMEGA_H_LAMBDA(const Omega_h::LO vert_id) {
    const Omega_h::Real value = phi[vert_id];
    p1_flux_sum_l[vert_id] += value;
    p1_flux_sum_sq_l[vert_id] += value * value;
    p1_rhs_l[vert_id] = 0.0;
  };
  Omega_h::parallel_for(p1_rhs.size(), accumulate_p1,
                        "accumulate batch P1 flux");

  n_realizations++;
}

void ParticleAtElemBoundary::DiscardBatchFlux() const {
  const auto flux_l = flux;
  auto reset = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    flux_l[elem_id] = 0.0;
  };
  Omega_h::parallel_for(flux.size(), reset, "reset batch flux");

  const auto p1_rhs_l = p1_rhs;
  auto reset_p1 = OMEGA_H_LAMBDA(const Omega_h::LO vert_id) {
    p1_rhs_l[vert_id] = 0.0;
  };
  Omega_h::parallel_for(p1_rhs.size(), reset_p1, "reset batch P1 rhs");
}

std::pair<Omega_h::Reals, Omega_h::Reals>
MeanAndStdDev(const Omega_h::Write<Omega_h::Real> &sum,
              const Omega_h::Write<Omega_h::Real> &sum_sq,
              const Omega_h::LO n_realizations) {
  OMEGA_H_CHECK_PRINTF(n_realizations > 0,
                       "Cannot compute statistics from %d batches\n",
                       n_realizations);

  const auto sum_l = sum;
  const auto sum_sq_l = sum_sq;
  const auto n = static_cast<Omega_h::Real>(n_realizations);

  const Omega_h::Write<Omega_h::Real> mean(sum.size(), 0.0, "flux_mean");
  const Omega_h::Write<Omega_h::Real> std_dev(sum.size(), 0.0,
                                              "flux_std_dev");

  auto compute_statistics = OMEGA_H_LAMBDA(const Omega_h::LO id) {
    const Omega_h::Real id_mean = sum_l[id] / n;
    mean[id] = id_mean;
    // a single realization gives no estimate of the spread
    if (n > 1.0) {
      const Omega_h::Real variance =
          (sum_sq_l[id] / n - id_mean * id_mean) / (n - 1.0);
      // round off can make the variance slightly negative for a flat tally
      std_dev[id] = (variance > 0.0) ? Kokkos::sqrt(variance) : 0.0;
    }
  };
  Omega_h::parallel_for(sum.size(), compute_statistics,
                        "compute flux mean and std dev");

  return {Omega_h::Reals(mean), Omega_h::Reals(std_dev)};
}

std::pair<Omega_h::Reals, Omega_h::Reals>
ParticleAtElemBoundary::ComputeMeanAndStdDev() const {
  return MeanAndStdDev(flux_sum, flux_sum_sq, n_realizations);
}

std::pair<Omega_h::Reals, Omega_h::Reals>
ParticleAtElemBoundary::ComputeP1MeanAndStdDev() const {
  return MeanAndStdDev(p1_flux_sum, p1_flux_sum_sq, n_realizations);
}

Omega_h::Reals ElementVolumes(Omega_h::Mesh &mesh) {
  const auto &el2n = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;
  const auto &coords = mesh.coords();

  const Omega_h::Write<Omega_h::Real> tet_volumes(mesh.nelems(), -1.0,
                                                  "tet_volumes");

  auto compute_volume = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    const auto elem_verts = Omega_h::gather_verts<4>(el2n, elem_id);
    const auto elem_vert_coords =
        Omega_h::gather_vectors<4, 3>(coords, elem_verts);

    const auto b = Omega_h::simplex_basis<3, 3>(elem_vert_coords);
    tet_volumes[elem_id] = Omega_h::simplex_size_from_basis(b);
  };
  Omega_h::parallel_for(tet_volumes.size(), compute_volume,
                        "compute element volumes");

  return {tet_volumes};
}

Omega_h::Reals DivideByVolume(const Omega_h::Reals &values,
                              const Omega_h::Reals &volumes) {
  OMEGA_H_CHECK_PRINTF(values.size() == volumes.size(),
                       "Cannot normalize %d values with %d volumes\n",
                       values.size(), volumes.size());
  const Omega_h::Write<Omega_h::Real> normalized_values(values.size(), -1.0,
                                                        "normalized values");
  auto normalize = OMEGA_H_LAMBDA(const Omega_h::LO elem_id) {
    normalized_values[elem_id] = values[elem_id] / volumes[elem_id];
  };
  Omega_h::parallel_for(normalized_values.size(), normalize,
                        "normalize by volume");
  return {normalized_values};
}

void ParticleAtElemBoundary::FinalizeTallies(Omega_h::Mesh &full_mesh,
                                             const std::string &filename) {
  // a driver that never closes a batch still gets its raw flux reported as the
  // mean of a single realization
  if (n_realizations == 0) {
    AccumulateBatchFlux(1.0);
  }

  const auto [mean, std_dev] = ComputeMeanAndStdDev();

  const auto volumes = ElementVolumes(full_mesh);
  full_mesh.add_tag(Omega_h::REGION, "volume", 1, volumes);

  printf("[INFO] Writing flux mean and standard deviation of %d batches\n",
         n_realizations);

  full_mesh.add_tag(Omega_h::REGION, "flux", 1, DivideByVolume(mean, volumes));
  full_mesh.add_tag(Omega_h::REGION, "flux_std_dev", 1,
                    DivideByVolume(std_dev, volumes));

  // P1 nodal flux: M^{-1} b already gives the FE coefficients, so it is not
  // divided by volume. The tracking mesh and full_mesh share vertex numbering
  // on a single rank, as they share element numbering for the histogram.
  OMEGA_H_CHECK_PRINTF(full_mesh.nverts() == p1_flux_sum.size(),
                       "P1 tally has %d vertices but output mesh has %d\n",
                       p1_flux_sum.size(), full_mesh.nverts());
  const auto [p1_mean, p1_std_dev] = ComputeP1MeanAndStdDev();
  full_mesh.add_tag(Omega_h::VERT, "flux_p1", 1, p1_mean);
  full_mesh.add_tag(Omega_h::VERT, "flux_p1_std_dev", 1, p1_std_dev);

  // Both .osh and .vtk are written. Later, .vtk will be dropped.
  Omega_h::binary::write(filename + ".osh", &full_mesh);
  Omega_h::vtk::write_parallel(filename + "_vtu", &full_mesh, 3);
}

void CommitParticlePositions(PPPS *ptcls) {
  auto particle_origin = ptcls->get<0>();
  auto particle_destination = ptcls->get<1>();
  auto update_particle_position =
      PS_LAMBDA(const int &, const int &pid, const bool &) {
    particle_origin(pid, 0) = particle_destination(pid, 0);
    particle_origin(pid, 1) = particle_destination(pid, 1);
    particle_origin(pid, 2) = particle_destination(pid, 2);
    particle_destination(pid, 0) = 0.0;
    particle_destination(pid, 1) = 0.0;
    particle_destination(pid, 2) = 0.0;
  };
  ps::parallel_for(ptcls, update_particle_position);
}

void PumiTallyImpl::SearchAndRebuild(const bool initial,
                                     const bool migrate) const {
  // the very first search has to be an initial (non tallying) one; every later
  // batch starts with an initial search again to locate its own source
  // particles, so `initial` may be true any number of times after that
  assert(is_pumipic_initialized || initial);
  p_pumi_particle_at_elem_boundary_handler->MarkAsInitial(initial);
  auto orig = pumipic_ptcls->get<0>();
  auto dest = pumipic_ptcls->get<1>();
  auto pid = pumipic_ptcls->get<2>();

  if (p_picparts->mesh() == nullptr || p_picparts->mesh()->nelems() == 0) {
    fprintf(stderr, "ERROR: Mesh is empty\n");
  }

  // total tracklengths are used to calculate the flux
  if (!initial) {
    p_pumi_particle_at_elem_boundary_handler->UpdatePreviousXPoints(
        pumipic_ptcls.get());
  }

  const bool found_all = p_particle_tracer->search(migrate);
  if (!found_all) {
    printf(
        "ERROR: Not all particles are found. May need more loops in search\n");
  }
}

std::unique_ptr<PPPS> CreateParticleDS(const Omega_h::Mesh &mesh,
                                       pumipic::lid_t num_ptcls) {
  Omega_h::Int ne = mesh.nelems();
  pumitally::PPPS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  pumitally::PPPS::kkGidView element_gids("element_gids", ne);

  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy;

  Omega_h::parallel_for(
      ne, OMEGA_H_LAMBDA(const Omega_h::LO &i) { element_gids(i) = i; });

  Omega_h::parallel_for(
      mesh.nelems(), OMEGA_H_LAMBDA(const Omega_h::LO id) {
        ptcls_per_elem[id] = (id == 0) ? num_ptcls : 0;
      });

#ifdef PUMI_USE_KOKKOS_CUDA
  printf("PumiPIC Using GPU for simulation...\n");
  policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, 32);
#else
  printf("PumiPIC Using CPU for simulation...\n");
  policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
#endif

  auto ptcls = std::make_unique<pumipic::DPS<pumitally::PPParticle>>(
      policy, ne, num_ptcls, ptcls_per_elem, element_gids);

  return ptcls;
}

void InitializeParticlesInElement0(Omega_h::Mesh &mesh,
                                   pumitally::PPPS *ptcls) {
  // find the centroid of the 0th element
  const auto &coords = mesh.coords();
  const auto &tet2node = mesh.ask_down(Omega_h::REGION, Omega_h::VERT).ab2b;

  Omega_h::Write<Omega_h::Real> centroid_of_el0(3, 0.0, "centroid");

  auto find_centroid_of_el0 = OMEGA_H_LAMBDA(const Omega_h::LO id) {
    const auto nodes = Omega_h::gather_verts<4>(tet2node, id);
    const Omega_h::Few<Omega_h::Vector<3>, 4> tet_node_coords =
        Omega_h::gather_vectors<4, 3>(coords, nodes);
    const auto centroid = o::average(tet_node_coords);
    centroid_of_el0[0] = centroid[0];
    centroid_of_el0[1] = centroid[1];
    centroid_of_el0[2] = centroid[2];
  };
  Omega_h::parallel_for(1, find_centroid_of_el0, "find centroid of element 0");

  // assign the location to all particles
  auto init_loc = ptcls->get<0>();
  auto pids = ptcls->get<2>();
  auto in_fly = ptcls->get<3>();

  auto set_initial_positions =
      PS_LAMBDA(const int &e, const int &pid, const int &mask) {
    if (mask > 0) {
      pids(pid) = pid;
      in_fly(pid) = 1;
      init_loc(pid, 0) = centroid_of_el0[0];
      init_loc(pid, 1) = centroid_of_el0[1];
      init_loc(pid, 2) = centroid_of_el0[2];
    }
  };
  pumipic::parallel_for(ptcls, set_initial_positions,
                        "set is_initial_track particle positions");
}

Omega_h::Mesh PumiTallyImpl::PartitionMesh() {
  const Omega_h::Write<Omega_h::LO> owners(full_mesh.nelems(), 0, "owners");
  // all the particles are initialized in element 0 to do an is_initial_track
  // search to find the starting locations of the openmc given particles.
  // p_picparts = new pumipic::Mesh(full_mesh, Omega_h::LOs(owners));
  p_picparts = std::make_unique<pumipic::Mesh>(full_mesh, Omega_h::LOs(owners));
  printf("PumiPIC mesh partitioned\n");

  return *p_picparts->mesh();
}

void PumiTallyImpl::InitializePUMIParticleStructure(Omega_h::Mesh &mesh) {
  pumipic_ptcls = CreateParticleDS(mesh, num_particles);
  InitializeParticlesInElement0(mesh, pumipic_ptcls.get());
  p_pumi_particle_at_elem_boundary_handler =
      std::make_unique<pumitally::ParticleAtElemBoundary>(
          mesh, pumipic_ptcls->capacity(), estimator);

  printf("PumiPIC Mesh and data structure created with %d and %d as particle "
         "structure capacity\n",
         p_picparts->mesh()->nelems(), pumipic_ptcls->capacity());
}

void PumiTallyImpl::ReadFullMesh(int &argc, char **&argv) {
  printf("Reading the Omega_h mesh %s to tally with tracklength estimator\n",
         oh_mesh_filename.c_str());
  InitPUMILibrary(argc, argv);

  if (oh_mesh_filename.empty()) {
    printf("[ERROR] Omega_h mesh for PumiPIC is not given. Provide --ohMesh = "
           "<osh file>");
  }
  full_mesh = Omega_h::binary::read(oh_mesh_filename, &oh_lib);
  if (full_mesh.dim() != 3) {
    printf("PumiPIC only works for 3D mesh now.\n");
  }
  printf("PumiPIC Loaded mesh %s with %d elements\n", oh_mesh_filename.c_str(),
         full_mesh.nelems());
}

void PumiTallyImpl::LoadMeshAndInitParticles(int &argc, char **&argv) {
  ReadFullMesh(argc, argv);
  Omega_h::Mesh mesh = PartitionMesh();
  InitializePUMIParticleStructure(mesh);
}
} // namespace pumitally
