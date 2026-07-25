/*
 * RestartReader.cpp - Restart-file reader implementation for iPIC3D
 *
 * Contains all checkpoint-reading logic, extracted from Collective.cpp.
 * Supports ADIOS2 (BP5) and HDF5 backends via compile-time guards.
 */

#include "RestartReader.h"

#include "CUDA/cudaTypeDef.cuh" // cudaCommonType, cudaPclType_ID
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "RestartRemapPlan.h"
#include "VCtopology3D.h"
#include "debug.h"    // eprintf
#include "ipicdefs.h" // DVECWIDTH
#include "ipichdf5.h" // HDF5 headers (guarded by NO_HDF5)
#include "ipicmath.h" // roundup_to_multiple

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_ADIOS2
#include "adios2.h"
#endif

using std::string;

namespace {

RestartMeshMetadata
makeDestinationMeshForRestart(const RestartMeshMetadata& source,
                              const VCtopology3D* vct) {
  RestartMeshMetadata destination = source;
  destination.xlen = vct->getXLEN();
  destination.ylen = vct->getYLEN();
  destination.zlen = vct->getZLEN();
  destination.nranks = vct->getNprocs();
  destination.periodicX = vct->getPERIODICX();
  destination.periodicY = vct->getPERIODICY();
  destination.periodicZ = vct->getPERIODICZ();
  destination.periodicParticleX = vct->getPERIODICX_P();
  destination.periodicParticleY = vct->getPERIODICY_P();
  destination.periodicParticleZ = vct->getPERIODICZ_P();
  destination.valid = true;
  return destination;
}

void validateDestinationGrid(const RestartRemapPlan& plan, const Grid* grid) {
  const RestartRankBox& destination = plan.destinationBox();
  if (destination.activeNodes.nx() != grid->getNXN() - 2 ||
      destination.activeNodes.ny() != grid->getNYN() - 2 ||
      destination.activeNodes.nz() != grid->getNZN() - 2) {
    eprintf("ERROR: restart remap destination active-node box does not "
            "match the current local grid");
  }
  if (destination.activeCells.nx() != grid->getNXC() - 2 ||
      destination.activeCells.ny() != grid->getNYC() - 2 ||
      destination.activeCells.nz() != grid->getNZC() - 2) {
    eprintf("ERROR: restart remap destination active-cell box does not "
            "match the current local grid");
  }
}

std::string restartRankFilePath(const std::string& restartDir,
                                const std::string& backend, int rank) {
  return restartDir + "/" + RestartSlotManager::rankFileName(backend, rank);
}

std::string restartBackendLabel(const std::string& backend) {
  if (backend == "adios2")
    return "ADIOS2";
  if (backend == "hdf5")
    return "HDF5";
  return backend;
}

void printRestartReadSignature(const char* objectName,
                               const std::string& backend,
                               const char* remapKind, int cycle) {
  std::cout << "[*] " << objectName << " Restarting ("
            << restartBackendLabel(backend) << ") with " << remapKind
            << " remap from cycle label: " << cycle << std::endl;
}

size_t boxValueCount(const RestartBox3D& box) {
  return static_cast<size_t>(box.nx()) * static_cast<size_t>(box.ny()) *
         static_cast<size_t>(box.nz());
}

template <typename T>
void scatterActiveNodeBufferToGuardedArray(const RestartNodeCopy& copy,
                                           const T* src, arr3_double dst) {
  size_t index = 0;
  const auto nx_ = copy.destinationLocal.nx();
  const auto ny_ = copy.destinationLocal.ny();
  const auto nz_ = copy.destinationLocal.nz();
  const auto x_begin = copy.destinationLocal.x.begin;
  const auto y_begin = copy.destinationLocal.y.begin;
  const auto z_begin = copy.destinationLocal.z.begin;

  for (int ix = 0; ix < nx_; ++ix) {
    for (int iy = 0; iy < ny_; ++iy) {
      for (int iz = 0; iz < nz_; ++iz) {
        dst[x_begin + ix + 1][y_begin + iy + 1][z_begin + iz + 1] =
            static_cast<double>(src[index++]);
      }
    }
  }
}

struct ParticleReadSpan {
  int sourceRank = 0;
  long long offset = 0;
  long long count = 0;
};

struct RestartParticleChunkBuffers {
  std::vector<double> u;
  std::vector<double> v;
  std::vector<double> w;
  std::vector<double> q;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<cudaPclType_ID> id;

  void resize(size_t count, bool trackParticleID) {
    u.resize(count);
    v.resize(count);
    w.resize(count);
    q.resize(count);
    x.resize(count);
    y.resize(count);
    z.resize(count);
    if (trackParticleID) {
      id.resize(count);
      std::fill(id.begin(), id.end(), PARTICLE_ID_INVALID);
    } else {
      id.clear();
    }
  }
};

int activeCellLinearIndex(int ix, int iy, int iz,
                          const std::array<int, 3>& dims) {
  return ix + iy * dims[0] + iz * dims[0] * dims[1];
}

void validateSourceActiveCellDims(const RestartCellCopy& copy,
                                  const RestartMeshMetadata& sourceMesh,
                                  const std::array<int, 3>& dims) {
  const RestartRankBox sourceBox =
      restartRankBoxFor(sourceMesh, copy.sourceRank);
  if (dims[0] != sourceBox.activeCells.nx() ||
      dims[1] != sourceBox.activeCells.ny() ||
      dims[2] != sourceBox.activeCells.nz()) {
    eprintf("ERROR: restart particle active-cell metadata dimensions "
            "do not match the source rank active-cell box");
  }
}

void appendMergedSpan(std::vector<ParticleReadSpan>& spans,
                      const ParticleReadSpan& span) {
  if (span.count <= 0)
    return;

  if (!spans.empty()) {
    ParticleReadSpan& last = spans.back();
    if (last.sourceRank == span.sourceRank &&
        last.offset + last.count == span.offset) {
      last.count += span.count;
      return;
    }
  }

  spans.push_back(span);
}

void appendParticleSpansForCopy(const RestartCellCopy& copy,
                                const std::array<int, 3>& activeCellDims,
                                const std::vector<int>& cellOffsets,
                                const std::vector<int>& cellCounts,
                                std::vector<ParticleReadSpan>& spans) {
  const int expectedCells =
      activeCellDims[0] * activeCellDims[1] * activeCellDims[2];
  if (static_cast<int>(cellOffsets.size()) != expectedCells ||
      static_cast<int>(cellCounts.size()) != expectedCells) {
    eprintf("ERROR: restart particle cell metadata size mismatch");
  }

  for (int iz = copy.sourceLocal.z.begin; iz < copy.sourceLocal.z.end; ++iz) {
    for (int iy = copy.sourceLocal.y.begin; iy < copy.sourceLocal.y.end; ++iy) {
      const int firstCell = activeCellLinearIndex(copy.sourceLocal.x.begin, iy,
                                                  iz, activeCellDims);
      const int lastCell = activeCellLinearIndex(copy.sourceLocal.x.end - 1, iy,
                                                 iz, activeCellDims);

      const long long begin = cellOffsets[firstCell];
      const long long end =
          static_cast<long long>(cellOffsets[lastCell]) + cellCounts[lastCell];

      ParticleReadSpan span;
      span.sourceRank = copy.sourceRank;
      span.offset = begin;
      span.count = end - begin;
      appendMergedSpan(spans, span);
    }
  }
}

struct RestartParticleCellLayout {
  std::array<int, 3> activeCellDims = {{0, 0, 0}};
  std::vector<int> offsets;
  std::vector<int> counts;

  size_t activeCellCount() const {
    return static_cast<size_t>(activeCellDims[0]) *
           static_cast<size_t>(activeCellDims[1]) *
           static_cast<size_t>(activeCellDims[2]);
  }
};

long long totalParticleSpanCount(const std::vector<ParticleReadSpan>& spans) {
  long long total = 0;
  for (const ParticleReadSpan& span : spans)
    total += span.count;
  return total;
}

void resizeParticleVectors(long long totalParticles, vector_double& u,
                           vector_double& v, vector_double& w, vector_double& q,
                           vector_double& x, vector_double& y, vector_double& z,
                           vector_cudaPclType_ID& id, bool trackParticleID) {
  if (totalParticles > std::numeric_limits<int>::max()) {
    eprintf("ERROR: restart particle count exceeds int-supported "
            "ParticleSoAHost size");
  }

  const int nop = static_cast<int>(totalParticles);
  const int padded_nop = roundup_to_multiple(nop, DVECWIDTH);

  u.reserve(padded_nop);
  v.reserve(padded_nop);
  w.reserve(padded_nop);
  q.reserve(padded_nop);
  x.reserve(padded_nop);
  y.reserve(padded_nop);
  z.reserve(padded_nop);
  if (trackParticleID)
    id.reserve(padded_nop);

  u.resize(nop);
  v.resize(nop);
  w.resize(nop);
  q.resize(nop);
  x.resize(nop);
  y.resize(nop);
  z.resize(nop);
  if (trackParticleID) {
    id.resize(nop);
  } else {
    id.clear();
  }
}

void copyChunkToParticleVectors(const RestartParticleChunkBuffers& buffers,
                                size_t count, long long destinationOffset,
                                vector_double& u, vector_double& v,
                                vector_double& w, vector_double& q,
                                vector_double& x, vector_double& y,
                                vector_double& z, vector_cudaPclType_ID& id,
                                bool trackParticleID) {
  const size_t dst = static_cast<size_t>(destinationOffset);
  std::copy_n(buffers.u.data(), count, &u[dst]);
  std::copy_n(buffers.v.data(), count, &v[dst]);
  std::copy_n(buffers.w.data(), count, &w[dst]);
  std::copy_n(buffers.q.data(), count, &q[dst]);
  std::copy_n(buffers.x.data(), count, &x[dst]);
  std::copy_n(buffers.y.data(), count, &y[dst]);
  std::copy_n(buffers.z.data(), count, &z[dst]);
  if (trackParticleID) {
    std::copy_n(buffers.id.data(), count, &id[dst]);
  }
}

constexpr long long kRestartParticleChunkSize = 1LL << 20;

template <typename ReadChunk>
void readParticleSpanChunks(const ParticleReadSpan& span,
                            RestartParticleChunkBuffers& buffers,
                            long long& destinationOffset, vector_double& u,
                            vector_double& v, vector_double& w,
                            vector_double& q, vector_double& x,
                            vector_double& y, vector_double& z,
                            vector_cudaPclType_ID& id, bool trackParticleID,
                            ReadChunk readChunk) {
  long long remaining = span.count;
  long long sourceOffset = span.offset;
  while (remaining > 0) {
    const size_t chunk = static_cast<size_t>(
        std::min<long long>(remaining, kRestartParticleChunkSize));

    readChunk(sourceOffset, chunk, buffers);
    copyChunkToParticleVectors(buffers, chunk, destinationOffset, u, v, w, q, x,
                               y, z, id, trackParticleID);

    sourceOffset += static_cast<long long>(chunk);
    destinationOffset += static_cast<long long>(chunk);
    remaining -= static_cast<long long>(chunk);
  }
}

#ifdef USE_ADIOS2

std::string dimsToString(const adios2::Dims& dims) {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << dims[i];
  }
  ss << "}";
  return ss.str();
}

bool beginLastAdiosStep(adios2::Engine& engine, int last_cycle) {
  const auto stepNum = engine.Steps();
  for (unsigned int step = 0; engine.BeginStep() == adios2::StepStatus::OK;
       ++step) {
    if (step < stepNum - 1) {
      engine.EndStep();
      continue;
    }

    int fileCycle = -1;
    engine.Get<int>("cycle", fileCycle, adios2::Mode::Sync);
    if (fileCycle != last_cycle) {
      engine.EndStep();
      eprintf("restart cycle label in source rank file does not "
              "match the selected checkpoint label");
    }
    return true;
  }
  return false;
}

class AdiosActiveNodeRemapReader {
public:
  void readInto(adios2::IO& io, adios2::Engine& engine,
                const std::string& variableName, const RestartNodeCopy& copy,
                const RestartMeshMetadata& sourceMesh, arr3_double dst) {
    auto var = io.InquireVariable<cudaCommonType>(variableName);
    if (!var) {
      eprintf("ERROR: ADIOS2 restart variable %s is missing",
              variableName.c_str());
    }

    const RestartRankBox sourceBox =
        restartRankBoxFor(sourceMesh, copy.sourceRank);
    const adios2::Dims expectedShape = {
        static_cast<size_t>(sourceBox.activeNodes.nx()),
        static_cast<size_t>(sourceBox.activeNodes.ny()),
        static_cast<size_t>(sourceBox.activeNodes.nz())};
    const adios2::Dims actualShape = var.Shape();
    if (actualShape != expectedShape) {
      const std::string expected = dimsToString(expectedShape);
      const std::string actual = dimsToString(actualShape);
      eprintf("ERROR: ADIOS2 restart variable %s has shape %s, "
              "expected source active-node shape %s",
              variableName.c_str(), actual.c_str(), expected.c_str());
    }

    const adios2::Dims start = {static_cast<size_t>(copy.sourceLocal.x.begin),
                                static_cast<size_t>(copy.sourceLocal.y.begin),
                                static_cast<size_t>(copy.sourceLocal.z.begin)};
    const adios2::Dims count = {static_cast<size_t>(copy.sourceLocal.nx()),
                                static_cast<size_t>(copy.sourceLocal.ny()),
                                static_cast<size_t>(copy.sourceLocal.nz())};

    buffer_.resize(boxValueCount(copy.sourceLocal));
    var.SetSelection({start, count});
    engine.Get<cudaCommonType>(var, buffer_.data(), adios2::Mode::Sync);

    scatterActiveNodeBufferToGuardedArray(copy, buffer_.data(), dst);
  }

private:
  std::vector<cudaCommonType> buffer_;
};

class AdiosParticleRemapReader {
public:
  void readMetadata(adios2::IO& io, adios2::Engine& engine, int species,
                    const RestartCellCopy& copy,
                    const RestartMeshMetadata& sourceMesh,
                    RestartParticleCellLayout& layout) {
    auto dimsVar = io.InquireVariable<int>("activeCellDims");
    if (!dimsVar) {
      eprintf("ERROR: ADIOS2 restart activeCellDims variable is missing");
    }
    int dimsRaw[3] = {0, 0, 0};
    dimsVar.SetSelection({{0}, {3}});
    engine.Get<int>(dimsVar, dimsRaw, adios2::Mode::Sync);

    layout.activeCellDims = {{dimsRaw[0], dimsRaw[1], dimsRaw[2]}};
    validateSourceActiveCellDims(copy, sourceMesh, layout.activeCellDims);

    const std::string spec = std::to_string(species);
    auto offsetsVar = io.InquireVariable<int>("part" + spec + "CellOffsets");
    auto countsVar = io.InquireVariable<int>("part" + spec + "CellCounts");
    if (!offsetsVar || !countsVar) {
      eprintf("ERROR: ADIOS2 restart particle cell metadata is missing");
    }

    const size_t activeCells = layout.activeCellCount();
    layout.offsets.resize(activeCells);
    layout.counts.resize(activeCells);
    offsetsVar.SetSelection({{0}, {activeCells}});
    countsVar.SetSelection({{0}, {activeCells}});
    engine.Get<int>(offsetsVar, layout.offsets.data(), adios2::Mode::Sync);
    engine.Get<int>(countsVar, layout.counts.data(), adios2::Mode::Sync);
  }

  void readChunk(adios2::IO& io, adios2::Engine& engine, int species,
                 long long offset, size_t count,
                 RestartParticleChunkBuffers& buffers, bool trackParticleID) {
    buffers.resize(count, trackParticleID);
    const std::string prefix = "part" + std::to_string(species);

    readDoubleVariable(io, engine, prefix + "VelocityU", offset, count,
                       buffers.u.data());
    readDoubleVariable(io, engine, prefix + "VelocityV", offset, count,
                       buffers.v.data());
    readDoubleVariable(io, engine, prefix + "VelocityW", offset, count,
                       buffers.w.data());
    readDoubleVariable(io, engine, prefix + "charge", offset, count,
                       buffers.q.data());
    readDoubleVariable(io, engine, prefix + "PositionX", offset, count,
                       buffers.x.data());
    readDoubleVariable(io, engine, prefix + "PositionY", offset, count,
                       buffers.y.data());
    readDoubleVariable(io, engine, prefix + "PositionZ", offset, count,
                       buffers.z.data());
    if (trackParticleID)
      readIDVariableIfPresent(io, engine, prefix + "ID", offset, count,
                              buffers.id.data());
  }

private:
  void readDoubleVariable(adios2::IO& io, adios2::Engine& engine,
                          const std::string& variableName, long long offset,
                          size_t count, double* dst) {
    auto var = io.InquireVariable<cudaCommonType>(variableName);
    if (!var) {
      eprintf("ERROR: ADIOS2 restart variable %s is missing",
              variableName.c_str());
    }

    var.SetSelection({{static_cast<size_t>(offset)}, {count}});
    engine.Get<cudaCommonType>(var, dst, adios2::Mode::Sync);
  }

  void readIDVariableIfPresent(adios2::IO& io, adios2::Engine& engine,
                               const std::string& variableName,
                               long long offset, size_t count,
                               cudaPclType_ID* dst) {
    auto var = io.InquireVariable<cudaPclType_ID>(variableName);
    if (!var) {
      auto legacyVar = io.InquireVariable<cudaCommonType>(variableName);
      if (!legacyVar)
        return;

      std::vector<cudaCommonType> legacy(count);
      legacyVar.SetSelection({{static_cast<size_t>(offset)}, {count}});
      engine.Get<cudaCommonType>(legacyVar, legacy.data(), adios2::Mode::Sync);
      for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<cudaPclType_ID>(legacy[i]);
      }
      return;
    }

    var.SetSelection({{static_cast<size_t>(offset)}, {count}});
    engine.Get<cudaPclType_ID>(var, dst, adios2::Mode::Sync);
  }
};

#endif

#ifndef NO_HDF5

class Hdf5ActiveNodeRemapReader {
public:
  void readInto(hid_t file_id, const std::string& datasetPath,
                const RestartNodeCopy& copy,
                const RestartMeshMetadata& sourceMesh, arr3_double dst) {
    hid_t dataset_id = H5Dopen2(file_id, datasetPath.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
      eprintf("ERROR: could not open dataset %s", datasetPath.c_str());
    }

    hid_t file_space = H5Dget_space(dataset_id);
    hsize_t dims[3] = {0, 0, 0};
    const int ndims = H5Sget_simple_extent_dims(file_space, dims, NULL);
    if (ndims != 3) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      eprintf("ERROR: HDF5 restart dataset %s is not rank-3",
              datasetPath.c_str());
    }

    const RestartRankBox sourceBox =
        restartRankBoxFor(sourceMesh, copy.sourceRank);
    const hsize_t expected[3] = {
        static_cast<hsize_t>(sourceBox.activeNodes.nx()),
        static_cast<hsize_t>(sourceBox.activeNodes.ny()),
        static_cast<hsize_t>(sourceBox.activeNodes.nz())};
    if (dims[0] != expected[0] || dims[1] != expected[1] ||
        dims[2] != expected[2]) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      eprintf("ERROR: HDF5 restart dataset %s has unexpected "
              "source active-node shape",
              datasetPath.c_str());
    }

    const hsize_t start[3] = {static_cast<hsize_t>(copy.sourceLocal.x.begin),
                              static_cast<hsize_t>(copy.sourceLocal.y.begin),
                              static_cast<hsize_t>(copy.sourceLocal.z.begin)};
    const hsize_t count[3] = {static_cast<hsize_t>(copy.sourceLocal.nx()),
                              static_cast<hsize_t>(copy.sourceLocal.ny()),
                              static_cast<hsize_t>(copy.sourceLocal.nz())};

    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
    hid_t mem_space = H5Screate_simple(3, count, NULL);

    buffer_.resize(boxValueCount(copy.sourceLocal));
    const herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, mem_space,
                                  file_space, H5P_DEFAULT, buffer_.data());

    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dataset_id);

    if (status < 0) {
      eprintf("ERROR: could not read hyperslab from HDF5 restart "
              "dataset %s",
              datasetPath.c_str());
    }

    scatterActiveNodeBufferToGuardedArray(copy, buffer_.data(), dst);
  }

private:
  std::vector<double> buffer_;
};

class Hdf5ParticleRemapReader {
public:
  void readMetadata(hid_t file_id, int species, int last_cycle,
                    const RestartCellCopy& copy,
                    const RestartMeshMetadata& sourceMesh,
                    RestartParticleCellLayout& layout) {
    readIntDataset(file_id, "/particles/active_cell_dims", dimsBuffer_);
    if (dimsBuffer_.size() != 3) {
      eprintf("ERROR: HDF5 restart active_cell_dims has invalid size");
    }
    layout.activeCellDims = {{dimsBuffer_[0], dimsBuffer_[1], dimsBuffer_[2]}};
    validateSourceActiveCellDims(copy, sourceMesh, layout.activeCellDims);

    const std::string cycle = "cycle_" + std::to_string(last_cycle);
    const std::string speciesPath =
        "/particles/species_" + std::to_string(species);

    readIntDataset(file_id, speciesPath + "/cell_offsets/" + cycle,
                   layout.offsets);
    readIntDataset(file_id, speciesPath + "/cell_counts/" + cycle,
                   layout.counts);
  }

  void readChunk(hid_t file_id, int species, int last_cycle, long long offset,
                 size_t count, RestartParticleChunkBuffers& buffers,
                 bool trackParticleID) {
    buffers.resize(count, trackParticleID);
    const std::string cycle = "cycle_" + std::to_string(last_cycle);
    const std::string speciesPath =
        "/particles/species_" + std::to_string(species);

    readDoubleSelection(file_id, speciesPath + "/u/" + cycle, offset, count,
                        buffers.u.data());
    readDoubleSelection(file_id, speciesPath + "/v/" + cycle, offset, count,
                        buffers.v.data());
    readDoubleSelection(file_id, speciesPath + "/w/" + cycle, offset, count,
                        buffers.w.data());
    readDoubleSelection(file_id, speciesPath + "/q/" + cycle, offset, count,
                        buffers.q.data());
    readDoubleSelection(file_id, speciesPath + "/x/" + cycle, offset, count,
                        buffers.x.data());
    readDoubleSelection(file_id, speciesPath + "/y/" + cycle, offset, count,
                        buffers.y.data());
    readDoubleSelection(file_id, speciesPath + "/z/" + cycle, offset, count,
                        buffers.z.data());
    if (trackParticleID)
      readIDSelectionIfPresent(file_id, speciesPath + "/ID/" + cycle, offset,
                               count, buffers.id.data());
  }

private:
  void readIntDataset(hid_t file_id, const std::string& datasetPath,
                      std::vector<int>& dst) {
    hid_t dataset_id = H5Dopen2(file_id, datasetPath.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
      eprintf("ERROR: could not open HDF5 restart dataset %s",
              datasetPath.c_str());
    }

    hid_t space_id = H5Dget_space(dataset_id);
    hsize_t dims[1] = {0};
    const int ndims = H5Sget_simple_extent_dims(space_id, dims, NULL);
    if (ndims != 1) {
      H5Sclose(space_id);
      H5Dclose(dataset_id);
      eprintf("ERROR: HDF5 restart dataset %s is not rank-1",
              datasetPath.c_str());
    }

    dst.resize(static_cast<size_t>(dims[0]));
    const herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, dst.data());
    H5Sclose(space_id);
    H5Dclose(dataset_id);

    if (status < 0) {
      eprintf("ERROR: could not read HDF5 restart dataset %s",
              datasetPath.c_str());
    }
  }

  template <typename T>
  void readSelection(hid_t file_id, const std::string& datasetPath,
                     hid_t nativeType, long long offset, size_t count, T* dst) {
    hid_t dataset_id = H5Dopen2(file_id, datasetPath.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
      eprintf("ERROR: could not open HDF5 restart dataset %s",
              datasetPath.c_str());
    }

    hid_t file_space = H5Dget_space(dataset_id);
    hsize_t dims[1] = {0};
    const int ndims = H5Sget_simple_extent_dims(file_space, dims, NULL);
    if (ndims != 1) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      eprintf("ERROR: HDF5 restart dataset %s is not rank-1",
              datasetPath.c_str());
    }
    if (offset < 0 || offset + static_cast<long long>(count) >
                          static_cast<long long>(dims[0])) {
      H5Sclose(file_space);
      H5Dclose(dataset_id);
      eprintf("ERROR: HDF5 restart particle selection is outside "
              "dataset %s",
              datasetPath.c_str());
    }

    const hsize_t start[1] = {static_cast<hsize_t>(offset)};
    const hsize_t selectedCount[1] = {static_cast<hsize_t>(count)};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, selectedCount,
                        NULL);
    hid_t mem_space = H5Screate_simple(1, selectedCount, NULL);

    const herr_t status = H5Dread(dataset_id, nativeType, mem_space, file_space,
                                  H5P_DEFAULT, dst);

    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dataset_id);

    if (status < 0) {
      eprintf("ERROR: could not read HDF5 restart particle selection "
              "from %s",
              datasetPath.c_str());
    }
  }

  void readDoubleSelection(hid_t file_id, const std::string& datasetPath,
                           long long offset, size_t count, double* dst) {
    readSelection(file_id, datasetPath, H5T_NATIVE_DOUBLE, offset, count, dst);
  }

  void readIDSelectionIfPresent(hid_t file_id, const std::string& datasetPath,
                                long long offset, size_t count,
                                cudaPclType_ID* dst) {
    const htri_t exists = H5Lexists(file_id, datasetPath.c_str(), H5P_DEFAULT);
    if (exists <= 0)
      return;

    readSelection(file_id, datasetPath, H5T_NATIVE_UINT64, offset, count, dst);
  }

  std::vector<int> dimsBuffer_;
};

#endif

struct RestartReadContext {
  RestartReadContext(const std::string& restartDir_,
                     const std::string& backend_, int lastCycle_,
                     const VCtopology3D* vct, RestartCheckpoint checkpoint_)
      : checkpoint(std::move(checkpoint_)),
        destinationMesh(makeDestinationMeshForRestart(checkpoint.mesh, vct)),
        plan(checkpoint.mesh, destinationMesh, vct->getCartesian_rank()),
        restartDir(restartDir_), backend(backend_), lastCycle(lastCycle_),
        rank(vct->getCartesian_rank()), xlen(vct->getXLEN()),
        ylen(vct->getYLEN()), zlen(vct->getZLEN()), nranks(vct->getNprocs()) {}

  bool matches(const std::string& restartDir_, const std::string& backend_,
               int lastCycle_, const VCtopology3D* vct) const {
    return restartDir == restartDir_ && backend == backend_ &&
           lastCycle == lastCycle_ && rank == vct->getCartesian_rank() &&
           xlen == vct->getXLEN() && ylen == vct->getYLEN() &&
           zlen == vct->getZLEN() && nranks == vct->getNprocs();
  }

  void validateGridOnce(const Grid* grid) {
    if (!gridValidated) {
      validateDestinationGrid(plan, grid);
      gridValidated = true;
    }
  }

  RestartCheckpoint checkpoint;
  RestartMeshMetadata destinationMesh;
  RestartRemapPlan plan;
  std::string restartDir;
  std::string backend;
  int lastCycle = -1;
  int rank = 0;
  int xlen = 0;
  int ylen = 0;
  int zlen = 0;
  int nranks = 0;
  bool gridValidated = false;
  RestartParticleCellLayout particleCellLayout;
  RestartParticleChunkBuffers particleBuffers;
  std::vector<ParticleReadSpan> particleSpans;
#ifdef USE_ADIOS2
  AdiosActiveNodeRemapReader adiosFieldReader;
  AdiosParticleRemapReader adiosParticleReader;
#endif
#ifndef NO_HDF5
  Hdf5ActiveNodeRemapReader hdf5FieldReader;
  Hdf5ParticleRemapReader hdf5ParticleReader;
#endif
};

std::unique_ptr<RestartReadContext> restartReadContext;

RestartReadContext& contextForRestartRead(const std::string& restartDir,
                                          int lastCycle,
                                          const VCtopology3D* vct,
                                          const Grid* grid) {
  const std::string backend = RestartSlotManager::backendName();
  if (backend.empty()) {
    eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without "
            "NO_HDF5).");
  }

  if (restartReadContext &&
      restartReadContext->matches(restartDir, backend, lastCycle, vct)) {
    if (grid)
      restartReadContext->validateGridOnce(grid);
    return *restartReadContext;
  }

  RestartCheckpoint checkpoint =
      RestartSlotManager::readManifest(restartDir, backend);
  if (!checkpoint.found) {
    eprintf("ERROR: restart directory %s does not contain a restart "
            "manifest",
            restartDir.c_str());
  }
  if (!checkpoint.mesh.valid) {
    eprintf("ERROR: restart manifest is missing source mesh metadata");
  }
  if (checkpoint.cycle != lastCycle) {
    eprintf("ERROR: selected restart cycle label %d does not match "
            "manifest cycle label %d",
            lastCycle, checkpoint.cycle);
  }

  std::unique_ptr<RestartReadContext> newContext(new RestartReadContext(
      restartDir, backend, lastCycle, vct, std::move(checkpoint)));
  if (grid)
    newContext->validateGridOnce(grid);
  restartReadContext = std::move(newContext);
  return *restartReadContext;
}

#ifdef USE_ADIOS2

void readAdiosFieldsRemapped(RestartReadContext& context, arr3_double Bxn,
                             arr3_double Byn, arr3_double Bzn, arr3_double Ex,
                             arr3_double Ey, arr3_double Ez,
                             array4_double* rhons, int ns, const Grid* grid) {
  const RestartRemapPlan& plan = context.plan;

  for (const RestartNodeCopy& copy : plan.nodeCopies()) {
    const std::string name_file = restartRankFilePath(
        context.restartDir, context.backend, copy.sourceRank);

    adios2::ADIOS adios;
    adios2::IO ioField =
        adios.DeclareIO("FieldRestartRemap" + std::to_string(copy.sourceRank));
    ioField.SetEngine("BP5");
    adios2::Engine engineField = ioField.Open(name_file, adios2::Mode::Read);

    if (!beginLastAdiosStep(engineField, context.lastCycle)) {
      engineField.Close();
      eprintf("ERROR: ADIOS2 restart file did not contain a readable step");
    }

    context.adiosFieldReader.readInto(ioField, engineField, "Bx", copy,
                                      plan.sourceMesh(), Bxn);
    context.adiosFieldReader.readInto(ioField, engineField, "By", copy,
                                      plan.sourceMesh(), Byn);
    context.adiosFieldReader.readInto(ioField, engineField, "Bz", copy,
                                      plan.sourceMesh(), Bzn);
    context.adiosFieldReader.readInto(ioField, engineField, "Ex", copy,
                                      plan.sourceMesh(), Ex);
    context.adiosFieldReader.readInto(ioField, engineField, "Ey", copy,
                                      plan.sourceMesh(), Ey);
    context.adiosFieldReader.readInto(ioField, engineField, "Ez", copy,
                                      plan.sourceMesh(), Ez);

    for (int i = 0; i < ns; i++) {
      arr3_double rhoSpecies(rhons->fetch_arr4()[i], grid->getNXN(),
                             grid->getNYN(), grid->getNZN());
      context.adiosFieldReader.readInto(ioField, engineField,
                                        "rhosSpecies" + std::to_string(i), copy,
                                        plan.sourceMesh(), rhoSpecies);
    }

    engineField.EndStep();
    engineField.Close();
  }
}

void collectAdiosParticleSpans(RestartReadContext& context, int species,
                               std::vector<ParticleReadSpan>& spans) {
  const RestartRemapPlan& plan = context.plan;

  for (const RestartCellCopy& copy : plan.cellCopies()) {
    const std::string name_file = restartRankFilePath(
        context.restartDir, context.backend, copy.sourceRank);

    adios2::ADIOS adios;
    adios2::IO ioParticle = adios.DeclareIO("ParticleRestartMetadata" +
                                            std::to_string(copy.sourceRank));
    ioParticle.SetEngine("BP5");
    adios2::Engine engineParticle =
        ioParticle.Open(name_file, adios2::Mode::Read);

    if (!beginLastAdiosStep(engineParticle, context.lastCycle)) {
      engineParticle.Close();
      eprintf("ERROR: ADIOS2 restart particle file did not contain "
              "a readable step");
    }

    context.adiosParticleReader.readMetadata(ioParticle, engineParticle,
                                             species, copy, plan.sourceMesh(),
                                             context.particleCellLayout);
    appendParticleSpansForCopy(copy, context.particleCellLayout.activeCellDims,
                               context.particleCellLayout.offsets,
                               context.particleCellLayout.counts, spans);

    engineParticle.EndStep();
    engineParticle.Close();
  }
}

void readAdiosParticlesRemapped(RestartReadContext& context, int species,
                                vector_double& u, vector_double& v,
                                vector_double& w, vector_double& q,
                                vector_double& x, vector_double& y,
                                vector_double& z, vector_cudaPclType_ID& id,
                                bool trackParticleID) {
  std::vector<ParticleReadSpan>& spans = context.particleSpans;
  spans.clear();
  collectAdiosParticleSpans(context, species, spans);

  const long long totalParticles = totalParticleSpanCount(spans);
  resizeParticleVectors(totalParticles, u, v, w, q, x, y, z, id,
                        trackParticleID);
  if (totalParticles == 0)
    return;

  long long destinationOffset = 0;
  size_t spanIndex = 0;
  while (spanIndex < spans.size()) {
    const int sourceRank = spans[spanIndex].sourceRank;
    const std::string name_file =
        restartRankFilePath(context.restartDir, context.backend, sourceRank);

    adios2::ADIOS adios;
    adios2::IO ioParticle =
        adios.DeclareIO("ParticleRestartRead" + std::to_string(sourceRank));
    ioParticle.SetEngine("BP5");
    adios2::Engine engineParticle =
        ioParticle.Open(name_file, adios2::Mode::Read);

    if (!beginLastAdiosStep(engineParticle, context.lastCycle)) {
      engineParticle.Close();
      eprintf("ERROR: ADIOS2 restart particle file did not contain "
              "a readable step");
    }

    while (spanIndex < spans.size() &&
           spans[spanIndex].sourceRank == sourceRank) {
      const ParticleReadSpan& span = spans[spanIndex];
      readParticleSpanChunks(span, context.particleBuffers, destinationOffset,
                             u, v, w, q, x, y, z, id, trackParticleID,
                             [&](long long sourceOffset, size_t chunk,
                                 RestartParticleChunkBuffers& buffers) {
                               context.adiosParticleReader.readChunk(
                                   ioParticle, engineParticle, species,
                                   sourceOffset, chunk, buffers,
                                   trackParticleID);
                             });
      ++spanIndex;
    }

    engineParticle.EndStep();
    engineParticle.Close();
  }
}

#endif

#ifndef NO_HDF5

void readHdf5FieldsRemapped(RestartReadContext& context, arr3_double Bxn,
                            arr3_double Byn, arr3_double Bzn, arr3_double Ex,
                            arr3_double Ey, arr3_double Ez,
                            array4_double* rhons, int ns, const Grid* grid) {
  const RestartRemapPlan& plan = context.plan;
  const std::string cycle_str = "cycle_" + std::to_string(context.lastCycle);

  for (const RestartNodeCopy& copy : plan.nodeCopies()) {
    const std::string name_file = restartRankFilePath(
        context.restartDir, context.backend, copy.sourceRank);

    hid_t file_id = H5Fopen(name_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      eprintf("ERROR: could not open HDF5 restart file: %s", name_file.c_str());
    }

    context.hdf5FieldReader.readInto(file_id, "/fields/Bx/" + cycle_str, copy,
                                     plan.sourceMesh(), Bxn);
    context.hdf5FieldReader.readInto(file_id, "/fields/By/" + cycle_str, copy,
                                     plan.sourceMesh(), Byn);
    context.hdf5FieldReader.readInto(file_id, "/fields/Bz/" + cycle_str, copy,
                                     plan.sourceMesh(), Bzn);
    context.hdf5FieldReader.readInto(file_id, "/fields/Ex/" + cycle_str, copy,
                                     plan.sourceMesh(), Ex);
    context.hdf5FieldReader.readInto(file_id, "/fields/Ey/" + cycle_str, copy,
                                     plan.sourceMesh(), Ey);
    context.hdf5FieldReader.readInto(file_id, "/fields/Ez/" + cycle_str, copy,
                                     plan.sourceMesh(), Ez);

    for (int i = 0; i < ns; i++) {
      arr3_double rhoSpecies(rhons->fetch_arr4()[i], grid->getNXN(),
                             grid->getNYN(), grid->getNZN());
      const string dsPath =
          "/moments/species_" + std::to_string(i) + "/rho/" + cycle_str;
      context.hdf5FieldReader.readInto(file_id, dsPath, copy, plan.sourceMesh(),
                                       rhoSpecies);
    }

    H5Fclose(file_id);
  }
}

void collectHdf5ParticleSpans(RestartReadContext& context, int species,
                              std::vector<ParticleReadSpan>& spans) {
  const RestartRemapPlan& plan = context.plan;

  for (const RestartCellCopy& copy : plan.cellCopies()) {
    const std::string name_file = restartRankFilePath(
        context.restartDir, context.backend, copy.sourceRank);

    hid_t file_id = H5Fopen(name_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      eprintf("ERROR: could not open HDF5 restart file: %s", name_file.c_str());
    }

    context.hdf5ParticleReader.readMetadata(file_id, species, context.lastCycle,
                                            copy, plan.sourceMesh(),
                                            context.particleCellLayout);
    appendParticleSpansForCopy(copy, context.particleCellLayout.activeCellDims,
                               context.particleCellLayout.offsets,
                               context.particleCellLayout.counts, spans);
    H5Fclose(file_id);
  }
}

void readHdf5ParticlesRemapped(RestartReadContext& context, int species,
                               vector_double& u, vector_double& v,
                               vector_double& w, vector_double& q,
                               vector_double& x, vector_double& y,
                               vector_double& z, vector_cudaPclType_ID& id,
                               bool trackParticleID) {
  std::vector<ParticleReadSpan>& spans = context.particleSpans;
  spans.clear();
  collectHdf5ParticleSpans(context, species, spans);

  const long long totalParticles = totalParticleSpanCount(spans);
  resizeParticleVectors(totalParticles, u, v, w, q, x, y, z, id,
                        trackParticleID);
  if (totalParticles == 0)
    return;

  long long destinationOffset = 0;
  size_t spanIndex = 0;
  while (spanIndex < spans.size()) {
    const int sourceRank = spans[spanIndex].sourceRank;
    const std::string name_file =
        restartRankFilePath(context.restartDir, context.backend, sourceRank);

    hid_t file_id = H5Fopen(name_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      eprintf("ERROR: could not open HDF5 restart file: %s", name_file.c_str());
    }

    while (spanIndex < spans.size() &&
           spans[spanIndex].sourceRank == sourceRank) {
      const ParticleReadSpan& span = spans[spanIndex];
      readParticleSpanChunks(span, context.particleBuffers, destinationOffset,
                             u, v, w, q, x, y, z, id, trackParticleID,
                             [&](long long sourceOffset, size_t chunk,
                                 RestartParticleChunkBuffers& buffers) {
                               context.hdf5ParticleReader.readChunk(
                                   file_id, species, context.lastCycle,
                                   sourceOffset, chunk, buffers,
                                   trackParticleID);
                             });
      ++spanIndex;
    }

    H5Fclose(file_id);
  }
}

#endif

} // namespace

RestartCheckpoint
RestartReader::resolveLatestCheckpoint(const std::string& restartDir) {
  const std::string backend = RestartSlotManager::backendName();
  if (backend.empty()) {
    eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without "
            "NO_HDF5).");
  }

#if !defined(USE_ADIOS2) && !defined(NO_HDF5)
  if (MPIdata::get_rank() == 0) {
    printf("\n");
    printf("==================================================================="
           "======\n");
    printf("  WARNING: HDF5 restart is a Beta feature. Use with caution!\n");
    printf("==================================================================="
           "======\n");
    printf("\n");
  }
#endif

  RestartCheckpoint checkpoint =
      RestartSlotManager::resolveLatest(restartDir, backend);
  if (checkpoint.found) {
    if (MPIdata::get_rank() == 0) {
      std::cout << "[*] Restart checkpoint ("
                << restartBackendLabel(checkpoint.backend) << ") = restart_"
                << checkpoint.slot << ", cycle label = " << checkpoint.cycle
                << std::endl;
    }
    return checkpoint;
  }

  eprintf("ERROR: no restart checkpoint found in %s. "
          "Legacy flat restarts are unsupported.",
          restartDir.c_str());
  return RestartCheckpoint();
}

// ===========================================================================
// readFields  —  EM fields + species charge densities
// ===========================================================================

void RestartReader::readFields(const VCtopology3D* vct, const Grid* grid,
                               arr3_double Bxn, arr3_double Byn,
                               arr3_double Bzn, arr3_double Ex, arr3_double Ey,
                               arr3_double Ez, array4_double* rhons_, int ns,
                               const std::string& restartDir, int last_cycle) {
#ifdef USE_ADIOS2
  RestartReadContext& context =
      contextForRestartRead(restartDir, last_cycle, vct, grid);
  if (context.checkpoint.mesh.ns < ns) {
    eprintf("ERROR: restart manifest has fewer species than this run");
  }

  if (vct->getCartesian_rank() == 0) {
    printRestartReadSignature("Fields", context.backend, "active-node",
                              last_cycle);
  }

  readAdiosFieldsRemapped(context, Bxn, Byn, Bzn, Ex, Ey, Ez, rhons_, ns, grid);

#elif !defined(NO_HDF5)
  RestartReadContext& context =
      contextForRestartRead(restartDir, last_cycle, vct, grid);
  if (context.checkpoint.mesh.ns < ns) {
    eprintf("ERROR: restart manifest has fewer species than this run");
  }

  if (vct->getCartesian_rank() == 0) {
    printRestartReadSignature("Fields", context.backend, "active-node",
                              last_cycle);
  }

  readHdf5FieldsRemapped(context, Bxn, Byn, Bzn, Ex, Ey, Ez, rhons_, ns, grid);

#else
  eprintf(
      "Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
#endif
}

// ===========================================================================
// readParticles  —  position, velocity, charge, and optional ID for one species
// ===========================================================================

void RestartReader::readParticles(
    const VCtopology3D* vct, int species_number, vector_double& u,
    vector_double& v, vector_double& w, vector_double& q, vector_double& x,
    vector_double& y, vector_double& z, vector_cudaPclType_ID& id,
    bool trackParticleID, const std::string& restartDir, int last_cycle) {
#ifdef USE_ADIOS2
  RestartReadContext& context =
      contextForRestartRead(restartDir, last_cycle, vct, nullptr);
  if (species_number >= context.checkpoint.mesh.ns) {
    eprintf("ERROR: requested restart species is not present in checkpoint");
  }

  if (vct->getCartesian_rank() == 0 && species_number == 0) {
    printRestartReadSignature("Particle", context.backend, "active-cell",
                              last_cycle);
  }

  readAdiosParticlesRemapped(context, species_number, u, v, w, q, x, y, z, id,
                             trackParticleID);

#elif !defined(NO_HDF5)
  RestartReadContext& context =
      contextForRestartRead(restartDir, last_cycle, vct, nullptr);
  if (species_number >= context.checkpoint.mesh.ns) {
    eprintf("ERROR: requested restart species is not present in checkpoint");
  }

  if (vct->getCartesian_rank() == 0 && species_number == 0) {
    printRestartReadSignature("Particle", context.backend, "active-cell",
                              last_cycle);
  }

  readHdf5ParticlesRemapped(context, species_number, u, v, w, q, x, y, z, id,
                            trackParticleID);

#else
  eprintf(
      "Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
#endif
}
