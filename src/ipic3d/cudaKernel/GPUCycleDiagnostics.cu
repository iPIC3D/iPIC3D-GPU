#include "GPUCycleDiagnostics.cuh"

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)

#include "ParticleSoADevice.cuh"
#include "cellSortBuffers.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr int DIAG_THREADS = 256;
constexpr int DIAG_MAX_BLOCKS = 256;
constexpr int DIAG_MAX_FIELDS = 16;
constexpr unsigned long long DIAG_INVALID_INDEX = ULLONG_MAX;

static_assert(std::is_same<cudaSolverType, double>::value,
              "GPU cycle diagnostics currently reduce double solver arrays");
static_assert(std::is_same<cudaPclType_X, double>::value &&
                  std::is_same<cudaPclType_U, double>::value &&
                  std::is_same<cudaPclType_Q, double>::value,
              "GPU cycle diagnostics currently reduce double particle SoA");
static_assert(DIAG_MAX_BLOCKS == CELL_SORT_DIAGNOSTIC_MAX_BLOCKS,
              "sorter and cycle diagnostics must use the same block bound");

// ======= Cell-sort diagnostic reduction records =======

__host__ __device__ unsigned long long magnitude(long long value) {
  return value < 0 ? 0ull - static_cast<unsigned long long>(value)
                   : static_cast<unsigned long long>(value);
}

struct SortHistogramPartial {
  long long sum;
  unsigned long long negativeCount;
  unsigned long long firstNegativeCell;
  int firstNegativeValue;
  int minCount;
  int maxCount;
  unsigned long long minCell;
  unsigned long long maxCell;
};

__host__ __device__ SortHistogramPartial emptySortHistogramPartial() {
  SortHistogramPartial p{};
  p.firstNegativeCell = DIAG_INVALID_INDEX;
  p.minCount = INT_MAX;
  p.maxCount = INT_MIN;
  p.minCell = DIAG_INVALID_INDEX;
  p.maxCell = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortHistogramPartial(SortHistogramPartial& dst,
                            const SortHistogramPartial& src) {
  dst.sum += src.sum;
  dst.negativeCount += src.negativeCount;
  if (src.firstNegativeCell < dst.firstNegativeCell) {
    dst.firstNegativeCell = src.firstNegativeCell;
    dst.firstNegativeValue = src.firstNegativeValue;
  }
  if (src.minCell != DIAG_INVALID_INDEX &&
      (dst.minCell == DIAG_INVALID_INDEX || src.minCount < dst.minCount ||
       (src.minCount == dst.minCount && src.minCell < dst.minCell))) {
    dst.minCount = src.minCount;
    dst.minCell = src.minCell;
  }
  if (src.maxCell != DIAG_INVALID_INDEX &&
      (dst.maxCell == DIAG_INVALID_INDEX || src.maxCount > dst.maxCount ||
       (src.maxCount == dst.maxCount && src.maxCell < dst.maxCell))) {
    dst.maxCount = src.maxCount;
    dst.maxCell = src.maxCell;
  }
}

struct SortPrefixPartial {
  unsigned long long recurrenceMismatchCount;
  unsigned long long firstRecurrenceCell;
  long long firstRecurrenceActual;
  long long firstRecurrenceExpected;
  unsigned long long maxRecurrenceError;
  unsigned long long worstRecurrenceCell;
  long long worstRecurrenceActual;
  long long worstRecurrenceExpected;
  unsigned long long invalidIntervalCount;
  unsigned long long firstInvalidIntervalCell;
  long long firstInvalidBegin;
  long long firstInvalidEnd;
  long long startZero;
  long long terminal;
};

__host__ __device__ SortPrefixPartial emptySortPrefixPartial() {
  SortPrefixPartial p{};
  p.firstRecurrenceCell = DIAG_INVALID_INDEX;
  p.worstRecurrenceCell = DIAG_INVALID_INDEX;
  p.firstInvalidIntervalCell = DIAG_INVALID_INDEX;
  p.startZero = LLONG_MIN;
  p.terminal = LLONG_MIN;
  return p;
}

__host__ __device__ void
combineSortPrefixPartial(SortPrefixPartial& dst, const SortPrefixPartial& src) {
  dst.recurrenceMismatchCount += src.recurrenceMismatchCount;
  if (src.firstRecurrenceCell < dst.firstRecurrenceCell) {
    dst.firstRecurrenceCell = src.firstRecurrenceCell;
    dst.firstRecurrenceActual = src.firstRecurrenceActual;
    dst.firstRecurrenceExpected = src.firstRecurrenceExpected;
  }
  if (src.worstRecurrenceCell != DIAG_INVALID_INDEX &&
      (dst.worstRecurrenceCell == DIAG_INVALID_INDEX ||
       src.maxRecurrenceError > dst.maxRecurrenceError ||
       (src.maxRecurrenceError == dst.maxRecurrenceError &&
        src.worstRecurrenceCell < dst.worstRecurrenceCell))) {
    dst.maxRecurrenceError = src.maxRecurrenceError;
    dst.worstRecurrenceCell = src.worstRecurrenceCell;
    dst.worstRecurrenceActual = src.worstRecurrenceActual;
    dst.worstRecurrenceExpected = src.worstRecurrenceExpected;
  }
  dst.invalidIntervalCount += src.invalidIntervalCount;
  if (src.firstInvalidIntervalCell < dst.firstInvalidIntervalCell) {
    dst.firstInvalidIntervalCell = src.firstInvalidIntervalCell;
    dst.firstInvalidBegin = src.firstInvalidBegin;
    dst.firstInvalidEnd = src.firstInvalidEnd;
  }
  if (src.startZero != LLONG_MIN)
    dst.startZero = src.startZero;
  if (src.terminal != LLONG_MIN)
    dst.terminal = src.terminal;
}

struct SortReservationPartial {
  long long actualSum;
  unsigned long long mismatchCount;
  unsigned long long firstMismatchCell;
  long long firstActual;
  long long firstExpected;
  unsigned long long maxError;
  unsigned long long worstMismatchCell;
  long long worstActual;
  long long worstExpected;
};

__host__ __device__ SortReservationPartial emptySortReservationPartial() {
  SortReservationPartial p{};
  p.firstMismatchCell = DIAG_INVALID_INDEX;
  p.worstMismatchCell = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortReservationPartial(SortReservationPartial& dst,
                              const SortReservationPartial& src) {
  dst.actualSum += src.actualSum;
  dst.mismatchCount += src.mismatchCount;
  if (src.firstMismatchCell < dst.firstMismatchCell) {
    dst.firstMismatchCell = src.firstMismatchCell;
    dst.firstActual = src.firstActual;
    dst.firstExpected = src.firstExpected;
  }
  if (src.worstMismatchCell != DIAG_INVALID_INDEX &&
      (dst.worstMismatchCell == DIAG_INVALID_INDEX ||
       src.maxError > dst.maxError ||
       (src.maxError == dst.maxError &&
        src.worstMismatchCell < dst.worstMismatchCell))) {
    dst.maxError = src.maxError;
    dst.worstMismatchCell = src.worstMismatchCell;
    dst.worstActual = src.worstActual;
    dst.worstExpected = src.worstExpected;
  }
}

struct SortTilePartial {
  unsigned long long baseMismatchCount;
  unsigned long long firstBaseCell;
  long long firstBaseActual;
  long long firstBaseExpected;
  unsigned long long localMismatchCount;
  unsigned long long firstLocalCell;
  long long firstLocalActual;
  long long firstLocalExpected;
  unsigned long long terminalMismatchCount;
  unsigned long long firstTerminalCell;
  long long firstTerminalActual;
  long long firstTerminalExpected;
};

__host__ __device__ SortTilePartial emptySortTilePartial() {
  SortTilePartial p{};
  p.firstBaseCell = DIAG_INVALID_INDEX;
  p.firstLocalCell = DIAG_INVALID_INDEX;
  p.firstTerminalCell = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void combineSortTilePartial(SortTilePartial& dst,
                                                const SortTilePartial& src) {
  dst.baseMismatchCount += src.baseMismatchCount;
  if (src.firstBaseCell < dst.firstBaseCell) {
    dst.firstBaseCell = src.firstBaseCell;
    dst.firstBaseActual = src.firstBaseActual;
    dst.firstBaseExpected = src.firstBaseExpected;
  }
  dst.localMismatchCount += src.localMismatchCount;
  if (src.firstLocalCell < dst.firstLocalCell) {
    dst.firstLocalCell = src.firstLocalCell;
    dst.firstLocalActual = src.firstLocalActual;
    dst.firstLocalExpected = src.firstLocalExpected;
  }
  dst.terminalMismatchCount += src.terminalMismatchCount;
  if (src.firstTerminalCell < dst.firstTerminalCell) {
    dst.firstTerminalCell = src.firstTerminalCell;
    dst.firstTerminalActual = src.firstTerminalActual;
    dst.firstTerminalExpected = src.firstTerminalExpected;
  }
}

struct SortPhase1TotalPartial {
  unsigned long long checkedCount;
  unsigned long long checkedCellCount;
  unsigned long long mismatchCount;
  long long rawSum;
  long long expectedSum;
  unsigned long long expectedOutOfIntRangeCount;
  unsigned long long firstBlock;
  long long firstRaw;
  long long firstExpected;
  long long firstDelta;
  unsigned long long maxError;
  unsigned long long worstBlock;
  long long worstRaw;
  long long worstExpected;
  long long worstDelta;
};

__host__ __device__ SortPhase1TotalPartial emptySortPhase1TotalPartial() {
  SortPhase1TotalPartial p{};
  p.firstBlock = DIAG_INVALID_INDEX;
  p.worstBlock = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortPhase1TotalPartial(SortPhase1TotalPartial& dst,
                              const SortPhase1TotalPartial& src) {
  dst.checkedCount += src.checkedCount;
  dst.checkedCellCount += src.checkedCellCount;
  dst.mismatchCount += src.mismatchCount;
  dst.rawSum += src.rawSum;
  dst.expectedSum += src.expectedSum;
  dst.expectedOutOfIntRangeCount += src.expectedOutOfIntRangeCount;
  if (src.firstBlock < dst.firstBlock) {
    dst.firstBlock = src.firstBlock;
    dst.firstRaw = src.firstRaw;
    dst.firstExpected = src.firstExpected;
    dst.firstDelta = src.firstDelta;
  }
  if (src.worstBlock != DIAG_INVALID_INDEX &&
      (dst.worstBlock == DIAG_INVALID_INDEX || src.maxError > dst.maxError ||
       (src.maxError == dst.maxError && src.worstBlock < dst.worstBlock))) {
    dst.maxError = src.maxError;
    dst.worstBlock = src.worstBlock;
    dst.worstRaw = src.worstRaw;
    dst.worstExpected = src.worstExpected;
    dst.worstDelta = src.worstDelta;
  }
}

struct SortBlockPrefixPartial {
  unsigned long long mismatchCount;
  unsigned long long firstBlock;
  long long firstActual;
  long long firstExpected;
  unsigned long long maxError;
  unsigned long long worstBlock;
  long long worstActual;
  long long worstExpected;
};

__host__ __device__ SortBlockPrefixPartial emptySortBlockPrefixPartial() {
  SortBlockPrefixPartial p{};
  p.firstBlock = DIAG_INVALID_INDEX;
  p.worstBlock = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortBlockPrefixPartial(SortBlockPrefixPartial& dst,
                              const SortBlockPrefixPartial& src) {
  dst.mismatchCount += src.mismatchCount;
  if (src.firstBlock < dst.firstBlock) {
    dst.firstBlock = src.firstBlock;
    dst.firstActual = src.firstActual;
    dst.firstExpected = src.firstExpected;
  }
  if (src.worstBlock != DIAG_INVALID_INDEX &&
      (dst.worstBlock == DIAG_INVALID_INDEX || src.maxError > dst.maxError ||
       (src.maxError == dst.maxError && src.worstBlock < dst.worstBlock))) {
    dst.maxError = src.maxError;
    dst.worstBlock = src.worstBlock;
    dst.worstActual = src.worstActual;
    dst.worstExpected = src.worstExpected;
  }
}

struct SortPermutationPartial {
  unsigned long long scanned;
  unsigned long long outOfBoundsCount;
  unsigned long long firstOutOfBoundsSource;
  unsigned long long firstOutOfBoundsDestination;
  unsigned long long intervalMismatchCount;
  unsigned long long firstIntervalSource;
  unsigned long long firstIntervalCell;
  unsigned long long firstIntervalDestination;
  long long firstIntervalBegin;
  long long firstIntervalEnd;
  unsigned long long nonfinitePositionCount;
  unsigned long long firstNonfiniteSource;
  unsigned long long firstNonfiniteCell;
  double firstNonfiniteX;
  double firstNonfiniteY;
  double firstNonfiniteZ;
  unsigned long long invalidCellCount;
  unsigned long long firstInvalidCellSource;
  long long firstInvalidCell;
  unsigned long long minDestination;
  unsigned long long minDestinationSource;
  unsigned long long maxDestination;
  unsigned long long maxDestinationSource;
};

__host__ __device__ SortPermutationPartial emptySortPermutationPartial() {
  SortPermutationPartial p{};
  p.firstOutOfBoundsSource = DIAG_INVALID_INDEX;
  p.firstOutOfBoundsDestination = DIAG_INVALID_INDEX;
  p.firstIntervalSource = DIAG_INVALID_INDEX;
  p.firstIntervalCell = DIAG_INVALID_INDEX;
  p.firstIntervalDestination = DIAG_INVALID_INDEX;
  p.firstNonfiniteSource = DIAG_INVALID_INDEX;
  p.firstNonfiniteCell = DIAG_INVALID_INDEX;
  p.firstInvalidCellSource = DIAG_INVALID_INDEX;
  p.minDestination = DIAG_INVALID_INDEX;
  p.minDestinationSource = DIAG_INVALID_INDEX;
  p.maxDestinationSource = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortPermutationPartial(SortPermutationPartial& dst,
                              const SortPermutationPartial& src) {
  dst.scanned += src.scanned;
  dst.outOfBoundsCount += src.outOfBoundsCount;
  if (src.firstOutOfBoundsSource < dst.firstOutOfBoundsSource) {
    dst.firstOutOfBoundsSource = src.firstOutOfBoundsSource;
    dst.firstOutOfBoundsDestination = src.firstOutOfBoundsDestination;
  }
  dst.intervalMismatchCount += src.intervalMismatchCount;
  if (src.firstIntervalSource < dst.firstIntervalSource) {
    dst.firstIntervalSource = src.firstIntervalSource;
    dst.firstIntervalCell = src.firstIntervalCell;
    dst.firstIntervalDestination = src.firstIntervalDestination;
    dst.firstIntervalBegin = src.firstIntervalBegin;
    dst.firstIntervalEnd = src.firstIntervalEnd;
  }
  dst.nonfinitePositionCount += src.nonfinitePositionCount;
  if (src.firstNonfiniteSource < dst.firstNonfiniteSource) {
    dst.firstNonfiniteSource = src.firstNonfiniteSource;
    dst.firstNonfiniteCell = src.firstNonfiniteCell;
    dst.firstNonfiniteX = src.firstNonfiniteX;
    dst.firstNonfiniteY = src.firstNonfiniteY;
    dst.firstNonfiniteZ = src.firstNonfiniteZ;
  }
  dst.invalidCellCount += src.invalidCellCount;
  if (src.firstInvalidCellSource < dst.firstInvalidCellSource) {
    dst.firstInvalidCellSource = src.firstInvalidCellSource;
    dst.firstInvalidCell = src.firstInvalidCell;
  }
  if (src.minDestinationSource != DIAG_INVALID_INDEX &&
      (dst.minDestinationSource == DIAG_INVALID_INDEX ||
       src.minDestination < dst.minDestination ||
       (src.minDestination == dst.minDestination &&
        src.minDestinationSource < dst.minDestinationSource))) {
    dst.minDestination = src.minDestination;
    dst.minDestinationSource = src.minDestinationSource;
  }
  if (src.maxDestinationSource != DIAG_INVALID_INDEX &&
      (dst.maxDestinationSource == DIAG_INVALID_INDEX ||
       src.maxDestination > dst.maxDestination ||
       (src.maxDestination == dst.maxDestination &&
        src.maxDestinationSource < dst.maxDestinationSource))) {
    dst.maxDestination = src.maxDestination;
    dst.maxDestinationSource = src.maxDestinationSource;
  }
}

struct SortOccupancyPartial {
  unsigned long long holes;
  unsigned long long firstHole;
  unsigned long long duplicateSlots;
  unsigned long long excessWrites;
  unsigned long long firstDuplicate;
  unsigned int maxMultiplicity;
  unsigned long long worstSlot;
};

__host__ __device__ SortOccupancyPartial emptySortOccupancyPartial() {
  SortOccupancyPartial p{};
  p.firstHole = DIAG_INVALID_INDEX;
  p.firstDuplicate = DIAG_INVALID_INDEX;
  p.worstSlot = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void
combineSortOccupancyPartial(SortOccupancyPartial& dst,
                            const SortOccupancyPartial& src) {
  dst.holes += src.holes;
  dst.duplicateSlots += src.duplicateSlots;
  dst.excessWrites += src.excessWrites;
  if (src.firstHole < dst.firstHole)
    dst.firstHole = src.firstHole;
  if (src.firstDuplicate < dst.firstDuplicate)
    dst.firstDuplicate = src.firstDuplicate;
  if (src.worstSlot != DIAG_INVALID_INDEX &&
      (dst.worstSlot == DIAG_INVALID_INDEX ||
       src.maxMultiplicity > dst.maxMultiplicity ||
       (src.maxMultiplicity == dst.maxMultiplicity &&
        src.worstSlot < dst.worstSlot))) {
    dst.maxMultiplicity = src.maxMultiplicity;
    dst.worstSlot = src.worstSlot;
  }
}

struct SortPostPartial {
  unsigned long long scanned;
  unsigned long long invalidRangeCount;
  unsigned long long firstInvalidCell;
  long long firstInvalidBegin;
  long long firstInvalidEnd;
  unsigned long long cellMismatchCount;
  unsigned long long firstMismatchPosition;
  unsigned long long firstExpectedCell;
  unsigned long long firstActualCell;
  unsigned long long nonfinitePositionCount;
  unsigned long long firstNonfinitePosition;
};

__host__ __device__ SortPostPartial emptySortPostPartial() {
  SortPostPartial p{};
  p.firstInvalidCell = DIAG_INVALID_INDEX;
  p.firstMismatchPosition = DIAG_INVALID_INDEX;
  p.firstExpectedCell = DIAG_INVALID_INDEX;
  p.firstActualCell = DIAG_INVALID_INDEX;
  p.firstNonfinitePosition = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void combineSortPostPartial(SortPostPartial& dst,
                                                const SortPostPartial& src) {
  dst.scanned += src.scanned;
  dst.invalidRangeCount += src.invalidRangeCount;
  if (src.firstInvalidCell < dst.firstInvalidCell) {
    dst.firstInvalidCell = src.firstInvalidCell;
    dst.firstInvalidBegin = src.firstInvalidBegin;
    dst.firstInvalidEnd = src.firstInvalidEnd;
  }
  dst.cellMismatchCount += src.cellMismatchCount;
  if (src.firstMismatchPosition < dst.firstMismatchPosition) {
    dst.firstMismatchPosition = src.firstMismatchPosition;
    dst.firstExpectedCell = src.firstExpectedCell;
    dst.firstActualCell = src.firstActualCell;
  }
  dst.nonfinitePositionCount += src.nonfinitePositionCount;
  if (src.firstNonfinitePosition < dst.firstNonfinitePosition)
    dst.firstNonfinitePosition = src.firstNonfinitePosition;
}

struct CellSortBlockReport {
  SortHistogramPartial histogram;
  SortPrefixPartial prefix;
  SortReservationPartial reservation;
  SortTilePartial tiles;
  SortPhase1TotalPartial phase1Totals;
  SortBlockPrefixPartial blockPrefix;
  SortPermutationPartial permutation;
  SortOccupancyPartial occupancy;
  SortPostPartial post;
};

struct CellSortLocalReport {
  SortHistogramPartial histogram;
  SortPrefixPartial prefix;
  SortReservationPartial reservation;
  SortTilePartial tiles;
  SortPhase1TotalPartial phase1Totals;
  SortBlockPrefixPartial blockPrefix;
  SortPermutationPartial permutation;
  SortOccupancyPartial occupancy;
  unsigned long long expectedParticles;
  unsigned long long requestedParticles;
  unsigned long long pendingNOP;
  unsigned long long nop;
  unsigned long long capacity;
  unsigned long long indexCapacity;
  unsigned long long scratchBytes;
  int numCells;
  int numScanBlocks;
  int nxc;
  int nyc;
  int nzc;
  int sortPending;
  int metadataFlags;
  int pointerAlias;
  int runtimeWarpSize;
  int compileWarpSize;
  int warpMaskBytes;
  int sorterCompileWarpSize;
  int sorterWarpMaskBytes;
  int sorterAlgorithmVersion;
  int invariantsChecked;
  int phase1TotalsChecked;
  int permutationChecked;
  int occupancyChecked;
  int pendingHostMatches;
  int enqueuePointerMutationMask;
  int enqueueSortBufferMutationMask;
  int enqueueAllocationMetadataMutationMask;
  int requestedCountWasClamped;
  int expectFullSort;
  int sorterPointerNullMask;
  int sorterPointerAlias;
  int firstAliasPointerA;
  int firstAliasPointerB;
  int enqueueFirstPointerField;
  std::uintptr_t enqueueExpectedPointer;
  std::uintptr_t enqueueCurrentPointer;
  int enqueueFirstSortBuffer;
  std::uintptr_t enqueueExpectedSortBuffer;
  std::uintptr_t enqueueCurrentSortBuffer;
  int enqueueFirstAllocationMetadata;
  unsigned long long enqueueExpectedAllocationMetadata;
  unsigned long long enqueueCurrentAllocationMetadata;
};

struct CellSortPostLocalReport {
  SortPostPartial post;
  CellSortStage4DiagnosticPartial stage4[CELL_SORT_DIAGNOSTIC_MAX_FIELDS];
  unsigned long long expectedParticles;
  unsigned long long nop;
  int active;
  int stage4FieldCount;
  int expectedFieldCount;
  int pointerRotationMask;
  int pointerAlias;
  int contextMismatch;
  int stage4Checked;
  int postLayoutChecked;
  int nxc;
  int nyc;
  int nzc;
  int expectFullSort;
  std::uintptr_t oldPointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS];
  std::uintptr_t newPointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS];
  std::uintptr_t oldScratch;
  std::uintptr_t newScratch;
};

struct CellSortPointerSnapshot {
  std::uintptr_t pointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS]{};
  std::uintptr_t scratch = 0;
  unsigned long long expectedParticles = 0;
  unsigned long long nop = 0;
  int cycle = -1;
  int species = -1;
  int fieldCount = 0;
  int active = 0;
  int expectFullSort = 0;
};

__device__ __forceinline__ int diagnosticCellIndex(double x, double y, double z,
                                                   const grid3DCUDA* grid) {
  int cx, cy, cz;
  grid->get_safe_cell(x, y, z, cx, cy, cz);
  return cx + cy * grid->nxc + cz * grid->nxc * grid->nyc;
}

__global__ void reduceSortHistogram(const int* counts, int numCells,
                                    CellSortBlockReport* results) {
  __shared__ SortHistogramPartial shared[DIAG_THREADS];
  SortHistogramPartial local = emptySortHistogramPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long cell =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < static_cast<unsigned long long>(numCells); cell += stride) {
    const int value = counts[cell];
    local.sum += value;
    if (value < 0) {
      ++local.negativeCount;
      if (cell < local.firstNegativeCell) {
        local.firstNegativeCell = cell;
        local.firstNegativeValue = value;
      }
    }
    if (local.minCell == DIAG_INVALID_INDEX || value < local.minCount ||
        (value == local.minCount && cell < local.minCell)) {
      local.minCount = value;
      local.minCell = cell;
    }
    if (local.maxCell == DIAG_INVALID_INDEX || value > local.maxCount ||
        (value == local.maxCount && cell < local.maxCell)) {
      local.maxCount = value;
      local.maxCell = cell;
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortHistogramPartial(shared[threadIdx.x],
                                  shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].histogram = shared[0];
}

__global__ void reduceSortPrefix(const int* starts, const int* counts,
                                 int numCells, std::uint32_t numToSort,
                                 CellSortBlockReport* results) {
  __shared__ SortPrefixPartial shared[DIAG_THREADS];
  SortPrefixPartial local = emptySortPrefixPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long cell =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < static_cast<unsigned long long>(numCells); cell += stride) {
    const long long actual = starts[cell];
    const long long expected =
        cell == 0 ? 0
                  : static_cast<long long>(starts[cell - 1]) +
                        static_cast<long long>(counts[cell - 1]);
    if (actual != expected) {
      const unsigned long long error = magnitude(actual - expected);
      ++local.recurrenceMismatchCount;
      if (cell < local.firstRecurrenceCell) {
        local.firstRecurrenceCell = cell;
        local.firstRecurrenceActual = actual;
        local.firstRecurrenceExpected = expected;
      }
      if (local.worstRecurrenceCell == DIAG_INVALID_INDEX ||
          error > local.maxRecurrenceError ||
          (error == local.maxRecurrenceError &&
           cell < local.worstRecurrenceCell)) {
        local.maxRecurrenceError = error;
        local.worstRecurrenceCell = cell;
        local.worstRecurrenceActual = actual;
        local.worstRecurrenceExpected = expected;
      }
    }
    const long long end = actual + static_cast<long long>(counts[cell]);
    if (actual < 0 || end < actual || end > static_cast<long long>(numToSort)) {
      ++local.invalidIntervalCount;
      if (cell < local.firstInvalidIntervalCell) {
        local.firstInvalidIntervalCell = cell;
        local.firstInvalidBegin = actual;
        local.firstInvalidEnd = end;
      }
    }
    if (cell == 0)
      local.startZero = actual;
    if (cell + 1 == static_cast<unsigned long long>(numCells))
      local.terminal = end;
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortPrefixPartial(shared[threadIdx.x],
                               shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].prefix = shared[0];
}

__global__ void reduceSortReservations(const int* starts, const int* ends,
                                       const int* counts, int numCells,
                                       CellSortBlockReport* results) {
  __shared__ SortReservationPartial shared[DIAG_THREADS];
  SortReservationPartial local = emptySortReservationPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long cell =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < static_cast<unsigned long long>(numCells); cell += stride) {
    const long long actual = static_cast<long long>(ends[cell]) - starts[cell];
    const long long expected = counts[cell];
    local.actualSum += actual;
    if (actual != expected) {
      const unsigned long long error = magnitude(actual - expected);
      ++local.mismatchCount;
      if (cell < local.firstMismatchCell) {
        local.firstMismatchCell = cell;
        local.firstActual = actual;
        local.firstExpected = expected;
      }
      if (local.worstMismatchCell == DIAG_INVALID_INDEX ||
          error > local.maxError ||
          (error == local.maxError && cell < local.worstMismatchCell)) {
        local.maxError = error;
        local.worstMismatchCell = cell;
        local.worstActual = actual;
        local.worstExpected = expected;
      }
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortReservationPartial(shared[threadIdx.x],
                                    shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].reservation = shared[0];
}

__global__ void reduceSortTiles(const int* starts, const int* counts,
                                const int* blockPrefixes, int numCells,
                                CellSortBlockReport* results) {
  __shared__ SortTilePartial shared[DIAG_THREADS];
  SortTilePartial local = emptySortTilePartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long cell =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < static_cast<unsigned long long>(numCells); cell += stride) {
    const int tile = static_cast<int>(cell) / SORT_SCAN_ELEMENTS_PER_BLOCK;
    const int tileStart = tile * SORT_SCAN_ELEMENTS_PER_BLOCK;
    const int uncappedTileEnd = tileStart + SORT_SCAN_ELEMENTS_PER_BLOCK;
    const int tileEnd = uncappedTileEnd < numCells ? uncappedTileEnd : numCells;
    const long long actual = starts[cell];
    if (static_cast<int>(cell) == tileStart) {
      const long long expected = blockPrefixes[tile];
      if (actual != expected) {
        ++local.baseMismatchCount;
        if (cell < local.firstBaseCell) {
          local.firstBaseCell = cell;
          local.firstBaseActual = actual;
          local.firstBaseExpected = expected;
        }
      }
    } else {
      const long long expected =
          static_cast<long long>(starts[cell - 1]) + counts[cell - 1];
      if (actual != expected) {
        ++local.localMismatchCount;
        if (cell < local.firstLocalCell) {
          local.firstLocalCell = cell;
          local.firstLocalActual = actual;
          local.firstLocalExpected = expected;
        }
      }
    }
    if (static_cast<int>(cell) + 1 == tileEnd) {
      long long tileTotal = 0;
      for (int i = tileStart; i < tileEnd; ++i)
        tileTotal += counts[i];
      const long long actualEnd = actual + counts[cell];
      const long long expectedEnd =
          static_cast<long long>(blockPrefixes[tile]) + tileTotal;
      if (actualEnd != expectedEnd) {
        ++local.terminalMismatchCount;
        if (cell < local.firstTerminalCell) {
          local.firstTerminalCell = cell;
          local.firstTerminalActual = actualEnd;
          local.firstTerminalExpected = expectedEnd;
        }
      }
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortTilePartial(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].tiles = shared[0];
}

/** Compare every preserved raw Phase-1 tile total with its 512-bin sum. */
__global__ void reduceSortPhase1Totals(const int* rawBlockTotals,
                                       const int* counts, int numCells,
                                       int numBlocks,
                                       CellSortBlockReport* results) {
  __shared__ SortPhase1TotalPartial shared[DIAG_THREADS];
  SortPhase1TotalPartial local = emptySortPhase1TotalPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long block =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       block < static_cast<unsigned long long>(numBlocks); block += stride) {
    const int tileStart =
        static_cast<int>(block) * SORT_SCAN_ELEMENTS_PER_BLOCK;
    const int uncappedTileEnd = tileStart + SORT_SCAN_ELEMENTS_PER_BLOCK;
    const int tileEnd = uncappedTileEnd < numCells ? uncappedTileEnd : numCells;
    long long expected = 0;
    for (int cell = tileStart; cell < tileEnd; ++cell)
      expected += static_cast<long long>(counts[cell]);

    const long long raw = rawBlockTotals[block];
    const long long delta = raw - expected;
    ++local.checkedCount;
    local.checkedCellCount +=
        static_cast<unsigned long long>(tileEnd - tileStart);
    local.rawSum += raw;
    local.expectedSum += expected;
    if (expected < INT_MIN || expected > INT_MAX)
      ++local.expectedOutOfIntRangeCount;
    if (delta == 0)
      continue;

    const unsigned long long error = magnitude(delta);
    ++local.mismatchCount;
    if (block < local.firstBlock) {
      local.firstBlock = block;
      local.firstRaw = raw;
      local.firstExpected = expected;
      local.firstDelta = delta;
    }
    if (local.worstBlock == DIAG_INVALID_INDEX || error > local.maxError ||
        (error == local.maxError && block < local.worstBlock)) {
      local.maxError = error;
      local.worstBlock = block;
      local.worstRaw = raw;
      local.worstExpected = expected;
      local.worstDelta = delta;
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortPhase1TotalPartial(shared[threadIdx.x],
                                   shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].phase1Totals = shared[0];
}

__global__ void reduceSortBlockPrefixes(const int* blockPrefixes,
                                        const int* counts, int numCells,
                                        int numBlocks,
                                        CellSortBlockReport* results) {
  __shared__ SortBlockPrefixPartial shared[DIAG_THREADS];
  SortBlockPrefixPartial local = emptySortBlockPrefixPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long block =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       block < static_cast<unsigned long long>(numBlocks); block += stride) {
    long long expected = 0;
    if (block > 0) {
      const int previousStart =
          (static_cast<int>(block) - 1) * SORT_SCAN_ELEMENTS_PER_BLOCK;
      const int uncappedPreviousEnd =
          previousStart + SORT_SCAN_ELEMENTS_PER_BLOCK;
      const int previousEnd =
          uncappedPreviousEnd < numCells ? uncappedPreviousEnd : numCells;
      expected = blockPrefixes[block - 1];
      for (int cell = previousStart; cell < previousEnd; ++cell)
        expected += counts[cell];
    }
    const long long actual = blockPrefixes[block];
    if (actual != expected) {
      const unsigned long long error = magnitude(actual - expected);
      ++local.mismatchCount;
      if (block < local.firstBlock) {
        local.firstBlock = block;
        local.firstActual = actual;
        local.firstExpected = expected;
      }
      if (local.worstBlock == DIAG_INVALID_INDEX || error > local.maxError ||
          (error == local.maxError && block < local.worstBlock)) {
        local.maxError = error;
        local.worstBlock = block;
        local.worstActual = actual;
        local.worstExpected = expected;
      }
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortBlockPrefixPartial(shared[threadIdx.x],
                                    shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].blockPrefix = shared[0];
}

__global__ void reduceSortPermutation(const unsigned int* indices,
                                      const double* x, const double* y,
                                      const double* z, const int* starts,
                                      const int* counts, const grid3DCUDA* grid,
                                      int numCells, std::uint32_t numToSort,
                                      CellSortBlockReport* results) {
  __shared__ SortPermutationPartial shared[DIAG_THREADS];
  SortPermutationPartial local = emptySortPermutationPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long source =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       source < numToSort; source += stride) {
    ++local.scanned;
    const unsigned long long destination = indices[source];
    if (local.minDestinationSource == DIAG_INVALID_INDEX ||
        destination < local.minDestination) {
      local.minDestination = destination;
      local.minDestinationSource = source;
    }
    if (local.maxDestinationSource == DIAG_INVALID_INDEX ||
        destination > local.maxDestination) {
      local.maxDestination = destination;
      local.maxDestinationSource = source;
    }
    if (destination >= numToSort) {
      ++local.outOfBoundsCount;
      if (source < local.firstOutOfBoundsSource) {
        local.firstOutOfBoundsSource = source;
        local.firstOutOfBoundsDestination = destination;
      }
    }
    const double px = x[source];
    const double py = y[source];
    const double pz = z[source];
    if (!(isfinite(px) && isfinite(py) && isfinite(pz))) {
      ++local.nonfinitePositionCount;
      if (source < local.firstNonfiniteSource) {
        local.firstNonfiniteSource = source;
        local.firstNonfiniteX = px;
        local.firstNonfiniteY = py;
        local.firstNonfiniteZ = pz;
      }
    }
    const int cell = diagnosticCellIndex(px, py, pz, grid);
    if (source == local.firstNonfiniteSource)
      local.firstNonfiniteCell = cell;
    if (cell < 0 || cell >= numCells) {
      ++local.invalidCellCount;
      if (source < local.firstInvalidCellSource) {
        local.firstInvalidCellSource = source;
        local.firstInvalidCell = cell;
      }
      continue;
    }
    const long long begin = starts[cell];
    const long long end = begin + static_cast<long long>(counts[cell]);
    const long long signedDestination = static_cast<long long>(destination);
    if (signedDestination < begin || signedDestination >= end) {
      ++local.intervalMismatchCount;
      if (source < local.firstIntervalSource) {
        local.firstIntervalSource = source;
        local.firstIntervalCell = cell;
        local.firstIntervalDestination = destination;
        local.firstIntervalBegin = begin;
        local.firstIntervalEnd = end;
      }
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortPermutationPartial(shared[threadIdx.x],
                                    shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].permutation = shared[0];
}

__global__ void markSortDestinations(const unsigned int* indices,
                                     std::uint32_t* occupancy,
                                     std::uint32_t numToSort) {
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long source =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       source < numToSort; source += stride) {
    const unsigned int destination = indices[source];
    if (destination < numToSort)
      atomicAdd(occupancy + destination, 1u);
  }
}

__global__ void reduceSortOccupancy(const std::uint32_t* occupancy,
                                    std::uint32_t numToSort,
                                    CellSortBlockReport* results) {
  __shared__ SortOccupancyPartial shared[DIAG_THREADS];
  SortOccupancyPartial local = emptySortOccupancyPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long slot =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       slot < numToSort; slot += stride) {
    const unsigned int value = occupancy[slot];
    if (value == 0) {
      ++local.holes;
      if (slot < local.firstHole)
        local.firstHole = slot;
    } else if (value > 1) {
      ++local.duplicateSlots;
      local.excessWrites += static_cast<unsigned long long>(value - 1);
      if (slot < local.firstDuplicate)
        local.firstDuplicate = slot;
    }
    if (local.worstSlot == DIAG_INVALID_INDEX ||
        value > local.maxMultiplicity ||
        (value == local.maxMultiplicity && slot < local.worstSlot)) {
      local.maxMultiplicity = value;
      local.worstSlot = slot;
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortOccupancyPartial(shared[threadIdx.x],
                                  shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].occupancy = shared[0];
}

__global__ void reducePostSortLayout(const double* x, const double* y,
                                     const double* z, const int* starts,
                                     const int* counts, int numCells,
                                     std::uint32_t numToSort,
                                     const grid3DCUDA* grid,
                                     CellSortBlockReport* results) {
  __shared__ SortPostPartial shared[DIAG_THREADS];
  SortPostPartial local = emptySortPostPartial();
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long cell =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < static_cast<unsigned long long>(numCells); cell += stride) {
    const long long begin = starts[cell];
    const long long end = begin + static_cast<long long>(counts[cell]);
    if (begin < 0 || end < begin || end > numToSort) {
      ++local.invalidRangeCount;
      if (cell < local.firstInvalidCell) {
        local.firstInvalidCell = cell;
        local.firstInvalidBegin = begin;
        local.firstInvalidEnd = end;
      }
      continue;
    }
    local.scanned += static_cast<unsigned long long>(end - begin);
    for (long long position = begin; position < end; ++position) {
      const double px = x[position];
      const double py = y[position];
      const double pz = z[position];
      if (!(isfinite(px) && isfinite(py) && isfinite(pz))) {
        ++local.nonfinitePositionCount;
        if (static_cast<unsigned long long>(position) <
            local.firstNonfinitePosition)
          local.firstNonfinitePosition = position;
      }
      const int actualCell = diagnosticCellIndex(px, py, pz, grid);
      if (actualCell != static_cast<int>(cell)) {
        ++local.cellMismatchCount;
        if (static_cast<unsigned long long>(position) <
            local.firstMismatchPosition) {
          local.firstMismatchPosition = position;
          local.firstExpectedCell = cell;
          local.firstActualCell = actualCell;
        }
      }
    }
  }
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineSortPostPartial(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    results[blockIdx.x].post = shared[0];
}

struct ArrayPartial {
  double minValue;
  double maxValue;
  double maxAbs;
  double maxAbsValue;
  unsigned long long minIndex;
  unsigned long long maxIndex;
  unsigned long long maxAbsIndex;
  unsigned long long nonfiniteCount;
  unsigned long long firstNonfiniteIndex;
};

__host__ __device__ ArrayPartial emptyArrayPartial() {
  ArrayPartial p;
  p.minValue = INFINITY;
  p.maxValue = -INFINITY;
  p.maxAbs = -1.0;
  p.maxAbsValue = 0.0;
  p.minIndex = DIAG_INVALID_INDEX;
  p.maxIndex = DIAG_INVALID_INDEX;
  p.maxAbsIndex = DIAG_INVALID_INDEX;
  p.nonfiniteCount = 0;
  p.firstNonfiniteIndex = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ bool earlierIndex(unsigned long long lhs,
                                      unsigned long long rhs) {
  return rhs == DIAG_INVALID_INDEX || lhs < rhs;
}

__host__ __device__ void combineArrayPartial(ArrayPartial& dst,
                                             const ArrayPartial& src) {
  if (src.minIndex != DIAG_INVALID_INDEX &&
      (src.minValue < dst.minValue ||
       (src.minValue == dst.minValue &&
        earlierIndex(src.minIndex, dst.minIndex)))) {
    dst.minValue = src.minValue;
    dst.minIndex = src.minIndex;
  }
  if (src.maxIndex != DIAG_INVALID_INDEX &&
      (src.maxValue > dst.maxValue ||
       (src.maxValue == dst.maxValue &&
        earlierIndex(src.maxIndex, dst.maxIndex)))) {
    dst.maxValue = src.maxValue;
    dst.maxIndex = src.maxIndex;
  }
  if (src.maxAbsIndex != DIAG_INVALID_INDEX &&
      (src.maxAbs > dst.maxAbs ||
       (src.maxAbs == dst.maxAbs &&
        earlierIndex(src.maxAbsIndex, dst.maxAbsIndex)))) {
    dst.maxAbs = src.maxAbs;
    dst.maxAbsValue = src.maxAbsValue;
    dst.maxAbsIndex = src.maxAbsIndex;
  }
  dst.nonfiniteCount += src.nonfiniteCount;
  if (earlierIndex(src.firstNonfiniteIndex, dst.firstNonfiniteIndex))
    dst.firstNonfiniteIndex = src.firstNonfiniteIndex;
}

__device__ void accumulateArrayValue(ArrayPartial& p, double value,
                                     unsigned long long index) {
  if (!isfinite(value)) {
    ++p.nonfiniteCount;
    if (index < p.firstNonfiniteIndex)
      p.firstNonfiniteIndex = index;
    return;
  }

  if (value < p.minValue || (value == p.minValue && index < p.minIndex)) {
    p.minValue = value;
    p.minIndex = index;
  }
  if (value > p.maxValue || (value == p.maxValue && index < p.maxIndex)) {
    p.maxValue = value;
    p.maxIndex = index;
  }
  const double absValue = fabs(value);
  if (absValue > p.maxAbs || (absValue == p.maxAbs && index < p.maxAbsIndex)) {
    p.maxAbs = absValue;
    p.maxAbsValue = value;
    p.maxAbsIndex = index;
  }
}

__global__ void reduceArray1D(const double* values, unsigned long long begin,
                              unsigned long long count, double scale,
                              ArrayPartial* blockResults) {
  __shared__ ArrayPartial shared[DIAG_THREADS];
  ArrayPartial local = emptyArrayPartial();

  for (unsigned long long logical =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       logical < count;
       logical += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const unsigned long long index = begin + logical;
    accumulateArrayValue(local, values[index] * scale, index);
  }

  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineArrayPartial(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    blockResults[blockIdx.x] = shared[0];
}

__global__ void reduceArray3D(const double* values, int ny, int nz, int ilo,
                              int ihi, int jlo, int jhi, int klo, int khi,
                              double scale, ArrayPartial* blockResults) {
  __shared__ ArrayPartial shared[DIAG_THREADS];
  ArrayPartial local = emptyArrayPartial();
  const unsigned long long ni = static_cast<unsigned long long>(ihi - ilo);
  const unsigned long long nj = static_cast<unsigned long long>(jhi - jlo);
  const unsigned long long nk = static_cast<unsigned long long>(khi - klo);
  const unsigned long long activeCount = ni * nj * nk;

  for (unsigned long long logical =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       logical < activeCount;
       logical += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const int i = ilo + static_cast<int>(logical / (nj * nk));
    const unsigned long long rem = logical % (nj * nk);
    const int j = jlo + static_cast<int>(rem / nk);
    const int k = klo + static_cast<int>(rem % nk);
    const unsigned long long index =
        (static_cast<unsigned long long>(i) * ny + j) * nz + k;
    accumulateArrayValue(local, values[index] * scale, index);
  }

  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineArrayPartial(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    blockResults[blockIdx.x] = shared[0];
}

struct ParticlePartial {
  unsigned long long nonfiniteParticles;
  unsigned long long firstNonfiniteIndex;
  unsigned long long outsideLocalCount;
  unsigned long long firstOutsideLocalIndex;
  double maxLocalDistance;
  unsigned long long worstLocalIndex;
  unsigned long long outsideGlobalCount;
  unsigned long long firstOutsideGlobalIndex;
  double maxGlobalDistance;
  unsigned long long worstGlobalIndex;
};

__host__ __device__ ParticlePartial emptyParticlePartial() {
  ParticlePartial p;
  p.nonfiniteParticles = 0;
  p.firstNonfiniteIndex = DIAG_INVALID_INDEX;
  p.outsideLocalCount = 0;
  p.firstOutsideLocalIndex = DIAG_INVALID_INDEX;
  p.maxLocalDistance = -1.0;
  p.worstLocalIndex = DIAG_INVALID_INDEX;
  p.outsideGlobalCount = 0;
  p.firstOutsideGlobalIndex = DIAG_INVALID_INDEX;
  p.maxGlobalDistance = -1.0;
  p.worstGlobalIndex = DIAG_INVALID_INDEX;
  return p;
}

__host__ __device__ void combineParticlePartial(ParticlePartial& dst,
                                                const ParticlePartial& src) {
  dst.nonfiniteParticles += src.nonfiniteParticles;
  if (src.firstNonfiniteIndex < dst.firstNonfiniteIndex)
    dst.firstNonfiniteIndex = src.firstNonfiniteIndex;

  dst.outsideLocalCount += src.outsideLocalCount;
  if (src.firstOutsideLocalIndex < dst.firstOutsideLocalIndex)
    dst.firstOutsideLocalIndex = src.firstOutsideLocalIndex;
  if (src.maxLocalDistance > dst.maxLocalDistance ||
      (src.maxLocalDistance == dst.maxLocalDistance &&
       src.worstLocalIndex < dst.worstLocalIndex)) {
    dst.maxLocalDistance = src.maxLocalDistance;
    dst.worstLocalIndex = src.worstLocalIndex;
  }

  dst.outsideGlobalCount += src.outsideGlobalCount;
  if (src.firstOutsideGlobalIndex < dst.firstOutsideGlobalIndex)
    dst.firstOutsideGlobalIndex = src.firstOutsideGlobalIndex;
  if (src.maxGlobalDistance > dst.maxGlobalDistance ||
      (src.maxGlobalDistance == dst.maxGlobalDistance &&
       src.worstGlobalIndex < dst.worstGlobalIndex)) {
    dst.maxGlobalDistance = src.maxGlobalDistance;
    dst.worstGlobalIndex = src.worstGlobalIndex;
  }
}

__device__ double distanceOutsideBox(double x, double y, double z,
                                     const double* lo, const double* hi) {
  const double dx = x < lo[0] ? lo[0] - x : (x > hi[0] ? x - hi[0] : 0.0);
  const double dy = y < lo[1] ? lo[1] - y : (y > hi[1] ? y - hi[1] : 0.0);
  const double dz = z < lo[2] ? lo[2] - z : (z > hi[2] ? z - hi[2] : 0.0);
  return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void reduceParticleBounds(const double* x, const double* y,
                                     const double* z, const double* u,
                                     const double* v, const double* w,
                                     const double* q, unsigned long long begin,
                                     unsigned long long count,
                                     GPUCycleDiagnosticParticleBounds bounds,
                                     ParticlePartial* blockResults) {
  __shared__ ParticlePartial shared[DIAG_THREADS];
  ParticlePartial local = emptyParticlePartial();

  for (unsigned long long logical =
           static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       logical < count;
       logical += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const unsigned long long index = begin + logical;
    const double px = x[index];
    const double py = y[index];
    const double pz = z[index];
    const bool finite = isfinite(px) && isfinite(py) && isfinite(pz) &&
                        isfinite(u[index]) && isfinite(v[index]) &&
                        isfinite(w[index]) && isfinite(q[index]);
    if (!finite) {
      ++local.nonfiniteParticles;
      if (index < local.firstNonfiniteIndex)
        local.firstNonfiniteIndex = index;
    }

    // A nonfinite position has no meaningful distance to either box.
    if (!(isfinite(px) && isfinite(py) && isfinite(pz)))
      continue;

    const double localDistance =
        distanceOutsideBox(px, py, pz, bounds.localMin, bounds.localMax);
    if (localDistance > 0.0) {
      ++local.outsideLocalCount;
      if (index < local.firstOutsideLocalIndex)
        local.firstOutsideLocalIndex = index;
      if (localDistance > local.maxLocalDistance ||
          (localDistance == local.maxLocalDistance &&
           index < local.worstLocalIndex)) {
        local.maxLocalDistance = localDistance;
        local.worstLocalIndex = index;
      }
    }

    const double globalDistance =
        distanceOutsideBox(px, py, pz, bounds.globalMin, bounds.globalMax);
    if (globalDistance > 0.0) {
      ++local.outsideGlobalCount;
      if (index < local.firstOutsideGlobalIndex)
        local.firstOutsideGlobalIndex = index;
      if (globalDistance > local.maxGlobalDistance ||
          (globalDistance == local.maxGlobalDistance &&
           index < local.worstGlobalIndex)) {
        local.maxGlobalDistance = globalDistance;
        local.worstGlobalIndex = index;
      }
    }
  }

  shared[threadIdx.x] = local;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      combineParticlePartial(shared[threadIdx.x], shared[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    blockResults[blockIdx.x] = shared[0];
}

struct LocalArraySummary {
  ArrayPartial stats;
  unsigned long long activeCount;
  int maxIJK[3];
  int firstNonfiniteIJK[3];
  double maxCoordinates[3];
  double firstNonfiniteCoordinates[3];
};

struct ParticleRecord {
  double values[7]; // x, y, z, u, v, w, q
  unsigned long long id;
  unsigned long long index;
  int valid;
  int hasId;
};

struct ParticleLocalReport {
  LocalArraySummary fields[7];
  ParticlePartial bounds;
  unsigned long long requestedBegin;
  unsigned long long requestedCount;
  unsigned long long rangeCount;
  unsigned long long nop;
  unsigned long long capacity;
  int invalidMetadataOrRange;
  ParticleRecord firstNonfinite;
  ParticleRecord firstOutsideLocal;
  ParticleRecord worstOutsideLocal;
  ParticleRecord firstOutsideGlobal;
  ParticleRecord worstOutsideGlobal;
};

int reductionBlocks(unsigned long long count) {
  if (count == 0)
    return 0;
  const unsigned long long needed = (count + DIAG_THREADS - 1) / DIAG_THREADS;
  return static_cast<int>(
      std::min<unsigned long long>(needed, DIAG_MAX_BLOCKS));
}

ArrayPartial foldArrayPartials(const ArrayPartial* partials, int count) {
  ArrayPartial result = emptyArrayPartial();
  for (int i = 0; i < count; ++i)
    combineArrayPartial(result, partials[i]);
  return result;
}

ParticlePartial foldParticlePartials(const ParticlePartial* partials,
                                     int count) {
  ParticlePartial result = emptyParticlePartial();
  for (int i = 0; i < count; ++i)
    combineParticlePartial(result, partials[i]);
  return result;
}

CellSortLocalReport foldCellSortPartials(const CellSortBlockReport* partials,
                                         int cellBlocks, int tileBlocks,
                                         int blockBlocks, int permutationBlocks,
                                         int occupancyBlocks) {
  CellSortLocalReport result{};
  result.histogram = emptySortHistogramPartial();
  result.prefix = emptySortPrefixPartial();
  result.reservation = emptySortReservationPartial();
  result.tiles = emptySortTilePartial();
  result.phase1Totals = emptySortPhase1TotalPartial();
  result.blockPrefix = emptySortBlockPrefixPartial();
  result.permutation = emptySortPermutationPartial();
  result.occupancy = emptySortOccupancyPartial();
  for (int i = 0; i < cellBlocks; ++i) {
    combineSortHistogramPartial(result.histogram, partials[i].histogram);
    combineSortPrefixPartial(result.prefix, partials[i].prefix);
    combineSortReservationPartial(result.reservation, partials[i].reservation);
  }
  for (int i = 0; i < tileBlocks; ++i)
    combineSortTilePartial(result.tiles, partials[i].tiles);
  for (int i = 0; i < blockBlocks; ++i) {
    combineSortPhase1TotalPartial(result.phase1Totals,
                                  partials[i].phase1Totals);
    combineSortBlockPrefixPartial(result.blockPrefix, partials[i].blockPrefix);
  }
  for (int i = 0; i < permutationBlocks; ++i)
    combineSortPermutationPartial(result.permutation, partials[i].permutation);
  for (int i = 0; i < occupancyBlocks; ++i)
    combineSortOccupancyPartial(result.occupancy, partials[i].occupancy);
  return result;
}

SortPostPartial foldPostSortPartials(const CellSortBlockReport* partials,
                                     int count) {
  SortPostPartial result = emptySortPostPartial();
  for (int i = 0; i < count; ++i)
    combineSortPostPartial(result, partials[i].post);
  return result;
}

void combineStage4Partial(CellSortStage4DiagnosticPartial& dst,
                          const CellSortStage4DiagnosticPartial& src) {
  dst.prefixPoisonCount += src.prefixPoisonCount;
  dst.tailPoisonCount += src.tailPoisonCount;
  dst.prefixValueMismatchCount += src.prefixValueMismatchCount;
  dst.tailValueMismatchCount += src.tailValueMismatchCount;
  dst.firstPrefixPoison =
      std::min(dst.firstPrefixPoison, src.firstPrefixPoison);
  dst.firstTailPoison = std::min(dst.firstTailPoison, src.firstTailPoison);
  if (src.prefixPoisonCount > 0)
    dst.lastPrefixPoison = std::max(dst.lastPrefixPoison, src.lastPrefixPoison);
  if (src.tailPoisonCount > 0)
    dst.lastTailPoison = std::max(dst.lastTailPoison, src.lastTailPoison);
  if (src.firstPrefixValueMismatchSource < dst.firstPrefixValueMismatchSource) {
    dst.firstPrefixValueMismatchSource = src.firstPrefixValueMismatchSource;
    dst.firstPrefixValueMismatchDestination =
        src.firstPrefixValueMismatchDestination;
  }
  dst.firstTailValueMismatch =
      std::min(dst.firstTailValueMismatch, src.firstTailValueMismatch);
}

CellSortStage4DiagnosticPartial emptyStage4Partial() {
  CellSortStage4DiagnosticPartial result{};
  result.firstPrefixPoison = DIAG_INVALID_INDEX;
  result.firstTailPoison = DIAG_INVALID_INDEX;
  result.firstPrefixValueMismatchSource = DIAG_INVALID_INDEX;
  result.firstPrefixValueMismatchDestination = DIAG_INVALID_INDEX;
  result.firstTailValueMismatch = DIAG_INVALID_INDEX;
  return result;
}

void decodeCell(unsigned long long cell, int nxc, int nyc, int xyz[3]) {
  if (cell == DIAG_INVALID_INDEX || nxc <= 0 || nyc <= 0) {
    xyz[0] = xyz[1] = xyz[2] = -1;
    return;
  }
  xyz[0] = static_cast<int>(cell % nxc);
  const unsigned long long yz = cell / nxc;
  xyz[1] = static_cast<int>(yz % nyc);
  xyz[2] = static_cast<int>(yz / nyc);
}

bool hasPointerAlias(const ParticleSoADevice& soa, const void* scratch) {
  const void* pointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS + 1];
  int count = 0;
  pointers[count++] = soa.u;
  pointers[count++] = soa.v;
  pointers[count++] = soa.w;
  pointers[count++] = soa.q;
  pointers[count++] = soa.x;
  pointers[count++] = soa.y;
  pointers[count++] = soa.z;
  if (soa.trackParticleID)
    pointers[count++] = soa.id;
  pointers[count++] = scratch;
  for (int i = 0; i < count; ++i) {
    if (!pointers[i])
      return true;
    for (int j = i + 1; j < count; ++j)
      if (pointers[i] == pointers[j])
        return true;
  }
  return false;
}

bool findSorterPointerAlias(
    const ParticleSoADevice& soa, const void* scratch,
    const void* const sorterBuffers[CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT],
    int& firstA, int& firstB) {
  // Stable indices are shared with the metadata names below:
  // u..id = 0..7, scratch = 8, sorter buffers = 9..14.
  constexpr int pointerCount = CELL_SORT_DIAGNOSTIC_MAX_FIELDS + 1 +
                               CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT;
  const void* pointers[pointerCount] = {
      soa.u,
      soa.v,
      soa.w,
      soa.q,
      soa.x,
      soa.y,
      soa.z,
      soa.id,
      scratch,
      sorterBuffers[0],
      sorterBuffers[1],
      sorterBuffers[2],
      sorterBuffers[3],
      sorterBuffers[4],
      sorterBuffers[5],
  };
  firstA = firstB = -1;
  for (int i = 0; i < pointerCount; ++i) {
    if (i == 7 && !soa.trackParticleID)
      continue;
    if (!pointers[i])
      continue;
    for (int j = i + 1; j < pointerCount; ++j) {
      if (j == 7 && !soa.trackParticleID)
        continue;
      if (!pointers[j] || pointers[i] != pointers[j])
        continue;
      // SoA-vs-SoA and SoA-vs-scratch are already reported by
      // hasPointerAlias().  This helper specifically protects operations that
      // reuse scratch while consuming the sorter buffers.
      if (i < 9 && j < 9)
        continue;
      firstA = i;
      firstB = j;
      return true;
    }
  }
  return false;
}

void decodeIndex(unsigned long long index, const GPUCycleDiagnosticRegion& r,
                 int ijk[3], double xyz[3]) {
  if (index == DIAG_INVALID_INDEX) {
    ijk[0] = ijk[1] = ijk[2] = -1;
    xyz[0] = xyz[1] = xyz[2] = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  const unsigned long long yz = static_cast<unsigned long long>(r.ny) * r.nz;
  ijk[0] = static_cast<int>(index / yz);
  const unsigned long long rem = index % yz;
  ijk[1] = static_cast<int>(rem / r.nz);
  ijk[2] = static_cast<int>(rem % r.nz);
  for (int d = 0; d < 3; ++d)
    xyz[d] = r.origin[d] + ijk[d] * r.spacing[d];
}

void cartesianCoordinates(MPI_Comm comm, int rank, int coords[3]) {
  coords[0] = coords[1] = coords[2] = -1;
  int topology = MPI_UNDEFINED;
  MPI_Topo_test(comm, &topology);
  if (topology == MPI_CART)
    MPI_Cart_coords(comm, rank, 3, coords);
}

std::string particleRecordText(const ParticleRecord& record) {
  if (!record.valid)
    return "none";
  std::ostringstream out;
  out << std::scientific << std::setprecision(17) << "index=" << record.index
      << " x=" << record.values[0] << " y=" << record.values[1]
      << " z=" << record.values[2] << " u=" << record.values[3]
      << " v=" << record.values[4] << " w=" << record.values[5]
      << " q=" << record.values[6];
  if (record.hasId)
    out << " id=" << record.id;
  return out.str();
}

} // namespace

struct GPUCycleDiagnostics::Impl {
  ArrayPartial* dArrayPartials = nullptr;
  ArrayPartial* hArrayPartials = nullptr;
  ParticlePartial* dParticlePartials = nullptr;
  ParticlePartial* hParticlePartials = nullptr;
  CellSortBlockReport* dCellSortPartials = nullptr;
  CellSortBlockReport* hCellSortPartials = nullptr;
  CellSortStage4DiagnosticPartial* dStage4Partials = nullptr;
  CellSortStage4DiagnosticPartial* hStage4Partials = nullptr;
  CellSortPointerSnapshot cellSortSnapshot;
  bool postSortLayoutSafe = false;
  std::vector<std::max_align_t> gatherStorage;

  void ensureReductionStorage() {
    if (dArrayPartials)
      return;
    const std::size_t arrayCount =
        static_cast<std::size_t>(DIAG_MAX_FIELDS) * DIAG_MAX_BLOCKS;
    cudaErrChk(cudaMalloc(&dArrayPartials, arrayCount * sizeof(ArrayPartial)));
    cudaErrChk(cudaHostAlloc(&hArrayPartials, arrayCount * sizeof(ArrayPartial),
                             cudaHostAllocDefault));
    cudaErrChk(cudaMalloc(&dParticlePartials,
                          DIAG_MAX_BLOCKS * sizeof(ParticlePartial)));
    cudaErrChk(cudaHostAlloc(&hParticlePartials,
                             DIAG_MAX_BLOCKS * sizeof(ParticlePartial),
                             cudaHostAllocDefault));
  }

  void ensureGatherStorage(std::size_t bytes) {
    const std::size_t elements =
        (bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    if (gatherStorage.size() < elements)
      gatherStorage.resize(elements);
  }

  void ensureCellSortStorage() {
    if (dCellSortPartials)
      return;
    cudaErrChk(cudaMalloc(&dCellSortPartials,
                          DIAG_MAX_BLOCKS * sizeof(CellSortBlockReport)));
    cudaErrChk(cudaHostAlloc(&hCellSortPartials,
                             DIAG_MAX_BLOCKS * sizeof(CellSortBlockReport),
                             cudaHostAllocDefault));
    const std::size_t stage4Count =
        static_cast<std::size_t>(CELL_SORT_DIAGNOSTIC_MAX_FIELDS) *
        CELL_SORT_DIAGNOSTIC_MAX_BLOCKS;
    cudaErrChk(
        cudaMalloc(&dStage4Partials,
                   stage4Count * sizeof(CellSortStage4DiagnosticPartial)));
    cudaErrChk(cudaHostAlloc(
        &hStage4Partials, stage4Count * sizeof(CellSortStage4DiagnosticPartial),
        cudaHostAllocDefault));
  }

  ~Impl() {
    if (dArrayPartials)
      cudaFree(dArrayPartials);
    if (hArrayPartials)
      cudaFreeHost(hArrayPartials);
    if (dParticlePartials)
      cudaFree(dParticlePartials);
    if (hParticlePartials)
      cudaFreeHost(hParticlePartials);
    if (dCellSortPartials)
      cudaFree(dCellSortPartials);
    if (hCellSortPartials)
      cudaFreeHost(hCellSortPartials);
    if (dStage4Partials)
      cudaFree(dStage4Partials);
    if (hStage4Partials)
      cudaFreeHost(hStage4Partials);
  }
};

GPUCycleDiagnostics::GPUCycleDiagnostics() : impl_(nullptr) {}

GPUCycleDiagnostics::~GPUCycleDiagnostics() { delete impl_; }

void GPUCycleDiagnostics::reportArraySet(
    const char* stage, int cycle, int species,
    const char* const* componentNames,
    const cudaSolverType* const* deviceArrays, const double* scale,
    int fieldCount, const GPUCycleDiagnosticRegion& region, cudaStream_t stream,
    MPI_Comm comm) {
  if (fieldCount <= 0)
    return;
  if (fieldCount > DIAG_MAX_FIELDS)
    throw std::runtime_error("GPU cycle diagnostic field batch exceeds 16");
  if (region.nx <= 0 || region.ny <= 0 || region.nz <= 0 || region.lo[0] < 0 ||
      region.lo[1] < 0 || region.lo[2] < 0 || region.hi[0] > region.nx ||
      region.hi[1] > region.ny || region.hi[2] > region.nz ||
      region.hi[0] < region.lo[0] || region.hi[1] < region.lo[1] ||
      region.hi[2] < region.lo[2])
    throw std::runtime_error("invalid GPU cycle diagnostic array region");

  if (!impl_)
    impl_ = new Impl();
  impl_->ensureReductionStorage();

  const unsigned long long activeCount =
      static_cast<unsigned long long>(region.hi[0] - region.lo[0]) *
      (region.hi[1] - region.lo[1]) * (region.hi[2] - region.lo[2]);
  const int blocks = reductionBlocks(activeCount);
  LocalArraySummary local[DIAG_MAX_FIELDS] = {};

  for (int field = 0; field < fieldCount; ++field) {
    local[field].stats = emptyArrayPartial();
    local[field].activeCount = activeCount;
    if (blocks == 0)
      continue;
    reduceArray3D<<<blocks, DIAG_THREADS, 0, stream>>>(
        deviceArrays[field], region.ny, region.nz, region.lo[0], region.hi[0],
        region.lo[1], region.hi[1], region.lo[2], region.hi[2],
        scale ? scale[field] : 1.0,
        impl_->dArrayPartials + field * DIAG_MAX_BLOCKS);
    cudaErrChk(cudaGetLastError());
  }

  if (blocks > 0) {
    for (int field = 0; field < fieldCount; ++field) {
      cudaErrChk(cudaMemcpyAsync(
          impl_->hArrayPartials + field * DIAG_MAX_BLOCKS,
          impl_->dArrayPartials + field * DIAG_MAX_BLOCKS,
          blocks * sizeof(ArrayPartial), cudaMemcpyDeviceToHost, stream));
    }
    cudaErrChk(cudaStreamSynchronize(stream));
    for (int field = 0; field < fieldCount; ++field) {
      local[field].stats = foldArrayPartials(
          impl_->hArrayPartials + field * DIAG_MAX_BLOCKS, blocks);
      decodeIndex(local[field].stats.maxAbsIndex, region, local[field].maxIJK,
                  local[field].maxCoordinates);
      decodeIndex(local[field].stats.firstNonfiniteIndex, region,
                  local[field].firstNonfiniteIJK,
                  local[field].firstNonfiniteCoordinates);
    }
  }

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int reportBytes =
      static_cast<int>(fieldCount * sizeof(LocalArraySummary));
  if (rank == 0)
    impl_->ensureGatherStorage(static_cast<std::size_t>(ranks) * reportBytes);
  MPI_Gather(local, reportBytes, MPI_BYTE,
             rank == 0 ? impl_->gatherStorage.data() : nullptr, reportBytes,
             MPI_BYTE, 0, comm);

  if (rank != 0)
    return;

  const auto* reports =
      reinterpret_cast<const LocalArraySummary*>(impl_->gatherStorage.data());
  for (int field = 0; field < fieldCount; ++field) {
    unsigned long long totalCount = 0;
    unsigned long long totalNonfinite = 0;
    int owner = -1;
    int firstNonfiniteOwner = -1;
    ArrayPartial global = emptyArrayPartial();
    double ownerCoordinates[3] = {};
    double firstNonfiniteCoordinates[3] = {};
    int ownerIJK[3] = {-1, -1, -1};
    int firstNonfiniteIJK[3] = {-1, -1, -1};

    for (int r = 0; r < ranks; ++r) {
      const LocalArraySummary& candidate = reports[r * fieldCount + field];
      totalCount += candidate.activeCount;
      totalNonfinite += candidate.stats.nonfiniteCount;

      if (candidate.stats.minIndex != DIAG_INVALID_INDEX &&
          candidate.stats.minValue < global.minValue)
        global.minValue = candidate.stats.minValue;
      if (candidate.stats.maxIndex != DIAG_INVALID_INDEX &&
          candidate.stats.maxValue > global.maxValue)
        global.maxValue = candidate.stats.maxValue;

      if (candidate.stats.maxAbsIndex != DIAG_INVALID_INDEX &&
          (owner < 0 || candidate.stats.maxAbs > global.maxAbs ||
           (candidate.stats.maxAbs == global.maxAbs && r < owner))) {
        owner = r;
        global.maxAbs = candidate.stats.maxAbs;
        global.maxAbsValue = candidate.stats.maxAbsValue;
        global.maxAbsIndex = candidate.stats.maxAbsIndex;
        std::copy(candidate.maxIJK, candidate.maxIJK + 3, ownerIJK);
        std::copy(candidate.maxCoordinates, candidate.maxCoordinates + 3,
                  ownerCoordinates);
      }
      if (candidate.stats.firstNonfiniteIndex != DIAG_INVALID_INDEX &&
          firstNonfiniteOwner < 0) {
        firstNonfiniteOwner = r;
        global.firstNonfiniteIndex = candidate.stats.firstNonfiniteIndex;
        std::copy(candidate.firstNonfiniteIJK, candidate.firstNonfiniteIJK + 3,
                  firstNonfiniteIJK);
        std::copy(candidate.firstNonfiniteCoordinates,
                  candidate.firstNonfiniteCoordinates + 3,
                  firstNonfiniteCoordinates);
      }
    }

    int ownerCart[3] = {-1, -1, -1};
    if (owner >= 0)
      cartesianCoordinates(comm, owner, ownerCart);

    std::ostringstream out;
    out << std::scientific << std::setprecision(17)
        << "[GPU-CYCLE-DIAG] cycle=" << cycle << " stage=" << stage;
    if (species >= 0)
      out << " species=" << species;
    out << " component=" << componentNames[field]
        << " diagnostic_scale=" << (scale ? scale[field] : 1.0) << " region=["
        << region.lo[0] << ':' << region.hi[0] << ',' << region.lo[1] << ':'
        << region.hi[1] << ',' << region.lo[2] << ':' << region.hi[2] << ")"
        << " samples=" << totalCount << " finite_min="
        << (owner >= 0 ? global.minValue
                       : std::numeric_limits<double>::quiet_NaN())
        << " finite_max="
        << (owner >= 0 ? global.maxValue
                       : std::numeric_limits<double>::quiet_NaN())
        << " maxabs="
        << (owner >= 0 ? global.maxAbs
                       : std::numeric_limits<double>::quiet_NaN())
        << " value="
        << (owner >= 0 ? global.maxAbsValue
                       : std::numeric_limits<double>::quiet_NaN())
        << " owner_rank=" << owner << " cart=(" << ownerCart[0] << ','
        << ownerCart[1] << ',' << ownerCart[2] << ")"
        << " local_index="
        << (owner >= 0 ? global.maxAbsIndex : DIAG_INVALID_INDEX) << " ijk=("
        << ownerIJK[0] << ',' << ownerIJK[1] << ',' << ownerIJK[2] << ")"
        << " xyz=(" << ownerCoordinates[0] << ',' << ownerCoordinates[1] << ','
        << ownerCoordinates[2] << ")"
        << " nonfinite=" << totalNonfinite;

    if (firstNonfiniteOwner >= 0) {
      out << " first_nonfinite_rank=" << firstNonfiniteOwner
          << " first_nonfinite_index=" << global.firstNonfiniteIndex
          << " first_nonfinite_ijk=(" << firstNonfiniteIJK[0] << ','
          << firstNonfiniteIJK[1] << ',' << firstNonfiniteIJK[2] << ")"
          << " first_nonfinite_xyz=(" << firstNonfiniteCoordinates[0] << ','
          << firstNonfiniteCoordinates[1] << ',' << firstNonfiniteCoordinates[2]
          << ')';
    }
    std::cout << out.str() << std::endl;
  }
}

void GPUCycleDiagnostics::reportParticleRange(
    const char* stage, int cycle, int species, const char* rangeName,
    const particleArrayCUDA& particles, std::uint32_t begin,
    std::uint32_t count, const GPUCycleDiagnosticParticleBounds& bounds,
    cudaStream_t stream, MPI_Comm comm) {
  if (!impl_)
    impl_ = new Impl();
  impl_->ensureReductionStorage();

  const ParticleSoADevice* soa = particles.getSoA();
  const double* fields[7] = {soa->x, soa->y, soa->z, soa->u,
                             soa->v, soa->w, soa->q};
  const unsigned long long requestedBegin = begin;
  const unsigned long long requestedCount = count;
  const unsigned long long nop = soa->nop;
  const unsigned long long capacity = soa->capacity;
  const unsigned long long readableLimit = std::min(nop, capacity);
  const unsigned long long safeBegin = std::min(requestedBegin, readableLimit);
  const unsigned long long safeCount =
      requestedBegin >= readableLimit
          ? 0
          : std::min(requestedCount, readableLimit - requestedBegin);
  const unsigned long long requestedEnd = requestedBegin + requestedCount;
  const bool invalidMetadataOrRange =
      nop > capacity || requestedBegin > nop || requestedEnd > nop ||
      requestedBegin > capacity || requestedEnd > capacity;
  const int blocks = reductionBlocks(safeCount);
  ParticleLocalReport local = {};
  local.requestedBegin = requestedBegin;
  local.requestedCount = requestedCount;
  local.rangeCount = safeCount;
  local.nop = nop;
  local.capacity = capacity;
  local.invalidMetadataOrRange = invalidMetadataOrRange ? 1 : 0;
  local.bounds = emptyParticlePartial();

  for (int field = 0; field < 7; ++field) {
    local.fields[field].stats = emptyArrayPartial();
    local.fields[field].activeCount = safeCount;
    if (blocks == 0)
      continue;
    reduceArray1D<<<blocks, DIAG_THREADS, 0, stream>>>(
        fields[field], safeBegin, safeCount, 1.0,
        impl_->dArrayPartials + field * DIAG_MAX_BLOCKS);
    cudaErrChk(cudaGetLastError());
  }
  if (blocks > 0) {
    reduceParticleBounds<<<blocks, DIAG_THREADS, 0, stream>>>(
        soa->x, soa->y, soa->z, soa->u, soa->v, soa->w, soa->q, safeBegin,
        safeCount, bounds, impl_->dParticlePartials);
    cudaErrChk(cudaGetLastError());
    for (int field = 0; field < 7; ++field) {
      cudaErrChk(cudaMemcpyAsync(
          impl_->hArrayPartials + field * DIAG_MAX_BLOCKS,
          impl_->dArrayPartials + field * DIAG_MAX_BLOCKS,
          blocks * sizeof(ArrayPartial), cudaMemcpyDeviceToHost, stream));
    }
    cudaErrChk(cudaMemcpyAsync(
        impl_->hParticlePartials, impl_->dParticlePartials,
        blocks * sizeof(ParticlePartial), cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    for (int field = 0; field < 7; ++field)
      local.fields[field].stats = foldArrayPartials(
          impl_->hArrayPartials + field * DIAG_MAX_BLOCKS, blocks);
    local.bounds = foldParticlePartials(impl_->hParticlePartials, blocks);
  }

  auto fillRecord = [&](ParticleRecord& record, unsigned long long index) {
    record = {};
    record.index = index;
    if (index == DIAG_INVALID_INDEX)
      return;
    record.valid = 1;
    record.hasId = soa->trackParticleID ? 1 : 0;
    for (int field = 0; field < 7; ++field)
      cudaErrChk(cudaMemcpy(&record.values[field], fields[field] + index,
                            sizeof(double), cudaMemcpyDeviceToHost));
    if (soa->trackParticleID)
      cudaErrChk(cudaMemcpy(&record.id, soa->id + index,
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  };
  fillRecord(local.firstNonfinite, local.bounds.firstNonfiniteIndex);
  fillRecord(local.firstOutsideLocal, local.bounds.firstOutsideLocalIndex);
  fillRecord(local.worstOutsideLocal, local.bounds.worstLocalIndex);
  fillRecord(local.firstOutsideGlobal, local.bounds.firstOutsideGlobalIndex);
  fillRecord(local.worstOutsideGlobal, local.bounds.worstGlobalIndex);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int reportBytes = static_cast<int>(sizeof(ParticleLocalReport));
  if (rank == 0)
    impl_->ensureGatherStorage(static_cast<std::size_t>(ranks) * reportBytes);
  MPI_Gather(&local, reportBytes, MPI_BYTE,
             rank == 0 ? impl_->gatherStorage.data() : nullptr, reportBytes,
             MPI_BYTE, 0, comm);

  if (rank != 0)
    return;
  const auto* reports =
      reinterpret_cast<const ParticleLocalReport*>(impl_->gatherStorage.data());
  const char* fieldNames[7] = {"x", "y", "z", "u", "v", "w", "q"};

  unsigned long long totalCount = 0;
  unsigned long long totalRequestedCount = 0;
  unsigned long long nonfiniteParticles = 0;
  unsigned long long outsideLocal = 0;
  unsigned long long outsideGlobal = 0;
  double maxOutsideLocalDistance = 0.0;
  double maxOutsideGlobalDistance = 0.0;
  int invalidMetadataRanks = 0;
  int firstInvalidMetadataRank = -1;
  for (int r = 0; r < ranks; ++r) {
    totalCount += reports[r].rangeCount;
    totalRequestedCount += reports[r].requestedCount;
    nonfiniteParticles += reports[r].bounds.nonfiniteParticles;
    outsideLocal += reports[r].bounds.outsideLocalCount;
    outsideGlobal += reports[r].bounds.outsideGlobalCount;
    if (reports[r].bounds.outsideLocalCount > 0)
      maxOutsideLocalDistance =
          std::max(maxOutsideLocalDistance, reports[r].bounds.maxLocalDistance);
    if (reports[r].bounds.outsideGlobalCount > 0)
      maxOutsideGlobalDistance = std::max(maxOutsideGlobalDistance,
                                          reports[r].bounds.maxGlobalDistance);
    if (reports[r].invalidMetadataOrRange) {
      ++invalidMetadataRanks;
      if (firstInvalidMetadataRank < 0)
        firstInvalidMetadataRank = r;
    }
  }

  std::ostringstream values;
  values << std::scientific << std::setprecision(17)
         << "[GPU-CYCLE-DIAG] cycle=" << cycle << " stage=" << stage
         << " species=" << species << " range=" << rangeName
         << " particles_requested=" << totalRequestedCount
         << " particles_scanned=" << totalCount;
  for (int field = 0; field < 7; ++field) {
    double globalMin = INFINITY;
    double globalMax = -INFINITY;
    unsigned long long nonfinite = 0;
    bool hasFinite = false;
    for (int r = 0; r < ranks; ++r) {
      const ArrayPartial& candidate = reports[r].fields[field].stats;
      nonfinite += candidate.nonfiniteCount;
      if (candidate.minIndex != DIAG_INVALID_INDEX) {
        hasFinite = true;
        globalMin = std::min(globalMin, candidate.minValue);
        globalMax = std::max(globalMax, candidate.maxValue);
      }
    }
    values << ' ' << fieldNames[field] << "=["
           << (hasFinite ? globalMin : std::numeric_limits<double>::quiet_NaN())
           << ','
           << (hasFinite ? globalMax : std::numeric_limits<double>::quiet_NaN())
           << "](" << "nonfinite=" << nonfinite << ')';
  }
  values << " nonfinite_particles=" << nonfiniteParticles
         << " outside_local=" << outsideLocal
         << " max_outside_local_distance=" << maxOutsideLocalDistance
         << " outside_global=" << outsideGlobal
         << " max_outside_global_distance=" << maxOutsideGlobalDistance
         << " invalid_metadata_ranks=" << invalidMetadataRanks;
  if (firstInvalidMetadataRank >= 0) {
    const ParticleLocalReport& invalid = reports[firstInvalidMetadataRank];
    values << " first_invalid_metadata_rank=" << firstInvalidMetadataRank
           << " requested_begin=" << invalid.requestedBegin
           << " requested_count=" << invalid.requestedCount
           << " nop=" << invalid.nop << " capacity=" << invalid.capacity
           << " scanned_count=" << invalid.rangeCount;
  }
  std::cout << values.str() << std::endl;

  auto firstRecord = [&](auto countMember, auto recordMember,
                         int& owner) -> const ParticleRecord* {
    owner = -1;
    for (int r = 0; r < ranks; ++r) {
      if (reports[r].bounds.*countMember > 0) {
        owner = r;
        return &(reports[r].*recordMember);
      }
    }
    return nullptr;
  };
  auto worstRecord = [&](auto countMember, auto distanceMember,
                         auto recordMember, int& owner,
                         double& distance) -> const ParticleRecord* {
    owner = -1;
    distance = -1.0;
    const ParticleRecord* selected = nullptr;
    for (int r = 0; r < ranks; ++r) {
      if (reports[r].bounds.*countMember == 0)
        continue;
      const double candidate = reports[r].bounds.*distanceMember;
      if (owner < 0 || candidate > distance) {
        owner = r;
        distance = candidate;
        selected = &(reports[r].*recordMember);
      }
    }
    return selected;
  };
  auto printRecord = [&](const char* kind, int owner, double distance,
                         const ParticleRecord* record) {
    if (!record)
      return;
    int cart[3];
    cartesianCoordinates(comm, owner, cart);
    std::ostringstream out;
    out << std::scientific << std::setprecision(17)
        << "[GPU-CYCLE-DIAG] cycle=" << cycle << " stage=" << stage
        << " species=" << species << " range=" << rangeName
        << " particle=" << kind << " owner_rank=" << owner << " cart=("
        << cart[0] << ',' << cart[1] << ',' << cart[2] << ')';
    if (distance >= 0.0)
      out << " distance=" << distance;
    out << ' ' << particleRecordText(*record);
    std::cout << out.str() << std::endl;
  };

  int owner = -1;
  const ParticleRecord* record =
      firstRecord(&ParticlePartial::nonfiniteParticles,
                  &ParticleLocalReport::firstNonfinite, owner);
  printRecord("first_nonfinite", owner, -1.0, record);

  record = firstRecord(&ParticlePartial::outsideLocalCount,
                       &ParticleLocalReport::firstOutsideLocal, owner);
  printRecord("first_outside_local", owner, -1.0, record);
  double distance = -1.0;
  record = worstRecord(
      &ParticlePartial::outsideLocalCount, &ParticlePartial::maxLocalDistance,
      &ParticleLocalReport::worstOutsideLocal, owner, distance);
  printRecord("worst_outside_local", owner, distance, record);

  record = firstRecord(&ParticlePartial::outsideGlobalCount,
                       &ParticleLocalReport::firstOutsideGlobal, owner);
  printRecord("first_outside_global", owner, -1.0, record);
  record = worstRecord(
      &ParticlePartial::outsideGlobalCount, &ParticlePartial::maxGlobalDistance,
      &ParticleLocalReport::worstOutsideGlobal, owner, distance);
  printRecord("worst_outside_global", owner, distance, record);
}

void GPUCycleDiagnostics::reportCellSorterBeforeScatter(
    int cycle, int species, CellSorter& sorter,
    const particleArrayCUDA& particles, const grid3DCUDA& hostGrid,
    const grid3DCUDA* deviceGrid, bool expectFullSort, cudaStream_t stream,
    MPI_Comm comm) {
  if (!impl_)
    impl_ = new Impl();
  impl_->ensureCellSortStorage();

  const ParticleSoADevice* soa = particles.getSoA();
  CellSortLocalReport local{};
  local.histogram = emptySortHistogramPartial();
  local.prefix = emptySortPrefixPartial();
  local.reservation = emptySortReservationPartial();
  local.tiles = emptySortTilePartial();
  local.phase1Totals = emptySortPhase1TotalPartial();
  local.blockPrefix = emptySortBlockPrefixPartial();
  local.permutation = emptySortPermutationPartial();
  local.occupancy = emptySortOccupancyPartial();
  local.sortPending = sorter.diagnosticSortPending() ? 1 : 0;
  local.expectedParticles =
      local.sortPending ? sorter.diagnosticPendingNumToSort() : 0;
  local.requestedParticles = sorter.diagnosticRequestedNumToSort();
  local.requestedCountWasClamped =
      sorter.diagnosticNumToSortWasClamped() ? 1 : 0;
  local.expectFullSort = expectFullSort ? 1 : 0;
  local.pendingNOP =
      local.sortPending ? sorter.diagnosticPendingNOP() : particles.getNOP();
  local.nop = particles.getNOP();
  local.capacity = particles.getCapacity();
  local.indexCapacity = sorter.diagnosticIndexCapacity();
  local.scratchBytes = sorter.diagnosticScratchBytes();
  local.numCells = sorter.getNumCells();
  local.numScanBlocks = sorter.diagnosticNumScanBlocks();
  local.nxc = hostGrid.nxc;
  local.nyc = hostGrid.nyc;
  local.nzc = hostGrid.nzc;
  local.pendingHostMatches =
      !local.sortPending || sorter.diagnosticPendingHostPointer() == &particles;
  local.compileWarpSize = WARP_SIZE;
  local.warpMaskBytes = sizeof(warp_mask_t);
  local.sorterCompileWarpSize = sorter.diagnosticCompiledWarpSize();
  local.sorterWarpMaskBytes = sorter.diagnosticCompiledWarpMaskBytes();
  local.sorterAlgorithmVersion = sorter.diagnosticAlgorithmVersion();

  int device = 0;
  cudaErrChk(cudaGetDevice(&device));
#if defined(HIPIFLY)
  cudaErrChk(hipDeviceGetAttribute(&local.runtimeWarpSize,
                                   hipDeviceAttributeWarpSize, device));
#else
  cudaErrChk(cudaDeviceGetAttribute(&local.runtimeWarpSize, cudaDevAttrWarpSize,
                                    device));
#endif

  // Flags: 0 pending/NOP, 1 N<=NOP, 2 NOP<=capacity, 3 index capacity,
  // 4 int-offset range, 5 scratch capacity, 6 cell geometry, 7 pointers,
  // 8 runtime/translation-unit warp configuration, 9 pending host object,
  // 10 pointer mutation between enqueue and pre-scatter inspection,
  // 11 full-sort count mismatch, 12 internal sorter buffers/geometry,
  // 13 caller count was clamped, 14 sorter buffer changed after enqueue,
  // 15 sorter buffer aliases another live allocation, 16 allocation metadata
  // changed after enqueue (detects same-address growth/reallocation).
  if ((local.nop > 0) != (local.sortPending != 0) ||
      local.pendingNOP != local.nop)
    local.metadataFlags |= 1 << 0;
  if (local.expectedParticles > local.nop)
    local.metadataFlags |= 1 << 1;
  if (local.nop > local.capacity)
    local.metadataFlags |= 1 << 2;
  if (local.expectedParticles > local.indexCapacity)
    local.metadataFlags |= 1 << 3;
  if (local.expectedParticles > static_cast<unsigned long long>(INT_MAX))
    local.metadataFlags |= 1 << 4;
  if (local.nop > local.scratchBytes / sizeof(std::uint64_t))
    local.metadataFlags |= 1 << 5;
  const long long gridCells =
      static_cast<long long>(hostGrid.nxc) * hostGrid.nyc * hostGrid.nzc;
  const int expectedScanBlocks =
      local.numCells > 0 ? (local.numCells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1) /
                               SORT_SCAN_ELEMENTS_PER_BLOCK
                         : 0;
  if (gridCells != local.numCells || local.numCells <= 0)
    local.metadataFlags |= 1 << 6;
  local.pointerAlias =
      hasPointerAlias(*soa, sorter.diagnosticScratchPointer()) ? 1 : 0;
  if (local.sortPending && local.pointerAlias)
    local.metadataFlags |= 1 << 7;
  local.enqueuePointerMutationMask =
      sorter.diagnosticEnqueuePointerMutationMask(particles);
  local.enqueueFirstPointerField = -1;
  const void* currentPointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS] = {
      soa->u, soa->v, soa->w, soa->q, soa->x, soa->y, soa->z, soa->id};
  for (int field = 0; field < CELL_SORT_DIAGNOSTIC_MAX_FIELDS; ++field) {
    if (local.enqueuePointerMutationMask & (1 << field)) {
      local.enqueueFirstPointerField = field;
      local.enqueueExpectedPointer = sorter.diagnosticEnqueuePointer(field);
      local.enqueueCurrentPointer =
          reinterpret_cast<std::uintptr_t>(currentPointers[field]);
      break;
    }
  }
  if (local.enqueueFirstPointerField < 0 &&
      (local.enqueuePointerMutationMask & (1 << 8))) {
    local.enqueueFirstPointerField = 8;
    local.enqueueExpectedPointer = sorter.diagnosticEnqueueScratchPointer();
    local.enqueueCurrentPointer =
        reinterpret_cast<std::uintptr_t>(sorter.diagnosticScratchPointer());
  }
  local.enqueueSortBufferMutationMask =
      sorter.diagnosticEnqueueSortBufferMutationMask();
  local.enqueueFirstSortBuffer = -1;
  const void* currentSortBuffers[CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT] = {
      sorter.getCellCounts(), sorter.diagnosticCellOffsets(),
      sorter.getCellStartOffsets(), sorter.diagnosticSortedIndices(),
      sorter.diagnosticBlockSums(), sorter.diagnosticPhase1BlockSums()};
  for (int buffer = 0; buffer < CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT;
       ++buffer) {
    if (!(local.enqueueSortBufferMutationMask & (1 << buffer)))
      continue;
    local.enqueueFirstSortBuffer = buffer;
    local.enqueueExpectedSortBuffer =
        sorter.diagnosticEnqueueSortBufferPointer(buffer);
    local.enqueueCurrentSortBuffer =
        reinterpret_cast<std::uintptr_t>(currentSortBuffers[buffer]);
    break;
  }
  local.enqueueAllocationMetadataMutationMask =
      sorter.diagnosticEnqueueAllocationMetadataMutationMask(particles);
  local.enqueueFirstAllocationMetadata = -1;
  for (int field = 0; field < CELL_SORT_DIAGNOSTIC_ALLOCATION_METADATA_COUNT;
       ++field) {
    if (!(local.enqueueAllocationMetadataMutationMask & (1 << field)))
      continue;
    local.enqueueFirstAllocationMetadata = field;
    local.enqueueExpectedAllocationMetadata =
        sorter.diagnosticEnqueueAllocationMetadata(field);
    local.enqueueCurrentAllocationMetadata =
        sorter.diagnosticCurrentAllocationMetadata(field, particles);
    break;
  }
  const int expectedMaskBytes = local.runtimeWarpSize > 32 ? 8 : 4;
  if (local.sorterCompileWarpSize != local.runtimeWarpSize ||
      local.sorterWarpMaskBytes != expectedMaskBytes ||
      local.sorterCompileWarpSize != local.compileWarpSize ||
      local.sorterWarpMaskBytes != local.warpMaskBytes)
    local.metadataFlags |= 1 << 8;
  if (!local.pendingHostMatches)
    local.metadataFlags |= 1 << 9;
  if (local.enqueuePointerMutationMask)
    local.metadataFlags |= 1 << 10;
  if (local.expectFullSort && (local.expectedParticles != local.nop ||
                               local.requestedParticles != local.nop))
    local.metadataFlags |= 1 << 11;
  if (!sorter.getCellCounts())
    local.sorterPointerNullMask |= 1 << 0;
  if (!sorter.diagnosticCellOffsets())
    local.sorterPointerNullMask |= 1 << 1;
  if (!sorter.getCellStartOffsets())
    local.sorterPointerNullMask |= 1 << 2;
  if (!sorter.diagnosticSortedIndices())
    local.sorterPointerNullMask |= 1 << 3;
  if (local.numCells > SORT_SCAN_ELEMENTS_PER_BLOCK &&
      !sorter.diagnosticBlockSums())
    local.sorterPointerNullMask |= 1 << 4;
  if (local.numCells > SORT_SCAN_ELEMENTS_PER_BLOCK &&
      !sorter.diagnosticPhase1BlockSums())
    local.sorterPointerNullMask |= 1 << 5;
  local.sorterPointerAlias =
      findSorterPointerAlias(*soa, sorter.diagnosticScratchPointer(),
                             currentSortBuffers, local.firstAliasPointerA,
                             local.firstAliasPointerB)
          ? 1
          : 0;
  const bool sorterPointersValid =
      local.sorterPointerNullMask == 0 && !local.sorterPointerAlias;
  if ((local.sortPending && local.sorterPointerNullMask != 0) ||
      local.numScanBlocks != expectedScanBlocks)
    local.metadataFlags |= 1 << 12;
  if (local.requestedCountWasClamped)
    local.metadataFlags |= 1 << 13;
  if (local.enqueueSortBufferMutationMask)
    local.metadataFlags |= 1 << 14;
  if (local.sortPending && local.sorterPointerAlias)
    local.metadataFlags |= 1 << 15;
  if (local.enqueueAllocationMetadataMutationMask)
    local.metadataFlags |= 1 << 16;

  const bool invariantsReadable =
      local.sortPending && local.expectedParticles <= local.nop &&
      local.pendingNOP == local.nop && local.nop <= local.capacity &&
      local.expectedParticles <= local.indexCapacity &&
      local.expectedParticles <= static_cast<unsigned long long>(INT_MAX) &&
      local.numCells > 0 && gridCells == local.numCells &&
      local.numScanBlocks == expectedScanBlocks && sorterPointersValid &&
      local.enqueueSortBufferMutationMask == 0 &&
      local.enqueueAllocationMetadataMutationMask == 0;
  const bool permutationReadable = invariantsReadable &&
                                   local.pendingHostMatches &&
                                   local.enqueuePointerMutationMask == 0 &&
                                   soa->x && soa->y && soa->z && deviceGrid;
  int cellBlocks = 0;
  int tileBlocks = 0;
  int blockBlocks = 0;
  int permutationBlocks = 0;
  int occupancyBlocks = 0;

  if (invariantsReadable) {
    local.invariantsChecked = 1;
    cellBlocks = reductionBlocks(local.numCells);
    reduceSortHistogram<<<cellBlocks, DIAG_THREADS, 0, stream>>>(
        sorter.getCellCounts(), local.numCells, impl_->dCellSortPartials);
    cudaErrChk(cudaGetLastError());
    reduceSortPrefix<<<cellBlocks, DIAG_THREADS, 0, stream>>>(
        sorter.getCellStartOffsets(), sorter.getCellCounts(), local.numCells,
        static_cast<std::uint32_t>(local.expectedParticles),
        impl_->dCellSortPartials);
    cudaErrChk(cudaGetLastError());
    reduceSortReservations<<<cellBlocks, DIAG_THREADS, 0, stream>>>(
        sorter.getCellStartOffsets(), sorter.diagnosticCellOffsets(),
        sorter.getCellCounts(), local.numCells, impl_->dCellSortPartials);
    cudaErrChk(cudaGetLastError());

    if (local.numCells > SORT_SCAN_ELEMENTS_PER_BLOCK) {
      tileBlocks = cellBlocks;
      reduceSortTiles<<<tileBlocks, DIAG_THREADS, 0, stream>>>(
          sorter.getCellStartOffsets(), sorter.getCellCounts(),
          sorter.diagnosticBlockSums(), local.numCells,
          impl_->dCellSortPartials);
      cudaErrChk(cudaGetLastError());
      blockBlocks = reductionBlocks(local.numScanBlocks);
      reduceSortPhase1Totals<<<blockBlocks, DIAG_THREADS, 0, stream>>>(
          sorter.diagnosticPhase1BlockSums(), sorter.getCellCounts(),
          local.numCells, local.numScanBlocks, impl_->dCellSortPartials);
      cudaErrChk(cudaGetLastError());
      local.phase1TotalsChecked = 1;
      reduceSortBlockPrefixes<<<blockBlocks, DIAG_THREADS, 0, stream>>>(
          sorter.diagnosticBlockSums(), sorter.getCellCounts(), local.numCells,
          local.numScanBlocks, impl_->dCellSortPartials);
      cudaErrChk(cudaGetLastError());
    }

    permutationBlocks =
        permutationReadable ? reductionBlocks(local.expectedParticles) : 0;
    if (permutationBlocks > 0) {
      local.permutationChecked = 1;
      reduceSortPermutation<<<permutationBlocks, DIAG_THREADS, 0, stream>>>(
          sorter.diagnosticSortedIndices(), soa->x, soa->y, soa->z,
          sorter.getCellStartOffsets(), sorter.getCellCounts(), deviceGrid,
          local.numCells, static_cast<std::uint32_t>(local.expectedParticles),
          impl_->dCellSortPartials);
      cudaErrChk(cudaGetLastError());
    }

    if (permutationBlocks > 0 && !local.pointerAlias &&
        sorter.diagnosticScratchPointer() &&
        local.expectedParticles <= local.scratchBytes / sizeof(std::uint32_t)) {
      occupancyBlocks = permutationBlocks;
      local.occupancyChecked = 1;
      auto* occupancy =
          static_cast<std::uint32_t*>(sorter.diagnosticScratchPointer());
      cudaErrChk(cudaMemsetAsync(
          occupancy, 0, local.expectedParticles * sizeof(std::uint32_t),
          stream));
      markSortDestinations<<<occupancyBlocks, DIAG_THREADS, 0, stream>>>(
          sorter.diagnosticSortedIndices(), occupancy,
          static_cast<std::uint32_t>(local.expectedParticles));
      cudaErrChk(cudaGetLastError());
      reduceSortOccupancy<<<occupancyBlocks, DIAG_THREADS, 0, stream>>>(
          occupancy, static_cast<std::uint32_t>(local.expectedParticles),
          impl_->dCellSortPartials);
      cudaErrChk(cudaGetLastError());
    }

    const int copiedBlocks = std::max({cellBlocks, tileBlocks, blockBlocks,
                                       permutationBlocks, occupancyBlocks});
    if (copiedBlocks > 0)
      cudaErrChk(cudaMemcpyAsync(impl_->hCellSortPartials,
                                 impl_->dCellSortPartials,
                                 copiedBlocks * sizeof(CellSortBlockReport),
                                 cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    const CellSortLocalReport folded =
        foldCellSortPartials(impl_->hCellSortPartials, cellBlocks, tileBlocks,
                             blockBlocks, permutationBlocks, occupancyBlocks);
    local.histogram = folded.histogram;
    local.prefix = folded.prefix;
    local.reservation = folded.reservation;
    local.tiles = folded.tiles;
    local.phase1Totals = folded.phase1Totals;
    local.blockPrefix = folded.blockPrefix;
    local.permutation = folded.permutation;
    local.occupancy = folded.occupancy;
  } else {
    cudaErrChk(cudaStreamSynchronize(stream));
  }

  impl_->postSortLayoutSafe =
      permutationReadable && local.histogram.negativeCount == 0 &&
      local.prefix.recurrenceMismatchCount == 0 &&
      local.prefix.invalidIntervalCount == 0 &&
      local.prefix.terminal == static_cast<long long>(local.expectedParticles);

  CellSortPointerSnapshot& snapshot = impl_->cellSortSnapshot;
  snapshot = CellSortPointerSnapshot{};
  snapshot.cycle = cycle;
  snapshot.species = species;
  snapshot.expectedParticles = local.expectedParticles;
  snapshot.nop = local.nop;
  snapshot.active = local.sortPending;
  snapshot.expectFullSort = local.expectFullSort;
  snapshot.fieldCount = soa->trackParticleID ? 8 : 7;
  const void* fields[CELL_SORT_DIAGNOSTIC_MAX_FIELDS] = {
      soa->u, soa->v, soa->w, soa->q, soa->x, soa->y, soa->z, soa->id};
  for (int field = 0; field < snapshot.fieldCount; ++field)
    snapshot.pointers[field] = reinterpret_cast<std::uintptr_t>(fields[field]);
  snapshot.scratch =
      reinterpret_cast<std::uintptr_t>(sorter.diagnosticScratchPointer());

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int reportBytes = sizeof(CellSortLocalReport);
  if (rank == 0)
    impl_->ensureGatherStorage(static_cast<std::size_t>(ranks) * reportBytes);
  MPI_Gather(&local, reportBytes, MPI_BYTE,
             rank == 0 ? impl_->gatherStorage.data() : nullptr, reportBytes,
             MPI_BYTE, 0, comm);

  if (rank == 0) {
    const auto* reports = reinterpret_cast<const CellSortLocalReport*>(
        impl_->gatherStorage.data());
    auto firstOwner = [&](auto predicate) {
      for (int r = 0; r < ranks; ++r)
        if (predicate(reports[r]))
          return r;
      return -1;
    };
    auto appendRankCell = [&](std::ostringstream& out, const char* label,
                              int owner, unsigned long long cell) {
      if (owner < 0)
        return;
      int cart[3];
      int cellXYZ[3];
      cartesianCoordinates(comm, owner, cart);
      decodeCell(cell, reports[owner].nxc, reports[owner].nyc, cellXYZ);
      out << ' ' << label << "_rank=" << owner << ' ' << label << "_cart=("
          << cart[0] << ',' << cart[1] << ',' << cart[2] << ") " << label
          << "_cell=" << cell << ' ' << label << "_cell_xyz=(" << cellXYZ[0]
          << ',' << cellXYZ[1] << ',' << cellXYZ[2] << ')';
    };

    unsigned long long globalExpected = 0;
    unsigned long long globalRequested = 0;
    unsigned long long checkedExpected = 0;
    unsigned long long globalNOP = 0;
    long long globalHistogram = 0;
    long long globalActual = 0;
    unsigned long long negativeCounts = 0;
    int invalidRanks = 0;
    int histogramMismatchRanks = 0;
    int actualMismatchRanks = 0;
    int minCount = INT_MAX;
    int maxCount = INT_MIN;
    int maxCountOwner = -1;
    unsigned long long maxCountCell = DIAG_INVALID_INDEX;
    int minRuntimeWarp = INT_MAX;
    int maxRuntimeWarp = INT_MIN;
    int activeRanks = 0;
    int invariantCheckedRanks = 0;
    int permutationCheckedRanks = 0;
    int occupancyCheckedRanks = 0;
    int clampedRequestRanks = 0;
    int fullSortCountMismatchRanks = 0;
    constexpr int metadataFlagCount = 17;
    int metadataFlagRanks[metadataFlagCount]{};
    int metadataFirstRank[metadataFlagCount];
    std::fill_n(metadataFirstRank, metadataFlagCount, -1);
    for (int r = 0; r < ranks; ++r) {
      globalExpected += reports[r].expectedParticles;
      globalRequested += reports[r].requestedParticles;
      globalNOP += reports[r].nop;
      if (reports[r].invariantsChecked) {
        checkedExpected += reports[r].expectedParticles;
        globalHistogram += reports[r].histogram.sum;
        globalActual += reports[r].reservation.actualSum;
      }
      negativeCounts += reports[r].histogram.negativeCount;
      invalidRanks += reports[r].metadataFlags != 0;
      histogramMismatchRanks +=
          reports[r].invariantsChecked &&
          reports[r].histogram.sum !=
              static_cast<long long>(reports[r].expectedParticles);
      actualMismatchRanks +=
          reports[r].invariantsChecked &&
          reports[r].reservation.actualSum !=
              static_cast<long long>(reports[r].expectedParticles);
      if (reports[r].histogram.minCell != DIAG_INVALID_INDEX)
        minCount = std::min(minCount, reports[r].histogram.minCount);
      if (reports[r].histogram.maxCell != DIAG_INVALID_INDEX &&
          (maxCountOwner < 0 || reports[r].histogram.maxCount > maxCount)) {
        maxCount = reports[r].histogram.maxCount;
        maxCountOwner = r;
        maxCountCell = reports[r].histogram.maxCell;
      }
      minRuntimeWarp = std::min(minRuntimeWarp, reports[r].runtimeWarpSize);
      maxRuntimeWarp = std::max(maxRuntimeWarp, reports[r].runtimeWarpSize);
      activeRanks += reports[r].sortPending != 0;
      invariantCheckedRanks += reports[r].invariantsChecked != 0;
      permutationCheckedRanks += reports[r].permutationChecked != 0;
      occupancyCheckedRanks += reports[r].occupancyChecked != 0;
      clampedRequestRanks += reports[r].requestedCountWasClamped != 0;
      fullSortCountMismatchRanks +=
          reports[r].expectFullSort &&
          (reports[r].expectedParticles != reports[r].nop ||
           reports[r].requestedParticles != reports[r].nop);
      for (int bit = 0; bit < metadataFlagCount; ++bit) {
        if (!(reports[r].metadataFlags & (1 << bit)))
          continue;
        ++metadataFlagRanks[bit];
        if (metadataFirstRank[bit] < 0)
          metadataFirstRank[bit] = r;
      }
    }

    int owner = firstOwner(
        [](const CellSortLocalReport& r) { return r.metadataFlags != 0; });
    const int enqueueMutationOwner =
        firstOwner([](const CellSortLocalReport& r) {
          return r.enqueuePointerMutationMask != 0;
        });
    std::ostringstream metadata;
    metadata << "[GPU-CYCLE-DIAG] cycle=" << cycle
             << " stage=cell_sort_metadata species=" << species
             << " requested_particles=" << globalRequested
             << " expected_particles=" << globalExpected << " nop=" << globalNOP
             << " active_ranks=" << activeRanks
             << " invalid_ranks=" << invalidRanks
             << " clamped_request_ranks=" << clampedRequestRanks
             << " full_sort_count_mismatch_ranks=" << fullSortCountMismatchRanks
             << " compile_warp_size=" << WARP_SIZE << " runtime_warp_size=["
             << minRuntimeWarp << ',' << maxRuntimeWarp
             << "] warp_mask_bytes=" << sizeof(warp_mask_t)
             << " sorter_compile_warp_size=" << reports[0].sorterCompileWarpSize
             << " sorter_warp_mask_bytes=" << reports[0].sorterWarpMaskBytes
             << " sorter_algorithm_version="
             << reports[0].sorterAlgorithmVersion;
    static const char* const flagNames[metadataFlagCount] = {
        "pending_nop",
        "sort_gt_nop",
        "nop_gt_capacity",
        "sort_gt_index",
        "int_offset_range",
        "scratch_capacity",
        "grid_geometry",
        "pointer_alias",
        "warp_configuration",
        "pending_host",
        "enqueue_pointer",
        "full_sort_count",
        "sorter_buffers",
        "request_clamped",
        "enqueue_sort_buffer",
        "sorter_alias",
        "enqueue_allocation_metadata"};
    for (int bit = 0; bit < metadataFlagCount; ++bit) {
      metadata << ' ' << flagNames[bit] << "_ranks=" << metadataFlagRanks[bit];
      if (metadataFirstRank[bit] >= 0) {
        int cart[3];
        cartesianCoordinates(comm, metadataFirstRank[bit], cart);
        metadata << " first_" << flagNames[bit]
                 << "_rank=" << metadataFirstRank[bit] << " cart=(" << cart[0]
                 << ',' << cart[1] << ',' << cart[2] << ')';
      }
    }
    if (owner >= 0) {
      metadata << " first_invalid_rank=" << owner << " flags=0x" << std::hex
               << reports[owner].metadataFlags << std::dec
               << " local_expected=" << reports[owner].expectedParticles
               << " local_requested=" << reports[owner].requestedParticles
               << " pending_nop=" << reports[owner].pendingNOP
               << " local_nop=" << reports[owner].nop
               << " capacity=" << reports[owner].capacity
               << " index_capacity=" << reports[owner].indexCapacity
               << " scratch_bytes=" << reports[owner].scratchBytes;
      if (!reports[owner].pendingHostMatches)
        metadata << " pending_host_pointer_mismatch=1";
    }
    if (enqueueMutationOwner >= 0) {
      static const char* const pointerNames[9] = {"u", "v", "w",  "q",      "x",
                                                  "y", "z", "id", "scratch"};
      const auto& mutation = reports[enqueueMutationOwner];
      int cart[3];
      cartesianCoordinates(comm, enqueueMutationOwner, cart);
      metadata << " enqueue_pointer_mutation_rank=" << enqueueMutationOwner
               << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
               << ") mask=0x" << std::hex << mutation.enqueuePointerMutationMask
               << std::dec;
      if (mutation.enqueueFirstPointerField >= 0 &&
          mutation.enqueueFirstPointerField < 9)
        metadata << " first_pointer="
                 << pointerNames[mutation.enqueueFirstPointerField]
                 << " enqueue_ptr=0x" << std::hex
                 << mutation.enqueueExpectedPointer << " pre_scatter_ptr=0x"
                 << mutation.enqueueCurrentPointer << std::dec;
    }
    const int enqueueSortBufferMutationOwner =
        firstOwner([](const CellSortLocalReport& r) {
          return r.enqueueSortBufferMutationMask != 0;
        });
    if (enqueueSortBufferMutationOwner >= 0) {
      static const char* const
          bufferNames[CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT] = {
              "cell_counts", "cell_offsets", "cell_start_offsets",
              "sorted_indices", "block_sums", "phase1_block_sums"};
      const auto& mutation = reports[enqueueSortBufferMutationOwner];
      int cart[3];
      cartesianCoordinates(comm, enqueueSortBufferMutationOwner, cart);
      metadata << " enqueue_sort_buffer_mutation_rank="
               << enqueueSortBufferMutationOwner << " cart=(" << cart[0] << ','
               << cart[1] << ',' << cart[2] << ") mask=0x" << std::hex
               << mutation.enqueueSortBufferMutationMask << std::dec;
      if (mutation.enqueueFirstSortBuffer >= 0 &&
          mutation.enqueueFirstSortBuffer <
              CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT)
        metadata << " first_sort_buffer="
                 << bufferNames[mutation.enqueueFirstSortBuffer]
                 << " enqueue_ptr=0x" << std::hex
                 << mutation.enqueueExpectedSortBuffer << " pre_scatter_ptr=0x"
                 << mutation.enqueueCurrentSortBuffer << std::dec;
    }
    const int enqueueAllocationMutationOwner =
        firstOwner([](const CellSortLocalReport& r) {
          return r.enqueueAllocationMetadataMutationMask != 0;
        });
    if (enqueueAllocationMutationOwner >= 0) {
      static const char* const
          metadataNames[CELL_SORT_DIAGNOSTIC_ALLOCATION_METADATA_COUNT] = {
              "particle_capacity", "scratch_bytes",    "index_capacity",
              "sorter_num_cells",  "buffer_num_cells", "scan_blocks"};
      const auto& mutation = reports[enqueueAllocationMutationOwner];
      int cart[3];
      cartesianCoordinates(comm, enqueueAllocationMutationOwner, cart);
      metadata << " enqueue_allocation_metadata_mutation_rank="
               << enqueueAllocationMutationOwner << " cart=(" << cart[0] << ','
               << cart[1] << ',' << cart[2] << ") mask=0x" << std::hex
               << mutation.enqueueAllocationMetadataMutationMask << std::dec;
      if (mutation.enqueueFirstAllocationMetadata >= 0 &&
          mutation.enqueueFirstAllocationMetadata <
              CELL_SORT_DIAGNOSTIC_ALLOCATION_METADATA_COUNT)
        metadata << " first_allocation_metadata="
                 << metadataNames[mutation.enqueueFirstAllocationMetadata]
                 << " enqueue_value="
                 << mutation.enqueueExpectedAllocationMetadata
                 << " pre_scatter_value="
                 << mutation.enqueueCurrentAllocationMetadata;
    }
    const int fullSortCountOwner = firstOwner([](const CellSortLocalReport& r) {
      return r.expectFullSort &&
             (r.expectedParticles != r.nop || r.requestedParticles != r.nop);
    });
    if (fullSortCountOwner >= 0) {
      const auto& count = reports[fullSortCountOwner];
      int cart[3];
      cartesianCoordinates(comm, fullSortCountOwner, cart);
      metadata << " first_full_sort_count_rank=" << fullSortCountOwner
               << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
               << ") requested=" << count.requestedParticles
               << " after_clamp=" << count.expectedParticles
               << " nop=" << count.nop
               << " was_clamped=" << count.requestedCountWasClamped;
    }
    const int sorterBufferOwner = firstOwner([](const CellSortLocalReport& r) {
      return (r.metadataFlags & (1 << 12)) != 0;
    });
    if (sorterBufferOwner >= 0) {
      const auto& buffer = reports[sorterBufferOwner];
      const int expectedBlocks =
          buffer.numCells > 0
              ? (buffer.numCells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1) /
                    SORT_SCAN_ELEMENTS_PER_BLOCK
              : 0;
      metadata << " first_sorter_buffer_rank=" << sorterBufferOwner
               << " null_mask=0x" << std::hex << buffer.sorterPointerNullMask
               << std::dec << " scan_blocks=" << buffer.numScanBlocks
               << " expected_scan_blocks=" << expectedBlocks;
    }
    const int sorterAliasOwner = firstOwner([](const CellSortLocalReport& r) {
      return r.sortPending && r.sorterPointerAlias != 0;
    });
    if (sorterAliasOwner >= 0) {
      static const char* const pointerNames[] = {"u",
                                                 "v",
                                                 "w",
                                                 "q",
                                                 "x",
                                                 "y",
                                                 "z",
                                                 "id",
                                                 "scratch",
                                                 "cell_counts",
                                                 "cell_offsets",
                                                 "cell_starts",
                                                 "sorted_indices",
                                                 "block_sums",
                                                 "phase1_block_sums"};
      const auto& alias = reports[sorterAliasOwner];
      int cart[3];
      cartesianCoordinates(comm, sorterAliasOwner, cart);
      metadata << " first_sorter_alias_rank=" << sorterAliasOwner << " cart=("
               << cart[0] << ',' << cart[1] << ',' << cart[2] << ')';
      constexpr int pointerNameCount =
          sizeof(pointerNames) / sizeof(pointerNames[0]);
      if (alias.firstAliasPointerA >= 0 &&
          alias.firstAliasPointerA < pointerNameCount &&
          alias.firstAliasPointerB >= 0 &&
          alias.firstAliasPointerB < pointerNameCount)
        metadata << " pointer_a=" << pointerNames[alias.firstAliasPointerA]
                 << " pointer_b=" << pointerNames[alias.firstAliasPointerB];
    }
    std::cout << metadata.str() << std::endl;

    owner = firstOwner([](const CellSortLocalReport& r) {
      return r.histogram.negativeCount > 0;
    });
    std::ostringstream histogram;
    histogram << "[GPU-CYCLE-DIAG] cycle=" << cycle
              << " stage=cell_sort_stage1_histogram species=" << species
              << " global_expected=" << globalExpected
              << " checked_expected=" << checkedExpected
              << " sum=" << globalHistogram << " delta="
              << (globalHistogram - static_cast<long long>(checkedExpected))
              << " checked_ranks=" << invariantCheckedRanks
              << " skipped_active_ranks="
              << (activeRanks - invariantCheckedRanks)
              << " mismatching_ranks=" << histogramMismatchRanks
              << " negative_cells=" << negativeCounts
              << " min_count=" << (minCount == INT_MAX ? 0 : minCount)
              << " max_count=" << (maxCount == INT_MIN ? 0 : maxCount);
    appendRankCell(histogram, "max", maxCountOwner, maxCountCell);
    if (owner >= 0) {
      appendRankCell(histogram, "first_negative", owner,
                     reports[owner].histogram.firstNegativeCell);
      histogram << " first_negative_value="
                << reports[owner].histogram.firstNegativeValue;
    }
    std::cout << histogram.str() << std::endl;

    int phase1ExpectedRanks = 0;
    int phase1CheckedRanks = 0;
    int phase1CoverageMismatchRanks = 0;
    int phase1MismatchRanks = 0;
    int firstPhase1MismatchOwner = -1;
    int worstPhase1MismatchOwner = -1;
    unsigned long long phase1ExpectedTiles = 0;
    unsigned long long phase1CheckedTiles = 0;
    unsigned long long phase1ExpectedCells = 0;
    unsigned long long phase1CheckedCells = 0;
    unsigned long long phase1RawMismatch = 0;
    unsigned long long phase1ExpectedOutOfIntRange = 0;
    long long phase1RawSum = 0;
    long long phase1ExpectedSum = 0;
    for (int r = 0; r < ranks; ++r) {
      if (reports[r].sortPending &&
          reports[r].numCells > SORT_SCAN_ELEMENTS_PER_BLOCK) {
        ++phase1ExpectedRanks;
        phase1ExpectedTiles += reports[r].numScanBlocks;
        phase1ExpectedCells += reports[r].numCells;
      }
      if (!reports[r].phase1TotalsChecked)
        continue;

      ++phase1CheckedRanks;
      phase1CheckedTiles += reports[r].phase1Totals.checkedCount;
      phase1CheckedCells += reports[r].phase1Totals.checkedCellCount;
      if (reports[r].phase1Totals.checkedCount !=
              static_cast<unsigned long long>(reports[r].numScanBlocks) ||
          reports[r].phase1Totals.checkedCellCount !=
              static_cast<unsigned long long>(reports[r].numCells))
        ++phase1CoverageMismatchRanks;
      phase1RawMismatch += reports[r].phase1Totals.mismatchCount;
      phase1RawSum += reports[r].phase1Totals.rawSum;
      phase1ExpectedSum += reports[r].phase1Totals.expectedSum;
      phase1ExpectedOutOfIntRange +=
          reports[r].phase1Totals.expectedOutOfIntRangeCount;
      if (reports[r].phase1Totals.mismatchCount) {
        ++phase1MismatchRanks;
        if (firstPhase1MismatchOwner < 0)
          firstPhase1MismatchOwner = r;
      }
      if (reports[r].phase1Totals.worstBlock == DIAG_INVALID_INDEX)
        continue;
      if (worstPhase1MismatchOwner < 0 ||
          reports[r].phase1Totals.maxError >
              reports[worstPhase1MismatchOwner].phase1Totals.maxError)
        worstPhase1MismatchOwner = r;
    }

    std::ostringstream phase1Totals;
    phase1Totals
        << "[GPU-CYCLE-DIAG] cycle=" << cycle
        << " stage=cell_sort_stage2_phase1_raw_totals species=" << species
        << " path="
        << (reports[0].numCells > SORT_SCAN_ELEMENTS_PER_BLOCK ? "multi_block"
                                                               : "single_block")
        << " expected_ranks=" << phase1ExpectedRanks
        << " checked_ranks=" << phase1CheckedRanks
        << " skipped_ranks=" << (phase1ExpectedRanks - phase1CheckedRanks)
        << " expected_tiles=" << phase1ExpectedTiles
        << " checked_tiles=" << phase1CheckedTiles
        << " expected_histogram_cells=" << phase1ExpectedCells
        << " checked_histogram_cells=" << phase1CheckedCells
        << " coverage_mismatch_ranks=" << phase1CoverageMismatchRanks
        << " mismatching_ranks=" << phase1MismatchRanks
        << " mismatching_tiles=" << phase1RawMismatch
        << " raw_sum=" << phase1RawSum
        << " expected_sum=" << phase1ExpectedSum
        << " delta=" << (phase1RawSum - phase1ExpectedSum)
        << " expected_out_of_int_range_tiles="
        << phase1ExpectedOutOfIntRange << " max_abs_delta="
        << (worstPhase1MismatchOwner >= 0
                ? reports[worstPhase1MismatchOwner].phase1Totals.maxError
                : 0);
    if (firstPhase1MismatchOwner >= 0) {
      const auto& first =
          reports[firstPhase1MismatchOwner].phase1Totals;
      int cart[3];
      cartesianCoordinates(comm, firstPhase1MismatchOwner, cart);
      const unsigned long long firstCell =
          first.firstBlock * SORT_SCAN_ELEMENTS_PER_BLOCK;
      const unsigned long long firstCellEnd = std::min<unsigned long long>(
          firstCell + SORT_SCAN_ELEMENTS_PER_BLOCK,
          reports[firstPhase1MismatchOwner].numCells);
      phase1Totals << " first_mismatch_rank=" << firstPhase1MismatchOwner
                   << " first_mismatch_cart=(" << cart[0] << ',' << cart[1]
                   << ',' << cart[2] << ") first_mismatch_block="
                   << first.firstBlock << " first_mismatch_cell_range=["
                   << firstCell << ':' << firstCellEnd << ") first_raw="
                   << first.firstRaw << " first_expected="
                   << first.firstExpected << " first_delta="
                   << first.firstDelta;
    }
    if (worstPhase1MismatchOwner >= 0) {
      const auto& worst =
          reports[worstPhase1MismatchOwner].phase1Totals;
      int cart[3];
      cartesianCoordinates(comm, worstPhase1MismatchOwner, cart);
      const unsigned long long worstCell =
          worst.worstBlock * SORT_SCAN_ELEMENTS_PER_BLOCK;
      const unsigned long long worstCellEnd = std::min<unsigned long long>(
          worstCell + SORT_SCAN_ELEMENTS_PER_BLOCK,
          reports[worstPhase1MismatchOwner].numCells);
      phase1Totals << " worst_mismatch_rank=" << worstPhase1MismatchOwner
                   << " worst_mismatch_cart=(" << cart[0] << ',' << cart[1]
                   << ',' << cart[2] << ") worst_mismatch_block="
                   << worst.worstBlock << " worst_mismatch_cell_range=["
                   << worstCell << ':' << worstCellEnd << ") worst_raw="
                   << worst.worstRaw << " worst_expected="
                   << worst.worstExpected << " worst_delta="
                   << worst.worstDelta;
    }
    std::cout << phase1Totals.str() << std::endl;

    unsigned long long tileBase = 0, tileLocal = 0, tileTerminal = 0;
    unsigned long long blockMismatch = 0;
    for (int r = 0; r < ranks; ++r) {
      tileBase += reports[r].tiles.baseMismatchCount;
      tileLocal += reports[r].tiles.localMismatchCount;
      tileTerminal += reports[r].tiles.terminalMismatchCount;
      blockMismatch += reports[r].blockPrefix.mismatchCount;
    }
    owner = firstOwner([](const CellSortLocalReport& r) {
      return r.tiles.baseMismatchCount || r.tiles.localMismatchCount ||
             r.tiles.terminalMismatchCount || r.blockPrefix.mismatchCount;
    });
    std::ostringstream phases;
    phases << "[GPU-CYCLE-DIAG] cycle=" << cycle
           << " stage=cell_sort_stage2_scan_phases species=" << species
           << " path="
           << (reports[0].numCells > SORT_SCAN_ELEMENTS_PER_BLOCK
                   ? "multi_block"
                   : "single_block")
           << " checked_ranks=" << invariantCheckedRanks
           << " skipped_active_ranks=" << (activeRanks - invariantCheckedRanks)
           << " tile_base_mismatches=" << tileBase
           << " tile_local_mismatches=" << tileLocal
           << " tile_terminal_mismatches=" << tileTerminal
           << " phase1_raw_total_mismatches=" << phase1RawMismatch
           << " block_prefix_mismatches=" << blockMismatch
           << " phase2_block_prefix_mismatches=" << blockMismatch;
    if (owner >= 0) {
      if (reports[owner].tiles.baseMismatchCount) {
        appendRankCell(phases, "first_tile_base", owner,
                       reports[owner].tiles.firstBaseCell);
        phases << " actual=" << reports[owner].tiles.firstBaseActual
               << " expected=" << reports[owner].tiles.firstBaseExpected;
      } else if (reports[owner].tiles.localMismatchCount) {
        appendRankCell(phases, "first_tile_local", owner,
                       reports[owner].tiles.firstLocalCell);
        phases << " actual=" << reports[owner].tiles.firstLocalActual
               << " expected=" << reports[owner].tiles.firstLocalExpected;
      } else if (reports[owner].tiles.terminalMismatchCount) {
        appendRankCell(phases, "first_tile_terminal", owner,
                       reports[owner].tiles.firstTerminalCell);
        phases << " actual=" << reports[owner].tiles.firstTerminalActual
               << " expected=" << reports[owner].tiles.firstTerminalExpected;
      } else {
        int cart[3];
        cartesianCoordinates(comm, owner, cart);
        phases << " first_block_rank=" << owner << " cart=(" << cart[0] << ','
               << cart[1] << ',' << cart[2]
               << ") block=" << reports[owner].blockPrefix.firstBlock
               << " actual=" << reports[owner].blockPrefix.firstActual
               << " expected=" << reports[owner].blockPrefix.firstExpected;
      }
    }
    std::cout << phases.str() << std::endl;

    unsigned long long recurrenceMismatch = 0;
    unsigned long long invalidIntervals = 0;
    int terminalMismatchRanks = 0;
    int worstPrefixOwner = -1;
    for (int r = 0; r < ranks; ++r) {
      recurrenceMismatch += reports[r].prefix.recurrenceMismatchCount;
      invalidIntervals += reports[r].prefix.invalidIntervalCount;
      terminalMismatchRanks +=
          reports[r].invariantsChecked &&
          reports[r].prefix.terminal !=
              static_cast<long long>(reports[r].expectedParticles);
      if (reports[r].prefix.worstRecurrenceCell != DIAG_INVALID_INDEX &&
          (worstPrefixOwner < 0 ||
           reports[r].prefix.maxRecurrenceError >
               reports[worstPrefixOwner].prefix.maxRecurrenceError))
        worstPrefixOwner = r;
    }
    owner = firstOwner([](const CellSortLocalReport& r) {
      return r.invariantsChecked &&
             (r.prefix.recurrenceMismatchCount > 0 ||
              r.prefix.invalidIntervalCount > 0 ||
              r.prefix.terminal != static_cast<long long>(r.expectedParticles));
    });
    std::ostringstream prefix;
    prefix << "[GPU-CYCLE-DIAG] cycle=" << cycle
           << " stage=cell_sort_stage2_final_prefix species=" << species
           << " checked_ranks=" << invariantCheckedRanks
           << " skipped_active_ranks=" << (activeRanks - invariantCheckedRanks)
           << " recurrence_mismatches=" << recurrenceMismatch
           << " invalid_intervals=" << invalidIntervals
           << " terminal_mismatch_ranks=" << terminalMismatchRanks;
    if (owner >= 0) {
      if (reports[owner].prefix.recurrenceMismatchCount) {
        appendRankCell(prefix, "first_recurrence", owner,
                       reports[owner].prefix.firstRecurrenceCell);
        prefix << " actual=" << reports[owner].prefix.firstRecurrenceActual
               << " expected=" << reports[owner].prefix.firstRecurrenceExpected;
      } else if (reports[owner].prefix.invalidIntervalCount) {
        appendRankCell(prefix, "first_invalid_interval", owner,
                       reports[owner].prefix.firstInvalidIntervalCell);
        prefix << " begin=" << reports[owner].prefix.firstInvalidBegin
               << " end=" << reports[owner].prefix.firstInvalidEnd;
      } else {
        int cart[3];
        cartesianCoordinates(comm, owner, cart);
        prefix << " first_terminal_rank=" << owner << " cart=(" << cart[0]
               << ',' << cart[1] << ',' << cart[2]
               << ") terminal=" << reports[owner].prefix.terminal
               << " expected=" << reports[owner].expectedParticles;
      }
    }
    if (worstPrefixOwner >= 0) {
      appendRankCell(prefix, "worst_recurrence", worstPrefixOwner,
                     reports[worstPrefixOwner].prefix.worstRecurrenceCell);
      prefix << " worst_actual="
             << reports[worstPrefixOwner].prefix.worstRecurrenceActual
             << " worst_expected="
             << reports[worstPrefixOwner].prefix.worstRecurrenceExpected
             << " max_abs_error="
             << reports[worstPrefixOwner].prefix.maxRecurrenceError;
    }
    std::cout << prefix.str() << std::endl;

    unsigned long long reservationMismatch = 0;
    int worstReservationOwner = -1;
    for (int r = 0; r < ranks; ++r) {
      reservationMismatch += reports[r].reservation.mismatchCount;
      if (reports[r].reservation.worstMismatchCell != DIAG_INVALID_INDEX &&
          (worstReservationOwner < 0 ||
           reports[r].reservation.maxError >
               reports[worstReservationOwner].reservation.maxError))
        worstReservationOwner = r;
    }
    owner = firstOwner([](const CellSortLocalReport& r) {
      return r.reservation.mismatchCount > 0;
    });
    std::ostringstream reservations;
    reservations << "[GPU-CYCLE-DIAG] cycle=" << cycle
                 << " stage=cell_sort_stage3_reservations species=" << species
                 << " global_expected=" << globalExpected
                 << " checked_expected=" << checkedExpected
                 << " actual_sum=" << globalActual << " delta="
                 << (globalActual - static_cast<long long>(checkedExpected))
                 << " checked_ranks=" << invariantCheckedRanks
                 << " skipped_active_ranks="
                 << (activeRanks - invariantCheckedRanks)
                 << " mismatching_ranks=" << actualMismatchRanks
                 << " mismatching_cells=" << reservationMismatch;
    if (owner >= 0) {
      appendRankCell(reservations, "first_mismatch", owner,
                     reports[owner].reservation.firstMismatchCell);
      reservations << " actual=" << reports[owner].reservation.firstActual
                   << " histogram=" << reports[owner].reservation.firstExpected;
    }
    if (worstReservationOwner >= 0) {
      appendRankCell(
          reservations, "worst_mismatch", worstReservationOwner,
          reports[worstReservationOwner].reservation.worstMismatchCell);
      reservations << " worst_actual="
                   << reports[worstReservationOwner].reservation.worstActual
                   << " worst_histogram="
                   << reports[worstReservationOwner].reservation.worstExpected
                   << " max_abs_error="
                   << reports[worstReservationOwner].reservation.maxError;
    }
    std::cout << reservations.str() << std::endl;

    unsigned long long scanned = 0, oob = 0, wrongInterval = 0;
    unsigned long long invalidCells = 0;
    unsigned long long nonfinite = 0, holes = 0, duplicateSlots = 0;
    unsigned long long excessWrites = 0;
    unsigned long long permutationExpected = 0, occupancyExpected = 0;
    unsigned int maxMultiplicity = 0;
    int worstMultiplicityOwner = -1;
    int firstNonfinitePermutationOwner = -1;
    int minDestinationOwner = -1, maxDestinationOwner = -1;
    unsigned long long worstSlot = DIAG_INVALID_INDEX;
    for (int r = 0; r < ranks; ++r) {
      if (reports[r].permutationChecked)
        permutationExpected += reports[r].expectedParticles;
      if (reports[r].occupancyChecked)
        occupancyExpected += reports[r].expectedParticles;
      scanned += reports[r].permutation.scanned;
      oob += reports[r].permutation.outOfBoundsCount;
      wrongInterval += reports[r].permutation.intervalMismatchCount;
      invalidCells += reports[r].permutation.invalidCellCount;
      nonfinite += reports[r].permutation.nonfinitePositionCount;
      holes += reports[r].occupancy.holes;
      duplicateSlots += reports[r].occupancy.duplicateSlots;
      excessWrites += reports[r].occupancy.excessWrites;
      if (firstNonfinitePermutationOwner < 0 &&
          reports[r].permutation.nonfinitePositionCount)
        firstNonfinitePermutationOwner = r;
      if (worstMultiplicityOwner < 0 ||
          reports[r].occupancy.maxMultiplicity > maxMultiplicity) {
        maxMultiplicity = reports[r].occupancy.maxMultiplicity;
        worstMultiplicityOwner = r;
        worstSlot = reports[r].occupancy.worstSlot;
      }
      if (reports[r].permutation.minDestinationSource != DIAG_INVALID_INDEX &&
          (minDestinationOwner < 0 ||
           reports[r].permutation.minDestination <
               reports[minDestinationOwner].permutation.minDestination))
        minDestinationOwner = r;
      if (reports[r].permutation.maxDestinationSource != DIAG_INVALID_INDEX &&
          (maxDestinationOwner < 0 ||
           reports[r].permutation.maxDestination >
               reports[maxDestinationOwner].permutation.maxDestination))
        maxDestinationOwner = r;
    }
    const int firstOobOwner = firstOwner([](const CellSortLocalReport& r) {
      return r.permutation.outOfBoundsCount > 0;
    });
    const int firstIntervalOwner = firstOwner([](const CellSortLocalReport& r) {
      return r.permutation.intervalMismatchCount > 0;
    });
    const int firstInvalidCellOwner =
        firstOwner([](const CellSortLocalReport& r) {
          return r.permutation.invalidCellCount > 0;
        });
    const int firstHoleOwner = firstOwner(
        [](const CellSortLocalReport& r) { return r.occupancy.holes > 0; });
    const int firstDuplicateOwner =
        firstOwner([](const CellSortLocalReport& r) {
          return r.occupancy.duplicateSlots > 0;
        });
    std::ostringstream permutation;
    permutation << "[GPU-CYCLE-DIAG] cycle=" << cycle
                << " stage=cell_sort_stage3_permutation species=" << species
                << " permutation_checked_ranks=" << permutationCheckedRanks
                << " permutation_skipped_active_ranks="
                << (activeRanks - permutationCheckedRanks)
                << " occupancy_checked_ranks=" << occupancyCheckedRanks
                << " occupancy_skipped_active_ranks="
                << (activeRanks - occupancyCheckedRanks)
                << " permutation_expected=" << permutationExpected
                << " occupancy_expected=" << occupancyExpected
                << " scanned=" << scanned << " out_of_bounds=" << oob
                << " invalid_source_cells=" << invalidCells
                << " wrong_cell_interval=" << wrongInterval
                << " nonfinite_positions=" << nonfinite << " holes=" << holes
                << " duplicate_slots=" << duplicateSlots
                << " excess_writes=" << excessWrites
                << " max_multiplicity=" << maxMultiplicity;
    if (minDestinationOwner >= 0) {
      const auto& minimum = reports[minDestinationOwner].permutation;
      int cart[3];
      cartesianCoordinates(comm, minDestinationOwner, cart);
      permutation << " min_destination_rank=" << minDestinationOwner
                  << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") destination=" << minimum.minDestination
                  << " source=" << minimum.minDestinationSource;
    }
    if (maxDestinationOwner >= 0) {
      const auto& maximum = reports[maxDestinationOwner].permutation;
      int cart[3];
      cartesianCoordinates(comm, maxDestinationOwner, cart);
      permutation << " max_destination_rank=" << maxDestinationOwner
                  << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") destination=" << maximum.maxDestination
                  << " source=" << maximum.maxDestinationSource;
    }
    if (worstMultiplicityOwner >= 0 && maxMultiplicity > 1) {
      int cart[3];
      cartesianCoordinates(comm, worstMultiplicityOwner, cart);
      permutation << " worst_slot_rank=" << worstMultiplicityOwner << " cart=("
                  << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") slot=" << worstSlot;
    }
    if (firstOobOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstOobOwner, cart);
      permutation
          << " first_oob_rank=" << firstOobOwner << " cart=(" << cart[0] << ','
          << cart[1] << ',' << cart[2] << ") source="
          << reports[firstOobOwner].permutation.firstOutOfBoundsSource
          << " destination="
          << reports[firstOobOwner].permutation.firstOutOfBoundsDestination;
    }
    if (firstIntervalOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstIntervalOwner, cart);
      permutation
          << " first_interval_rank=" << firstIntervalOwner << " cart=("
          << cart[0] << ',' << cart[1] << ',' << cart[2] << ") source="
          << reports[firstIntervalOwner].permutation.firstIntervalSource
          << " cell="
          << reports[firstIntervalOwner].permutation.firstIntervalCell
          << " destination="
          << reports[firstIntervalOwner].permutation.firstIntervalDestination
          << " interval=["
          << reports[firstIntervalOwner].permutation.firstIntervalBegin << ','
          << reports[firstIntervalOwner].permutation.firstIntervalEnd << ')';
    }
    if (firstInvalidCellOwner >= 0) {
      const auto& first = reports[firstInvalidCellOwner].permutation;
      int cart[3];
      cartesianCoordinates(comm, firstInvalidCellOwner, cart);
      permutation << " first_invalid_cell_rank=" << firstInvalidCellOwner
                  << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") source=" << first.firstInvalidCellSource
                  << " cell=" << first.firstInvalidCell;
    }
    if (firstHoleOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstHoleOwner, cart);
      permutation << " first_hole_rank=" << firstHoleOwner << " cart=("
                  << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") slot=" << reports[firstHoleOwner].occupancy.firstHole;
    }
    if (firstDuplicateOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstDuplicateOwner, cart);
      permutation << " first_duplicate_rank=" << firstDuplicateOwner
                  << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") slot="
                  << reports[firstDuplicateOwner].occupancy.firstDuplicate;
    }
    if (firstNonfinitePermutationOwner >= 0) {
      const auto& first = reports[firstNonfinitePermutationOwner].permutation;
      int cart[3];
      int cellXYZ[3];
      cartesianCoordinates(comm, firstNonfinitePermutationOwner, cart);
      decodeCell(first.firstNonfiniteCell,
                 reports[firstNonfinitePermutationOwner].nxc,
                 reports[firstNonfinitePermutationOwner].nyc, cellXYZ);
      permutation << " first_nonfinite_rank=" << firstNonfinitePermutationOwner
                  << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
                  << ") source=" << first.firstNonfiniteSource
                  << " cell=" << first.firstNonfiniteCell << " cell_xyz=("
                  << cellXYZ[0] << ',' << cellXYZ[1] << ',' << cellXYZ[2]
                  << ") position=(" << std::scientific << std::setprecision(17)
                  << first.firstNonfiniteX << ',' << first.firstNonfiniteY
                  << ',' << first.firstNonfiniteZ << ')';
    }
    std::cout << permutation.str() << std::endl;
  }

  // finishSort() will write its bounded per-field poison reductions here.
  const bool safeToPoison =
      local.sortPending && local.pendingNOP == local.nop &&
      local.expectedParticles <= local.nop && local.nop <= local.capacity &&
      local.expectedParticles <= local.indexCapacity &&
      local.expectedParticles <= static_cast<unsigned long long>(INT_MAX) &&
      local.nop <= local.scratchBytes / sizeof(std::uint64_t) &&
      !local.pointerAlias && local.pendingHostMatches &&
      local.enqueuePointerMutationMask == 0 &&
      local.enqueueSortBufferMutationMask == 0 &&
      local.enqueueAllocationMetadataMutationMask == 0 && sorterPointersValid;
  sorter.armStage4Diagnostics(safeToPoison ? impl_->dStage4Partials : nullptr,
                              CELL_SORT_DIAGNOSTIC_MAX_BLOCKS);
}

void GPUCycleDiagnostics::reportCellSorterAfterScatter(
    int cycle, int species, const CellSorter& sorter,
    const particleArrayCUDA& particles, const grid3DCUDA& hostGrid,
    const grid3DCUDA* deviceGrid, cudaStream_t stream, MPI_Comm comm) {
  if (!impl_)
    impl_ = new Impl();
  impl_->ensureCellSortStorage();

  CellSortPostLocalReport local{};
  local.post = emptySortPostPartial();
  for (int field = 0; field < CELL_SORT_DIAGNOSTIC_MAX_FIELDS; ++field)
    local.stage4[field] = emptyStage4Partial();

  const CellSortPointerSnapshot& snapshot = impl_->cellSortSnapshot;
  local.contextMismatch =
      snapshot.cycle != cycle || snapshot.species != species;
  local.expectedParticles = snapshot.expectedParticles;
  local.nop = snapshot.nop;
  local.active = snapshot.active;
  local.expectFullSort = snapshot.expectFullSort;
  local.stage4FieldCount = sorter.diagnosticStage4FieldCount();
  local.expectedFieldCount = snapshot.fieldCount;
  local.nxc = hostGrid.nxc;
  local.nyc = hostGrid.nyc;
  local.nzc = hostGrid.nzc;
  const ParticleSoADevice* soa = particles.getSoA();
  const void* fields[CELL_SORT_DIAGNOSTIC_MAX_FIELDS] = {
      soa->u, soa->v, soa->w, soa->q, soa->x, soa->y, soa->z, soa->id};
  const int currentFieldCount = soa->trackParticleID ? 8 : 7;
  for (int field = 0; field < currentFieldCount; ++field)
    local.newPointers[field] = reinterpret_cast<std::uintptr_t>(fields[field]);
  for (int field = 0; field < snapshot.fieldCount; ++field)
    local.oldPointers[field] = snapshot.pointers[field];
  local.oldScratch = snapshot.scratch;
  local.newScratch =
      reinterpret_cast<std::uintptr_t>(sorter.diagnosticScratchPointer());
  local.pointerAlias =
      hasPointerAlias(*soa, sorter.diagnosticScratchPointer()) ? 1 : 0;

  if (snapshot.active && !local.contextMismatch) {
    if (currentFieldCount != snapshot.fieldCount)
      local.pointerRotationMask |= 1 << 15;
    const int comparableFields =
        std::min(currentFieldCount, snapshot.fieldCount);
    for (int field = 0; field < comparableFields; ++field) {
      const std::uintptr_t expected =
          field == 0 ? snapshot.scratch : snapshot.pointers[field - 1];
      if (local.newPointers[field] != expected)
        local.pointerRotationMask |= 1 << field;
    }
    if (local.newScratch != snapshot.pointers[snapshot.fieldCount - 1])
      local.pointerRotationMask |= 1 << 8;
  }

  const int stage4Blocks = sorter.diagnosticStage4Blocks();
  local.stage4Checked = snapshot.active && !local.contextMismatch &&
                        stage4Blocks > 0 &&
                        local.stage4FieldCount == snapshot.fieldCount;
  if (local.stage4FieldCount > 0 && stage4Blocks > 0) {
    const std::size_t stage4Count =
        static_cast<std::size_t>(local.stage4FieldCount) *
        CELL_SORT_DIAGNOSTIC_MAX_BLOCKS;
    cudaErrChk(
        cudaMemcpyAsync(impl_->hStage4Partials, impl_->dStage4Partials,
                        stage4Count * sizeof(CellSortStage4DiagnosticPartial),
                        cudaMemcpyDeviceToHost, stream));
  }

  int postBlocks = 0;
  const bool postPointersValid = currentFieldCount == snapshot.fieldCount &&
                                 local.pointerRotationMask == 0 &&
                                 !local.pointerAlias && soa->x && soa->y &&
                                 soa->z && sorter.getCellStartOffsets() &&
                                 sorter.getCellCounts();
  if (snapshot.active && !local.contextMismatch && impl_->postSortLayoutSafe &&
      postPointersValid && snapshot.expectedParticles <= UINT32_MAX &&
      deviceGrid) {
    postBlocks = reductionBlocks(sorter.getNumCells());
    local.postLayoutChecked = postBlocks > 0;
    reducePostSortLayout<<<postBlocks, DIAG_THREADS, 0, stream>>>(
        soa->x, soa->y, soa->z, sorter.getCellStartOffsets(),
        sorter.getCellCounts(), sorter.getNumCells(),
        static_cast<std::uint32_t>(snapshot.expectedParticles), deviceGrid,
        impl_->dCellSortPartials);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaMemcpyAsync(impl_->hCellSortPartials,
                               impl_->dCellSortPartials,
                               postBlocks * sizeof(CellSortBlockReport),
                               cudaMemcpyDeviceToHost, stream));
  }
  cudaErrChk(cudaStreamSynchronize(stream));

  if (local.stage4FieldCount > 0 && stage4Blocks > 0) {
    for (int field = 0; field < local.stage4FieldCount; ++field) {
      CellSortStage4DiagnosticPartial folded = emptyStage4Partial();
      const CellSortStage4DiagnosticPartial* partials =
          impl_->hStage4Partials + field * CELL_SORT_DIAGNOSTIC_MAX_BLOCKS;
      for (int block = 0; block < stage4Blocks; ++block)
        combineStage4Partial(folded, partials[block]);
      local.stage4[field] = folded;
    }
  }
  if (postBlocks > 0)
    local.post = foldPostSortPartials(impl_->hCellSortPartials, postBlocks);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &ranks);
  const int reportBytes = sizeof(CellSortPostLocalReport);
  if (rank == 0)
    impl_->ensureGatherStorage(static_cast<std::size_t>(ranks) * reportBytes);
  MPI_Gather(&local, reportBytes, MPI_BYTE,
             rank == 0 ? impl_->gatherStorage.data() : nullptr, reportBytes,
             MPI_BYTE, 0, comm);

  if (rank != 0)
    return;

  const auto* reports = reinterpret_cast<const CellSortPostLocalReport*>(
      impl_->gatherStorage.data());
  const char* fieldNames[CELL_SORT_DIAGNOSTIC_MAX_FIELDS] = {
      "u", "v", "w", "q", "x", "y", "z", "id"};
  int maxExpectedFieldCount = 0;
  int pointerMismatchRanks = 0;
  int pointerAliasRanks = 0;
  int contextMismatchRanks = 0;
  int activeRanks = 0;
  int stage4CheckedRanks = 0;
  int postLayoutCheckedRanks = 0;
  int fullSortCountMismatchRanks = 0;
  int firstPointerMismatch = -1;
  for (int r = 0; r < ranks; ++r) {
    if (reports[r].active)
      maxExpectedFieldCount =
          std::max(maxExpectedFieldCount, reports[r].expectedFieldCount);
    if (reports[r].pointerRotationMask) {
      ++pointerMismatchRanks;
      if (firstPointerMismatch < 0)
        firstPointerMismatch = r;
    }
    pointerAliasRanks += reports[r].active && reports[r].pointerAlias != 0;
    contextMismatchRanks += reports[r].contextMismatch != 0;
    activeRanks += reports[r].active != 0;
    stage4CheckedRanks += reports[r].stage4Checked != 0;
    postLayoutCheckedRanks += reports[r].postLayoutChecked != 0;
    fullSortCountMismatchRanks +=
        reports[r].active && reports[r].expectFullSort &&
        reports[r].expectedParticles != reports[r].nop;
  }

  std::ostringstream pointers;
  pointers << "[GPU-CYCLE-DIAG] cycle=" << cycle
           << " stage=cell_sort_stage4_pointer_rotation species=" << species
           << " active_ranks=" << activeRanks
           << " mismatch_ranks=" << pointerMismatchRanks
           << " alias_ranks=" << pointerAliasRanks
           << " scatter_checked_ranks=" << stage4CheckedRanks
           << " scatter_skipped_active_ranks="
           << (activeRanks - stage4CheckedRanks)
           << " full_sort_count_mismatch_ranks=" << fullSortCountMismatchRanks
           << " context_mismatch_ranks=" << contextMismatchRanks;
  if (firstPointerMismatch >= 0) {
    const CellSortPostLocalReport& bad = reports[firstPointerMismatch];
    int cart[3];
    cartesianCoordinates(comm, firstPointerMismatch, cart);
    pointers << " first_rank=" << firstPointerMismatch << " cart=(" << cart[0]
             << ',' << cart[1] << ',' << cart[2] << ") rotation_mask=0x"
             << std::hex << bad.pointerRotationMask << std::dec;
    int firstField = -1;
    for (int field = 0; field < 8; ++field)
      if (bad.pointerRotationMask & (1 << field)) {
        firstField = field;
        break;
      }
    if (firstField >= 0) {
      const std::uintptr_t expected =
          firstField == 0 ? bad.oldScratch : bad.oldPointers[firstField - 1];
      pointers << " first_field=" << fieldNames[firstField]
               << " expected_ptr=0x" << std::hex << expected << " actual_ptr=0x"
               << bad.newPointers[firstField] << std::dec;
    } else if (bad.pointerRotationMask & (1 << 8)) {
      pointers << " first_field=scratch expected_ptr=0x" << std::hex
               << bad.oldPointers[bad.expectedFieldCount - 1]
               << " actual_ptr=0x" << bad.newScratch << std::dec;
    }
  }
  std::cout << pointers.str() << std::endl;

  maxExpectedFieldCount =
      std::min(maxExpectedFieldCount, CELL_SORT_DIAGNOSTIC_MAX_FIELDS);
  if (maxExpectedFieldCount == 0 && activeRanks > 0) {
    std::cout << "[GPU-CYCLE-DIAG] cycle=" << cycle
              << " stage=cell_sort_stage4_scatter species=" << species
              << " component=none checked_ranks=0 skipped_active_ranks="
              << activeRanks << std::endl;
  }

  for (int field = 0; field < maxExpectedFieldCount; ++field) {
    unsigned long long prefixPoison = 0;
    unsigned long long tailPoison = 0;
    unsigned long long prefixValueMismatches = 0;
    unsigned long long tailValueMismatches = 0;
    int firstPrefixOwner = -1;
    int firstTailOwner = -1;
    int lastPrefixOwner = -1;
    int lastTailOwner = -1;
    unsigned long long firstPrefix = DIAG_INVALID_INDEX;
    unsigned long long firstTail = DIAG_INVALID_INDEX;
    unsigned long long lastPrefix = 0;
    unsigned long long lastTail = 0;
    int firstPrefixValueOwner = -1;
    int firstTailValueOwner = -1;
    unsigned long long firstPrefixValueSource = DIAG_INVALID_INDEX;
    unsigned long long firstPrefixValueDestination = DIAG_INVALID_INDEX;
    unsigned long long firstTailValue = DIAG_INVALID_INDEX;
    int fieldCheckedRanks = 0;
    for (int r = 0; r < ranks; ++r) {
      if (!reports[r].stage4Checked || field >= reports[r].stage4FieldCount)
        continue;
      ++fieldCheckedRanks;
      const auto& candidate = reports[r].stage4[field];
      prefixPoison += candidate.prefixPoisonCount;
      tailPoison += candidate.tailPoisonCount;
      prefixValueMismatches += candidate.prefixValueMismatchCount;
      tailValueMismatches += candidate.tailValueMismatchCount;
      if (candidate.prefixPoisonCount > 0 && firstPrefixOwner < 0) {
        firstPrefixOwner = r;
        firstPrefix = candidate.firstPrefixPoison;
      }
      if (candidate.tailPoisonCount > 0 && firstTailOwner < 0) {
        firstTailOwner = r;
        firstTail = candidate.firstTailPoison;
      }
      if (candidate.prefixPoisonCount > 0 &&
          (lastPrefixOwner < 0 || candidate.lastPrefixPoison > lastPrefix)) {
        lastPrefixOwner = r;
        lastPrefix = candidate.lastPrefixPoison;
      }
      if (candidate.tailPoisonCount > 0 &&
          (lastTailOwner < 0 || candidate.lastTailPoison > lastTail)) {
        lastTailOwner = r;
        lastTail = candidate.lastTailPoison;
      }
      if (candidate.prefixValueMismatchCount > 0 && firstPrefixValueOwner < 0) {
        firstPrefixValueOwner = r;
        firstPrefixValueSource = candidate.firstPrefixValueMismatchSource;
        firstPrefixValueDestination =
            candidate.firstPrefixValueMismatchDestination;
      }
      if (candidate.tailValueMismatchCount > 0 && firstTailValueOwner < 0) {
        firstTailValueOwner = r;
        firstTailValue = candidate.firstTailValueMismatch;
      }
    }
    std::ostringstream coverage;
    coverage << "[GPU-CYCLE-DIAG] cycle=" << cycle
             << " stage=cell_sort_stage4_scatter species=" << species
             << " component=" << fieldNames[field]
             << " checked_ranks=" << fieldCheckedRanks
             << " skipped_active_ranks=" << (activeRanks - fieldCheckedRanks)
             << " unwritten_prefix=" << prefixPoison
             << " unwritten_identity_tail=" << tailPoison
             << " mapped_value_mismatch_prefix=" << prefixValueMismatches
             << " identity_tail_value_mismatch=" << tailValueMismatches;
    if (firstPrefixOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstPrefixOwner, cart);
      coverage << " first_prefix_rank=" << firstPrefixOwner << " cart=("
               << cart[0] << ',' << cart[1] << ',' << cart[2]
               << ") first_prefix_index=" << firstPrefix
               << " last_prefix_rank=" << lastPrefixOwner
               << " last_prefix_index=" << lastPrefix;
    }
    if (firstTailOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstTailOwner, cart);
      coverage << " first_tail_rank=" << firstTailOwner << " cart=(" << cart[0]
               << ',' << cart[1] << ',' << cart[2]
               << ") first_tail_index=" << firstTail
               << " last_tail_rank=" << lastTailOwner
               << " last_tail_index=" << lastTail;
    }
    if (firstPrefixValueOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstPrefixValueOwner, cart);
      coverage << " first_mapped_value_mismatch_rank=" << firstPrefixValueOwner
               << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
               << ") source=" << firstPrefixValueSource
               << " destination=" << firstPrefixValueDestination;
    }
    if (firstTailValueOwner >= 0) {
      int cart[3];
      cartesianCoordinates(comm, firstTailValueOwner, cart);
      coverage << " first_tail_value_mismatch_rank=" << firstTailValueOwner
               << " cart=(" << cart[0] << ',' << cart[1] << ',' << cart[2]
               << ") index=" << firstTailValue;
    }
    std::cout << coverage.str() << std::endl;
  }

  unsigned long long scanned = 0;
  unsigned long long invalidRanges = 0;
  unsigned long long cellMismatches = 0;
  unsigned long long nonfinite = 0;
  int firstInvalidOwner = -1;
  int firstMismatchOwner = -1;
  int firstNonfiniteOwner = -1;
  for (int r = 0; r < ranks; ++r) {
    scanned += reports[r].post.scanned;
    invalidRanges += reports[r].post.invalidRangeCount;
    cellMismatches += reports[r].post.cellMismatchCount;
    nonfinite += reports[r].post.nonfinitePositionCount;
    if (firstInvalidOwner < 0 && reports[r].post.invalidRangeCount)
      firstInvalidOwner = r;
    if (firstMismatchOwner < 0 && reports[r].post.cellMismatchCount)
      firstMismatchOwner = r;
    if (firstNonfiniteOwner < 0 && reports[r].post.nonfinitePositionCount)
      firstNonfiniteOwner = r;
  }
  std::ostringstream layout;
  layout << "[GPU-CYCLE-DIAG] cycle=" << cycle
         << " stage=cell_sort_stage4_sorted_layout species=" << species
         << " checked_ranks=" << postLayoutCheckedRanks
         << " skipped_active_ranks=" << (activeRanks - postLayoutCheckedRanks)
         << " scanned=" << scanned << " invalid_ranges=" << invalidRanges
         << " wrong_cell=" << cellMismatches
         << " nonfinite_positions=" << nonfinite;
  if (firstInvalidOwner >= 0) {
    const auto& first = reports[firstInvalidOwner].post;
    int cart[3];
    int cellXYZ[3];
    cartesianCoordinates(comm, firstInvalidOwner, cart);
    decodeCell(first.firstInvalidCell, reports[firstInvalidOwner].nxc,
               reports[firstInvalidOwner].nyc, cellXYZ);
    layout << " first_invalid_rank=" << firstInvalidOwner << " cart=("
           << cart[0] << ',' << cart[1] << ',' << cart[2]
           << ") cell=" << first.firstInvalidCell << " cell_xyz=(" << cellXYZ[0]
           << ',' << cellXYZ[1] << ',' << cellXYZ[2] << ") interval=["
           << first.firstInvalidBegin << ',' << first.firstInvalidEnd << ')';
  }
  if (firstMismatchOwner >= 0) {
    const auto& first = reports[firstMismatchOwner].post;
    int cart[3];
    int expectedXYZ[3];
    int actualXYZ[3];
    cartesianCoordinates(comm, firstMismatchOwner, cart);
    decodeCell(first.firstExpectedCell, reports[firstMismatchOwner].nxc,
               reports[firstMismatchOwner].nyc, expectedXYZ);
    decodeCell(first.firstActualCell, reports[firstMismatchOwner].nxc,
               reports[firstMismatchOwner].nyc, actualXYZ);
    layout << " first_mismatch_rank=" << firstMismatchOwner << " cart=("
           << cart[0] << ',' << cart[1] << ',' << cart[2]
           << ") position=" << first.firstMismatchPosition
           << " expected_cell=" << first.firstExpectedCell << " expected_xyz=("
           << expectedXYZ[0] << ',' << expectedXYZ[1] << ',' << expectedXYZ[2]
           << ") actual_cell=" << first.firstActualCell << " actual_xyz=("
           << actualXYZ[0] << ',' << actualXYZ[1] << ',' << actualXYZ[2] << ')';
  }
  if (firstNonfiniteOwner >= 0) {
    int cart[3];
    cartesianCoordinates(comm, firstNonfiniteOwner, cart);
    layout << " first_nonfinite_rank=" << firstNonfiniteOwner << " cart=("
           << cart[0] << ',' << cart[1] << ',' << cart[2] << ") position="
           << reports[firstNonfiniteOwner].post.firstNonfinitePosition;
  }
  std::cout << layout.str() << std::endl;
}

#endif // IPIC3D_GPU_CYCLE_DIAGNOSTICS
