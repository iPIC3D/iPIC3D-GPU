/**
 * @file GPUFieldArray.cuh
 * @brief GPU-resident field array wrapper for the iPIC3D GPU field solver.
 *
 * Provides GPU-allocated 3D/4D arrays that mirror the host array3_double /
 * array4_double layout (row-major, contiguous) but live entirely in device
 * memory.  Host↔device transfers are explicit and only triggered for I/O or
 * diagnostic purposes.
 *
 * The MPI halo exchange can pass the raw device pointer directly to
 * GPU-aware MPI calls (using the same MPI derived datatypes already
 * defined in EMfields3D for the host arrays, since the memory layout
 * is identical).
 *
 * All three wrapper types are parameterised on the element type T.
 * Backward-compatible aliases at the bottom of this file map the plain
 * names (GPUFieldArray3, GPUFieldArray4, GPUKrylovVector) to the
 * cudaSolverType instantiation so that existing call-sites compile
 * without modification.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_FIELD_ARRAY_CUH
#define GPU_FIELD_ARRAY_CUH

#include "cudaTypeDef.cuh"
#include <cstddef>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  GPUFieldArray3Impl<T>  –  3-dimensional device array  (S3 × S2 × S1)
// ---------------------------------------------------------------------------
template <typename T> class GPUFieldArray3Impl {
public:
  // ---- construction / destruction ----

  GPUFieldArray3Impl()
      : d_ptr_(nullptr), S3_(0), S2_(0), S1_(0), size_(0), owning_(true) {}

  GPUFieldArray3Impl(size_t s3, size_t s2, size_t s1)
      : S3_(s3), S2_(s2), S1_(s1), size_(s3 * s2 * s1), owning_(true) {
    cudaErrChk(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    cudaErrChk(cudaMemset(d_ptr_, 0, size_ * sizeof(T)));
  }

  ~GPUFieldArray3Impl() { free(); }

  // Not copyable (owns device memory)
  GPUFieldArray3Impl(const GPUFieldArray3Impl&) = delete;
  GPUFieldArray3Impl& operator=(const GPUFieldArray3Impl&) = delete;

  // Movable
  GPUFieldArray3Impl(GPUFieldArray3Impl&& o) noexcept
      : d_ptr_(o.d_ptr_), S3_(o.S3_), S2_(o.S2_), S1_(o.S1_), size_(o.size_),
        owning_(o.owning_) {
    o.d_ptr_ = nullptr;
    o.size_ = 0;
  }

  GPUFieldArray3Impl& operator=(GPUFieldArray3Impl&& o) noexcept {
    if (this != &o) {
      free();
      d_ptr_ = o.d_ptr_;
      S3_ = o.S3_;
      S2_ = o.S2_;
      S1_ = o.S1_;
      size_ = o.size_;
      owning_ = o.owning_;
      o.d_ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  /** Create a non-owning view that wraps an existing device pointer.
   *  The view does NOT own the memory — free() and the destructor are no-ops.
   *  Use this to pass per-species slices of GPUFieldArray4Impl to functions
   *  that take GPUFieldArray3Impl& (e.g. GPU halo exchange wrappers). */
  static GPUFieldArray3Impl wrapDevice(T* devPtr, size_t s3, size_t s2,
                                       size_t s1) {
    GPUFieldArray3Impl v;
    v.d_ptr_ = devPtr;
    v.S3_ = s3;
    v.S2_ = s2;
    v.S1_ = s1;
    v.size_ = s3 * s2 * s1;
    v.owning_ = false;
    return v;
  }

  void free() {
    if (d_ptr_ && owning_) {
      cudaFree(d_ptr_);
    }
    d_ptr_ = nullptr;
    size_ = 0;
  }

  // ---- accessors ----

  /** Raw device pointer (row-major, contiguous). */
  T* devPtr() { return d_ptr_; }
  const T* devPtr() const { return d_ptr_; }

  size_t dim1() const { return S3_; } // outermost
  size_t dim2() const { return S2_; }
  size_t dim3() const { return S1_; }   // innermost (contiguous)
  size_t size() const { return size_; } // total elements

  // ---- device ↔ host transfers (synchronous) ----

  /** Copy entire array from host (arr3_double flat buffer) to device. */
  void copyFromHost(const T* h_ptr) {
    cudaErrChk(
        cudaMemcpy(d_ptr_, h_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  /** Copy entire array from device to host (arr3_double flat buffer). */
  void copyToHost(T* h_ptr) const {
    cudaErrChk(
        cudaMemcpy(h_ptr, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

  // ---- async variants ----

  void copyFromHostAsync(const T* h_ptr, cudaStream_t stream) {
    cudaErrChk(cudaMemcpyAsync(d_ptr_, h_ptr, size_ * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  void copyToHostAsync(T* h_ptr, cudaStream_t stream) const {
    cudaErrChk(cudaMemcpyAsync(h_ptr, d_ptr_, size_ * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  // ---- device-side utilities ----

  void setAll(T val, cudaStream_t stream = 0) {
    if (val == T{})
      cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(T), stream));
    else {
      // For non-zero, we do a small kernel launch (defined in
      // GPUFieldArray.cu).
      fillKernelLaunch(d_ptr_, val, size_, stream);
    }
  }

  // ---- 3D → device pointer with offset (for MPI halo exchange) ----
  // Returns a pointer offset to element [i3][i2][i1] so that
  // MPI derived datatypes (which encode strides from the base pointer)
  // work identically to the host arrays.
  T* ptrAt(size_t i3, size_t i2, size_t i1) {
    return d_ptr_ + (i3 * S2_ + i2) * S1_ + i1;
  }

  // helper: fill device array with a constant value
  // (declared here, defined + explicitly instantiated in GPUFieldArray.cu)
  static void fillKernelLaunch(T* d, T val, size_t n, cudaStream_t stream);

private:
  T* d_ptr_;
  size_t S3_, S2_, S1_, size_;
  bool owning_; ///< true = this object owns d_ptr_ (will cudaFree)
};

// ---------------------------------------------------------------------------
//  GPUFieldArray4Impl<T>  –  4-dimensional device array  (S4 × S3 × S2 × S1)
// ---------------------------------------------------------------------------
template <typename T> class GPUFieldArray4Impl {
public:
  GPUFieldArray4Impl()
      : d_ptr_(nullptr), S4_(0), S3_(0), S2_(0), S1_(0), size_(0) {}

  GPUFieldArray4Impl(size_t s4, size_t s3, size_t s2, size_t s1)
      : S4_(s4), S3_(s3), S2_(s2), S1_(s1), size_(s4 * s3 * s2 * s1) {
    cudaErrChk(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
    cudaErrChk(cudaMemset(d_ptr_, 0, size_ * sizeof(T)));
  }

  ~GPUFieldArray4Impl() { free(); }

  GPUFieldArray4Impl(const GPUFieldArray4Impl&) = delete;
  GPUFieldArray4Impl& operator=(const GPUFieldArray4Impl&) = delete;

  GPUFieldArray4Impl(GPUFieldArray4Impl&& o) noexcept
      : d_ptr_(o.d_ptr_), S4_(o.S4_), S3_(o.S3_), S2_(o.S2_), S1_(o.S1_),
        size_(o.size_) {
    o.d_ptr_ = nullptr;
    o.size_ = 0;
  }

  GPUFieldArray4Impl& operator=(GPUFieldArray4Impl&& o) noexcept {
    if (this != &o) {
      free();
      d_ptr_ = o.d_ptr_;
      S4_ = o.S4_;
      S3_ = o.S3_;
      S2_ = o.S2_;
      S1_ = o.S1_;
      size_ = o.size_;
      o.d_ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  void free() {
    if (d_ptr_) {
      cudaFree(d_ptr_);
      d_ptr_ = nullptr;
      size_ = 0;
    }
  }

  T* devPtr() { return d_ptr_; }
  const T* devPtr() const { return d_ptr_; }

  size_t dim1() const { return S4_; }
  size_t dim2() const { return S3_; }
  size_t dim3() const { return S2_; }
  size_t dim4() const { return S1_; }
  size_t size() const { return size_; }

  /** Pointer to the start of species-slice [is], i.e. element [is][0][0][0]. */
  T* speciesPtr(size_t is) { return d_ptr_ + is * S3_ * S2_ * S1_; }

  void copyFromHost(const T* h_ptr) {
    cudaErrChk(
        cudaMemcpy(d_ptr_, h_ptr, size_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyToHost(T* h_ptr) const {
    cudaErrChk(
        cudaMemcpy(h_ptr, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copyFromHostAsync(const T* h_ptr, cudaStream_t stream) {
    cudaErrChk(cudaMemcpyAsync(d_ptr_, h_ptr, size_ * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  void copyToHostAsync(T* h_ptr, cudaStream_t stream) const {
    cudaErrChk(cudaMemcpyAsync(h_ptr, d_ptr_, size_ * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  void setAll(T val, cudaStream_t stream = 0) {
    if (val == T{})
      cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(T), stream));
    else
      GPUFieldArray3Impl<T>::fillKernelLaunch(d_ptr_, val, size_, stream);
  }

  // For MPI: pointer at [is][i3][i2][i1]
  T* ptrAt(size_t is, size_t i3, size_t i2, size_t i1) {
    return d_ptr_ + ((is * S3_ + i3) * S2_ + i2) * S1_ + i1;
  }

private:
  T* d_ptr_;
  size_t S4_, S3_, S2_, S1_, size_;
};

// ---------------------------------------------------------------------------
//  GPUKrylovVectorImpl<T> – 1-D device array for Krylov solver vectors
// ---------------------------------------------------------------------------
template <typename T> class GPUKrylovVectorImpl {
public:
  GPUKrylovVectorImpl() : d_ptr_(nullptr), size_(0) {}

  explicit GPUKrylovVectorImpl(size_t n) : size_(n) {
    cudaErrChk(cudaMalloc(&d_ptr_, n * sizeof(T)));
    cudaErrChk(cudaMemset(d_ptr_, 0, n * sizeof(T)));
  }

  ~GPUKrylovVectorImpl() { free(); }

  GPUKrylovVectorImpl(const GPUKrylovVectorImpl&) = delete;
  GPUKrylovVectorImpl& operator=(const GPUKrylovVectorImpl&) = delete;

  GPUKrylovVectorImpl(GPUKrylovVectorImpl&& o) noexcept
      : d_ptr_(o.d_ptr_), size_(o.size_) {
    o.d_ptr_ = nullptr;
    o.size_ = 0;
  }

  GPUKrylovVectorImpl& operator=(GPUKrylovVectorImpl&& o) noexcept {
    if (this != &o) {
      free();
      d_ptr_ = o.d_ptr_;
      size_ = o.size_;
      o.d_ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  void free() {
    if (d_ptr_) {
      cudaFree(d_ptr_);
      d_ptr_ = nullptr;
      size_ = 0;
    }
  }

  T* devPtr() { return d_ptr_; }
  const T* devPtr() const { return d_ptr_; }
  size_t size() const { return size_; }

  void setZero(cudaStream_t stream = 0) {
    cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(T), stream));
  }

  void copyFromHost(const T* h, cudaStream_t s = 0) {
    cudaErrChk(cudaMemcpyAsync(d_ptr_, h, size_ * sizeof(T),
                               cudaMemcpyHostToDevice, s));
  }

  void copyToHost(T* h, cudaStream_t s = 0) const {
    cudaErrChk(cudaMemcpyAsync(h, d_ptr_, size_ * sizeof(T),
                               cudaMemcpyDeviceToHost, s));
  }

private:
  T* d_ptr_;
  size_t size_;
};

// ---------------------------------------------------------------------------
//  Backward-compatible type aliases
//
//  All existing code that uses GPUFieldArray3, GPUFieldArray4, or
//  GPUKrylovVector without template arguments continues to compile
//  unchanged; it receives the cudaSolverType instantiation.
// ---------------------------------------------------------------------------
using GPUFieldArray3 = GPUFieldArray3Impl<cudaSolverType>;
using GPUFieldArray4 = GPUFieldArray4Impl<cudaSolverType>;
using GPUKrylovVector = GPUKrylovVectorImpl<cudaSolverType>;

#endif // GPU_FIELD_ARRAY_CUH
