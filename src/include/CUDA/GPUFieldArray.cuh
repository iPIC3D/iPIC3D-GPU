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
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_FIELD_ARRAY_CUH
#define GPU_FIELD_ARRAY_CUH

#include "cudaTypeDef.cuh"
#include <cstddef>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  GPUFieldArray3  –  3-dimensional device array  (S3 × S2 × S1)
// ---------------------------------------------------------------------------
class GPUFieldArray3
{
public:
    // ---- construction / destruction ----

    GPUFieldArray3() : d_ptr_(nullptr), S3_(0), S2_(0), S1_(0), size_(0), owning_(true) {}

    GPUFieldArray3(size_t s3, size_t s2, size_t s1)
        : S3_(s3), S2_(s2), S1_(s1), size_(s3 * s2 * s1), owning_(true)
    {
        cudaErrChk(cudaMalloc(&d_ptr_, size_ * sizeof(double)));
        cudaErrChk(cudaMemset(d_ptr_, 0, size_ * sizeof(double)));
    }

    ~GPUFieldArray3() { free(); }

    // Not copyable (owns device memory)
    GPUFieldArray3(const GPUFieldArray3&) = delete;
    GPUFieldArray3& operator=(const GPUFieldArray3&) = delete;

    // Movable
    GPUFieldArray3(GPUFieldArray3&& o) noexcept
        : d_ptr_(o.d_ptr_), S3_(o.S3_), S2_(o.S2_), S1_(o.S1_), size_(o.size_), owning_(o.owning_)
    { o.d_ptr_ = nullptr; o.size_ = 0; }

    GPUFieldArray3& operator=(GPUFieldArray3&& o) noexcept
    {
        if (this != &o) { free(); d_ptr_ = o.d_ptr_; S3_ = o.S3_; S2_ = o.S2_; S1_ = o.S1_; size_ = o.size_; owning_ = o.owning_; o.d_ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }

    /** Create a non-owning view that wraps an existing device pointer.
     *  The view does NOT own the memory — free() and the destructor are no-ops.
     *  Use this to pass per-species slices of GPUFieldArray4 to functions that
     *  take GPUFieldArray3& (e.g. GPU halo exchange wrappers). */
    static GPUFieldArray3 wrapDevice(double* devPtr, size_t s3, size_t s2, size_t s1)
    {
        GPUFieldArray3 v;
        v.d_ptr_  = devPtr;
        v.S3_     = s3;
        v.S2_     = s2;
        v.S1_     = s1;
        v.size_   = s3 * s2 * s1;
        v.owning_ = false;
        return v;
    }

    void free()
    {
        if (d_ptr_ && owning_) { cudaFree(d_ptr_); }
        d_ptr_ = nullptr; size_ = 0;
    }

    // ---- accessors ----

    /** Raw device pointer (row-major, contiguous). */
    double*       devPtr()       { return d_ptr_; }
    const double* devPtr() const { return d_ptr_; }

    size_t dim1() const { return S3_; }   // outermost
    size_t dim2() const { return S2_; }
    size_t dim3() const { return S1_; }   // innermost (contiguous)
    size_t size() const { return size_; } // total elements

    // ---- device ↔ host transfers (synchronous) ----

    /** Copy entire array from host (arr3_double flat buffer) to device. */
    void copyFromHost(const double* h_ptr)
    {
        cudaErrChk(cudaMemcpy(d_ptr_, h_ptr, size_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    /** Copy entire array from device to host (arr3_double flat buffer). */
    void copyToHost(double* h_ptr) const
    {
        cudaErrChk(cudaMemcpy(h_ptr, d_ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ---- async variants ----

    void copyFromHostAsync(const double* h_ptr, cudaStream_t stream)
    {
        cudaErrChk(cudaMemcpyAsync(d_ptr_, h_ptr, size_ * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(double* h_ptr, cudaStream_t stream) const
    {
        cudaErrChk(cudaMemcpyAsync(h_ptr, d_ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }

    // ---- device-side utilities ----

    void setAll(double val, cudaStream_t stream = 0)
    {
        if (val == 0.0)
            cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(double), stream));
        else
        {
            // For non-zero, we do a small kernel launch (defined below).
            fillKernelLaunch(d_ptr_, val, size_, stream);
        }
    }

    // ---- 3D → device pointer with offset (for MPI halo exchange) ----
    // Returns a pointer offset to element [i3][i2][i1] so that
    // MPI derived datatypes (which encode strides from the base pointer)
    // work identically to the host arrays.
    double* ptrAt(size_t i3, size_t i2, size_t i1)
    {
        return d_ptr_ + (i3 * S2_ + i2) * S1_ + i1;
    }

    // helper: fill device array with a constant value (public so GPUFieldArray4 can reuse)
    static void fillKernelLaunch(double* d, double val, size_t n, cudaStream_t stream);

private:
    double* d_ptr_;
    size_t  S3_, S2_, S1_, size_;
    bool    owning_;   ///< true = this object owns d_ptr_ (will cudaFree)
};

// ---------------------------------------------------------------------------
//  GPUFieldArray4  –  4-dimensional device array  (S4 × S3 × S2 × S1)
// ---------------------------------------------------------------------------
class GPUFieldArray4
{
public:
    GPUFieldArray4() : d_ptr_(nullptr), S4_(0), S3_(0), S2_(0), S1_(0), size_(0) {}

    GPUFieldArray4(size_t s4, size_t s3, size_t s2, size_t s1)
        : S4_(s4), S3_(s3), S2_(s2), S1_(s1), size_(s4 * s3 * s2 * s1)
    {
        cudaErrChk(cudaMalloc(&d_ptr_, size_ * sizeof(double)));
        cudaErrChk(cudaMemset(d_ptr_, 0, size_ * sizeof(double)));
    }

    ~GPUFieldArray4() { free(); }

    GPUFieldArray4(const GPUFieldArray4&) = delete;
    GPUFieldArray4& operator=(const GPUFieldArray4&) = delete;

    GPUFieldArray4(GPUFieldArray4&& o) noexcept
        : d_ptr_(o.d_ptr_), S4_(o.S4_), S3_(o.S3_), S2_(o.S2_), S1_(o.S1_), size_(o.size_)
    { o.d_ptr_ = nullptr; o.size_ = 0; }

    GPUFieldArray4& operator=(GPUFieldArray4&& o) noexcept
    {
        if (this != &o) { free(); d_ptr_ = o.d_ptr_; S4_ = o.S4_; S3_ = o.S3_; S2_ = o.S2_; S1_ = o.S1_; size_ = o.size_; o.d_ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }

    void free()
    {
        if (d_ptr_) { cudaFree(d_ptr_); d_ptr_ = nullptr; size_ = 0; }
    }

    double*       devPtr()       { return d_ptr_; }
    const double* devPtr() const { return d_ptr_; }

    size_t dim1() const { return S4_; }
    size_t dim2() const { return S3_; }
    size_t dim3() const { return S2_; }
    size_t dim4() const { return S1_; }
    size_t size() const { return size_; }

    /** Pointer to the start of species-slice [is], i.e. element [is][0][0][0]. */
    double* speciesPtr(size_t is)
    {
        return d_ptr_ + is * S3_ * S2_ * S1_;
    }

    void copyFromHost(const double* h_ptr)
    {
        cudaErrChk(cudaMemcpy(d_ptr_, h_ptr, size_ * sizeof(double), cudaMemcpyHostToDevice));
    }

    void copyToHost(double* h_ptr) const
    {
        cudaErrChk(cudaMemcpy(h_ptr, d_ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void copyFromHostAsync(const double* h_ptr, cudaStream_t stream)
    {
        cudaErrChk(cudaMemcpyAsync(d_ptr_, h_ptr, size_ * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    void copyToHostAsync(double* h_ptr, cudaStream_t stream) const
    {
        cudaErrChk(cudaMemcpyAsync(h_ptr, d_ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }

    void setAll(double val, cudaStream_t stream = 0)
    {
        if (val == 0.0)
            cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(double), stream));
        else
            GPUFieldArray3::fillKernelLaunch(d_ptr_, val, size_, stream); // reuse
    }

    // For MPI: pointer at [is][i3][i2][i1]
    double* ptrAt(size_t is, size_t i3, size_t i2, size_t i1)
    {
        return d_ptr_ + ((is * S3_ + i3) * S2_ + i2) * S1_ + i1;
    }

private:
    double* d_ptr_;
    size_t  S4_, S3_, S2_, S1_, size_;
};

// ---------------------------------------------------------------------------
//  GPUKrylovVector – 1-D device array for Krylov solver vectors
// ---------------------------------------------------------------------------
class GPUKrylovVector
{
public:
    GPUKrylovVector() : d_ptr_(nullptr), size_(0) {}

    explicit GPUKrylovVector(size_t n) : size_(n)
    {
        cudaErrChk(cudaMalloc(&d_ptr_, n * sizeof(double)));
        cudaErrChk(cudaMemset(d_ptr_, 0, n * sizeof(double)));
    }

    ~GPUKrylovVector() { free(); }

    GPUKrylovVector(const GPUKrylovVector&) = delete;
    GPUKrylovVector& operator=(const GPUKrylovVector&) = delete;

    GPUKrylovVector(GPUKrylovVector&& o) noexcept : d_ptr_(o.d_ptr_), size_(o.size_)
    { o.d_ptr_ = nullptr; o.size_ = 0; }

    GPUKrylovVector& operator=(GPUKrylovVector&& o) noexcept
    { if (this != &o) { free(); d_ptr_ = o.d_ptr_; size_ = o.size_; o.d_ptr_ = nullptr; o.size_ = 0; } return *this; }

    void free() { if (d_ptr_) { cudaFree(d_ptr_); d_ptr_ = nullptr; size_ = 0; } }

    double* devPtr() { return d_ptr_; }
    const double* devPtr() const { return d_ptr_; }
    size_t size() const { return size_; }

    void setZero(cudaStream_t stream = 0)
    {
        cudaErrChk(cudaMemsetAsync(d_ptr_, 0, size_ * sizeof(double), stream));
    }

    void copyFromHost(const double* h, cudaStream_t s = 0)
    { cudaErrChk(cudaMemcpyAsync(d_ptr_, h, size_ * sizeof(double), cudaMemcpyHostToDevice, s)); }

    void copyToHost(double* h, cudaStream_t s = 0) const
    { cudaErrChk(cudaMemcpyAsync(h, d_ptr_, size_ * sizeof(double), cudaMemcpyDeviceToHost, s)); }

private:
    double* d_ptr_;
    size_t  size_;
};

#endif // GPU_FIELD_ARRAY_CUH
