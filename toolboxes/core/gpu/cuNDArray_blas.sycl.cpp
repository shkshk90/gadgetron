#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_blas.h"
#include "complext.h"
#include "GadgetronCuException.h"
#include "cudaDeviceManager.h"
#include <dpct/blas_utils.hpp>

#include <cmath>

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace Gadgetron{

#define CUBLAS_CALL(fun) { int err = fun; if (err != 0) { throw cuda_error(gadgetron_getCublasErrorString(err)); } }

  // Persistent per-BLAS-handle scratch buffers for scalar reduction outputs
  // (dot/nrm2/asum/amin/amax). SYCLomatic's dpct::blas::wrapper_*_out helpers
  // malloc a fresh device buffer and free it (via an async host_task) on
  // *every* call whenever the result pointer is host memory (always true
  // here - callers pass a stack variable) or whenever source/target types
  // differ (always true for amin/amax's int->int64 conversion). That
  // malloc/free pair, not the actual reduction, dominates these calls'
  // runtime (measured 3.6-5.1x slower than CUDA on identical workloads).
  // Routing results through a buffer that's allocated once and reused
  // avoids it. Handles are already serialized per-device by
  // cudaDeviceManager::lockHandle's mutex, so this cache's own mutex only
  // ever guards a first-touch insert, never contends with the actual BLAS
  // call.
  namespace {
    struct BlasScratch {
      float* f = nullptr;
      double* d = nullptr;
      sycl::float2* fc = nullptr;
      sycl::double2* dc = nullptr;
      std::int64_t* i64 = nullptr;
    };

    std::mutex g_blas_scratch_mutex;
    std::unordered_map<dpct::blas::descriptor_ptr, BlasScratch> g_blas_scratch_cache;

    template <class T> T*& scratch_slot(BlasScratch& s);
    template <> float*& scratch_slot<float>(BlasScratch& s) { return s.f; }
    template <> double*& scratch_slot<double>(BlasScratch& s) { return s.d; }
    template <> sycl::float2*& scratch_slot<sycl::float2>(BlasScratch& s) { return s.fc; }
    template <> sycl::double2*& scratch_slot<sycl::double2>(BlasScratch& s) { return s.dc; }
    template <> std::int64_t*& scratch_slot<std::int64_t>(BlasScratch& s) { return s.i64; }

    template <class T> T* get_blas_scratch(dpct::blas::descriptor_ptr hndl) {
      std::lock_guard<std::mutex> lock(g_blas_scratch_mutex);
      auto& entry = g_blas_scratch_cache[hndl];
      auto& ptr = scratch_slot<T>(entry);
      if (!ptr)
        ptr = sycl::malloc_device<T>(1, hndl->get_queue());
      return ptr;
    }

    // Growable per-BLAS-handle scratch arrays for the two-level reductions
    // below (amin/amax, and nrm2/asum/dot). Any of these called with a
    // small explicit batchSize blocks on a device->host sync every loop
    // iteration - fine for CUDA's cheap per-call sync, but under
    // SYCL/Level-Zero that dominates runtime once num_splits gets into the
    // tens of thousands. The fix in each is the same shape: submit all
    // per-chunk reductions without waiting, combine their partial results
    // on-device (either directly, or via one small gather/reduce step),
    // then read back once - turning O(num_splits) host syncs into O(1).
    // Each role below is a dedicated slot for one function's intermediate
    // state, so different functions' scratch never aliases.
    enum SplitScratchRole {
      kAminAmaxLocalIdx = 0,
      kAminAmaxChunkVal = 1,
      kAminAmaxChunkAbsIdx = 2,
      kNrm2Partial = 3,
      kAsumPartial = 4,
      kDotPartial = 5,
      kDotFinalSum = 6,
      kNumSplitScratchRoles = 7
    };

    template <class T> T* get_split_scratch(dpct::blas::descriptor_ptr hndl, int role, size_t count) {
      struct Entry { T* ptr = nullptr; size_t capacity = 0; };
      static std::mutex mtx;
      static std::unordered_map<dpct::blas::descriptor_ptr, std::array<Entry, kNumSplitScratchRoles>> cache;
      std::lock_guard<std::mutex> lock(mtx);
      auto& entry = cache[hndl][role];
      if (entry.capacity < count) {
        if (entry.ptr)
          sycl::free(entry.ptr, hndl->get_queue());
        entry.ptr = sycl::malloc_device<T>(count, hndl->get_queue());
        entry.capacity = count;
      }
      return entry.ptr;
    }

    // Argmin/argmax of a device array, result left on the device (1-based
    // index, matching oneMKL's index_base::one) with no readback - used
    // both for the per-chunk reduction and the final cross-chunk reduction
    // in amin<T>/amax<T> below, so submissions can be pipelined without a
    // host sync between them.
    template <class T> int device_iamin(dpct::blas::descriptor_ptr hndl, int n, const T* x, int incx, std::int64_t* result);
    template <class T> int device_iamax(dpct::blas::descriptor_ptr hndl, int n, const T* x, int incx, std::int64_t* result);

    template <> int device_iamin<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamin<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamin<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (const std::complex<float>*)x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamin<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (const std::complex<double>*)x, incx, result, oneapi::mkl::index_base::one));
    }

    template <> int device_iamax<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamax<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamax<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (const std::complex<float>*)x, incx, result, oneapi::mkl::index_base::one));
    }
    template <> int device_iamax<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx, std::int64_t* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (const std::complex<double>*)x, incx, result, oneapi::mkl::index_base::one));
    }

    // Same no-readback idea as device_iamin/device_iamax, for nrm2/asum
    // (result left on device, no wrapper, no copy) - used for the
    // per-chunk pass in nrm2<T>/asum<T> below.
    template <class T> int device_nrm2(dpct::blas::descriptor_ptr hndl, int n, const T* x, int incx, typename realType<T>::Type* result);
    template <class T> int device_asum(dpct::blas::descriptor_ptr hndl, int n, const T* x, int incx, typename realType<T>::Type* result);

    template <> int device_nrm2<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, float* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, incx, result));
    }
    template <> int device_nrm2<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, double* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, incx, result));
    }
    template <> int device_nrm2<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx, float* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (const std::complex<float>*)x, incx, result));
    }
    template <> int device_nrm2<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx, double* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (const std::complex<double>*)x, incx, result));
    }

    template <> int device_asum<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, float* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, result));
    }
    template <> int device_asum<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, double* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, result));
    }
    template <> int device_asum<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx, float* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (const std::complex<float>*)x, incx, result));
    }
    template <> int device_asum<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx, double* result) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (const std::complex<double>*)x, incx, result));
    }

    // Same idea for dot, which additionally carries the conjugate/cc flag
    // for complex types.
    template <class T> int device_dot(dpct::blas::descriptor_ptr hndl, int n, const T* x, int incx, const T* y, int incy, T* result, bool cc);

    template <> int device_dot<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, const float* y, int incy, float* result, bool) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, result));
    }
    template <> int device_dot<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, const double* y, int incy, double* result, bool) {
      return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, result));
    }
    template <> int device_dot<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx, const float_complext* y, int incy, float_complext* result, bool cc) {
      if (cc)
        return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (const std::complex<float>*)x, incx, (const std::complex<float>*)y, incy, (std::complex<float>*)result));
      else
        return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (const std::complex<float>*)x, incx, (const std::complex<float>*)y, incy, (std::complex<float>*)result));
    }
    template <> int device_dot<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx, const double_complext* y, int incy, double_complext* result, bool cc) {
      if (cc)
        return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (const std::complex<double>*)x, incx, (const std::complex<double>*)y, incy, (std::complex<double>*)result));
      else
        return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (const std::complex<double>*)x, incx, (const std::complex<double>*)y, incy, (std::complex<double>*)result));
    }
  }

  //NRM2
  //

  template <class T>
  int cublas_axpy(dpct::blas::descriptor_ptr hndl, int n, const T* a, const T* x, int incx, T* y, int incy);
  template <class T> int cublas_dot(dpct::blas::descriptor_ptr, int, const T*, int, const T*, int, T*, bool cc = true);
  template <class T> int cublas_nrm2(dpct::blas::descriptor_ptr, int, const T*, int, typename realType<T>::Type* result);
  template <class T> int cublas_amax(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, int* result);
  template <class T> int cublas_amin(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, int* result);
  template <class T>
  int cublas_asum(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, typename realType<T>::Type* result);

  template <> int cublas_nrm2<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int inc, float* res) try {
    float* scratch = get_blas_scratch<float>(hndl);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, inc, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(float)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <> int cublas_nrm2<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int inc, double* res) try {
    double* scratch = get_blas_scratch<double>(hndl);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, inc, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(double)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_nrm2<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int inc,
                                  float* res) try {
    float* scratch = get_blas_scratch<float>(hndl);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (std::complex<float>*)x, inc, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(float)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_nrm2<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int inc,
                                   double* res) try {
    double* scratch = get_blas_scratch<double>(hndl);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (std::complex<double>*)x, inc, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(double)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //DOT
  //

  template <>
  int cublas_dot<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, const float* y, int incy,
                        float* res, bool cc) try {
    float* scratch = get_blas_scratch<float>(hndl);
    oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(float)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_dot<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, const double* y, int incy,
                         double* res, bool cc) try {
    double* scratch = get_blas_scratch<double>(hndl);
    oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(double)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_dot<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                 const float_complext* y, int incy, float_complext* res, bool cc) try {
    sycl::float2* scratch = get_blas_scratch<sycl::float2>(hndl);
    if(cc)
      oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                            (std::complex<float>*)y, incy,
                                            (std::complex<float>*)scratch);
    else
      oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                            (std::complex<float>*)y, incy,
                                            (std::complex<float>*)scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(sycl::float2)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_dot<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                  const double_complext* y, int incy, double_complext* res, bool cc) try {
    sycl::double2* scratch = get_blas_scratch<sycl::double2>(hndl);
    if(cc)
      oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                            (std::complex<double>*)y, incy,
                                            (std::complex<double>*)scratch);
    else
      oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                            (std::complex<double>*)y, incy,
                                            (std::complex<double>*)scratch);
    hndl->get_queue().memcpy(res, scratch, sizeof(sycl::double2)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  // AXPY
  //

  template <>
  int cublas_axpy<float>(dpct::blas::descriptor_ptr hndl, int n, const float* a, const float* x, int incx, float* y,
                         int incy) try {
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value(a, hndl->get_queue()), x, incx, y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_axpy<double>(dpct::blas::descriptor_ptr hndl, int n, const double* a, const double* x, int incx, double* y,
                          int incy) try {
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value(a, hndl->get_queue()), x, incx, y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_axpy<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* a,
                                  const float_complext* x, int incx, float_complext* y, int incy) try {
    auto alpha = *reinterpret_cast<const std::complex<float>*>(a);
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, alpha, (std::complex<float>*)x, incx,
        (std::complex<float>*)y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_axpy<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* a,
                                   const double_complext* x, int incx, double_complext* y, int incy) try {
    // Workaround: oneMath cublas backend passes std::complex<double> (alignof=8)
    // to cublasZaxpy_v2 which expects cuDoubleComplex (alignof=16), causing
    // SIGSEGV on aligned SSE/AVX loads. Use a simple SYCL kernel instead.
    double_complext alpha = *a;
    hndl->get_queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i * incy] = alpha * x[i * incx] + y[i * incy];
    }).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //SUM
  //

  template <>
  int cublas_asum<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, float* result) try {
    float* scratch = get_blas_scratch<float>(hndl);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, scratch);
    hndl->get_queue().memcpy(result, scratch, sizeof(float)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_asum<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, double* result) try {
    double* scratch = get_blas_scratch<double>(hndl);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, scratch);
    hndl->get_queue().memcpy(result, scratch, sizeof(double)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_asum<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  float* result) try {
    float* scratch = get_blas_scratch<float>(hndl);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (std::complex<float>*)x, incx, scratch);
    hndl->get_queue().memcpy(result, scratch, sizeof(float)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_asum<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   double* result) try {
    double* scratch = get_blas_scratch<double>(hndl);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (std::complex<double>*)x, incx, scratch);
    hndl->get_queue().memcpy(result, scratch, sizeof(double)).wait();
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //AMIN
  //

  template <> int cublas_amin<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amin<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amin<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                           scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amin<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                           scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //AMAX
  //

  template <> int cublas_amax<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amax<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amax<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                           scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <>
  int cublas_amax<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   int* result) try {
    std::int64_t* scratch = get_blas_scratch<std::int64_t>(hndl);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                           scratch, oneapi::mkl::index_base::one);
    std::int64_t host_result;
    hndl->get_queue().memcpy(&host_result, scratch, sizeof(std::int64_t)).wait();
    *result = static_cast<int>(host_result);
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> typename realType<T>::Type nrm2(cuNDArray<T>* arr, size_t batchSize) try {
    if( arr == 0x0 )
        throw std::runtime_error("Gadgetron::nrm2(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    typedef typename realType<T>::Type REAL;
    REAL ret = 0;
    size_t n_elements = arr->get_number_of_elements();

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = n_elements / batchSize + 1;
    int remainder = n_elements - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    if (num_splits == 1) {
        CUBLAS_CALL(cublas_nrm2<T>(handle, (int)n_elements, arr->get_data_ptr(), 1, &ret));
    } else {
        // Two-level reduction (see get_split_scratch's comment): compute
        // each chunk's norm without waiting between chunks, then combine
        // via nrm2 itself - sqrt(sum of squares) of the per-chunk norms is
        // exactly the overall norm, so the combine step reuses cublas_nrm2
        // as-is. One host sync total instead of one per chunk.
        REAL* partial = get_split_scratch<REAL>(handle, kNrm2Partial, num_splits);
        for (int ii = 0; ii < num_splits; ii++) {
            int chunk_n = (ii == num_splits - 1) ? remainder : batchSize;
            CUBLAS_CALL(device_nrm2<T>(handle, chunk_n, arr->get_data_ptr() + (size_t)batchSize * ii, 1, partial + ii));
        }
        CUBLAS_CALL(cublas_nrm2<REAL>(handle, num_splits, partial, 1, &ret));
    }

    cudaDeviceManager::Instance()->unlockHandle(device);

    return ret;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> T dot(cuNDArray<T>* arr1, cuNDArray<T>* arr2, size_t batchSize, bool cc) try {
    if (arr1 == 0x0 || arr2 == 0x0)
        throw std::runtime_error("Gadgetron::dot(): Invalid input array");

    if (arr1->get_number_of_elements() != arr2->get_number_of_elements())
        throw std::runtime_error("Gadgetron::dot(): Array sizes mismatch");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    T ret = 0;
    size_t n_elements = arr1->get_number_of_elements();

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = n_elements / batchSize + 1;
    int remainder = n_elements - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);
    auto& queue = handle->get_queue();

    if (num_splits == 1) {
        CUBLAS_CALL(cublas_dot(handle, (int)n_elements, arr1->get_data_ptr(), 1, arr2->get_data_ptr(), 1, &ret, cc));
    } else {
        // Two-level reduction (see get_split_scratch's comment): compute
        // each chunk's dot product without waiting between chunks, then
        // sum the partial results with one tiny single-item kernel (num_
        // splits is at most in the tens of thousands - a sequential sum
        // is still far cheaper than one host sync per chunk) and read
        // back once. Unlike nrm2/asum, the combine here is a plain sum
        // (can be negative or complex), so it can't reuse an existing
        // reduction primitive the way nrm2/asum's combine does.
        T* partial = get_split_scratch<T>(handle, kDotPartial, num_splits);
        for (int ii = 0; ii < num_splits; ii++) {
            int chunk_n = (ii == num_splits - 1) ? remainder : batchSize;
            CUBLAS_CALL(device_dot<T>(handle, chunk_n, arr1->get_data_ptr() + (size_t)batchSize * ii, 1,
                                       arr2->get_data_ptr() + (size_t)batchSize * ii, 1, partial + ii, cc));
        }

        T* sum_scratch = get_split_scratch<T>(handle, kDotFinalSum, 1);
        queue.single_task([=]() {
            T sum = T(0);
            for (int ii = 0; ii < num_splits; ii++)
                sum = sum + partial[ii];
            sum_scratch[0] = sum;
        });
        queue.memcpy(&ret, sum_scratch, sizeof(T)).wait();
    }

    cudaDeviceManager::Instance()->unlockHandle(device);

    return ret;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> void axpy(T a, cuNDArray<T>* x, cuNDArray<T>* y, size_t batchSize) try {
    if (x == 0x0 || y == 0x0)
        throw std::runtime_error("Gadgetron::axpy(): Invalid input array");

    if (x->get_number_of_elements() != y->get_number_of_elements())
        throw std::runtime_error("Gadgetron::axpy(): Array sizes mismatch");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t n_elements = x->get_number_of_elements();

    // Unlike the reduction functions above (dot/nrm2/asum/amin/amax), axpy
    // is a pure element-wise op (y[i] = a*x[i] + y[i]): each output
    // element depends on nothing but itself, so - unlike a reduction -
    // splitting the work has no numerical effect at all. batchSize-driven
    // chunking here was only ever a copy of the reductions' overflow
    // guard, not something axpy itself needs; it just forced thousands of
    // pointless dispatches whenever a caller (e.g. a test exercising the
    // OTHER functions' chunking) passed a small batchSize. The only real
    // constraint is cublas_axpy's own `int n` parameter, so chunk by that
    // instead - for every realistic array (well under INT_MAX elements)
    // this collapses to a single dispatch, regardless of batchSize.
    (void)batchSize; // intentionally unused - see above
    constexpr size_t max_chunk = (size_t)std::numeric_limits<int>::max();
    int num_splits = n_elements / max_chunk + 1;
    int remainder = n_elements - max_chunk * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    // Ensure alpha is 16-byte aligned for cuBLAS (cuDoubleComplex requires alignof=16,
    // but complext<double> only has alignof=8, causing segfault in SSE/AVX loads)
    alignas(16) T a_aligned = a;

    for (int ii = 0; ii < num_splits; ii++) {

        CUBLAS_CALL(cublas_axpy(handle,
                                (ii == num_splits - 1) ? remainder : (int)max_chunk, &a_aligned, x->get_data_ptr() + max_chunk * ii, 1,
                                y->get_data_ptr() + max_chunk * ii, 1));

    }
    cudaDeviceManager::Instance()->unlockHandle(device);
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> void axpy(T a, cuNDArray<complext<T>>* x, cuNDArray<complext<T>>* y, size_t batchSize ) { axpy(complext<T>(a), x, y, batchSize); }

  template <class T> typename realType<T>::Type asum(cuNDArray<T>* x, size_t batchSize) try {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::asum(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    typedef typename realType<T>::Type REAL;
    REAL result = 0;
    size_t n_elements = x->get_number_of_elements();

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = n_elements / batchSize + 1;
    int remainder = n_elements - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    if (num_splits == 1) {
        CUBLAS_CALL(cublas_asum(handle, (int)n_elements, x->get_data_ptr(), 1, &result));
    } else {
        // Two-level reduction (see get_split_scratch's comment): compute
        // each chunk's absolute-sum without waiting between chunks, then
        // combine via asum itself - the per-chunk sums are already
        // non-negative, so summing them equals summing their absolute
        // values, letting the combine step reuse cublas_asum as-is. One
        // host sync total instead of one per chunk.
        REAL* partial = get_split_scratch<REAL>(handle, kAsumPartial, num_splits);
        for (int ii = 0; ii < num_splits; ii++) {
            int chunk_n = (ii == num_splits - 1) ? remainder : batchSize;
            CUBLAS_CALL(device_asum<T>(handle, chunk_n, x->get_data_ptr() + (size_t)batchSize * ii, 1, partial + ii));
        }
        CUBLAS_CALL(cublas_asum<REAL>(handle, num_splits, partial, 1, &result));
    }
    cudaDeviceManager::Instance()->unlockHandle(device);

    return result;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> size_t amin(cuNDArray<T>* x, size_t batchSize )
  {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::amin(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t n_elements = x->get_number_of_elements();
    size_t result = n_elements + 1;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = n_elements / batchSize + 1;
    int remainder = n_elements - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);
    auto& queue = handle->get_queue();

    if (num_splits == 1) {
        // Common case (default batchSize): a single reduction, no need for
        // the two-level gather machinery below.
        int interim_result;
        CUBLAS_CALL(cublas_amin(handle, (int)n_elements, x->get_data_ptr(), 1, &interim_result));
        result = (size_t)interim_result - 1;
    } else {
        // Two-level reduction: find each chunk's local winner without
        // waiting between chunks, gather their values on-device in one
        // kernel, then reduce across chunks - one host sync total instead
        // of one per chunk. See get_split_scratch's comment for why.
        std::int64_t* local_idx = get_split_scratch<std::int64_t>(handle, 0, num_splits);
        T* chunk_val = get_split_scratch<T>(handle, 1, num_splits);
        std::int64_t* chunk_absidx = get_split_scratch<std::int64_t>(handle, 2, num_splits);

        for (int ii = 0; ii < num_splits; ii++) {
            int chunk_n = (ii == num_splits - 1) ? remainder : batchSize;
            CUBLAS_CALL(device_iamin<T>(handle, chunk_n, x->get_data_ptr() + (size_t)batchSize * ii, 1, local_idx + ii));
        }

        auto data_ptr = x->get_data_ptr();
        size_t bs = batchSize;
        queue.parallel_for(sycl::range<1>(num_splits), [=](sycl::id<1> i) {
            size_t idx = i[0];
            std::int64_t local = local_idx[idx];
            std::int64_t abs_idx = (std::int64_t)(bs * idx) + local - 1;
            chunk_val[idx] = data_ptr[abs_idx];
            chunk_absidx[idx] = abs_idx;
        });

        std::int64_t* winner_chunk = get_blas_scratch<std::int64_t>(handle);
        CUBLAS_CALL(device_iamin<T>(handle, num_splits, chunk_val, 1, winner_chunk));

        std::int64_t host_winner_chunk;
        queue.memcpy(&host_winner_chunk, winner_chunk, sizeof(std::int64_t)).wait();
        std::int64_t host_abs_idx;
        queue.memcpy(&host_abs_idx, chunk_absidx + (host_winner_chunk - 1), sizeof(std::int64_t)).wait();
        result = (size_t)host_abs_idx;
    }

    cudaDeviceManager::Instance()->unlockHandle(device);
    if (result > n_elements) {
        throw std::runtime_error("Gadgetron::amin(): computed index is out of bounds");
    }

    return result; // result - 1;
  }

  template <class T> size_t amax(cuNDArray<T>* x, size_t batchSize )
  {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::amax(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t n_elements = x->get_number_of_elements();
    size_t result = n_elements + 1;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = n_elements / batchSize + 1;
    int remainder = n_elements - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);
    auto& queue = handle->get_queue();

    if (num_splits == 1) {
        // Common case (default batchSize): a single reduction, no need for
        // the two-level gather machinery below.
        int interim_result;
        CUBLAS_CALL(cublas_amax(handle, (int)n_elements, x->get_data_ptr(), 1, &interim_result));
        result = (size_t)interim_result - 1;
    } else {
        // Two-level reduction: see amin<T> above for the full rationale -
        // same approach, mirrored with device_iamax.
        std::int64_t* local_idx = get_split_scratch<std::int64_t>(handle, 0, num_splits);
        T* chunk_val = get_split_scratch<T>(handle, 1, num_splits);
        std::int64_t* chunk_absidx = get_split_scratch<std::int64_t>(handle, 2, num_splits);

        for (int ii = 0; ii < num_splits; ii++) {
            int chunk_n = (ii == num_splits - 1) ? remainder : batchSize;
            CUBLAS_CALL(device_iamax<T>(handle, chunk_n, x->get_data_ptr() + (size_t)batchSize * ii, 1, local_idx + ii));
        }

        auto data_ptr = x->get_data_ptr();
        size_t bs = batchSize;
        queue.parallel_for(sycl::range<1>(num_splits), [=](sycl::id<1> i) {
            size_t idx = i[0];
            std::int64_t local = local_idx[idx];
            std::int64_t abs_idx = (std::int64_t)(bs * idx) + local - 1;
            chunk_val[idx] = data_ptr[abs_idx];
            chunk_absidx[idx] = abs_idx;
        });

        std::int64_t* winner_chunk = get_blas_scratch<std::int64_t>(handle);
        CUBLAS_CALL(device_iamax<T>(handle, num_splits, chunk_val, 1, winner_chunk));

        std::int64_t host_winner_chunk;
        queue.memcpy(&host_winner_chunk, winner_chunk, sizeof(std::int64_t)).wait();
        std::int64_t host_abs_idx;
        queue.memcpy(&host_abs_idx, chunk_absidx + (host_winner_chunk - 1), sizeof(std::int64_t)).wait();
        result = (size_t)host_abs_idx;
    }

    cudaDeviceManager::Instance()->unlockHandle(device);
    if (result > n_elements) {
        throw std::runtime_error("Gadgetron::amax(): computed index is out of bounds");
    }

    return result; //(size_t)result - 1;
  }

  std::string gadgetron_getCublasErrorString(int err)
  {
    switch (err) {
    case 1:
        return "NOT INITIALIZED";
    case 3:
        return "ALLOC FAILED";
    case 7:
        return "INVALID VALUE";
    case 8:
        return "ARCH MISMATCH";
    case 11:
        return "MAPPING ERROR";
    case 13:
        return "EXECUTION FAILED";
    case 14:
        return "INTERNAL ERROR";
    case 0:
        return "SUCCES";
    default:
        return "UNKNOWN CUBLAS ERROR";
    }
  }

  //
  // Instantiation
  //

  template float dot(cuNDArray<float>*, cuNDArray<float>*, size_t, bool);
  template float nrm2(cuNDArray<float>*, size_t);
  template void axpy(float, cuNDArray<float>*, cuNDArray<float>*, size_t);
  template size_t amin(cuNDArray<float>*, size_t);
  template size_t amax(cuNDArray<float>*, size_t);
  template float asum(cuNDArray<float>*, size_t);

  template double dot(cuNDArray<double>*, cuNDArray<double>*, size_t, bool);
  template double nrm2(cuNDArray<double>*, size_t);
  template void axpy(double, cuNDArray<double>*, cuNDArray<double>*, size_t);
  template size_t amin(cuNDArray<double>*, size_t);
  template size_t amax(cuNDArray<double>*, size_t);
  template double asum(cuNDArray<double>*, size_t);

  template float_complext dot(cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t, bool);
  template float nrm2(cuNDArray<float_complext>*, size_t);
  template void axpy(float_complext, cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t);
  template void axpy(float, cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t);
  template size_t amin(cuNDArray<float_complext>*, size_t);
  template size_t amax(cuNDArray<float_complext>*, size_t);
  template float asum(cuNDArray<float_complext>*, size_t);

  template double_complext dot(cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t, bool);
  template double nrm2(cuNDArray<double_complext>*, size_t);
  template void axpy(double_complext, cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t);
  template void axpy(double, cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t);
  template size_t amin(cuNDArray<double_complext>*, size_t);
  template size_t amax(cuNDArray<double_complext>*, size_t);
  template double asum(cuNDArray<double_complext>*, size_t);
} // namespace Gadgetron
