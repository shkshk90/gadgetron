#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "hoCuNDArray_blas.h"
#include "cuNDArray_blas.h"
#include "complext.h"
#include "check_CUDA.h"
#include <dpct/blas_utils.hpp>

namespace Gadgetron{

#define CUBLAS_CALL(fun)                                                       \
    { int err = fun; if (err != 0) {                                           \
        throw cuda_error(gadgetron_getCublasErrorString(err));                 \
    } }

  // These are defined in cuNDArray_blas.cu
  //

  template <class T>
  int cublas_axpy(dpct::blas::descriptor_ptr hndl, int n, const T *a,
                  const T *x, int incx, T *y, int incy);
  template <class T>
  int cublas_dot(dpct::blas::descriptor_ptr, int, const T *, int, const T *,
                 int, T *, bool cc = true);
  template <class T>
  int cublas_nrm2(dpct::blas::descriptor_ptr, int, const T *, int,
                  typename realType<T>::Type *result);
  template <class T>
  int cublas_amax(dpct::blas::descriptor_ptr handle, int n, const T *x,
                  int incx, int *result);
  template <class T>
  int cublas_amin(dpct::blas::descriptor_ptr handle, int n, const T *x,
                  int incx, int *result);
  template <class T>
  int cublas_asum(dpct::blas::descriptor_ptr handle, int n, const T *x,
                  int incx, typename realType<T>::Type *result);

  template <class T> void axpy(T a, hoCuNDArray<T> *x, hoCuNDArray<T> *y) try {
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*2*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    T* y_ptr = y->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    cuNDArray<T> cuY(dims);

    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){

      size_t curSize = std::min(batchSize,remaining);

      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuY.get_data_ptr(),
                                             y_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      CUBLAS_CALL(cublas_axpy(cudaDeviceManager::Instance()->lockHandle(device), curSize,
			      &a, cuX.get_data_ptr(), 1, cuY.get_data_ptr(), 1));

      cudaDeviceManager::Instance()->unlockHandle(device);

      CUDA_CALL(DPCT_CHECK_ERROR(
          dpct::get_in_order_queue()
              .memcpy(y_ptr, cuY.get_data_ptr(), curSize * sizeof(T))
              .wait()));
      remaining -= batchSize;
    }
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template<class T> void axpy( T a, hoCuNDArray< complext<T> >*x, hoCuNDArray< complext<T> > *y )
  {
    axpy( complext<T>(a), x, y );
  }

  template <class T> T dot(hoCuNDArray<T> *x, hoCuNDArray<T> *y, bool cc) try {
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*2*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    T* y_ptr = y->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    cuNDArray<T> cuY(dims);
    T ret = T(0);

    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){
    
      size_t curSize = std::min(batchSize,remaining);

      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuY.get_data_ptr(),
                                             y_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      T cur_ret;
      CUBLAS_CALL(cublas_dot( cudaDeviceManager::Instance()->lockHandle(device), curSize,
			      cuX.get_data_ptr(), 1,
			      cuY.get_data_ptr(), 1,
			      &cur_ret, cc ));

      cudaDeviceManager::Instance()->unlockHandle(device);

      remaining -= batchSize;
      ret += cur_ret;
    }
    return ret;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> typename realType<T>::Type nrm2(hoCuNDArray<T> *x) try {
    typedef typename realType<T>::Type REAL;
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    REAL ret = 0;

    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){

      size_t curSize = std::min(batchSize,remaining);
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      REAL cur_ret;
      CUBLAS_CALL(cublas_nrm2<T>( cudaDeviceManager::Instance()->lockHandle(device), batchSize,
				  cuX.get_data_ptr(), 1, &cur_ret));

      cudaDeviceManager::Instance()->unlockHandle(device);

      remaining -= batchSize;
      ret += cur_ret*cur_ret;
    }
    return std::sqrt(ret);
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> typename realType<T>::Type asum(hoCuNDArray<T> *x) try {
    typedef typename realType<T>::Type REAL;
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    REAL ret = 0;

    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){

      size_t curSize = std::min(batchSize,remaining);
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      REAL cur_ret;
      CUBLAS_CALL(cublas_asum( cudaDeviceManager::Instance()->lockHandle(device), batchSize,
			       cuX.get_data_ptr(), 1,
			       &cur_ret));

      cudaDeviceManager::Instance()->unlockHandle(device);

      remaining -= batchSize;
      ret += cur_ret;
    }
    return ret;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> size_t amin(hoCuNDArray<T> *x) try {
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    std::vector<size_t> results;
 
    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){

      size_t curSize = std::min(batchSize,remaining);
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      int cur_ret;
      CUBLAS_CALL(cublas_amin( cudaDeviceManager::Instance()->lockHandle(device), batchSize,
			       cuX.get_data_ptr(), 1,
			       &cur_ret));

      cudaDeviceManager::Instance()->unlockHandle(device);

      remaining -= batchSize;
      results.push_back(cur_ret+i*batchSize-1);
    }

    size_t res =0;
    for (size_t i =0; i < results.size(); i++){
      if (abs(x_ptr[results[i]]) < abs(x_ptr[res])) res = results[i];
    }
    return res;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> size_t amax(hoCuNDArray<T> *x) try {
    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t free = cudaDeviceManager::Instance()->getFreeMemory(device);
    size_t batchSize = 1024*1024*(free/(sizeof(T)*1024*1024)); //Ensure 1Mb allocations
    size_t remaining = x->get_number_of_elements();
    batchSize = std::min(batchSize,remaining);
    T* x_ptr = x->get_data_ptr();
    std::vector<size_t> dims;
    dims.push_back(batchSize);
    cuNDArray<T> cuX(dims);
    std::vector<size_t> results;

    for (size_t i = 0; i < (x->get_number_of_elements()-1)/batchSize+1; i++){

      size_t curSize = std::min(batchSize,remaining);
      CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(cuX.get_data_ptr(),
                                             x_ptr + i * batchSize,
                                             curSize * sizeof(T))
                                     .wait()));

      int cur_ret;
      CUBLAS_CALL(cublas_amax( cudaDeviceManager::Instance()->lockHandle(device), batchSize,
			       cuX.get_data_ptr(), 1,
			       &cur_ret));

      cudaDeviceManager::Instance()->unlockHandle(device);

      remaining -= batchSize;
      results.push_back(cur_ret+i*batchSize-1);
    }

    size_t res =0;
    for (size_t i =0; i < results.size(); i++){
      if (abs(x_ptr[results[i]]) > abs(x_ptr[res])) res = results[i];
    }
    return res;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //
  // Instantiation
  //

  template float dot(hoCuNDArray<float>*,hoCuNDArray<float>*,bool);
  template float nrm2(hoCuNDArray<float>*);
  template void axpy(float,hoCuNDArray<float>*,hoCuNDArray<float>*);
  template size_t amin(hoCuNDArray<float>*);
  template size_t amax(hoCuNDArray<float>*);
  template float asum(hoCuNDArray<float>*);

  template double dot(hoCuNDArray<double>*,hoCuNDArray<double>*,bool);
  template double nrm2(hoCuNDArray<double>*);
  template void axpy(double,hoCuNDArray<double>*,hoCuNDArray<double>*);
  template size_t amin(hoCuNDArray<double>*);
  template size_t amax(hoCuNDArray<double>*);
  template double asum(hoCuNDArray<double>*);

  template float_complext dot(hoCuNDArray<float_complext>*,hoCuNDArray<float_complext>*,bool);
  template float nrm2(hoCuNDArray<float_complext>*);
  template void axpy(float_complext,hoCuNDArray<float_complext>*,hoCuNDArray<float_complext>*);
  template void axpy(float,hoCuNDArray<float_complext>*,hoCuNDArray<float_complext>*);
  template size_t amin(hoCuNDArray<float_complext>*);
  template size_t amax(hoCuNDArray<float_complext>*);
  template float asum(hoCuNDArray<float_complext>*);

  template double_complext dot(hoCuNDArray<double_complext>*,hoCuNDArray<double_complext>*,bool);
  template double nrm2(hoCuNDArray<double_complext>*);
  template void axpy(double_complext,hoCuNDArray<double_complext>*,hoCuNDArray<double_complext>*);
  template void axpy(double,hoCuNDArray<double_complext>*,hoCuNDArray<double_complext>*);
  template size_t amin(hoCuNDArray<double_complext>*);
  template size_t amax(hoCuNDArray<double_complext>*);
  template double asum(hoCuNDArray<double_complext>*);
}
