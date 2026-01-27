#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <dpct/blas_utils.hpp>

#include <dpct/sparse_utils.hpp>

/* DPCT_ORIG #include <cublas_v2.h>*/
/* DPCT_ORIG #include <cusparse.h>*/

/* DPCT_ORIG #include <library_types.h>*/
#include "complext.h"
#include <dpct/lib_common_utils.hpp>

namespace Gadgetron{

  class cudaDeviceManager 
  {
  public:

    // This class is used as a singleton.
    // Use Instance() to access the public member functions.
    //

    static cudaDeviceManager* Instance();
    
    // Public member functions.
    // If the function does not take a device id, it will use the current device.
    //

    inline size_t total_global_mem(int device){ return _total_global_mem[device]; }
    inline size_t shared_mem_per_block(int device){ return _shared_mem_per_block[device]; }
    inline int warp_size(int device){ return _warp_size[device]; }
    inline int max_blockdim(int device){ return _max_blockdim[device]; }
    inline int max_griddim(int device){ return _max_griddim[device]; }
    inline int major_version(int device){ return _major[device]; }
    inline int minor_version(int device){ return _minor[device]; }

    size_t total_global_mem();
    size_t shared_mem_per_block();
    int major_version();
    int minor_version();
    int warp_size();
    int max_blockdim();
    int max_griddim();

    int getCurrentDevice();

    int getTotalNumberOfDevice();

    size_t getFreeMemory();
    size_t getFreeMemory(int device);

    size_t getTotalMemory();
    size_t getTotalMemory(int device);

    // Access to Cublas is protected by a mutex
    // Despite what the Cublas manual claims, we have not found it thread safe.

/* DPCT_ORIG     cublasHandle_t lockHandle();*/
    dpct::blas::descriptor_ptr lockHandle();
/* DPCT_ORIG     cublasHandle_t lockHandle(int device);*/
    dpct::blas::descriptor_ptr lockHandle(int device);

    void unlockHandle();
    void unlockHandle(int device);

/* DPCT_ORIG     cusparseHandle_t lockSparseHandle();*/
    dpct::sparse::descriptor_ptr lockSparseHandle();
/* DPCT_ORIG     cusparseHandle_t lockSparseHandle(int device);*/
    dpct::sparse::descriptor_ptr lockSparseHandle(int device);

    void unlockSparseHandle();
    void unlockSparseHandle(int device);


  private:

    // Use the Instance() method to access the singleton
    //

    cudaDeviceManager();
    ~cudaDeviceManager();

    static void CleanUp();
    
    int _num_devices;
    std::vector<size_t> _total_global_mem; // in bytes
    std::vector<size_t> _shared_mem_per_block; // in bytes
    std::vector<int> _warp_size;
    std::vector<int> _max_blockdim;
    std::vector<int> _max_griddim;
    std::vector<int> _major;
    std::vector<int> _minor;
/* DPCT_ORIG     std::vector<cublasHandle_t> _handle;*/
    std::vector<dpct::blas::descriptor_ptr> _handle;
/* DPCT_ORIG     std::vector<cusparseHandle_t> _sparse_handle;*/
    std::vector<dpct::sparse::descriptor_ptr> _sparse_handle;
    static cudaDeviceManager * _instance;
  };

  template<class T>
  struct cudaDataType {
  };

  template<>
  struct cudaDataType<float>{
/* DPCT_ORIG     static constexpr cudaDataType_t value = CUDA_R_32F;*/
    static constexpr dpct::library_data_t value = dpct::library_data_t::real_float;
  };

  template<>
  struct cudaDataType<double>{
/* DPCT_ORIG     static constexpr cudaDataType_t value = CUDA_R_64F;*/
    static constexpr dpct::library_data_t value = dpct::library_data_t::real_double;
  };

  template<>
  struct cudaDataType<complext<float>>{
/* DPCT_ORIG     static constexpr cudaDataType_t value = CUDA_C_32F;*/
    static constexpr dpct::library_data_t value = dpct::library_data_t::complex_float;
  };

  template<>
  struct cudaDataType<complext<double>>{
/* DPCT_ORIG     static constexpr cudaDataType_t value = CUDA_C_64F;*/
    static constexpr dpct::library_data_t value = dpct::library_data_t::complex_double;
  };

/* DPCT_ORIG   template<class T>
  constexpr cudaDataType_t cuda_datatype(){return cudaDataType<T>::value;}*/
  template <class T> constexpr dpct::library_data_t cuda_datatype() { return cudaDataType<T>::value; }

/* DPCT_ORIG 	std::string gadgetron_getCusparseErrorString(cusparseStatus_t err);*/
        std::string gadgetron_getCusparseErrorString(int err);
}



