#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudaDeviceManager.h"
#include "check_CUDA.h"
#include "cuNDArray_blas.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_array.hpp>
/* DPCT_ORIG #include <cuda_runtime_api.h>*/
#include <stdlib.h>
#include <sstream>
#include <dpct/blas_utils.hpp>

/* DPCT_ORIG  std::string Gadgetron::gadgetron_getCusparseErrorString(cusparseStatus_t err)*/
 std::string Gadgetron::gadgetron_getCusparseErrorString(int err)
  {
    switch (err)
    {
/* DPCT_ORIG     case CUSPARSE_STATUS_NOT_INITIALIZED:*/
    case 1:
      return "NOT INITIALIZED";
/* DPCT_ORIG     case CUSPARSE_STATUS_ALLOC_FAILED:*/
    case 2:
      return "ALLOC FAILED";
/* DPCT_ORIG     case CUSPARSE_STATUS_INVALID_VALUE:*/
    case 3:
      return "INVALID VALUE";
/* DPCT_ORIG     case CUSPARSE_STATUS_ARCH_MISMATCH:*/
    case 4:
      return "ARCH MISMATCH";
/* DPCT_ORIG     case CUSPARSE_STATUS_MAPPING_ERROR:*/
    case 5:
      return "MAPPING ERROR";
/* DPCT_ORIG     case CUSPARSE_STATUS_EXECUTION_FAILED:*/
    case 6:
      return "EXECUTION FAILED";
/* DPCT_ORIG     case CUSPARSE_STATUS_INTERNAL_ERROR:*/
    case 7:
      return "INTERNAL ERROR";
/* DPCT_ORIG     case CUSPARSE_STATUS_SUCCESS:*/
    case 0:
      return "SUCCES";
/* DPCT_ORIG     case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:*/
    case 8:
      return "MATRIX TYPE NOT SUPPORTED";
    default:
      return "UNKNOWN CUSPARSE ERROR";
    }
  }
namespace Gadgetron
{

 

  static boost::shared_array<boost::mutex> _mutex;
  static boost::shared_array<boost::mutex> _sparseMutex;

  cudaDeviceManager *cudaDeviceManager::_instance = 0;

  cudaDeviceManager::cudaDeviceManager() try {

    // This constructor is executed only once for a singleton
    //

    atexit(&CleanUp);

/* DPCT_ORIG     if (auto res = cudaGetDeviceCount(&_num_devices); res != cudaSuccess)*/
    /*
    DPCT1000:269: Error handling if-stmt was detected but could not be rewritten.
    */
    if (auto res = DPCT_CHECK_ERROR(_num_devices = dpct::device_count()); res != 0)
    {
      /*
      DPCT1001:268: The statement could not be removed.
      */
      _num_devices = 0;
      throw cuda_error(res);
    }

    _mutex = boost::shared_array<boost::mutex>(new boost::mutex[_num_devices]);
    _sparseMutex = boost::shared_array<boost::mutex>(new boost::mutex[_num_devices]);

    int old_device;
/* DPCT_ORIG     if (auto res = cudaGetDevice(&old_device); res != cudaSuccess)*/
    /*
    DPCT1000:271: Error handling if-stmt was detected but could not be rewritten.
    */
    if (auto res = DPCT_CHECK_ERROR(old_device = dpct::get_current_device_id()); res != 0)
    {
      /*
      DPCT1001:270: The statement could not be removed.
      */
      throw cuda_error(res);
    }

    _total_global_mem = std::vector<size_t>(_num_devices, 0);
    _shared_mem_per_block = std::vector<size_t>(_num_devices, 0);
    _warp_size = std::vector<int>(_num_devices, 0);
    _max_blockdim = std::vector<int>(_num_devices, 0);
    _max_griddim = std::vector<int>(_num_devices, 0);
    _major = std::vector<int>(_num_devices, 0);
    _minor = std::vector<int>(_num_devices, 0);
/* DPCT_ORIG     _handle = std::vector<cublasHandle_t>(_num_devices, (cublasContext *)0x0);*/
    _handle = std::vector<dpct::blas::descriptor_ptr>(_num_devices, nullptr);
/* DPCT_ORIG     _sparse_handle = std::vector<cusparseHandle_t>(_num_devices, (cusparseHandle_t)0x0);*/
    _sparse_handle = std::vector<dpct::sparse::descriptor_ptr>(_num_devices, nullptr);

    for (int device = 0; device < _num_devices; device++)
    {

/* DPCT_ORIG       if (auto res = cudaSetDevice(device); res != cudaSuccess)*/
      /*
      DPCT1000:275: Error handling if-stmt was detected but could not be rewritten.
      */
      /*
      DPCT1093:278: The "device" device may be not the one intended for use. Adjust the selected device if needed.
      */
      if (auto res = DPCT_CHECK_ERROR(dpct::select_device(device)); res != 0)
      {
        /*
        DPCT1001:274: The statement could not be removed.
        */
        throw cuda_error(res);
      }

/* DPCT_ORIG       cudaDeviceProp deviceProp;*/
      dpct::device_info deviceProp;

/* DPCT_ORIG       if (auto res = cudaGetDeviceProperties(&deviceProp, device); res != cudaSuccess)*/
      /*
      DPCT1000:277: Error handling if-stmt was detected but could not be rewritten.
      */
      if (auto res = DPCT_CHECK_ERROR(dpct::get_device(device).get_device_info(deviceProp)); res != 0)
      {
        /*
        DPCT1001:276: The statement could not be removed.
        */
        throw cuda_error(res);
      }

/* DPCT_ORIG       _total_global_mem[device] = deviceProp.totalGlobalMem;*/
      _total_global_mem[device] = deviceProp.get_global_mem_size();
/* DPCT_ORIG       _shared_mem_per_block[device] = deviceProp.sharedMemPerBlock;*/
      /*
      DPCT1019:279: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to
      adjust the code.
      */
      _shared_mem_per_block[device] = deviceProp.get_local_mem_size();
/* DPCT_ORIG       _warp_size[device] = deviceProp.warpSize;*/
      _warp_size[device] = deviceProp.get_max_sub_group_size();
/* DPCT_ORIG       _max_blockdim[device] = deviceProp.maxThreadsDim[0];*/
      _max_blockdim[device] = deviceProp.get_max_work_item_sizes<int*>()[0];
/* DPCT_ORIG       _max_griddim[device] = deviceProp.maxGridSize[0];*/
      /*
      DPCT1022:280: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of
      the code.
      */
      _max_griddim[device] = deviceProp.get_max_nd_range_size<int*>()[0];
/* DPCT_ORIG       _major[device] = deviceProp.major;*/
      /*
      DPCT1005:281: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this
      code.
      */
      _major[device] = deviceProp.get_major_version();
/* DPCT_ORIG       _minor[device] = deviceProp.minor;*/
      /*
      DPCT1005:282: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this
      code.
      */
      _minor[device] = deviceProp.get_minor_version();
    }

/* DPCT_ORIG     if (auto res = cudaSetDevice(old_device); res != cudaSuccess)*/
    /*
    DPCT1000:273: Error handling if-stmt was detected but could not be rewritten.
    */
    /*
    DPCT1093:283: The "old_device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    if (auto res = DPCT_CHECK_ERROR(dpct::select_device(old_device)); res != 0)
    {
      /*
      DPCT1001:272: The statement could not be removed.
      */
      throw cuda_error(res);
    }
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  cudaDeviceManager::~cudaDeviceManager()
  {

    for (int device = 0; device < _num_devices; device++)
    {
      if (_handle[device] != NULL)
/* DPCT_ORIG         cublasDestroy(_handle[device]);*/
        delete (_handle[device]);
      if (_sparse_handle[device] != NULL)
/* DPCT_ORIG         cusparseDestroy(_sparse_handle[device]);*/
        delete (_sparse_handle[device]);
    }
  }

  size_t cudaDeviceManager::total_global_mem()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _total_global_mem[device];
  }

  size_t cudaDeviceManager::shared_mem_per_block()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _shared_mem_per_block[device];
  }

  int cudaDeviceManager::max_blockdim()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _max_blockdim[device];
  }

  int cudaDeviceManager::max_griddim()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _max_griddim[device];
  }

  int cudaDeviceManager::warp_size()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _warp_size[device];
  }

  int cudaDeviceManager::major_version()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _major[device];
  }

  int cudaDeviceManager::minor_version()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _minor[device];
  }

  size_t cudaDeviceManager::getFreeMemory()
  {
    size_t free, total;
/* DPCT_ORIG     CUDA_CALL(cudaMemGetInfo(&free, &total));*/
    /*
    DPCT1106:284: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(free, total)));
    return free;
  }

  size_t cudaDeviceManager::getTotalMemory()
  {
    size_t free, total;
/* DPCT_ORIG     CUDA_CALL(cudaMemGetInfo(&free, &total));*/
    /*
    DPCT1106:285: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(free, total)));
    return total;
  }

  size_t cudaDeviceManager::getFreeMemory(int device)
  {
    int oldDevice;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&oldDevice));*/
    CUDA_CALL(DPCT_CHECK_ERROR(oldDevice = dpct::get_current_device_id()));
/* DPCT_ORIG     CUDA_CALL(cudaSetDevice(device));*/
    /*
    DPCT1093:286: The "device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device)));
    size_t ret = getFreeMemory();
/* DPCT_ORIG     CUDA_CALL(cudaSetDevice(oldDevice));*/
    /*
    DPCT1093:287: The "oldDevice" device may be not the one intended for use. Adjust the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(oldDevice)));
    return ret;
  }

  size_t cudaDeviceManager::getTotalMemory(int device)
  {
    int oldDevice;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&oldDevice));*/
    CUDA_CALL(DPCT_CHECK_ERROR(oldDevice = dpct::get_current_device_id()));
/* DPCT_ORIG     CUDA_CALL(cudaSetDevice(device));*/
    /*
    DPCT1093:288: The "device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device)));
    size_t ret = getTotalMemory();
/* DPCT_ORIG     CUDA_CALL(cudaSetDevice(oldDevice));*/
    /*
    DPCT1093:289: The "oldDevice" device may be not the one intended for use. Adjust the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(oldDevice)));
    return ret;
  }

  cudaDeviceManager *cudaDeviceManager::Instance()
  {
    if (_instance == 0)
      _instance = new cudaDeviceManager;
    return _instance;
  }

/* DPCT_ORIG   cublasHandle_t cudaDeviceManager::lockHandle()*/
  dpct::blas::descriptor_ptr cudaDeviceManager::lockHandle()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return lockHandle(device);
  }

/* DPCT_ORIG   cublasHandle_t cudaDeviceManager::lockHandle(int device)*/
  dpct::blas::descriptor_ptr cudaDeviceManager::lockHandle(int device) try {
    _mutex[device].lock();
    if (_handle[device] == NULL)
    {
/* DPCT_ORIG       cublasStatus_t ret = cublasCreate(&_handle[device]);*/
      int ret = DPCT_CHECK_ERROR(_handle[device] = new dpct::blas::descriptor());
/* DPCT_ORIG       if (ret != CUBLAS_STATUS_SUCCESS)*/
      if (ret != 0)
      {
        std::stringstream ss;
        ss << "Error: unable to create cublas handle for device " << device << " : ";
        ss << gadgetron_getCublasErrorString(ret) << std::endl;
        throw cuda_error(ss.str());
      }
/* DPCT_ORIG       cublasSetPointerMode(_handle[device], CUBLAS_POINTER_MODE_HOST);*/
      /*
      DPCT1026:290: The call to cublasSetPointerMode was removed because this functionality is redundant in SYCL.
      */
    }
    return _handle[device];
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void cudaDeviceManager::unlockHandle()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return unlockHandle(device);
  }

  void cudaDeviceManager::unlockHandle(int device)
  {
    _mutex[device].unlock();
  }

/* DPCT_ORIG   cusparseHandle_t cudaDeviceManager::lockSparseHandle()*/
  dpct::sparse::descriptor_ptr cudaDeviceManager::lockSparseHandle()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return lockSparseHandle(device);
  }

/* DPCT_ORIG   cusparseHandle_t cudaDeviceManager::lockSparseHandle(int device)*/
  dpct::sparse::descriptor_ptr cudaDeviceManager::lockSparseHandle(int device) try {
    _sparseMutex[device].lock();
    if (_sparse_handle[device] == NULL)
    {
/* DPCT_ORIG       cusparseStatus_t ret = cusparseCreate(&_sparse_handle[device]);*/
      int ret = DPCT_CHECK_ERROR(_sparse_handle[device] = new dpct::sparse::descriptor());
/* DPCT_ORIG       if (ret != CUSPARSE_STATUS_SUCCESS)*/
      if (ret != 0)
      {
        std::stringstream ss;
        ss << "Error: unable to create cusparse handle for device " << device << " : ";
        ss << gadgetron_getCusparseErrorString(ret) << std::endl;
        throw cuda_error(ss.str());
      }
/* DPCT_ORIG       cusparseSetPointerMode(_sparse_handle[device], CUSPARSE_POINTER_MODE_HOST);*/
      /*
      DPCT1026:291: The call to cusparseSetPointerMode was removed because this functionality is redundant in SYCL.
      */
      // cublasSetPointerMode( _handle[device], CUBLAS_POINTER_MODE_HOST );
    }
    return _sparse_handle[device];
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void cudaDeviceManager::unlockSparseHandle()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return unlockSparseHandle(device);
  }

  void cudaDeviceManager::unlockSparseHandle(int device)
  {
    _sparseMutex[device].unlock();
  }

  int cudaDeviceManager::getCurrentDevice()
  {
    int device;
/* DPCT_ORIG     CUDA_CALL(cudaGetDevice(&device));*/
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return device;
  }

  int cudaDeviceManager::getTotalNumberOfDevice()
  {
    int number_of_devices;
/* DPCT_ORIG     CUDA_CALL(cudaGetDeviceCount(&number_of_devices));*/
    CUDA_CALL(DPCT_CHECK_ERROR(number_of_devices = dpct::device_count()));
    return number_of_devices;
  }

  void cudaDeviceManager::CleanUp()
  {
    delete _instance;
    _instance = 0;
  }
} // namespace Gadgetron
