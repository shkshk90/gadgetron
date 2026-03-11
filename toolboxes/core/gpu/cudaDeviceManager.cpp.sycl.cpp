#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudaDeviceManager.h"
#include "check_CUDA.h"
#include "cuNDArray_blas.h"

#include <boost/thread/mutex.hpp>
#include <boost/shared_array.hpp>
#include <stdlib.h>
#include <sstream>
#include <dpct/blas_utils.hpp>

 std::string Gadgetron::gadgetron_getCusparseErrorString(int err)
  {
    switch (err)
    {
    case 1:
      return "NOT INITIALIZED";
    case 2:
      return "ALLOC FAILED";
    case 3:
      return "INVALID VALUE";
    case 4:
      return "ARCH MISMATCH";
    case 5:
      return "MAPPING ERROR";
    case 6:
      return "EXECUTION FAILED";
    case 7:
      return "INTERNAL ERROR";
    case 0:
      return "SUCCES";
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

    /*
    DPCT1000:45: Error handling if-stmt was detected but could not be rewritten.
    */
    if (auto res = DPCT_CHECK_ERROR(_num_devices = dpct::device_count());
        res != 0)
    {
      /*
      DPCT1001:44: The statement could not be removed.
      */
      _num_devices = 0;
      throw cuda_error(res);
    }

    _mutex = boost::shared_array<boost::mutex>(new boost::mutex[_num_devices]);
    _sparseMutex = boost::shared_array<boost::mutex>(new boost::mutex[_num_devices]);

    int old_device;
    /*
    DPCT1000:47: Error handling if-stmt was detected but could not be rewritten.
    */
    if (auto res = DPCT_CHECK_ERROR(old_device = dpct::get_current_device_id());
        res != 0)
    {
      /*
      DPCT1001:46: The statement could not be removed.
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
    _handle = std::vector<dpct::blas::descriptor_ptr>(_num_devices, nullptr);
    _sparse_handle = std::vector<dpct::sparse::descriptor_ptr>(
        _num_devices, (dpct::sparse::descriptor_ptr)0x0);

    for (int device = 0; device < _num_devices; device++)
    {

      /*
      DPCT1000:51: Error handling if-stmt was detected but could not be
      rewritten.
      */
      /*
      DPCT1093:54: The "device" device may be not the one intended for use.
      Adjust the selected device if needed.
      */
      if (auto res = DPCT_CHECK_ERROR(dpct::select_device(device)); res != 0)
      {
        /*
        DPCT1001:50: The statement could not be removed.
        */
        throw cuda_error(res);
      }

      dpct::device_info deviceProp;

      /*
      DPCT1000:53: Error handling if-stmt was detected but could not be
      rewritten.
      */
      if (auto res = DPCT_CHECK_ERROR(
              dpct::get_device(device).get_device_info(deviceProp));
          res != 0)
      {
        /*
        DPCT1001:52: The statement could not be removed.
        */
        throw cuda_error(res);
      }

      _total_global_mem[device] = deviceProp.get_global_mem_size();
      /*
      DPCT1019:55: local_mem_size in SYCL is not a complete equivalent of
      sharedMemPerBlock in CUDA. You may need to adjust the code.
      */
      _shared_mem_per_block[device] = deviceProp.get_local_mem_size();
      _warp_size[device] = deviceProp.get_max_sub_group_size();
      _max_blockdim[device] = deviceProp.get_max_work_group_size();
      /*
      DPCT1022:56: There is no exact match between the maxGridSize and the
      max_nd_range size. Verify the correctness of the code.
      */
      _max_griddim[device] = deviceProp.get_max_nd_range_size<int *>()[2];
      /*
      DPCT1005:57: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      _major[device] = deviceProp.get_major_version();
      /*
      DPCT1005:58: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      _minor[device] = deviceProp.get_minor_version();
    }

    /*
    DPCT1000:49: Error handling if-stmt was detected but could not be rewritten.
    */
    /*
    DPCT1093:59: The "old_device" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    if (auto res = DPCT_CHECK_ERROR(dpct::select_device(old_device)); res != 0)
    {
      /*
      DPCT1001:48: The statement could not be removed.
      */
      throw cuda_error(res);
    }
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  cudaDeviceManager::~cudaDeviceManager()
  {

    for (int device = 0; device < _num_devices; device++)
    {
      if (_handle[device] != NULL)
        delete (_handle[device]);
      if (_sparse_handle[device] != NULL)
        delete (_sparse_handle[device]);
    }
  }

  size_t cudaDeviceManager::total_global_mem()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _total_global_mem[device];
  }

  size_t cudaDeviceManager::shared_mem_per_block()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _shared_mem_per_block[device];
  }

  int cudaDeviceManager::max_blockdim()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _max_blockdim[device];
  }

  int cudaDeviceManager::max_griddim()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _max_griddim[device];
  }

  int cudaDeviceManager::warp_size()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _warp_size[device];
  }

  int cudaDeviceManager::major_version()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _major[device];
  }

  int cudaDeviceManager::minor_version()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return _minor[device];
  }

  size_t cudaDeviceManager::getFreeMemory()
  {
    size_t free, total;
    /*
    DPCT1106:60: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(
        dpct::get_current_device().get_memory_info(free, total)));
    return free;
  }

  size_t cudaDeviceManager::getTotalMemory()
  {
    size_t free, total;
    /*
    DPCT1106:61: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(
        dpct::get_current_device().get_memory_info(free, total)));
    return total;
  }

  size_t cudaDeviceManager::getFreeMemory(int device)
  {
    int oldDevice;
    CUDA_CALL(DPCT_CHECK_ERROR(oldDevice = dpct::get_current_device_id()));
    /*
    DPCT1093:62: The "device" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device)));
    size_t ret = getFreeMemory();
    /*
    DPCT1093:63: The "oldDevice" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(oldDevice)));
    return ret;
  }

  size_t cudaDeviceManager::getTotalMemory(int device)
  {
    int oldDevice;
    CUDA_CALL(DPCT_CHECK_ERROR(oldDevice = dpct::get_current_device_id()));
    /*
    DPCT1093:64: The "device" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device)));
    size_t ret = getTotalMemory();
    /*
    DPCT1093:65: The "oldDevice" device may be not the one intended for use.
    Adjust the selected device if needed.
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

  dpct::blas::descriptor_ptr cudaDeviceManager::lockHandle()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return lockHandle(device);
  }

  dpct::blas::descriptor_ptr cudaDeviceManager::lockHandle(int device) try {
    _mutex[device].lock();
    if (_handle[device] == NULL)
    {
      int ret =
          DPCT_CHECK_ERROR(_handle[device] = new dpct::blas::descriptor());
      if (ret != 0)
      {
        std::stringstream ss;
        ss << "Error: unable to create cublas handle for device " << device << " : ";
        ss << gadgetron_getCublasErrorString(ret) << std::endl;
        throw cuda_error(ss.str());
      }
      /*
      DPCT1026:66: The call to cublasSetPointerMode was removed because this
      functionality is redundant in SYCL.
      */
    }
    return _handle[device];
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void cudaDeviceManager::unlockHandle()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return unlockHandle(device);
  }

  void cudaDeviceManager::unlockHandle(int device)
  {
    _mutex[device].unlock();
  }

  dpct::sparse::descriptor_ptr cudaDeviceManager::lockSparseHandle()
  {
    int device;
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return lockSparseHandle(device);
  }

  dpct::sparse::descriptor_ptr
  cudaDeviceManager::lockSparseHandle(int device) try {
    _sparseMutex[device].lock();
    if (_sparse_handle[device] == NULL)
    {
      int ret = DPCT_CHECK_ERROR(_sparse_handle[device] =
                                     new dpct::sparse::descriptor());
      if (ret != 0)
      {
        std::stringstream ss;
        ss << "Error: unable to create cusparse handle for device " << device << " : ";
        ss << gadgetron_getCusparseErrorString(ret) << std::endl;
        throw cuda_error(ss.str());
      }
      /*
      DPCT1026:67: The call to cusparseSetPointerMode was removed because this
      functionality is redundant in SYCL.
      */
      // cublasSetPointerMode( _handle[device], CUBLAS_POINTER_MODE_HOST );
    }
    return _sparse_handle[device];
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  void cudaDeviceManager::unlockSparseHandle()
  {
    int device;
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
    CUDA_CALL(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    return device;
  }

  int cudaDeviceManager::getTotalNumberOfDevice()
  {
    int number_of_devices;
    CUDA_CALL(DPCT_CHECK_ERROR(number_of_devices = dpct::device_count()));
    return number_of_devices;
  }

  void cudaDeviceManager::CleanUp()
  {
    delete _instance;
    _instance = 0;
  }
} // namespace Gadgetron
