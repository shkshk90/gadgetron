#include "syclDeviceManager.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

_Pragma("clang diagnostic push") 
_Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") //  
_Pragma("clang diagnostic ignored \"-Wbitwise-op-parentheses\"")
_Pragma("clang diagnostic ignored \"-Wsign-compare\"")
_Pragma("clang diagnostic ignored \"-Wlogical-op-parentheses\"")
_Pragma("clang diagnostic ignored \"-Wreorder-ctor\"")
_Pragma("clang diagnostic ignored \"-Wunneeded-internal-declaration\"")
_Pragma("clang diagnostic ignored \"-Wunused-function\"")

#include <sycl/sycl.hpp>

_Pragma("clang diagnostic pop")

namespace Gadgetron {

syclDeviceManager* syclDeviceManager::_instance = 0;

syclDeviceManager::syclDeviceManager() try {

    atexit(&CleanUp);

    int old_device;

    _num_devices = dpct::device_count();
    old_device = dpct::get_current_device_id();

    _total_global_mem = std::vector<size_t>(_num_devices, 0);
    _shared_mem_per_block = std::vector<size_t>(_num_devices, 0);
    _warp_size = std::vector<int>(_num_devices, 0);
    _max_blockdim = std::vector<int>(_num_devices, 0);
    _max_griddim = std::vector<int>(_num_devices, 0);
    _major = std::vector<int>(_num_devices, 0);
    _minor = std::vector<int>(_num_devices, 0);
    _handle = std::vector<dpct::blas::descriptor_ptr>(_num_devices, nullptr);
    _sparse_handle = std::vector<dpct::sparse::descriptor_ptr>(_num_devices, nullptr);

    for (int device = 0; device < _num_devices; device++) {

        dpct::select_device(device);
        dpct::device_info deviceProp;
        dpct::get_device(device).get_device_info(deviceProp);

        _total_global_mem[device] = deviceProp.get_global_mem_size();
        /*
        DPCT1019:15: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to
        adjust the code.
        */
        _shared_mem_per_block[device] = deviceProp.get_local_mem_size();
        _warp_size[device] = deviceProp.get_max_sub_group_size();
        _max_blockdim[device] = deviceProp.get_max_work_item_sizes<int*>()[0];
        /*
        DPCT1022:16: There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness
        of the code.
        */
        _max_griddim[device] = deviceProp.get_max_nd_range_size<int*>()[0];
        /*
        DPCT1005:17: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this
        code.
        */
        _major[device] = deviceProp.get_major_version();
        /*
        DPCT1005:18: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this
        code.
        */
        _minor[device] = deviceProp.get_minor_version();
    }

    dpct::select_device(old_device);

} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    throw;
}

syclDeviceManager::~syclDeviceManager() {

    for (int device = 0; device < _num_devices; device++) {
        if (_handle[device] != NULL)
            delete (_handle[device]);
        if (_sparse_handle[device] != NULL)
            delete (_sparse_handle[device]);
    }
}

size_t syclDeviceManager::total_global_mem() {
    int device;
    device = dpct::get_current_device_id();
    return _total_global_mem[device];
}

size_t syclDeviceManager::shared_mem_per_block() {
    int device;
    device = dpct::get_current_device_id();
    return _shared_mem_per_block[device];
}

int syclDeviceManager::max_blockdim() {
    int device;
    device = dpct::get_current_device_id();
    return _max_blockdim[device];
}

int syclDeviceManager::max_griddim() {
    int device;
    device = dpct::get_current_device_id();
    return _max_griddim[device];
}

int syclDeviceManager::warp_size() {
    int device;
    device = dpct::get_current_device_id();
    return _warp_size[device];
}

int syclDeviceManager::major_version() {
    int device;
    device = dpct::get_current_device_id();
    return _major[device];
}

int syclDeviceManager::minor_version() {
    int device;
    device = dpct::get_current_device_id();
    return _minor[device];
}

size_t syclDeviceManager::getFreeMemory() {
    size_t free, total;
    /*
    DPCT1106:22: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(free, total);
    return free;
}

size_t syclDeviceManager::getTotalMemory() {
    size_t free, total;
    /*
    DPCT1106:23: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
    supported by all compilers or runtimes. You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(free, total);
    return total;
}

size_t syclDeviceManager::getFreeMemory(int device) {
    int oldDevice;
    oldDevice = dpct::get_current_device_id();
    /*
    DPCT1093:24: The "device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(device);
    size_t ret = getFreeMemory();
    /*
    DPCT1093:25: The "oldDevice" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(oldDevice);
    return ret;
}

size_t syclDeviceManager::getTotalMemory(int device) {
    int oldDevice;
    oldDevice = dpct::get_current_device_id();
    /*
    DPCT1093:26: The "device" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(device);
    size_t ret = getTotalMemory();
    /*
    DPCT1093:27: The "oldDevice" device may be not the one intended for use. Adjust the selected device if needed.
    */
    dpct::select_device(oldDevice);
    return ret;
}

syclDeviceManager* syclDeviceManager::Instance() {
    if (_instance == 0)
        _instance = new syclDeviceManager;
    return _instance;
}

dpct::blas::descriptor_ptr syclDeviceManager::lockHandle() {
    int device;
    device = dpct::get_current_device_id();
    return lockHandle(device);
}

dpct::blas::descriptor_ptr syclDeviceManager::lockHandle(int device) try {
    if (_handle[device] == NULL)
        _handle[device] = new dpct::blas::descriptor();
    return _handle[device];
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    throw;
}

void syclDeviceManager::unlockHandle() {
    int device;
    device = dpct::get_current_device_id();
    return unlockHandle(device);
}

void syclDeviceManager::unlockHandle(int device) {}

dpct::sparse::descriptor_ptr syclDeviceManager::lockSparseHandle() {
    int device;
    device = dpct::get_current_device_id();
    return lockSparseHandle(device);
}

dpct::sparse::descriptor_ptr syclDeviceManager::lockSparseHandle(int device) try {
    if (_sparse_handle[device] == NULL)
        _sparse_handle[device] = new dpct::sparse::descriptor();
    return _sparse_handle[device];
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    throw;
}

void syclDeviceManager::unlockSparseHandle() {
    int device;
    device = dpct::get_current_device_id();
    return unlockSparseHandle(device);
}

void syclDeviceManager::unlockSparseHandle(int device) {}

int syclDeviceManager::getCurrentDevice() {
    int device;
    device = dpct::get_current_device_id();
    return device;
}

int syclDeviceManager::getTotalNumberOfDevice() {
    int number_of_devices;
    number_of_devices = dpct::device_count();
    return number_of_devices;
}

void syclDeviceManager::CleanUp() {
    delete _instance;
    _instance = 0;
}
} // namespace Gadgetron