#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUBLASContextProvider.h"
#include <dpct/blas_utils.hpp>

/* DPCT_ORIG #include <cuda_runtime_api.h>*/

#ifdef _WITH_CULA_SUPPORT
#include <cula_lapack_device.h>
#endif


CUBLASContextProvider* CUBLASContextProvider::instance()
{
		if (!instance_) instance_ = new CUBLASContextProvider();
		return instance_;
}

CUBLASContextProvider::~CUBLASContextProvider() try {
/* DPCT_ORIG 	std::map<int, cublasHandle_t>::iterator it = handles_.begin();*/
        std::map<int, dpct::blas::descriptor_ptr>::iterator it = handles_.begin();

        while (it != handles_.end()) {
/* DPCT_ORIG 		if (cudaSetDevice(it->first)!= cudaSuccess) {*/
                /*
                DPCT1093:265: The "it->first" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(it->first)) != 0) {
                    std::cerr << "Error: unable to set CUDA device." << std::endl;
		}

#ifdef _WITH_CULA_SUPPORT
		culaShutdown();
#endif

/* DPCT_ORIG 		cublasDestroy_v2(it->second);*/
                delete (it->second);
                it++;
	}
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/* DPCT_ORIG cublasHandle_t* CUBLASContextProvider::getCublasHandle(int device_no)*/
dpct::blas::descriptor_ptr* CUBLASContextProvider::getCublasHandle(int device_no) try {
/* DPCT_ORIG 	std::map<int, cublasHandle_t>::iterator it;*/
        std::map<int, dpct::blas::descriptor_ptr>::iterator it;

        //Let's see if we have the handle already:
	it = handles_.find(device_no);

	if (it != handles_.end()) {
		return &handles_[device_no];
	}


	//We don't have the handle yet, let's check if it makes sense to create one

	int number_of_devices = 0;
/* DPCT_ORIG 	if (cudaGetDeviceCount(&number_of_devices)!= cudaSuccess) {*/
        if (DPCT_CHECK_ERROR(number_of_devices = dpct::device_count()) != 0) {
            std::cerr << "Error: unable to query number of CUDA devices.\n" << std::endl;
	    return 0;
	}

	if (number_of_devices == 0) {
	      std::cerr << "Error: No available CUDA devices.\n" << std::endl;
	      return 0;
     }

	  if (device_no >= number_of_devices) {
	      std::cerr << "Requested device number exceeds number of devices." << std::endl;
		  return 0;
	  }

	  //OK, so we are OK to create the handle. Before we do that, let's capture the current cuda device.

	  int current_device_no;
/* DPCT_ORIG 	if (cudaGetDevice(&current_device_no)!= cudaSuccess) {*/
        if (DPCT_CHECK_ERROR(current_device_no = dpct::get_current_device_id()) != 0) {
                 std::cerr << "Error: unable to get current CUDA device.\n" << std::endl;
		      return 0;
	}

	if (current_device_no != device_no) {
		//We must switch context
/* DPCT_ORIG 		if (cudaSetDevice(device_no)!= cudaSuccess) {*/
                /*
                DPCT1093:266: The "device_no" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(device_no)) != 0) {
                    std::cerr << "Error: unable to set CUDA device." << std::endl;
		      return 0;
		}
	}

/* DPCT_ORIG 	cublasHandle_t handle; */
        dpct::blas::descriptor_ptr handle; // this is a struct pointer

        //GDEBUG_STREAM("*********   CREATING NEW CONTEXT ************" << std::endl);

/* DPCT_ORIG 	if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {*/
        if (DPCT_CHECK_ERROR(handle = new dpct::blas::descriptor()) != 0) {
                std::cerr << "CUBLASContextProvider: unable to create cublas handle\n" << std::endl;
		return 0;
	}

	handles_[device_no] = handle;

#ifdef _WITH_CULA_SUPPORT
	culaStatus s;
	s = culaInitialize();
	if(s != culaNoError) {
		std::cerr << "CUBLASContextProvider: failed to initialize CULA" << std::endl;
		return 0;
	}
#endif

	if (current_device_no != device_no) {
		//We must switch context back
/* DPCT_ORIG 		if (cudaSetDevice(current_device_no)!= cudaSuccess) {*/
                /*
                DPCT1093:267: The "current_device_no" device may be not the one intended for use. Adjust the selected
                device if needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(current_device_no)) != 0) {
                   std::cerr << "Error: unable to set CUDA device.\n" << std::endl;
		    return 0;
		}
	}

	return &handles_[device_no];
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

CUBLASContextProvider* CUBLASContextProvider::instance_ = 0;

