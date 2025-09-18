#include "SYCLBLASContextProvider.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <iostream>


SYCLBLASContextProvider* SYCLBLASContextProvider::instance()
{
		if (!instance_) instance_ = new SYCLBLASContextProvider();
		return instance_;
}


SYCLBLASContextProvider::~SYCLBLASContextProvider() try {
	std::map<int, dpct::blas::descriptor_ptr>::iterator it = handles_.begin();

	while (it != handles_.end()) {
		/*
		DPCT1093:0: The "it->first" device may be not the one intended for use. Adjust the selected device if
		needed.
		*/
		dpct::select_device(it->first);

		delete (it->second);
		it++;
	}
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  return 0;
}

dpct::blas::descriptor_ptr* SYCLBLASContextProvider::getCublasHandle(int device_no) try {
        std::map<int, dpct::blas::descriptor_ptr>::iterator it;

        //Let's see if we have the handle already:
	it = handles_.find(device_no);

	if (it != handles_.end()) {
		return &handles_[device_no];
	}


	//We don't have the handle yet, let's check if it makes sense to create one

	int number_of_devices = dpct::device_count();
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
        current_device_no = dpct::get_current_device_id();

	if (current_device_no != device_no) {
		//We must switch context
                /*
                DPCT1093:1: The "device_no" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                dpct::select_device(device_no);
	}

        dpct::blas::descriptor_ptr handle; // this is a struct pointer


        handle = new dpct::blas::descriptor();
    if (false)  {
                std::cerr << "SYCLBLASContextProvider: unable to create cublas handle\n" << std::endl;
		return 0;
	}

	handles_[device_no] = handle;

	if (current_device_no != device_no) {
		//We must switch context back
		/*
		DPCT1093:2: The "current_device_no" device may be not the one intended for use. Adjust the selected
		device if needed.
		*/
		dpct::select_device(current_device_no);

	}

	return &handles_[device_no];
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  return 0;
}

SYCLBLASContextProvider* SYCLBLASContextProvider::instance_ = 0;

