

#include <dpct/blas_utils.hpp>

#include <map>


class SYCLBLASContextProvider
{

public:
	static SYCLBLASContextProvider* instance();

    dpct::blas::descriptor_ptr* getCublasHandle(int device_no = 0);

private:
	SYCLBLASContextProvider() {}
	virtual ~SYCLBLASContextProvider();

	static SYCLBLASContextProvider* instance_;

    std::map<int, dpct::blas::descriptor_ptr> handles_;
};

// #ifdef _WITH_CULA_SUPPORT
// #include <cula_lapack_device.h>
// #endif

