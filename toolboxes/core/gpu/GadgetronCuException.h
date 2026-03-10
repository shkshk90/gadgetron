#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdexcept>

namespace Gadgetron{
  
  class cuda_error : public std::runtime_error
  {
  public:
    cuda_error(std::string msg) : std::runtime_error(msg) {}
    /*
    DPCT1009:1: SYCL reports errors using exceptions and does not use error codes. Please replace the
    "get_error_string_dummy(...)" with a real error-handling function.
    */
    cuda_error(dpct::err0 errN) : std::runtime_error(dpct::get_error_string_dummy(errN)) {
    }
  };
}
