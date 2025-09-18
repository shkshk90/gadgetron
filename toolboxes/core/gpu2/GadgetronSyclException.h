#pragma once

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <system_error>

namespace Gadgetron::sycl {
  
  class sycl_error : public std::runtime_error
  {
  public:
    sycl_error(std::string msg) : std::runtime_error(msg) {}
    sycl_error(sycl::exception const& e) : std::runtime_error(e.what()) {}
    sycl_error(std::error_code ec) : std::runtime_error(ec.message()) {}
  };
}
