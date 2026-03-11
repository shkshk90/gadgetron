/** \file check_CUDA.h
    \brief Macroes to check whether GPU-based code has caused any errors, and if so, throw a runtime exception accordingly.
*/

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GadgetronCuException.h"

namespace Gadgetron {

  /**
   *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
   *  inspired by cutil.h: CUT_CHECK_ERROR
   */
  inline void CHECK_FOR_CUDA_ERROR(char const * cur_fun, const char* file, const int line) {
    /*
    DPCT1010:4: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
    call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 errorCode = 0;
    /*
    DPCT1000:3: Error handling if-stmt was detected but could not be rewritten.
    */
    if (errorCode != 0) {
      /*
      DPCT1001:2: The statement could not be removed.
      */
      throw cuda_error(errorCode);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess) {
      throw cuda_error(errorCode);
    }
#endif
  }
}

/**
 *  Checks for CUDA errors and throws an exception if an error was detected.
 */
#define CHECK_FOR_CUDA_ERROR(); CHECK_FOR_CUDA_ERROR(BOOST_CURRENT_FUNCTION,__FILE__,__LINE__);

/**
 *  Call "res", checks for CUDA errors and throws an exception if an error was detected.
 */
/*
DPCT1001:12: The statement could not be removed.
*/
/*
DPCT1000:13: Error handling if-stmt was detected but could not be rewritten.
*/
#define CUDA_CALL(res) { dpct::err0 errorCode = res; if (errorCode != 0) { throw cuda_error(errorCode); } }

#define CUSPARSE_CALL(res) {int errorCode = res; if (errorCode != 0){ std::stringstream ss; ss << "CUSPARSE failed with error: " <<  gadgetron_getCusparseErrorString(errorCode); throw cuda_error(ss.str());}}
