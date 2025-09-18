/** \file check_SYCL.h
    \brief Macroes to check whether GPU-based code has caused any errors, and if so, throw a runtime exception accordingly.
*/

#pragma once

#include "GadgetronSyclException.h"


namespace Gadgetron::sycl {

  // inline void CHECK_FOR_SYCL_ERROR(char const * cur_fun, const char* file, const int line) {
  //   cudaError_t errorCode = cudaGetLastError();
  //   if (errorCode != cudaSuccess) {
  //     throw cuda_error(errorCode);
  //   }
  // }
}

#define CHECK_FOR_SYCL_ERROR() #warning FIXME
//; Gadgetron::sycl::CHECK_FOR_SYCL_ERROR(__func__,__FILE__,__LINE__);

#define SYCL_CALL(res) #warning FIXME2
// {cudaError_t errorCode = res; if (errorCode != cudaSuccess) { throw cuda_error(errorCode); }}

#define SYCLSPARSE_CALL(res) #warning FIXME3
// {cusparseStatus_t errorCode = res; if (errorCode != CUSPARSE_STATUS_SUCCESS){ std::stringstream ss; ss << "CUSPARSE failed with error: " <<  gadgetron_getCusparseErrorString(errorCode); throw cuda_error(ss.str());}}
