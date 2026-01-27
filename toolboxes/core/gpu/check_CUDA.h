/** \file check_CUDA.h
    \brief Macroes to check whether GPU-based code has caused any errors, and if so, throw a runtime exception accordingly.
*/

#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "GadgetronCuException.h"

namespace Gadgetron {

  /**
   *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
   *  inspired by cutil.h: CUT_CHECK_ERROR
   */
  inline void CHECK_FOR_CUDA_ERROR(char const * cur_fun, const char* file, const int line) {
/* DPCT_ORIG     cudaError_t errorCode = cudaGetLastError();*/
    /*
    DPCT1010:121: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
    call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 errorCode = 0;
/* DPCT_ORIG     if (errorCode != cudaSuccess) {*/
    /*
    DPCT1000:120: Error handling if-stmt was detected but could not be rewritten.
    */
    if (errorCode != 0) {
      /*
      DPCT1001:119: The statement could not be removed.
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
/* DPCT_ORIG #define CUDA_CALL(res) {cudaError_t errorCode = res; if (errorCode != cudaSuccess) { throw
 * cuda_error(errorCode); }}*/
/*
DPCT1001:129: The statement could not be removed.
*/
/*
DPCT1000:130: Error handling if-stmt was detected but could not be rewritten.
*/
#define CUDA_CALL(res) { dpct::err0 errorCode = res; if (errorCode != 0) { throw cuda_error(errorCode); } }

/* DPCT_ORIG #define CUSPARSE_CALL(res) {cusparseStatus_t errorCode = res; if (errorCode != CUSPARSE_STATUS_SUCCESS){
 * std::stringstream ss; ss << "CUSPARSE failed with error: " <<  gadgetron_getCusparseErrorString(errorCode); throw
 * cuda_error(ss.str());}}*/
#define CUSPARSE_CALL(res)                                                                                             \
  { int errorCode = res; if (errorCode != 0) {                                                                         \
    std::stringstream ss; ss << "CUSPARSE failed with error: " << gadgetron_getCusparseErrorString(errorCode);         \
    throw cuda_error(ss.str());                                                                                        \
  } }
