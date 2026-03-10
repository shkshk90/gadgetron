/*
 * CUBLASContextProvider.h
 *
 *  Created on: Mar 22, 2012
 *      Author: Michael S. Hansen
 */

#ifndef CUBLASCONTEXTPROVIDER_H_
#define CUBLASCONTEXTPROVIDER_H_
#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <map>
#include <iostream>

class CUBLASContextProvider
{

public:
	static CUBLASContextProvider* instance();

        dpct::blas::descriptor_ptr* getCublasHandle(int device_no = 0);

private:
	CUBLASContextProvider() {}
	virtual ~CUBLASContextProvider();

	static CUBLASContextProvider* instance_;

        std::map<int, dpct::blas::descriptor_ptr> handles_;
};

#endif /* CUBLASCONTEXTPROVIDER_H_ */
