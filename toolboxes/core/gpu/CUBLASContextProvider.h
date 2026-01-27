/*
 * CUBLASContextProvider.h
 *
 *  Created on: Mar 22, 2012
 *      Author: Michael S. Hansen
 */

#ifndef CUBLASCONTEXTPROVIDER_H_
#define CUBLASCONTEXTPROVIDER_H_
#pragma once

/* DPCT_ORIG #include <cublas_v2.h>*/
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

/* DPCT_ORIG 	cublasHandle_t* getCublasHandle(int device_no = 0);*/
        dpct::blas::descriptor_ptr* getCublasHandle(int device_no = 0);

private:
	CUBLASContextProvider() {}
	virtual ~CUBLASContextProvider();

	static CUBLASContextProvider* instance_;

/* DPCT_ORIG 	std::map<int, cublasHandle_t> handles_;*/
        std::map<int, dpct::blas::descriptor_ptr> handles_;
};

#endif /* CUBLASCONTEXTPROVIDER_H_ */
