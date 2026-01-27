/*
 * CUSPARSE.h
 *
 *  Created on: Jan 28, 2015
 *      Author: u051747
 */

#pragma once
/* DPCT_ORIG #include "cusparse.h"*/
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/dpl_utils.hpp>
/* DPCT_ORIG #include <thrust/device_vector.h>*/
#include "cuNDArray.h"
#include "cudaDeviceManager.h"

namespace Gadgetron
{


	template <class T>
	struct cuCsrMatrix
	{

/* DPCT_ORIG 		cuCsrMatrix(size_t rows, size_t cols, thrust::device_vector<int> csrRow,
 * thrust::device_vector<int> csrColdnd, thrust::device_vector<T> data) : csrRow{std::move(csrRow)},
 * csrColdnd{std::move(csrColdnd)}, data{std::move(data)}, rows{rows}, cols{cols}*/
                cuCsrMatrix(size_t rows, size_t cols, dpct::device_vector<int> csrRow,
                            dpct::device_vector<int> csrColdnd, dpct::device_vector<T> data)
                    : csrRow{std::move(csrRow)}, csrColdnd{std::move(csrColdnd)}, data{std::move(data)}, rows{rows},
                      cols{cols}
                {
/* DPCT_ORIG 			cusparseCreateCsr(&descr, rows, cols, this->data.size(),
                                thrust::raw_pointer_cast(this->csrRow.data()),
   thrust::raw_pointer_cast(this->csrColdnd.data()), thrust::raw_pointer_cast(this->data.data()), CUSPARSE_INDEX_32I,
   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, Gadgetron::cuda_datatype<T>());*/
                        descr = std::make_shared<dpct::sparse::sparse_matrix_desc>(
                            rows, cols, this->data.size(), dpct::get_raw_pointer(this->csrRow.data()),
                            dpct::get_raw_pointer(this->csrColdnd.data()), dpct::get_raw_pointer(this->data.data()),
                            dpct::library_data_t::real_int32, dpct::library_data_t::real_int32,
                            oneapi::mkl::index_base::zero, Gadgetron::cuda_datatype<T>(),
                            dpct::sparse::matrix_format::csr);
                }

		~cuCsrMatrix()
		{
			if (this->descr)
/* DPCT_ORIG 				cusparseDestroySpMat(this->descr);*/
                                (this->descr).reset();
                }

		cuCsrMatrix(cuCsrMatrix &&other)
		{
			*this = std::move(other);
		}

		cuCsrMatrix &operator=(cuCsrMatrix &&other)
		{
			this->descr = other.descr;
			other.descr = nullptr;
			this->csrColdnd = std::move(other.csrColdnd);
			this->csrRow = std::move(other.csrRow);
			this->data = std::move(this->data);
			return *this;
		}

		size_t rows, cols;
/* DPCT_ORIG 		thrust::device_vector<int> csrRow, csrColdnd;*/
                dpct::device_vector<int> csrRow, csrColdnd;
/* DPCT_ORIG 		thrust::device_vector<T> data;*/
                dpct::device_vector<T> data;
/* DPCT_ORIG 		cusparseSpMatDescr_t descr;*/
                dpct::sparse::sparse_matrix_desc_t descr;
        };

	/**
 * Performs a sparse matrix vector multiplication: vec_out = alpha*Mat * beta*vec_in
 * @param alpha
 * @param beta
 * @param mat
 * @param vec_in
 * @param vec_out
 * @param adjoint
 */
	template <class T>
	void sparseMV(T alpha, T beta, const cuCsrMatrix<T> &mat, const cuNDArray<T> &vec_in, cuNDArray<T> &vec_out, bool adjoint = false);

	template <class T>
	void sparseMM(T alpha, T beta, const cuCsrMatrix<T> &mat, const cuNDArray<T> &mat_in, cuNDArray<T> &mat_out, bool adjoint = false);

} // namespace Gadgetron
