/*
 * CUSPARSE.h
 *
 *  Created on: Jan 28, 2015
 *      Author: u051747
 */

#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/dpl_utils.hpp>
#include "cuNDArray.h"
#include "cudaDeviceManager.h"

namespace Gadgetron
{


	template <class T>
	struct cuCsrMatrix
	{

                cuCsrMatrix(size_t rows, size_t cols,
                            dpct::device_vector<int> csrRow,
                            dpct::device_vector<int> csrColdnd,
                            dpct::device_vector<T> data)
                    : csrRow{std::move(csrRow)},
                      csrColdnd{std::move(csrColdnd)}, data{std::move(data)},
                      rows{rows}, cols{cols}
                {
                        cusparseCreateCsr(
                            &descr, rows, cols, this->data.size(),
                            dpct::get_raw_pointer(this->csrRow.data()),
                            dpct::get_raw_pointer(this->csrColdnd.data()),
                            dpct::get_raw_pointer(this->data.data()),
                            dpct::library_data_t::real_int32,
                            dpct::library_data_t::real_int32,
                            oneapi::mkl::index_base::zero,
                            Gadgetron::cuda_datatype<T>());
                }

		~cuCsrMatrix()
		{
			if (this->descr)
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
                dpct::device_vector<int> csrRow, csrColdnd;
                dpct::device_vector<T> data;
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
