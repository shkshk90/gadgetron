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
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include "cuNDArray.h"
#include "cudaDeviceManager.h"
#include <oneapi/math/sparse_blas.hpp>

namespace Gadgetron
{

	// Map complext<T> to std::complex<T> for oneMath API
	template <class T> struct to_std_type { using type = T; };
	template <class T> struct to_std_type<complext<T>> { using type = std::complex<T>; };
	template <class T> using to_std_type_t = typename to_std_type<T>::type;

	template <class T>
	struct cuCsrMatrix
	{

                cuCsrMatrix(size_t rows, size_t cols, dpct::device_vector<int> csrRow,
                            dpct::device_vector<int> csrColdnd, dpct::device_vector<T> data)
                    : csrRow{std::move(csrRow)}, csrColdnd{std::move(csrColdnd)}, data{std::move(data)}, rows{rows},
                      cols{cols}, descr{nullptr}
                {
                        using stdT = to_std_type_t<T>;
                        oneapi::math::sparse::init_csr_matrix(
                            dpct::get_in_order_queue(), &descr,
                            static_cast<std::int64_t>(rows),
                            static_cast<std::int64_t>(cols),
                            static_cast<std::int64_t>(this->data.size()),
                            oneapi::math::index_base::zero,
                            dpct::get_raw_pointer(this->csrRow.data()),
                            dpct::get_raw_pointer(this->csrColdnd.data()),
                            reinterpret_cast<stdT*>(dpct::get_raw_pointer(this->data.data())));
                }

		~cuCsrMatrix()
		{
			if (this->descr)
                                oneapi::math::sparse::release_sparse_matrix(dpct::get_in_order_queue(), this->descr);
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
			this->data = std::move(other.data);
			this->rows = other.rows;
			this->cols = other.cols;
			return *this;
		}

		size_t rows, cols;
                dpct::device_vector<int> csrRow, csrColdnd;
                dpct::device_vector<T> data;
                oneapi::math::sparse::matrix_handle_t descr;
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
