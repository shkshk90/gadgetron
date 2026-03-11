#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuSparseMatrix.h"
#include <dpct/dpl_utils.hpp>

#include "cuNDArray_math.h"
#include <oneapi/math/sparse_blas.hpp>

using namespace Gadgetron;


template <class T>
void Gadgetron::sparseMV(T alpha, T beta, const cuCsrMatrix<T>& mat, const cuNDArray<T>& vec_in, cuNDArray<T>& vec_out,
                         bool adjoint) {

        if (vec_in.get_number_of_elements() != (adjoint ? mat.rows : mat.cols))
		throw std::runtime_error("Matrix and input vector have mismatching dimensions");
	if (vec_out.get_number_of_elements() != (adjoint ? mat.rows : mat.cols))
		throw std::runtime_error("Matrix and output vector have mismatching dimensions");

        using stdT = to_std_type_t<T>;
        oneapi::math::transpose trans = adjoint ? oneapi::math::transpose::conjtrans : oneapi::math::transpose::nontrans;

        auto& queue = dpct::get_in_order_queue();

        // Create sparse matrix handle from CSR data
        oneapi::math::sparse::matrix_handle_t A_handle = nullptr;
        oneapi::math::sparse::init_csr_matrix(queue, &A_handle,
                static_cast<std::int64_t>(mat.rows),
                static_cast<std::int64_t>(mat.cols),
                static_cast<std::int64_t>(mat.data.size()),
                oneapi::math::index_base::zero,
                const_cast<int*>(dpct::get_raw_pointer(mat.csrRow.data())),
                const_cast<int*>(dpct::get_raw_pointer(mat.csrColdnd.data())),
                reinterpret_cast<stdT*>(const_cast<T*>(dpct::get_raw_pointer(mat.data.data()))));

        // Create dense vector handles
        oneapi::math::sparse::dense_vector_handle_t x_handle = nullptr;
        oneapi::math::sparse::init_dense_vector(queue, &x_handle,
                static_cast<std::int64_t>(vec_in.get_number_of_elements()),
                reinterpret_cast<stdT*>(const_cast<T*>(vec_in.get_data_ptr())));

        oneapi::math::sparse::dense_vector_handle_t y_handle = nullptr;
        oneapi::math::sparse::init_dense_vector(queue, &y_handle,
                static_cast<std::int64_t>(vec_out.get_number_of_elements()),
                reinterpret_cast<stdT*>(vec_out.get_data_ptr()));

        // Create spmv descriptor
        oneapi::math::sparse::spmv_descr_t spmv_descr = nullptr;
        oneapi::math::sparse::init_spmv_descr(queue, &spmv_descr);

        auto alg = oneapi::math::sparse::spmv_alg::default_alg;
        auto A_view = oneapi::math::sparse::matrix_view();

        stdT std_alpha = reinterpret_cast<const stdT&>(alpha);
        stdT std_beta = reinterpret_cast<const stdT&>(beta);

        // Query buffer size
        std::size_t workspace_size = 0;
        oneapi::math::sparse::spmv_buffer_size(queue, trans, &std_alpha,
                A_view, A_handle, x_handle, &std_beta, y_handle,
                alg, spmv_descr, workspace_size);

        // Allocate workspace and optimize
        void* workspace = nullptr;
        if (workspace_size > 0)
                workspace = sycl::malloc_device(workspace_size, queue);

        auto optimize_ev = oneapi::math::sparse::spmv_optimize(queue, trans, &std_alpha,
                A_view, A_handle, x_handle, &std_beta, y_handle,
                alg, spmv_descr, workspace);

        // Execute spmv
        auto spmv_ev = oneapi::math::sparse::spmv(queue, trans, &std_alpha,
                A_view, A_handle, x_handle, &std_beta, y_handle,
                alg, spmv_descr, {optimize_ev});
        spmv_ev.wait();

        // Cleanup
        oneapi::math::sparse::release_spmv_descr(queue, spmv_descr);
        oneapi::math::sparse::release_dense_vector(queue, x_handle);
        oneapi::math::sparse::release_dense_vector(queue, y_handle);
        oneapi::math::sparse::release_sparse_matrix(queue, A_handle);
        if (workspace)
                sycl::free(workspace, queue);
}

template<class T> void Gadgetron::sparseMM(T alpha,T beta, const cuCsrMatrix<T> & mat, const cuNDArray<T> & mat_in, cuNDArray<T>& mat_out, bool adjoint) {

	if (mat_in.get_size(1) != mat_out.get_size(1)) throw std::runtime_error("In and out dense matrix must have same second dimension");
	if (mat_in.get_size(0) != mat.rows) throw std::runtime_error("Input matrix and sparse matrix have mismatched dimensions");
	if (mat_out.get_size(0) != mat.cols) throw std::runtime_error("Output matrix and sparse matrix have mismatched dimensions");

        using stdT = to_std_type_t<T>;
        oneapi::math::transpose trans = adjoint ? oneapi::math::transpose::conjtrans : oneapi::math::transpose::nontrans;

        auto& queue = dpct::get_in_order_queue();

        // Create sparse matrix handle
        oneapi::math::sparse::matrix_handle_t A_handle = nullptr;
        oneapi::math::sparse::init_csr_matrix(queue, &A_handle,
                static_cast<std::int64_t>(mat.rows),
                static_cast<std::int64_t>(mat.cols),
                static_cast<std::int64_t>(mat.data.size()),
                oneapi::math::index_base::zero,
                const_cast<int*>(dpct::get_raw_pointer(mat.csrRow.data())),
                const_cast<int*>(dpct::get_raw_pointer(mat.csrColdnd.data())),
                reinterpret_cast<stdT*>(const_cast<T*>(dpct::get_raw_pointer(mat.data.data()))));

        // Create dense matrix handles (column-major)
        oneapi::math::sparse::dense_matrix_handle_t B_handle = nullptr;
        oneapi::math::sparse::init_dense_matrix(queue, &B_handle,
                static_cast<std::int64_t>(mat_in.get_size(0)),
                static_cast<std::int64_t>(mat_in.get_size(1)),
                static_cast<std::int64_t>(mat_in.get_size(0)),
                oneapi::math::layout::col_major,
                reinterpret_cast<stdT*>(const_cast<T*>(mat_in.get_data_ptr())));

        oneapi::math::sparse::dense_matrix_handle_t C_handle = nullptr;
        oneapi::math::sparse::init_dense_matrix(queue, &C_handle,
                static_cast<std::int64_t>(mat_out.get_size(0)),
                static_cast<std::int64_t>(mat_out.get_size(1)),
                static_cast<std::int64_t>(mat_out.get_size(0)),
                oneapi::math::layout::col_major,
                reinterpret_cast<stdT*>(mat_out.get_data_ptr()));

        // Create spmm descriptor
        oneapi::math::sparse::spmm_descr_t spmm_descr = nullptr;
        oneapi::math::sparse::init_spmm_descr(queue, &spmm_descr);

        auto alg = oneapi::math::sparse::spmm_alg::default_alg;
        auto A_view = oneapi::math::sparse::matrix_view();

        stdT std_alpha = reinterpret_cast<const stdT&>(alpha);
        stdT std_beta = reinterpret_cast<const stdT&>(beta);

        // Query buffer size
        std::size_t workspace_size = 0;
        oneapi::math::sparse::spmm_buffer_size(queue, trans, oneapi::math::transpose::nontrans,
                &std_alpha, A_view, A_handle, B_handle, &std_beta, C_handle,
                alg, spmm_descr, workspace_size);

        // Allocate workspace and optimize
        void* workspace = nullptr;
        if (workspace_size > 0)
                workspace = sycl::malloc_device(workspace_size, queue);

        auto optimize_ev = oneapi::math::sparse::spmm_optimize(queue, trans, oneapi::math::transpose::nontrans,
                &std_alpha, A_view, A_handle, B_handle, &std_beta, C_handle,
                alg, spmm_descr, workspace);

        // Execute spmm
        auto spmm_ev = oneapi::math::sparse::spmm(queue, trans, oneapi::math::transpose::nontrans,
                &std_alpha, A_view, A_handle, B_handle, &std_beta, C_handle,
                alg, spmm_descr, {optimize_ev});
        spmm_ev.wait();

        // Cleanup
        oneapi::math::sparse::release_spmm_descr(queue, spmm_descr);
        oneapi::math::sparse::release_dense_matrix(queue, B_handle);
        oneapi::math::sparse::release_dense_matrix(queue, C_handle);
        oneapi::math::sparse::release_sparse_matrix(queue, A_handle);
        if (workspace)
                sycl::free(workspace, queue);
}


template void Gadgetron::sparseMV<float>(float alpha,float beta, const cuCsrMatrix<float> & mat, const cuNDArray<float> & vec_in, cuNDArray<float>& vec_out, bool adjoint);
template void Gadgetron::sparseMV<double>(double alpha,double beta, const cuCsrMatrix<double> & mat, const cuNDArray<double> & vec_in, cuNDArray<double>& vec_out, bool adjoint);
template void Gadgetron::sparseMV<complext<float> >(complext<float> alpha,complext<float> beta, const cuCsrMatrix<complext<float> > & mat, const cuNDArray<complext<float> > & vec_in, cuNDArray<complext<float> >& vec_out, bool adjoint);
template void Gadgetron::sparseMV<complext<double> >(complext<double> alpha,complext<double> beta, const cuCsrMatrix<complext<double> > & mat, const cuNDArray<complext<double> > & vec_in, cuNDArray<complext<double> >& vec_out, bool adjoint);

template void Gadgetron::sparseMM<float>(float alpha,float beta, const cuCsrMatrix<float> & mat, const cuNDArray<float> & vec_in, cuNDArray<float>& vec_out, bool adjoint);
template void Gadgetron::sparseMM<double>(double alpha,double beta, const cuCsrMatrix<double> & mat, const cuNDArray<double> & vec_in, cuNDArray<double>& vec_out, bool adjoint);
template void Gadgetron::sparseMM<complext<float> >(complext<float> alpha,complext<float> beta, const cuCsrMatrix<complext<float> > & mat, const cuNDArray<complext<float> > & vec_in, cuNDArray<complext<float> >& vec_out, bool adjoint);
template void Gadgetron::sparseMM<complext<double> >(complext<double> alpha,complext<double> beta, const cuCsrMatrix<complext<double> > & mat, const cuNDArray<complext<double> > & vec_in, cuNDArray<complext<double> >& vec_out, bool adjoint);
