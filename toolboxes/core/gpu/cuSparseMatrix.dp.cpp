#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuSparseMatrix.h"
#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/extrema.h>*/
/* DPCT_ORIG #include <thrust/device_ptr.h>*/
#include "cuNDArray_math.h"
#include <dpct/sparse_utils.hpp>

/* DPCT_ORIG #include <cusparse.h>*/

using namespace Gadgetron;

template <class T> static auto create_DnVec(cuNDArray<T>& vec) try {

/* DPCT_ORIG 	cusparseDnVecDescr_t dnvec;	*/
        std::shared_ptr<dpct::sparse::dense_vector_desc> dnvec;
/* DPCT_ORIG 	cusparseCreateDnVec(&dnvec,vec.size(),vec.data(),cuda_datatype<T>());*/
        dnvec = std::make_shared<dpct::sparse::dense_vector_desc>(vec.size(), vec.data(), cuda_datatype<T>());
/* DPCT_ORIG 	auto deleter = [](cusparseDnVecDescr_t val){cusparseDestroyDnVec(val);};*/
        auto deleter = [](std::shared_ptr<dpct::sparse::dense_vector_desc> val) {
                                                                                val.reset();
        };

        return std::unique_ptr<std::decay_t<decltype(*dnvec)>,decltype(deleter)>(dnvec.get(),deleter);
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <class T>
void Gadgetron::sparseMV(T alpha, T beta, const cuCsrMatrix<T>& mat, const cuNDArray<T>& vec_in, cuNDArray<T>& vec_out,
                         bool adjoint) try {

        if (vec_in.get_number_of_elements() != (adjoint ? mat.rows : mat.cols))
		throw std::runtime_error("Matrix and input vector have mismatching dimensions");
	if (vec_out.get_number_of_elements() != (adjoint ? mat.rows : mat.cols))
		throw std::runtime_error("Matrix and output vector have mismatching dimensions");

/* DPCT_ORIG 	cusparseOperation_t trans = adjoint ?  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE :
 * CUSPARSE_OPERATION_NON_TRANSPOSE;*/
        oneapi::mkl::transpose trans = adjoint ? oneapi::mkl::transpose::conjtrans : oneapi::mkl::transpose::nontrans;
        //cusparseStatus_t status = sparseCSRMV(cudaDeviceManager::Instance()->lockSparseHandle(),trans,mat.m,mat.n,mat.nnz,&alpha, mat.descr,
	//		thrust::raw_pointer_cast(&mat.data[0]),thrust::raw_pointer_cast(&mat.csrRow[0]),thrust::raw_pointer_cast(&mat.csrColdnd[0]),vec_in.get_data_ptr(),&beta,vec_out.get_data_ptr());


	auto dnvec_in = create_DnVec(const_cast<cuNDArray<T>&>(vec_in));
	auto dnvec_out = create_DnVec(vec_out);

	size_t bufferSize;
	auto handle =  cudaDeviceManager::Instance()->lockSparseHandle();
/* DPCT_ORIG 	cusparseSpMV(handle, trans,
 * &alpha,mat.descr,dnvec_in.get(),&beta,dnvec_out.get(),cuda_datatype<T>(),CUSPARSE_SPMV_CSR_ALG2,&bufferSize);*/
        dpct::sparse::spmv(handle->get_queue(), trans, &alpha, mat.descr, dnvec_in.get(), &beta, dnvec_out.get(),
                           cuda_datatype<T>());
        cuNDArray<char> buffer(bufferSize);

/* DPCT_ORIG 	cusparseStatus_t status = cusparseSpMV(handle, trans,
 * &alpha,mat.descr,dnvec_in.get(),&beta,dnvec_out.get(),cuda_datatype<T>(),CUSPARSE_SPMV_CSR_ALG2, buffer.data());*/
        int status = DPCT_CHECK_ERROR(dpct::sparse::spmv(handle->get_queue(), trans, &alpha, mat.descr, dnvec_in.get(),
                                                         &beta, dnvec_out.get(), cuda_datatype<T>()));

        cudaDeviceManager::Instance()->unlockSparseHandle();
/* DPCT_ORIG 	if (status != CUSPARSE_STATUS_SUCCESS){*/
        if (status != 0) {
                std::stringstream ss;
		ss << "Sparse Matrix Vector multiplication failed. Error: ";
		ss << gadgetron_getCusparseErrorString(status);
		throw cuda_error(ss.str());
	}
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
template <class T> static auto create_DnMat(cuNDArray<T>& mat) try {

/* DPCT_ORIG 	cusparseDnMatDescr_t dnmat;	*/
        std::shared_ptr<dpct::sparse::dense_matrix_desc> dnmat;
/* DPCT_ORIG cusparseCreateDnMat(&dnmat,mat.get_size(0),mat.get_size(1),mat.get_size(0),mat.data(),cuda_datatype<T>(),
 * CUSPARSE_ORDER_COL);*/
        dnmat = std::make_shared<dpct::sparse::dense_matrix_desc>(mat.get_size(0), mat.get_size(1), mat.get_size(0),
                                                                  mat.data(), cuda_datatype<T>(),
                                                                  oneapi::mkl::layout::col_major);
/* DPCT_ORIG 	auto deleter = [](cusparseDnMatDescr_t val){cusparseDestroyDnMat(val);};*/
        auto deleter = [](std::shared_ptr<dpct::sparse::dense_matrix_desc> val) {
                                                                                val.reset();
        };
        return std::unique_ptr<std::decay_t<decltype(*dnmat)>,decltype(deleter)>(dnmat,deleter);
	//return std::unique_ptr<std::decay_t<decltype(*dnmat)>,decltype(&cusparseDestroyDnMat)>(dnmat);
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template<class T> void Gadgetron::sparseMM(T alpha,T beta, const cuCsrMatrix<T> & mat, const cuNDArray<T> & mat_in, cuNDArray<T>& mat_out, bool adjoint) {

	if (mat_in.get_size(1) != mat_out.get_size(1)) throw std::runtime_error("In and out dense matrix must have same second dimension");
	if (mat_in.get_size(0) != mat.rows) throw std::runtime_error("Input matrix and sparse matrix have mismatched dimensions");
	if (mat_out.get_size(0) != mat.cols) throw std::runtime_error("Output matrix and sparse matrix have mismatched dimensions");

/* DPCT_ORIG 	cusparseOperation_t trans = adjoint ?  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE :
 * CUSPARSE_OPERATION_NON_TRANSPOSE;*/
        oneapi::mkl::transpose trans = adjoint ? oneapi::mkl::transpose::conjtrans : oneapi::mkl::transpose::nontrans;
        auto handle = cudaDeviceManager::Instance()->lockSparseHandle();

	auto dnmat_in = create_DnMat(const_cast<cuNDArray<T>&>(mat_in));
	auto dnmat_out = create_DnMat(mat_out);
	size_t bufferSize;
/* DPCT_ORIG 	CUSPARSE_CALL(cusparseSpMM_bufferSize(handle, trans, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
 * mat.descr, dnmat_in.get(), &beta, dnmat_out.get(), cuda_datatype<T>(),CUSPARSE_SPMM_CSR_ALG1, &bufferSize));*/
        CUSPARSE_CALL(DPCT_CHECK_ERROR(bufferSize = 0));
        cuNDArray<char> buffer(bufferSize);

/* DPCT_ORIG 	CUSPARSE_CALL(cusparseSpMM(handle, trans, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat.descr,
 * dnmat_in.get(), &beta, dnmat_out.get(), cuda_datatype<T>(),CUSPARSE_SPMM_CSR_ALG1, buffer.data()));*/
        CUSPARSE_CALL(DPCT_CHECK_ERROR(dpct::sparse::spmm(handle->get_queue(), trans, oneapi::mkl::transpose::nontrans,
                                                          &alpha, mat.descr, dnmat_in.get(), &beta, dnmat_out.get(),
                                                          cuda_datatype<T>())));
        cudaDeviceManager::Instance()->unlockSparseHandle();

}


template void Gadgetron::sparseMV<float>(float alpha,float beta, const cuCsrMatrix<float> & mat, const cuNDArray<float> & vec_in, cuNDArray<float>& vec_out, bool adjoint);
template void Gadgetron::sparseMV<double>(double alpha,double beta, const cuCsrMatrix<double> & mat, const cuNDArray<double> & vec_in, cuNDArray<double>& vec_out, bool adjoint);
template void Gadgetron::sparseMV<complext<float> >(complext<float> alpha,complext<float> beta, const cuCsrMatrix<complext<float> > & mat, const cuNDArray<complext<float> > & vec_in, cuNDArray<complext<float> >& vec_out, bool adjoint);
template void Gadgetron::sparseMV<complext<double> >(complext<double> alpha,complext<double> beta, const cuCsrMatrix<complext<double> > & mat, const cuNDArray<complext<double> > & vec_in, cuNDArray<complext<double> >& vec_out, bool adjoint);

template void Gadgetron::sparseMM<float>(float alpha,float beta, const cuCsrMatrix<float> & mat, const cuNDArray<float> & vec_in, cuNDArray<float>& vec_out, bool adjoint);
template void Gadgetron::sparseMM<double>(double alpha,double beta, const cuCsrMatrix<double> & mat, const cuNDArray<double> & vec_in, cuNDArray<double>& vec_out, bool adjoint);
template void Gadgetron::sparseMM<complext<float> >(complext<float> alpha,complext<float> beta, const cuCsrMatrix<complext<float> > & mat, const cuNDArray<complext<float> > & vec_in, cuNDArray<complext<float> >& vec_out, bool adjoint);
template void Gadgetron::sparseMM<complext<double> >(complext<double> alpha,complext<double> beta, const cuCsrMatrix<complext<double> > & mat, const cuNDArray<complext<double> > & vec_in, cuNDArray<complext<double> >& vec_out, bool adjoint);
