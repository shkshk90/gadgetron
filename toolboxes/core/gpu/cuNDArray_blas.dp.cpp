#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_blas.h"
#include "complext.h"
#include "GadgetronCuException.h"
#include "cudaDeviceManager.h"
#include <dpct/blas_utils.hpp>

#include <cmath>

#include <complex>

/* DPCT_ORIG #include <cublas_v2.h>*/

namespace Gadgetron{

/* DPCT_ORIG #define CUBLAS_CALL(fun) {cublasStatus_t err = fun; if (err != CUBLAS_STATUS_SUCCESS) {throw
 * cuda_error(gadgetron_getCublasErrorString(err));}}*/
#define CUBLAS_CALL(fun) { int err = fun; if (err != 0) { throw cuda_error(gadgetron_getCublasErrorString(err)); } }

  //NRM2
  //

/* DPCT_ORIG   template<class T> cublasStatus_t cublas_axpy(cublasHandle_t hndl, int n, const T* a , const T* x , int
 * incx,  T* y, int incy);*/
  template <class T>
  int cublas_axpy(dpct::blas::descriptor_ptr hndl, int n, const T* a, const T* x, int incx, T* y, int incy);
/* DPCT_ORIG   template<class T> cublasStatus_t cublas_dot(cublasHandle_t, int, const T*, int, const  T*, int, T*, bool
 * cc = true);*/
  template <class T> int cublas_dot(dpct::blas::descriptor_ptr, int, const T*, int, const T*, int, T*, bool cc = true);
/* DPCT_ORIG   template<class T> cublasStatus_t cublas_nrm2(cublasHandle_t, int, const T*, int, typename
 * realType<T>::Type *result);*/
  template <class T> int cublas_nrm2(dpct::blas::descriptor_ptr, int, const T*, int, typename realType<T>::Type* result);
/* DPCT_ORIG   template<class T> cublasStatus_t cublas_amax(cublasHandle_t handle, int n,const T *x, int incx, int
 * *result);*/
  template <class T> int cublas_amax(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, int* result);
/* DPCT_ORIG   template<class T> cublasStatus_t cublas_amin(cublasHandle_t handle, int n,const T *x, int incx, int
 * *result);*/
  template <class T> int cublas_amin(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, int* result);
/* DPCT_ORIG   template<class T> cublasStatus_t cublas_asum(cublasHandle_t handle, int n,const T *x, int incx, typename
 * realType<T>::Type *result);*/
  template <class T>
  int cublas_asum(dpct::blas::descriptor_ptr handle, int n, const T* x, int incx, typename realType<T>::Type* result);

/* DPCT_ORIG   template<> cublasStatus_t cublas_nrm2<float>(cublasHandle_t hndl, int n, const float*  x, int inc, float*
 * res){*/
  template <> int cublas_nrm2<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int inc, float* res) try {
/* DPCT_ORIG     return cublasSnrm2(hndl,n,x,inc,res);*/
    /*
    DPCT1034:189: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_float_out res_wrapper_ct4(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, inc, res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_nrm2<double>(cublasHandle_t hndl, int n, const double*  x, int inc,
 * double* res){*/
  template <> int cublas_nrm2<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int inc, double* res) try {
/* DPCT_ORIG     return cublasDnrm2(hndl,n,x,inc,res);*/
    /*
    DPCT1034:190: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_double_out res_wrapper_ct4(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, x, inc, res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_nrm2<float_complext>(cublasHandle_t hndl, int n, const float_complext*
 * x, int inc, float* res){*/
  template <>
  int cublas_nrm2<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int inc,
                                  float* res) try {
/* DPCT_ORIG     return cublasScnrm2(hndl,n,(const cuComplex*)x,inc,res);*/
    /*
    DPCT1034:191: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_float_out res_wrapper_ct4(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (std::complex<float>*)x, inc,
                                          res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_nrm2<double_complext>(cublasHandle_t hndl, int n, const double_complext*
 * x, int inc, double* res){*/
  template <>
  int cublas_nrm2<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int inc,
                                   double* res) try {
/* DPCT_ORIG     return cublasDznrm2(hndl,n,(const cuDoubleComplex*) x,inc,res);*/
    /*
    DPCT1034:192: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_double_out res_wrapper_ct4(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::nrm2(hndl->get_queue(), n, (std::complex<double>*)x, inc,
                                          res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //DOT
  //

/* DPCT_ORIG   template<> cublasStatus_t cublas_dot<float>(cublasHandle_t hndl, int n , const float* x , int incx, const
 * float* y , int incy, float* res, bool cc){*/
  template <>
  int cublas_dot<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, const float* y, int incy,
                        float* res, bool cc) try {
/* DPCT_ORIG     return cublasSdot( hndl, n, x, incx, y, incy, res);*/
    /*
    DPCT1034:193: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_float_out res_wrapper_ct6(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_dot<double>(cublasHandle_t hndl, int n , const double* x , int incx,
 * const  double* y , int incy, double* res, bool cc){*/
  template <>
  int cublas_dot<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, const double* y, int incy,
                         double* res, bool cc) try {
/* DPCT_ORIG     return cublasDdot( hndl, n, x, incx, y, incy, res);*/
    /*
    DPCT1034:194: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_double_out res_wrapper_ct6(hndl->get_queue(), res);
    oneapi::mkl::blas::column_major::dot(hndl->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_dot<float_complext>(cublasHandle_t hndl, int n , const float_complext* x
 * ,*/
  template <>
  int cublas_dot<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                 const float_complext* y, int incy, float_complext* res, bool cc) try {
    if(cc)
/* DPCT_ORIG       return cublasCdotc( hndl, n, (const cuComplex*) x, incx, (const cuComplex*) y, incy, (cuComplex*)
 * res);*/
      /*
      DPCT1034:195: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite
      this code.
      */
      return [&]() {
      dpct::blas::wrapper_float2_out res_wrapper_ct6(hndl->get_queue(), (sycl::float2*)res);
      oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                            (std::complex<float>*)y, incy,
                                            (std::complex<float>*)res_wrapper_ct6.get_ptr());
      return 0;
      }();
    else
/* DPCT_ORIG       return cublasCdotu( hndl, n, (const cuComplex*) x, incx, (const cuComplex*) y, incy, (cuComplex*)
 * res);*/
      /*
      DPCT1034:196: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite
      this code.
      */
      return [&]() {
      dpct::blas::wrapper_float2_out res_wrapper_ct6(hndl->get_queue(), (sycl::float2*)res);
      oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                            (std::complex<float>*)y, incy,
                                            (std::complex<float>*)res_wrapper_ct6.get_ptr());
      return 0;
      }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_dot<double_complext>(cublasHandle_t hndl, int n , const double_complext*
 * x ,*/
  template <>
  int cublas_dot<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                  const double_complext* y, int incy, double_complext* res, bool cc) try {
    if(cc)
/* DPCT_ORIG       return cublasZdotc( hndl, n, (const cuDoubleComplex*) x, incx, (const cuDoubleComplex*) y, incy,
 * (cuDoubleComplex*) res);*/
      /*
      DPCT1034:197: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite
      this code.
      */
      return [&]() {
      dpct::blas::wrapper_double2_out res_wrapper_ct6(hndl->get_queue(), (sycl::double2*)res);
      oneapi::mkl::blas::column_major::dotc(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                            (std::complex<double>*)y, incy,
                                            (std::complex<double>*)res_wrapper_ct6.get_ptr());
      return 0;
      }();
    else
/* DPCT_ORIG       return cublasZdotu( hndl, n, (const cuDoubleComplex*) x, incx, (const cuDoubleComplex*) y, incy,
 * (cuDoubleComplex*) res);*/
      /*
      DPCT1034:198: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite
      this code.
      */
      return [&]() {
      dpct::blas::wrapper_double2_out res_wrapper_ct6(hndl->get_queue(), (sycl::double2*)res);
      oneapi::mkl::blas::column_major::dotu(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                            (std::complex<double>*)y, incy,
                                            (std::complex<double>*)res_wrapper_ct6.get_ptr());
      return 0;
      }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  // AXPY
  //

/* DPCT_ORIG   template<> cublasStatus_t cublas_axpy<float>(cublasHandle_t hndl , int n , const float* a , const float*
 * x , int incx ,  float* y , int incy){*/
  template <>
  int cublas_axpy<float>(dpct::blas::descriptor_ptr hndl, int n, const float* a, const float* x, int incx, float* y,
                         int incy) try {
/* DPCT_ORIG     return cublasSaxpy(hndl,n,a,x,incx,y,incy);*/
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value(a, hndl->get_queue()), x, incx, y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_axpy<double>(cublasHandle_t hndl , int n , const double* a , const
 * double* x , int incx ,  double* y , int incy){*/
  template <>
  int cublas_axpy<double>(dpct::blas::descriptor_ptr hndl, int n, const double* a, const double* x, int incx, double* y,
                          int incy) try {
/* DPCT_ORIG     return cublasDaxpy(hndl,n,a,x,incx,y,incy);*/
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value(a, hndl->get_queue()), x, incx, y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_axpy<float_complext>(cublasHandle_t hndl , int n , const float_complext*
 * a , const float_complext* x , int incx ,  float_complext* y , int incy){*/
  template <>
  int cublas_axpy<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* a,
                                  const float_complext* x, int incx, float_complext* y, int incy) try {
/* DPCT_ORIG     return cublasCaxpy(hndl,n,(const cuComplex*) a, (const cuComplex*) x,incx, (cuComplex*)y,incy);*/
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value((const sycl::float2*)a, hndl->get_queue()), (std::complex<float>*)x, incx,
        (std::complex<float>*)y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_axpy<double_complext>(cublasHandle_t hndl , int n , const
 * double_complext* a , const double_complext* x , int incx ,  double_complext* y , int incy){*/
  template <>
  int cublas_axpy<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* a,
                                   const double_complext* x, int incx, double_complext* y, int incy) try {
/* DPCT_ORIG     return cublasZaxpy(hndl,n,(const cuDoubleComplex*) a, (const cuDoubleComplex*) x,incx,
 * (cuDoubleComplex*)y,incy);*/
    return DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::axpy(
        hndl->get_queue(), n, dpct::get_value((const sycl::double2*)a, hndl->get_queue()), (std::complex<double>*)x,
        incx, (std::complex<double>*)y, incy));
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //SUM
  //

/* DPCT_ORIG   template<> cublasStatus_t cublas_asum<float>(cublasHandle_t hndl, int n,const float *x, int incx, float
 * *result){*/
  template <>
  int cublas_asum<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, float* result) try {
/* DPCT_ORIG     return cublasSasum(hndl,n,x,incx,result);*/
    /*
    DPCT1034:199: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_float_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_asum<double>(cublasHandle_t hndl, int n,const double *x, int incx,
 * double *result){*/
  template <>
  int cublas_asum<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, double* result) try {
/* DPCT_ORIG     return cublasDasum(hndl,n,x,incx,result);*/
    /*
    DPCT1034:200: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_double_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_asum<float_complext>(cublasHandle_t hndl, int n,const float_complext *x,
 * int incx, float *result){*/
  template <>
  int cublas_asum<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  float* result) try {
/* DPCT_ORIG     return cublasScasum(hndl,n,(const cuComplex*) x,incx,result);*/
    /*
    DPCT1034:201: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_float_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                          res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_asum<double_complext>(cublasHandle_t hndl, int n,const double_complext
 * *x, int incx, double *result){*/
  template <>
  int cublas_asum<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   double* result) try {
/* DPCT_ORIG     return cublasDzasum(hndl,n,(const cuDoubleComplex*) x,incx,result);*/
    /*
    DPCT1034:202: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_double_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::asum(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                          res_wrapper_ct4.get_ptr());
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //AMIN
  //

/* DPCT_ORIG   template<> cublasStatus_t cublas_amin<float>(cublasHandle_t hndl, int n,const float *x, int incx, int
 * *result){*/
  template <> int cublas_amin<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, int* result) try {
/* DPCT_ORIG     return cublasIsamin(hndl,n,x,incx,result);*/
    /*
    DPCT1034:203: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(),
                                           oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amin<double>(cublasHandle_t hndl, int n,const double *x, int incx, int
 * *result){*/
  template <>
  int cublas_amin<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, int* result) try {
/* DPCT_ORIG     return cublasIdamin(hndl,n,x,incx,result);*/
    /*
    DPCT1034:204: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(),
                                           oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amin<float_complext>(cublasHandle_t hndl, int n,const float_complext *x,
 * int incx, int *result){*/
  template <>
  int cublas_amin<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  int* result) try {
/* DPCT_ORIG     return cublasIcamin(hndl,n, (const cuComplex* ) x,incx,result);*/
    /*
    DPCT1034:205: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                           res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amin<double_complext>(cublasHandle_t hndl, int n,const double_complext
 * *x, int incx, int *result){*/
  template <>
  int cublas_amin<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   int* result) try {
/* DPCT_ORIG     return cublasIzamin(hndl,n, (const cuDoubleComplex* ) x,incx,result);*/
    /*
    DPCT1034:206: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamin(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                           res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //AMAX
  //

/* DPCT_ORIG   template<> cublasStatus_t cublas_amax<float>(cublasHandle_t hndl, int n,const float *x, int incx, int
 * *result){*/
  template <> int cublas_amax<float>(dpct::blas::descriptor_ptr hndl, int n, const float* x, int incx, int* result) try {
/* DPCT_ORIG     return cublasIsamax(hndl,n,x,incx,result);*/
    /*
    DPCT1034:207: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(),
                                           oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amax<double>(cublasHandle_t hndl, int n,const double *x, int incx, int
 * *result){*/
  template <>
  int cublas_amax<double>(dpct::blas::descriptor_ptr hndl, int n, const double* x, int incx, int* result) try {
/* DPCT_ORIG     return cublasIdamax(hndl,n,x,incx,result);*/
    /*
    DPCT1034:208: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(),
                                           oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amax<float_complext>(cublasHandle_t hndl, int n,const float_complext *x,
 * int incx, int *result){*/
  template <>
  int cublas_amax<float_complext>(dpct::blas::descriptor_ptr hndl, int n, const float_complext* x, int incx,
                                  int* result) try {
/* DPCT_ORIG     return cublasIcamax(hndl,n, (const cuComplex* ) x,incx,result);*/
    /*
    DPCT1034:209: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (std::complex<float>*)x, incx,
                                           res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template<> cublasStatus_t cublas_amax<double_complext>(cublasHandle_t hndl, int n,const double_complext
 * *x, int incx, int *result){*/
  template <>
  int cublas_amax<double_complext>(dpct::blas::descriptor_ptr hndl, int n, const double_complext* x, int incx,
                                   int* result) try {
/* DPCT_ORIG     return cublasIzamax(hndl,n, (const cuDoubleComplex* ) x,incx,result);*/
    /*
    DPCT1034:210: Migrated API does not return an error code. 0 is returned in the lambda. You may need to rewrite this
    code.
    */
    return [&]() {
    dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(hndl->get_queue(), result);
    oneapi::mkl::blas::column_major::iamax(hndl->get_queue(), n, (std::complex<double>*)x, incx,
                                           res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
    return 0;
    }();
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> typename realType<T>::Type nrm2(cuNDArray<T>* arr, size_t batchSize) try {
    if( arr == 0x0 )
        throw std::runtime_error("Gadgetron::nrm2(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    typedef typename realType<T>::Type REAL;
    REAL ret = 0;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = arr->get_number_of_elements() / batchSize + 1;
    int remainder = arr->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    for (int ii = 0; ii < num_splits; ii++) {

        REAL val;

        CUBLAS_CALL(cublas_nrm2<T>(handle,
                                   (ii == num_splits - 1) ? remainder : batchSize, // n number of elements
                                   arr->get_data_ptr() + batchSize * ii, 1, &val));


        if (ii == 0)
            ret = val;
        else
/* DPCT_ORIG             ret = sqrt(pow(ret, 2) + pow(val, 2));*/
            ret = sqrt(ret * ret + val * val);
    }

    cudaDeviceManager::Instance()->unlockHandle(device);

    return ret;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> T dot(cuNDArray<T>* arr1, cuNDArray<T>* arr2, size_t batchSize, bool cc) try {
    if (arr1 == 0x0 || arr2 == 0x0)
        throw std::runtime_error("Gadgetron::dot(): Invalid input array");

    if (arr1->get_number_of_elements() != arr2->get_number_of_elements())
        throw std::runtime_error("Gadgetron::dot(): Array sizes mismatch");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    T ret = 0;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = arr1->get_number_of_elements() / batchSize + 1;
    int remainder = arr1->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    for (int ii = 0; ii < num_splits; ii++) {

        T val;
        CUBLAS_CALL(cublas_dot(handle,
                               (ii == num_splits - 1) ? remainder : batchSize, // n number of elements
                               arr1->get_data_ptr() + batchSize * ii, 1, arr2->get_data_ptr() + batchSize * ii, 1, &val,
                               cc));

        if (ii == 0)
            ret = val;
        else
            ret += val;
    }
    
    cudaDeviceManager::Instance()->unlockHandle(device);

    return ret;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> void axpy(T a, cuNDArray<T>* x, cuNDArray<T>* y, size_t batchSize) try {
    if (x == 0x0 || y == 0x0)
        throw std::runtime_error("Gadgetron::axpy(): Invalid input array");

    if (x->get_number_of_elements() != y->get_number_of_elements())
        throw std::runtime_error("Gadgetron::axpy(): Array sizes mismatch");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays
    int num_splits = x->get_number_of_elements() / batchSize + 1;
    int remainder = x->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    for (int ii = 0; ii < num_splits; ii++) {

        CUBLAS_CALL(cublas_axpy(handle,
                                (ii == num_splits - 1) ? remainder : batchSize, &a, x->get_data_ptr() + batchSize * ii, 1,
                                y->get_data_ptr() + batchSize * ii, 1));

    }
    cudaDeviceManager::Instance()->unlockHandle(device);
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> void axpy(T a, cuNDArray<complext<T>>* x, cuNDArray<complext<T>>* y, size_t batchSize ) { axpy(complext<T>(a), x, y, batchSize); }

  template <class T> typename realType<T>::Type asum(cuNDArray<T>* x, size_t batchSize) try {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::asum(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    typename realType<T>::Type result = 0;
    
    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = x->get_number_of_elements() / batchSize + 1;
    int remainder = x->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);

    for (int ii = 0; ii < num_splits; ii++) {
        typename realType<T>::Type interim_result;

        CUBLAS_CALL(cublas_asum(handle,
                                (ii == num_splits - 1) ? remainder : batchSize, // n number of elements
                                x->get_data_ptr() + batchSize * ii, 1, &interim_result));


        if (ii == 0)
            result = interim_result;
        else
            result += interim_result;
    }
    cudaDeviceManager::Instance()->unlockHandle(device);

    return result;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> size_t amin(cuNDArray<T>* x, size_t batchSize ) 
  {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::amin(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t result = x->get_number_of_elements() + 1;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = x->get_number_of_elements() / batchSize + 1;
    int remainder = x->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);
    T saved_value = T(0);
    
    // Since operators cannot be overloaded inside functions - this lambda function should do the comparison and mimic cublas comparisons
    auto lessThan = [](auto c1, auto c2)
    {
        if constexpr (std::is_same_v<decltype(c1),float> || std::is_same_v<decltype(c1),double>)
            return (abs(c1) < abs(c2));
        else if constexpr (std::is_same_v<decltype(c1),complext<float>> || std::is_same_v<decltype(c1),complext<double>>)
            return (abs(c1.real()) + abs(c1.imag())) < (abs(c2.real()) + abs(c2.imag()));
    };

    for (int ii = 0; ii < num_splits; ii++) {

        int interim_result;

        CUBLAS_CALL(cublas_amin(handle,
                                (ii == num_splits - 1) ? remainder : batchSize, // n number of elements
                                x->get_data_ptr() + batchSize * ii, 1, &interim_result));

        auto interim_value = (*x)[batchSize * ii + (size_t)interim_result - 1];

        if (ii == 0)
            result = (size_t)interim_result - 1;
        else if (lessThan((interim_value), (saved_value)))
            result = batchSize * ii + (size_t)interim_result - 1;
        
        saved_value = (*x)[result];

    }
    cudaDeviceManager::Instance()->unlockHandle(device);
    if (result > x->get_number_of_elements()) {
        throw std::runtime_error("Gadgetron::amin(): computed index is out of bounds");
    }

    return result; // result - 1;
  }

  template <class T> size_t amax(cuNDArray<T>* x, size_t batchSize ) 
  {
    if (x == 0x0)
        throw std::runtime_error("Gadgetron::amax(): Invalid input array");

    int device = cudaDeviceManager::Instance()->getCurrentDevice();
    size_t result = x->get_number_of_elements() + 1;

    // If number of elements in the array is greater than batchSize break it up and perform calculations this is done to
    // support large data arrays

    int num_splits = x->get_number_of_elements() / batchSize + 1;
    int remainder = x->get_number_of_elements() - batchSize * (num_splits - 1);
    auto handle = cudaDeviceManager::Instance()->lockHandle(device);
    T saved_value = T(0);

    // Since operators cannot be overloaded inside functions - this lambda function should do the comparison and mimic cublas comparisons
    auto greaterThan = [](auto c1, auto c2)
    {
        if constexpr (std::is_same_v<decltype(c1),float> || std::is_same_v<decltype(c1),double>)
            return (abs(c1) > abs(c2));
        else if constexpr (std::is_same_v<decltype(c1),complext<float>> || std::is_same_v<decltype(c1),complext<double>>)
            return (abs(c1.real()) + abs(c1.imag())) > (abs(c2.real()) + abs(c2.imag()));
    };

    for (int ii = 0; ii < num_splits; ii++) {

        int interim_result;
        CUBLAS_CALL(cublas_amax(
            handle,
            (ii == num_splits - 1) ? remainder : batchSize, // n number of elements (int)x->get_number_of_elements(),
            x->get_data_ptr() + batchSize * ii, 1, &interim_result));
       
         auto interim_value = (*x)[batchSize * ii + (size_t)interim_result - 1];

        if (ii == 0)
            result = (size_t)interim_result - 1;
        else if (greaterThan((interim_value), (saved_value)))
            result = batchSize * ii + (size_t)interim_result - 1;
        
        saved_value = (*x)[result];

    }
            cudaDeviceManager::Instance()->unlockHandle(device);
    if (result > x->get_number_of_elements()) {
        throw std::runtime_error("Gadgetron::amax(): computed index is out of bounds");
    }

    return result; //(size_t)result - 1;
  }

/* DPCT_ORIG   std::string gadgetron_getCublasErrorString(cublasStatus_t err) */
  std::string gadgetron_getCublasErrorString(int err)
  {
    switch (err) {
/* DPCT_ORIG     case CUBLAS_STATUS_NOT_INITIALIZED:*/
    case 1:
        return "NOT INITIALIZED";
/* DPCT_ORIG     case CUBLAS_STATUS_ALLOC_FAILED:*/
    case 3:
        return "ALLOC FAILED";
/* DPCT_ORIG     case CUBLAS_STATUS_INVALID_VALUE:*/
    case 7:
        return "INVALID VALUE";
/* DPCT_ORIG     case CUBLAS_STATUS_ARCH_MISMATCH:*/
    case 8:
        return "ARCH MISMATCH";
/* DPCT_ORIG     case CUBLAS_STATUS_MAPPING_ERROR:*/
    case 11:
        return "MAPPING ERROR";
/* DPCT_ORIG     case CUBLAS_STATUS_EXECUTION_FAILED:*/
    case 13:
        return "EXECUTION FAILED";
/* DPCT_ORIG     case CUBLAS_STATUS_INTERNAL_ERROR:*/
    case 14:
        return "INTERNAL ERROR";
/* DPCT_ORIG     case CUBLAS_STATUS_SUCCESS:*/
    case 0:
        return "SUCCES";
    default:
        return "UNKNOWN CUBLAS ERROR";
    }
  }

  //
  // Instantiation
  //

  template float dot(cuNDArray<float>*, cuNDArray<float>*, size_t, bool);
  template float nrm2(cuNDArray<float>*, size_t);
  template void axpy(float, cuNDArray<float>*, cuNDArray<float>*, size_t);
  template size_t amin(cuNDArray<float>*, size_t);
  template size_t amax(cuNDArray<float>*, size_t);
  template float asum(cuNDArray<float>*, size_t);

  template double dot(cuNDArray<double>*, cuNDArray<double>*, size_t, bool);
  template double nrm2(cuNDArray<double>*, size_t);
  template void axpy(double, cuNDArray<double>*, cuNDArray<double>*, size_t);
  template size_t amin(cuNDArray<double>*, size_t);
  template size_t amax(cuNDArray<double>*, size_t);
  template double asum(cuNDArray<double>*, size_t);

  template float_complext dot(cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t, bool);
  template float nrm2(cuNDArray<float_complext>*, size_t);
  template void axpy(float_complext, cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t);
  template void axpy(float, cuNDArray<float_complext>*, cuNDArray<float_complext>*, size_t);
  template size_t amin(cuNDArray<float_complext>*, size_t);
  template size_t amax(cuNDArray<float_complext>*, size_t);
  template float asum(cuNDArray<float_complext>*, size_t);

  template double_complext dot(cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t, bool);
  template double nrm2(cuNDArray<double_complext>*, size_t);
  template void axpy(double_complext, cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t);
  template void axpy(double, cuNDArray<double_complext>*, cuNDArray<double_complext>*, size_t);
  template size_t amin(cuNDArray<double_complext>*, size_t);
  template size_t amax(cuNDArray<double_complext>*, size_t);
  template double asum(cuNDArray<double_complext>*, size_t);
} // namespace Gadgetron
