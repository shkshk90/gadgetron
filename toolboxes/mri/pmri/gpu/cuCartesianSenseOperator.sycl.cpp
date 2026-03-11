#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuCartesianSenseOperator.h"
#include "cuNDFFT.h"

#include <sstream>
#include <cmath>

using namespace Gadgetron;

template<class REAL> void 
sample_array_kernel( const complext<REAL> * __restrict__ in, complext<REAL> * __restrict__ out,
		     unsigned int *idx, 
		     unsigned int image_elements,
		     unsigned int samples,
		     unsigned int coils )
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  unsigned int idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  if (idx_in < samples) {
    for (unsigned int i = 0; i < coils; i++) {
      out[idx_in + i*samples]._real += in[idx[idx_in] + i*image_elements]._real;
      out[idx_in + i*samples]._imag += in[idx[idx_in] + i*image_elements]._imag;
    }
  }
}

template<class REAL> void 
insert_samples_kernel( const complext<REAL> * __restrict__ in, complext<REAL> * __restrict__ out,
		       unsigned int *idx, 
		       unsigned int image_elements,
		       unsigned int samples,
		       unsigned int coils )
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  unsigned int idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  if (idx_in < samples) {
    for (unsigned int i = 0; i < coils; i++) {
      out[idx[idx_in] + i*image_elements]._real += in[idx_in + i*samples]._real;
      out[idx[idx_in] + i*image_elements]._imag += in[idx_in + i*samples]._imag;
    }
  }
}

template<class REAL, unsigned int D> void
cuCartesianSenseOperator<REAL,D>::mult_M( cuNDArray< complext<REAL> > *in, cuNDArray< complext<REAL> > *out, bool accumulate )
{
  if (!(in->dimensions_equal(this->get_domain_dimensions())) || !(out->dimensions_equal(this->get_codomain_dimensions())) ) {
    throw std::runtime_error("cuCartesianSenseOperator::mult_M dimensions mismatch");
  }
  
  std::vector<size_t> full_dimensions = this->get_domain_dimensions();
  full_dimensions.push_back(this->ncoils_);
  cuNDArray< complext<REAL> > tmp(full_dimensions);

  this->mult_csm(in,&tmp);

  std::vector<size_t> ft_dims;
   for (unsigned int i = 0; i < this->get_domain_dimensions().size(); i++) {
     ft_dims.push_back(i);
   }

   cuNDFFT<REAL>::instance()->fft(&tmp, &ft_dims);
   
  if (!accumulate) 
    clear(out);

  dpct::dim3 blockDim(512, 1, 1);
  dpct::dim3 gridDim((unsigned int)std::ceil((double)idx_->get_number_of_elements() / blockDim.x), 1, 1);
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
      auto tmp_get_data_ptr_ct0 = tmp.get_data_ptr();
      auto out_get_data_ptr_ct1 = out->get_data_ptr();
      auto idx__get_data_ptr_ct2 = idx_->get_data_ptr();
      auto in_get_number_of_elements_ct3 = in->get_number_of_elements();
      auto idx__get_number_of_elements_ct4 = idx_->get_number_of_elements();
      auto this_ncoils__ct5 = this->ncoils_;

      cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

      cgh.parallel_for<dpct_kernel_name<class sample_array_kernel_20d023, REAL, dpct_kernel_scalar<D>>>(
          sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
            sample_array_kernel<REAL>(tmp_get_data_ptr_ct0, out_get_data_ptr_ct1, idx__get_data_ptr_ct2,
                                      in_get_number_of_elements_ct3, idx__get_number_of_elements_ct4, this_ncoils__ct5);
          });
    });
  }
  /*
  DPCT1010:35: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
  call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 err = 0;
  /*
  DPCT1000:34: Error handling if-stmt was detected but could not be rewritten.
  */
  if (err != 0) {
    std::stringstream ss;
    /*
    DPCT1001:33: The statement could not be removed.
    */
    ss << "cuCartesianSenseOperator::mult_M : Unable to sample data: " <<
        /*
        DPCT1009:36: SYCL reports errors using exceptions and does not use error codes. Please replace the
        "get_error_string_dummy(...)" with a real error-handling function.
        */
        dpct::get_error_string_dummy(err);
    throw cuda_error(ss.str());
  }
}

template<class REAL, unsigned int D> void
cuCartesianSenseOperator<REAL,D>::mult_MH(cuNDArray< complext<REAL> > *in, cuNDArray< complext<REAL> > *out, bool accumulate)
{
  if (!(out->dimensions_equal(this->get_domain_dimensions())) || 
      !(in->dimensions_equal(this->get_codomain_dimensions())) ) {
    throw std::runtime_error( "cuCartesianSenseOperator::mult_MH dimensions mismatch");

  }

  std::vector<size_t> tmp_dimensions = this->get_domain_dimensions();
  tmp_dimensions.push_back(this->ncoils_);

  cuNDArray< complext<REAL> > tmp(tmp_dimensions);
  clear(&tmp);

  dpct::dim3 blockDim(512, 1, 1);
  dpct::dim3 gridDim((unsigned int)std::ceil((double)idx_->get_number_of_elements() / blockDim.x), 1, 1);
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
      auto in_get_data_ptr_ct0 = in->get_data_ptr();
      auto tmp_get_data_ptr_ct1 = tmp.get_data_ptr();
      auto idx__get_data_ptr_ct2 = idx_->get_data_ptr();
      auto out_get_number_of_elements_ct3 = out->get_number_of_elements();
      auto idx__get_number_of_elements_ct4 = idx_->get_number_of_elements();
      auto this_ncoils__ct5 = this->ncoils_;

      cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

      cgh.parallel_for<dpct_kernel_name<class insert_samples_kernel_58879d, REAL, dpct_kernel_scalar<D>>>(
          sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
            insert_samples_kernel<REAL>(in_get_data_ptr_ct0, tmp_get_data_ptr_ct1, idx__get_data_ptr_ct2,
                                        out_get_number_of_elements_ct3, idx__get_number_of_elements_ct4,
                                        this_ncoils__ct5);
          });
    });
  }

  /*
  DPCT1010:39: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
  call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 err = 0;
  /*
  DPCT1000:38: Error handling if-stmt was detected but could not be rewritten.
  */
  if (err != 0) {
    std::stringstream ss;
    /*
    DPCT1001:37: The statement could not be removed.
    */
    ss << "cuCartesianSenseOperator::mult_EM : Unable to insert samples into array: " <<
        /*
        DPCT1009:40: SYCL reports errors using exceptions and does not use error codes. Please replace the
        "get_error_string_dummy(...)" with a real error-handling function.
        */
        dpct::get_error_string_dummy(err);
    throw cuda_error(ss.str());
  }


  std::vector<size_t> ft_dims;
  for (unsigned int i = 0; i < this->get_domain_dimensions().size(); i++) {
    ft_dims.push_back(i);
  }

  cuNDFFT<REAL>::instance()->ifft(&tmp, &ft_dims);

  if (!accumulate) 
    clear(out);
  
  this->mult_csm_conj_sum(&tmp,out);
}

//
// Instantiations
//
namespace Gadgetron {

  template class EXPORTGPUPMRI cuCartesianSenseOperator<float,1>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<float,2>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<float,3>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<float,4>;
  
  template class EXPORTGPUPMRI cuCartesianSenseOperator<double,1>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<double,2>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<double,3>;
  template class EXPORTGPUPMRI cuCartesianSenseOperator<double,4>;
}

