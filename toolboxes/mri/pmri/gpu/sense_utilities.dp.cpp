#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sense_utilities.h"
#include "vector_td_utilities.h"
#include <sstream>

namespace Gadgetron{

/* DPCT_ORIG   template<class REAL> __global__ void
  mult_csm_kernel( const complext<REAL> * __restrict__ in, complext<REAL> * __restrict__ out, complext<REAL> *csm,
                   size_t image_elements, unsigned int nframes, unsigned int ncoils )*/
  template <class REAL>
  void mult_csm_kernel(const complext<REAL>* __restrict__ in, complext<REAL>* __restrict__ out, complext<REAL>* csm,
                       size_t image_elements, unsigned int nframes, unsigned int ncoils)
  {
/* DPCT_ORIG     unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if( idx < image_elements) {
/* DPCT_ORIG       complext<REAL> _in = in[idx+blockIdx.y*image_elements];*/
      complext<REAL> _in = in[idx + item_ct1.get_group(1) * image_elements];
      for( unsigned int i=0; i<ncoils; i++) {
/* DPCT_ORIG 	out[idx + blockIdx.y*image_elements + i*image_elements*nframes] =  _in * csm[idx+i*image_elements];*/
        out[idx + item_ct1.get_group(1) * image_elements + i * image_elements * nframes] =
            _in * csm[idx + i * image_elements];
      }
    }
  }

  template <class REAL, unsigned int D>
  void csm_mult_M(cuNDArray<complext<REAL>>* in, cuNDArray<complext<REAL>>* out, cuNDArray<complext<REAL>>* csm) try {
    int device;
/* DPCT_ORIG     if( cudaGetDevice( &device ) != cudaSuccess ){*/
    if (DPCT_CHECK_ERROR(device = dpct::get_current_device_id()) != 0) {
      throw cuda_error( "mult_csm: unable to query current device");
    }
  
    if( !in || in->get_device() != device || !out || out->get_device() != device || !csm || csm->get_device() != device ){
      throw cuda_error("mult_csm: array not residing current device");
    }
  
    if( in->get_number_of_dimensions() < D  || in->get_number_of_dimensions() > D+1 ){
      throw std::runtime_error("mult_csm: unexpected input dimensionality");
    }
  
    if( in->get_number_of_dimensions() > out->get_number_of_dimensions() ){
      throw std::runtime_error("mult_csm: input dimensionality cannot exceed output dimensionality");
    }

    if( csm->get_number_of_dimensions() != D+1 ) {
      throw std::runtime_error("mult_csm: input dimensionality of csm not as expected");
    }

    unsigned int num_image_elements = 1;
    for( unsigned int d=0; d<D; d++ )
      num_image_elements *= in->get_size(d);
  
    unsigned int num_frames = in->get_number_of_elements() / num_image_elements;

/* DPCT_ORIG     dim3 blockDim(256);*/
    dpct::dim3 blockDim(256);
/* DPCT_ORIG     dim3 gridDim((num_image_elements+blockDim.x-1)/blockDim.x, num_frames);*/
    dpct::dim3 gridDim((num_image_elements + blockDim.x - 1) / blockDim.x, num_frames);

/* DPCT_ORIG     mult_csm_kernel<REAL><<< gridDim, blockDim >>>
      ( in->get_data_ptr(), out->get_data_ptr(), csm->get_data_ptr(), num_image_elements, num_frames, csm->get_size(D)
   );*/
    /*
    DPCT1049:62: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct0 = in->get_data_ptr();
            auto out_get_data_ptr_ct1 = out->get_data_ptr();
            auto csm_get_data_ptr_ct2 = csm->get_data_ptr();
            auto csm_get_size_D_ct5 = csm->get_size(D);

            cgh.parallel_for<dpct_kernel_name<class mult_csm_kernel_3fe95a, REAL>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    mult_csm_kernel<REAL>(in_get_data_ptr_ct0, out_get_data_ptr_ct1, csm_get_data_ptr_ct2,
                                          num_image_elements, num_frames, csm_get_size_D_ct5);
                });
        });
    }

/* DPCT_ORIG     cudaError_t err = cudaGetLastError();*/
    /*
    DPCT1010:240: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
    call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 err = 0;
/* DPCT_ORIG     if( err != cudaSuccess ){*/
    /*
    DPCT1000:239: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
      std::stringstream ss;
      /*
      DPCT1001:238: The statement could not be removed.
      */
      ss << "mult_csm: unable to multiply with coil sensitivities: " <<
          /* DPCT_ORIG 	cudaGetErrorString(err);*/
          /*
          DPCT1009:241: SYCL reports errors using exceptions and does not use error codes. Please replace the
          "get_error_string_dummy(...)" with a real error-handling function.
          */
          dpct::get_error_string_dummy(err);
      throw cuda_error(ss.str());

    }
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

/* DPCT_ORIG   template <class REAL> __global__ void
  mult_csm_conj_sum_kernel(const  complext<REAL> * __restrict__ in, complext<REAL> * __restrict__ out, const
  complext<REAL> * __restrict__ csm, size_t image_elements, unsigned int nframes, unsigned int ncoils )*/
  template <class REAL>
  void mult_csm_conj_sum_kernel(const complext<REAL>* __restrict__ in, complext<REAL>* __restrict__ out,
                                const complext<REAL>* __restrict__ csm, size_t image_elements, unsigned int nframes,
                                unsigned int ncoils)
  {
/* DPCT_ORIG     unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if( idx < image_elements ) {
      complext<REAL> _out =complext<REAL>(0);
      for( unsigned int i = 0; i < ncoils; i++ ) {
/* DPCT_ORIG 	_out += in[idx+blockIdx.y*image_elements+i*nframes*image_elements] * conj(csm[idx+i*image_elements]);*/
        _out += in[idx + item_ct1.get_group(1) * image_elements + i * nframes * image_elements] *
                conj(csm[idx + i * image_elements]);
      }
/* DPCT_ORIG       out[idx+blockIdx.y*image_elements] = _out;*/
      out[idx + item_ct1.get_group(1) * image_elements] = _out;
    }
  }

  template <class REAL, unsigned int D>
  void csm_mult_MH(cuNDArray<complext<REAL>>* in, cuNDArray<complext<REAL>>* out, cuNDArray<complext<REAL>>* csm) try {
    int device;
/* DPCT_ORIG     if( cudaGetDevice( &device ) != cudaSuccess ){*/
    if (DPCT_CHECK_ERROR(device = dpct::get_current_device_id()) != 0) {
      throw cuda_error("mult_csm_conj_sum: unable to query current device");
    }
  
    if( !in || in->get_device() != device || !out || out->get_device() != device || !csm || csm->get_device() != device ){
      throw std::runtime_error("mult_csm_conj_sum: array not residing current device");
    }
  
    if( out->get_number_of_dimensions() < D  || out->get_number_of_dimensions() > D+1 ){
      throw std::runtime_error("mult_csm_conj_sum: unexpected output dimensionality");
    }

    if( out->get_number_of_dimensions() > in->get_number_of_dimensions() ){
      throw std::runtime_error("mult_csm_conj_sum: output dimensionality cannot exceed input dimensionality");
    }

    if( csm->get_number_of_dimensions() != D+1 ) {
      throw std::runtime_error("mult_csm_conj_sum: input dimensionality of csm not as expected");
    }

    unsigned int num_image_elements = 1;
    for( unsigned int d=0; d<D; d++ )
      num_image_elements *= out->get_size(d);
  
    unsigned int num_frames = out->get_number_of_elements() / num_image_elements;

/* DPCT_ORIG     dim3 blockDim(256);*/
    dpct::dim3 blockDim(256);
/* DPCT_ORIG     dim3 gridDim((num_image_elements+blockDim.x-1)/blockDim.x, num_frames);*/
    dpct::dim3 gridDim((num_image_elements + blockDim.x - 1) / blockDim.x, num_frames);

/* DPCT_ORIG     mult_csm_conj_sum_kernel<REAL><<< gridDim, blockDim >>>
      ( in->get_data_ptr(), out->get_data_ptr(), csm->get_data_ptr(), num_image_elements, num_frames, csm->get_size(D)
   );*/
    /*
    DPCT1049:63: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct0 = in->get_data_ptr();
            auto out_get_data_ptr_ct1 = out->get_data_ptr();
            auto csm_get_data_ptr_ct2 = csm->get_data_ptr();
            auto csm_get_size_D_ct5 = csm->get_size(D);

            cgh.parallel_for<dpct_kernel_name<class mult_csm_conj_sum_kernel_8aa13b, REAL>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    mult_csm_conj_sum_kernel<REAL>(in_get_data_ptr_ct0, out_get_data_ptr_ct1, csm_get_data_ptr_ct2,
                                                   num_image_elements, num_frames, csm_get_size_D_ct5);
                });
        });
    }

/* DPCT_ORIG     cudaError_t err = cudaGetLastError();*/
    /*
    DPCT1010:244: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
    call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 err = 0;
/* DPCT_ORIG     if( err != cudaSuccess ){*/
    /*
    DPCT1000:243: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
      std::stringstream ss;
      /*
      DPCT1001:242: The statement could not be removed.
      */
      ss << "mult_csm_conj_sum: unable to combine coils " <<
          /* DPCT_ORIG 	cudaGetErrorString(err);*/
          /*
          DPCT1009:245: SYCL reports errors using exceptions and does not use error codes. Please replace the
          "get_error_string_dummy(...)" with a real error-handling function.
          */
          dpct::get_error_string_dummy(err);
      throw cuda_error(ss.str());
    }
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  // Instantiation

  template EXPORTGPUPMRI void csm_mult_M<float,1>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_M<float,2>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_M<float,3>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_M<float,4>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);

  template EXPORTGPUPMRI void csm_mult_M<double,1>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_M<double,2>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_M<double,3>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_M<double,4>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);

  template EXPORTGPUPMRI void csm_mult_MH<float,1>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_MH<float,2>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_MH<float,3>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);
  template EXPORTGPUPMRI void csm_mult_MH<float,4>( cuNDArray< complext<float> >*, cuNDArray< complext<float> >*, cuNDArray< complext<float> >*);

  template EXPORTGPUPMRI void csm_mult_MH<double,1>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_MH<double,2>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_MH<double,3>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
  template EXPORTGPUPMRI void csm_mult_MH<double,4>( cuNDArray< complext<double> >*, cuNDArray< complext<double> >*, cuNDArray< complext<double> >*);
}
