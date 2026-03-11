#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuConvolutionOperator.h"
#include "vector_td_utilities.h"
#include "cudaDeviceManager.h"
#include "setup_grid.h"

namespace Gadgetron {

  // Mirror, but keep the origin unchanged
  template<class T, unsigned int D> void
  origin_mirror_kernel( vector_td<unsigned int,D> matrix_size, vector_td<unsigned int,D> origin, const T * __restrict__ in, T * __restrict__ out, bool zero_fill )
  {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                             item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if( idx < prod(matrix_size) ){
      
      vector_td<unsigned int,D> in_co = idx_to_co( idx, matrix_size );
      vector_td<unsigned int,D> out_co = matrix_size-in_co;
    
      bool wrap = false;
      for( unsigned int d=0; d<D; d++ ){
	if( out_co.vec[d] == matrix_size.vec[d] ){
	  out_co.vec[d] = 0;
	  wrap = true;
	}
      }
    
      const unsigned int in_idx = co_to_idx(in_co, matrix_size);
      const unsigned int out_idx = co_to_idx(out_co, matrix_size);

      if( wrap && zero_fill )
	out[out_idx] = T(0);
      else
	out[out_idx] = in[in_idx];
    }
  }
  
  // Mirror around the origin -- !! leaving the origin unchanged !!
  // This creates empty space "on the left" that can be filled by zero (default) or the left-over entry.
  template<class REAL, unsigned int D> void
  cuConvolutionOperator<REAL,D>::origin_mirror( cuNDArray< complext<REAL> > *in, cuNDArray< complext<REAL> > *out )
  {
    if( in == 0x0 || out == 0x0 ){
      throw std::runtime_error( "origin_mirror: 0x0 ndarray provided");
    }
    
    if( !in->dimensions_equal(out) ){
      throw std::runtime_error("origin_mirror: image dimensions mismatch");
    }
    
    if( in->get_number_of_dimensions() != D ){
      std::stringstream ss;
      ss << "origin_mirror: number of image dimensions is not " << D;
      throw std::runtime_error(ss.str());
    }

    typename uint64d<D>::Type matrix_size = from_std_vector<size_t,D>( in->get_dimensions() );
  
    // Setup block/grid dimensions
    dpct::dim3 blockDim; dpct::dim3 gridDim;
    setup_grid( prod(matrix_size), &blockDim, &gridDim );

    // Invoke kernel
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    /*
    DPCT1129:0: The type "vector_td<unsigned int, D>" is used in the SYCL kernel, but it is not device copyable. The
    sycl::is_device_copyable specialization has been added for this type. Please review the code.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto vector_td_unsigned_int_D_matrix_size_ct0 = vector_td<unsigned int, D>(matrix_size);
            auto vector_td_unsigned_int_D_matrix_size_ct1 = vector_td<unsigned int, D>(matrix_size >> 1);
            auto in_get_data_ptr_ct2 = in->get_data_ptr();
            auto out_get_data_ptr_ct3 = out->get_data_ptr();

            cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for<
                dpct_kernel_name<class origin_mirror_kernel_a16b38, complext<REAL>, dpct_kernel_scalar<D>>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    origin_mirror_kernel<complext<REAL>, D>(vector_td_unsigned_int_D_matrix_size_ct0,
                                                            vector_td_unsigned_int_D_matrix_size_ct1,
                                                            in_get_data_ptr_ct2, out_get_data_ptr_ct3, true);
                });
        });
    }

    CHECK_FOR_CUDA_ERROR();
  }


  template <class REAL, unsigned int D> void 
  cuConvolutionOperator<REAL,D>::operator_fft( bool forwards_transform, cuNDArray< complext<REAL> > *image )
  {
    if( forwards_transform )
      cuNDFFT<REAL>::instance()->fft(image);
    else
      cuNDFFT<REAL>::instance()->ifft(image);
  }    
  
  template EXPORTGPUOPERATORS class cuConvolutionOperator<float,1>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<float,2>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<float,3>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<float,4>;

  template EXPORTGPUOPERATORS class cuConvolutionOperator<double,1>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<double,2>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<double,3>;
  template EXPORTGPUOPERATORS class cuConvolutionOperator<double,4>;
  
}
