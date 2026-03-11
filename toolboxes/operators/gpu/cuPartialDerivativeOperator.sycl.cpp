/** \file cuPartialDerivativeOperator.h
    \brief Implementation of the partial derivative operator for the gpu.
*/

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuPartialDerivativeOperator.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "vector_td_utilities.h"
#include "check_CUDA.h"

namespace Gadgetron{

  template<class T, unsigned int D> void
  first_order_partial_derivative_kernel( typename intd<D>::Type stride, 
                                         typename intd<D>::Type dims, 
                                         const T  * __restrict__ in, T * __restrict__ out )
  {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if( idx < prod(dims) ){

      T valN, valC;

      typename intd<D>::Type co = idx_to_co(idx, dims);
      typename intd<D>::Type coN = (co+dims+stride)%dims;
    
      valN = in[co_to_idx(coN, dims)];
      valC = in[co_to_idx(co, dims)];
    
      T val = valN-valC;
    
      out[idx] += val;
    }
  }

  template<class T, unsigned int D> void
  second_order_partial_derivative_kernel( typename intd<D>::Type forwards_stride, 
                                          typename intd<D>::Type adjoint_stride, 
                                          typename intd<D>::Type dims, 
                                          const T  * __restrict__ in, T * __restrict__ out )
  {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if( idx < prod(dims) ){

      T valN1, valN2, valC;

      typename intd<D>::Type co = idx_to_co(idx, dims);
      typename intd<D>::Type coN1 = (co+dims+forwards_stride)%dims;
      typename intd<D>::Type coN2 = (co+dims+adjoint_stride)%dims;
    
      valN1 = in[co_to_idx(coN1, dims)];
      valN2 = in[co_to_idx(coN2, dims)];
      valC = in[co_to_idx(co, dims)];
    
      T val = valC+valC-valN1-valN2;
    
      out[idx] += val;
    }
  }

  template< class T, unsigned int D> void
  cuPartialDerivativeOperator<T,D>::compute_partial_derivative( typename int64d<D>::Type stride,
                                                                cuNDArray<T> *in, 
                                                                cuNDArray<T> *out, 
                                                                bool accumulate )
  {
    if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "partialDerivativeOperator::compute_partial_derivative : array dimensions mismatch.");

    }


    if (!accumulate) clear(out);
    
    typename int64d<D>::Type dims = vector_td<long long,D>( from_std_vector<size_t,D>( in->get_dimensions()) );
    dpct::dim3 dimBlock(dims.vec[0]);
    dpct::dim3 dimGrid(1, dims.vec[D - 1]);

    for(int d=1; d<D-1; d++ )
      dimGrid.x *= dims.vec[d];
  
    size_t elements = in->get_number_of_elements();

    // Invoke kernel
    for (size_t i = 0; i < elements/prod(dims); i++)
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      /*
      DPCT1129:0: The type "vector_td<int, D>" is used in the SYCL kernel, but it is not device copyable. The
      sycl::is_device_copyable specialization has been added for this type. Please review the code.
      */
      {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                  auto vector_td_int_D_stride_ct0 = vector_td<int, D>(stride);
                  auto vector_td_int_D_dims_ct1 = vector_td<int, D>(dims);
                  auto in_get_data_ptr_i_prod_dims_ct2 = in->get_data_ptr() + i * prod(dims);
                  auto out_get_data_ptr_i_prod_dims_ct3 = out->get_data_ptr() + i * prod(dims);

                  cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                  cgh.parallel_for<
                      dpct_kernel_name<class first_order_partial_derivative_kernel_894512, T, dpct_kernel_scalar<D>>>(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                            first_order_partial_derivative_kernel<T, D>(
                                vector_td_int_D_stride_ct0, vector_td_int_D_dims_ct1, in_get_data_ptr_i_prod_dims_ct2,
                                out_get_data_ptr_i_prod_dims_ct3);
                      });
            });
      }

    CHECK_FOR_CUDA_ERROR();
  }

  template<class T, unsigned int D> void
  cuPartialDerivativeOperator<T,D>::compute_second_order_partial_derivative( typename int64d<D>::Type forwards_stride,
                                                                             typename int64d<D>::Type adjoint_stride, 
                                                                             cuNDArray<T> *in, cuNDArray<T> *out, 
                                                                             bool accumulate )
  {  
    if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "partialDerivativeOperator::compute_second_order_partial_derivative : array dimensions mismatch.");
    }
    
    if (!accumulate) clear(out);

    typename int64d<D>::Type dims = vector_td<long long,D>( from_std_vector<size_t,D>( in->get_dimensions()) );
    dpct::dim3 dimBlock(dims.vec[0]);
    dpct::dim3 dimGrid(1, dims.vec[D - 1]);

    for(int d=1; d<D-1; d++ )
      dimGrid.x *= dims.vec[d];
  
    size_t elements = in->get_number_of_elements();

    // Invoke kernel
		for (size_t i = 0; i < elements/prod(dims); i++)
                        /*
                        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the limit. To get the
                        device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
                        */
      /*
      DPCT1129:2: The type "vector_td<int, D>" is used in the SYCL kernel, but it is not device
      copyable. The sycl::is_device_copyable specialization has been added for this type. Please
      review the code.
      */
      {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                  auto vector_td_int_D_forwards_stride_ct0 = vector_td<int, D>(forwards_stride);
                  auto vector_td_int_D_adjoint_stride_ct1 = vector_td<int, D>(adjoint_stride);
                  auto vector_td_int_D_dims_ct2 = vector_td<int, D>(dims);
                  auto in_get_data_ptr_i_prod_dims_ct3 = in->get_data_ptr() + i * prod(dims);
                  auto out_get_data_ptr_i_prod_dims_ct4 = out->get_data_ptr() + i * prod(dims);

                  cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                  cgh.parallel_for<
                      dpct_kernel_name<class second_order_partial_derivative_kernel_97579d, T, dpct_kernel_scalar<D>>>(
                      sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                            second_order_partial_derivative_kernel<T, D>(
                                vector_td_int_D_forwards_stride_ct0, vector_td_int_D_adjoint_stride_ct1,
                                vector_td_int_D_dims_ct2, in_get_data_ptr_i_prod_dims_ct3,
                                out_get_data_ptr_i_prod_dims_ct4);
                      });
            });
      }

    CHECK_FOR_CUDA_ERROR();
  }

  //
  // Instantiations
  //

  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float, 1>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float, 2>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float, 3>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float, 4>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float, 5>;

  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float_complext, 1>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float_complext, 2>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float_complext, 3>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float_complext, 4>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<float_complext, 5>;

  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double, 1>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double, 2>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double, 3>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double, 4>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double, 5>;

  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double_complext, 1>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double_complext, 2>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double_complext, 3>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double_complext, 4>;
  template class EXPORTGPUOPERATORS cuPartialDerivativeOperator<double_complext, 5>;
}


