#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_reductions.h"
#include "setup_grid.h"
#include <dpct/dpl_utils.hpp>

namespace Gadgetron {

  template<class T> static void 
  find_stride( cuNDArray<T> *in, size_t dim, size_t *stride, std::vector<size_t> *dims )
  {
    *stride = 1;
    for( unsigned int i=0; i<in->get_number_of_dimensions(); i++ ){
      if( i != dim )
        dims->push_back(in->get_size(i));
      if( i < dim )
        *stride *= in->get_size(i);
    }
  }
  
  // Sum
  //
  template<class T> 
  void sum_kernel( T *in, T *out, 
                              unsigned int stride, unsigned int number_of_batches, unsigned int number_of_elements )
  {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                             item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if( idx < number_of_elements ){

      unsigned int in_idx = (idx/stride)*stride*number_of_batches+(idx%stride);

      T val = in[in_idx];

      for( unsigned int i=1; i<number_of_batches; i++ ) 
        val += in[i*stride+in_idx];

      out[idx] = val; 
    }
  }

  // Sum
  //
  template<class T>  boost::shared_ptr< cuNDArray<T> > sum( cuNDArray<T> *in, unsigned int dim )
  {
    // Some validity checks
    if( !(in->get_number_of_dimensions()>1) ){
      throw std::runtime_error("sum: underdimensioned.");;
    }

    if( dim > in->get_number_of_dimensions()-1 ){
      throw std::runtime_error( "sum: dimension out of range.");;
    }

    unsigned int number_of_batches = in->get_size(dim);
    unsigned int number_of_elements = in->get_number_of_elements()/number_of_batches;

    // Setup block/grid dimensions
    dpct::dim3 blockDim; dpct::dim3 gridDim;
    setup_grid( number_of_elements, &blockDim, &gridDim );

    // Find element stride
    size_t stride; std::vector<size_t> dims;
    find_stride<T>( in, dim, &stride, &dims );

    // Invoke kernel
    boost::shared_ptr< cuNDArray<T> > out(new cuNDArray<T>());
    out->create(dims);

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct0 = in->get_data_ptr();
            auto out_get_data_ptr_ct1 = out->get_data_ptr();

            cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for<dpct_kernel_name<class sum_kernel_3a78db, T>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), exp_props, [=](sycl::nd_item<3> item_ct1) {
                    sum_kernel<T>(in_get_data_ptr_ct0, out_get_data_ptr_ct1, stride, number_of_batches,
                                  number_of_elements);
                });
        });
    }

    CHECK_FOR_CUDA_ERROR();
    return out;
  }

  template<class T> T mean(cuNDArray<T>* in)
  {
    return std::reduce(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), in->begin(), in->end(),
                       T(0), std::plus<T>()) /
           T(in->get_number_of_elements());
  }

  template<class T> T min(cuNDArray<T>* in)
	{
        return *std::min_element(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), in->begin(),
                                 in->end());
        }

  template<class T> T max(cuNDArray<T>* in)
	{
                return *std::max_element(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                                         in->begin(), in->end());
        }

  template boost::shared_ptr< cuNDArray<float> > sum<float>( cuNDArray<float>*, unsigned int);
  template boost::shared_ptr< cuNDArray<double> > sum<double>( cuNDArray<double>*, unsigned int);
  template boost::shared_ptr< cuNDArray<float_complext> > sum<float_complext>( cuNDArray<float_complext>*, unsigned int);
  template boost::shared_ptr< cuNDArray<double_complext> > sum<double_complext>( cuNDArray<double_complext>*, unsigned int);  

  template float mean<float>(cuNDArray<float>*);
  template float_complext mean<float_complext>(cuNDArray<float_complext>*);
  template double mean<double>(cuNDArray<double>*);
  template double_complext mean<double_complext>(cuNDArray<double_complext>*);

  template float min<float>(cuNDArray<float>*);
  template float max<float>(cuNDArray<float>*);
  template double min<double>(cuNDArray<double>*);
	template double max<double>(cuNDArray<double>*);
}
