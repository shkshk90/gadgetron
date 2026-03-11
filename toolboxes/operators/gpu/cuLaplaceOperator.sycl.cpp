#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuLaplaceOperator.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "vector_td.h"
#include "vector_td_utilities.h"
#include "check_CUDA.h"

namespace Gadgetron{

  // Template Power function
  template<unsigned int i, unsigned int j>
  struct Pow
  {
    enum { Value = i*Pow<i,j-1>::Value};
  };

  template <unsigned int i>
  struct Pow<i,1>
  {
    enum { Value = i};
  };

  template<class T, unsigned int D, unsigned int dim> class inner_laplace_functor{
  public:
		static __inline__ void apply(T& val,const T* __restrict__ in, const typename intd<D>::Type dims,const typename intd<D>::Type co, typename intd<D>::Type& stride){
			for (int d = -1; d < 2; d++)
				stride[dim]=d;
				inner_laplace_functor<T,D,dim-1>::apply(val,in,dims,co,stride);
		}
  };
  template<class T, unsigned int D> class inner_laplace_functor<T,D,0>{
  public:
  	static __inline__ void apply(T& val,const T* __restrict__ in, const typename intd<D>::Type dims,const typename intd<D>::Type co, typename intd<D>::Type& stride){
  		typename intd<D>::Type coN = (co+dims+stride)%dims;
  		val -= in[co_to_idx(coN,dims)];
  	}
  };

  template<class REAL, class T, unsigned int D> void
  laplace_kernel( typename intd<D>::Type dims, const T * __restrict__ in, T * __restrict__ out )
  {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    if( idx < prod(dims) ){
    
      T val = T(0);

      typename intd<D>::Type co = idx_to_co(idx, dims);
      typename intd<D>::Type stride(0);

      inner_laplace_functor<T,D,D-1>::apply(val,in,dims,co,stride);
      out[idx] = val+in[co_to_idx(co, dims)]*((REAL) Pow<3,D>::Value);
    }
  }

  template< class T, unsigned int D> void
  cuLaplaceOperator<T,D>::compute_laplace( cuNDArray<T> *in, cuNDArray<T> *out, bool accumulate )
  {
  
    if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error("laplaceOperator::compute_laplace : array dimensions mismatch.");

    }
  
    typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>( in->get_dimensions()) );

    dpct::dim3 dimBlock(dims[0]);
    dpct::dim3 dimGrid(prod(dims) / dims[0]);

    // Invoke kernel
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct1 = in->get_data_ptr();
            auto out_get_data_ptr_ct2 = out->get_data_ptr();

            cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for<
                dpct_kernel_name<class laplace_kernel_e20ef5, typename realType<T>::Type, T, dpct_kernel_scalar<D>>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    laplace_kernel<typename realType<T>::Type, T, D>(dims, in_get_data_ptr_ct1, out_get_data_ptr_ct2);
                });
        });
    }

    CHECK_FOR_CUDA_ERROR();
  }
  
  // Instantiations

  template class EXPORTGPUOPERATORS cuLaplaceOperator<float, 1>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<float, 2>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<float, 3>;

  template class EXPORTGPUOPERATORS cuLaplaceOperator<float_complext, 1>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<float_complext, 2>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<float_complext, 3>;

  template class EXPORTGPUOPERATORS cuLaplaceOperator<double, 1>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<double, 2>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<double, 3>;

  template class EXPORTGPUOPERATORS cuLaplaceOperator<double_complext, 1>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<double_complext, 2>;
  template class EXPORTGPUOPERATORS cuLaplaceOperator<double_complext, 3>;
}
