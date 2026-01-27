#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuLinearResampleOperator.h"
#include "cuNDArray_reductions.h"
#include "cuResampleOperator_macros.h"
#include "vector_td_utilities.h"
#include "check_CUDA.h"
#include "setup_grid.h"

namespace Gadgetron{

  //
  // Check if all neighbors required for the linear interpolation exists
  //

/* DPCT_ORIG   template<class REAL, unsigned int D> __device__
  bool is_border_pixel( typename reald<REAL,D>::Type co, typename uintd<D>::Type dims )*/
  template <class REAL, unsigned int D>
  bool is_border_pixel(typename reald<REAL, D>::Type co, typename uintd<D>::Type dims)
  {
    for( unsigned int dim=0; dim<D; dim++ ){
      if( dims[dim] > 1 && ( co[dim] < REAL(0) || co[dim] >= (REAL(dims[dim])-REAL(1)) ) )
        return true;
    }
    return false;
  }

/* DPCT_ORIG   template<unsigned int D> static __inline__ __host__ __device__
  unsigned int _get_num_neighbors()*/
  template <unsigned int D> SYCL_EXTERNAL static __inline__ unsigned int _get_num_neighbors()
  {
    return 1 << D;
  }

  template<class T, unsigned int D> unsigned int
  cuLinearResampleOperator<T,D>::get_num_neighbors()
  {
    return _get_num_neighbors<D>();
  }

  //
  // Linear interpolation
  //

/* DPCT_ORIG   template<class T, unsigned int D> __device__
  T interpolate( unsigned int batch_no,
                 typename reald<typename realType<T>::Type,D>::Type co,
                 typename uintd<D>::Type matrix_size,
                 const T * __restrict__ image )*/
  template <class T, unsigned int D>
  SYCL_EXTERNAL T interpolate(unsigned int batch_no, typename reald<typename realType<T>::Type, D>::Type co,
                              typename uintd<D>::Type matrix_size, const T* __restrict__ image)
  {
    typedef typename realType<T>::Type REAL;

    // We will only proceed if all neighbours exist
    //

    if( is_border_pixel<REAL,D>(co, matrix_size) )
      return T(0);

    // To hold the result
    //

    T res = T(0);

    // Iterate over all neighbors
    //

    const typename uintd<D>::Type twos(2);
    const unsigned int num_neighbors = _get_num_neighbors<D>();
  
    for( unsigned int i=0; i<num_neighbors; i++ ){
    
      // Determine image coordinate of current neighbor
      //

      const typename uintd<D>::Type stride = idx_to_co( i, twos );

      if( weak_greater_equal( stride, matrix_size ) ) continue; // For dimensions of size 1

      typename reald<REAL,D>::Type co_stride;

      for( unsigned int dim=0; dim<D; dim++ ){
        if( stride.vec[dim] == 0 ){
/* DPCT_ORIG           co_stride.vec[dim] = ::floor(co.vec[dim]);*/
          co_stride.vec[dim] = sycl::floor(co.vec[dim]);
        }
        else{
/* DPCT_ORIG           co_stride.vec[dim] = ::ceil(co.vec[dim]);*/
          co_stride.vec[dim] = sycl::ceil(co.vec[dim]);
          if( co_stride.vec[dim] == co.vec[dim] )
            co_stride.vec[dim] += REAL(1.0);
        }
      }
      
      // Read corresponding pixel value
      //
    
      T image_value = image[co_to_idx(vector_td<unsigned int,D>(co_stride), matrix_size) + batch_no*prod(matrix_size)];
    
      // Determine weight
      //

      REAL weight = REAL(1);

      for( unsigned int dim=0; dim<D; dim++ ){

        if( stride.vec[dim] == 0 ){
          weight *= (REAL(1.0)-(co.vec[dim]-co_stride.vec[dim]));
        }
        else{
          weight *= (REAL(1.0)-(co_stride.vec[dim]-co.vec[dim]));
        }
      }
      
      // Accumulate result
      //
    
      res += (weight * image_value);
    }

    // All done, return result
    //

    return res;
  }

/* DPCT_ORIG   template<class REAL, unsigned int D> __global__ void
  write_sort_arrays_kernel( typename uintd<D>::Type matrix_size, unsigned int extended_size, const REAL * __restrict__
  displacements, unsigned int * __restrict__ sort_keys,  unsigned int * __restrict__ sort_values_indices, REAL *
  __restrict__ sort_values_weights )*/
  /*
DPCT1110:106: The total declared local variable size in device function write_sort_arrays_kernel exceeds 128 bytes and
may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
  template <class REAL, unsigned int D>
  void write_sort_arrays_kernel(typename uintd<D>::Type matrix_size, unsigned int extended_size,
                                const REAL* __restrict__ displacements, unsigned int* __restrict__ sort_keys,
                                unsigned int* __restrict__ sort_values_indices, REAL* __restrict__ sort_values_weights)
  {
/* DPCT_ORIG     const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                             item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int num_elements_mat = prod(matrix_size);
    const unsigned int num_elements_ext = prod(matrix_size)*extended_size;
  
    if( idx < num_elements_ext ){

      const unsigned int batch_no = idx/num_elements_mat;
      const unsigned int idx_in_batch = idx-batch_no*num_elements_mat;
    
      const typename uintd<D>::Type co = idx_to_co( idx_in_batch, matrix_size );

      typename reald<REAL,D>::Type co_disp = vector_td<REAL,D>(co);
      for( unsigned int dim=0; dim<D; dim++ )
        co_disp.vec[dim] +=  displacements[dim*num_elements_ext+batch_no*num_elements_mat+idx_in_batch];
    
      // Determine the number of neighbors
      //
    
      const typename uintd<D>::Type twos(2);
      const unsigned int num_neighbors = _get_num_neighbors<D>();

      // Weights are non-zero only if all neighbors exist
      //
    
      bool non_zero = !is_border_pixel<REAL,D>(co_disp, matrix_size);

      // Iterate over all neighbors
      //
    
      for( unsigned int i=0; i<num_neighbors; i++ ){
      
        // Write out the sort values/indices
        //
        
        sort_values_indices[idx+i*num_elements_ext] = idx;
        
        // Determine image coordinate of current neighbor
        //
        
        const typename uintd<D>::Type stride = idx_to_co( i, twos );
        
        if( weak_greater_equal( stride, matrix_size ) ) non_zero = false; // For dimensions of size 1
        
        typename reald<REAL,D>::Type co_stride;
        
        if( non_zero ){
          for( unsigned int dim=0; dim<D; dim++ ){
            if( stride.vec[dim] == 0 ){
/* DPCT_ORIG               co_stride.vec[dim] = ::floor(co_disp.vec[dim]);*/
              co_stride.vec[dim] = sycl::floor(co_disp.vec[dim]);
            }
            else{
/* DPCT_ORIG               co_stride.vec[dim] = ::ceil(co_disp.vec[dim]);*/
              co_stride.vec[dim] = sycl::ceil(co_disp.vec[dim]);
              if( co_stride.vec[dim] == co_disp.vec[dim] )
                co_stride.vec[dim] += REAL(1.0);
            }
          }
          
          // Write out sort keys (moving image resampling indices).
          //
          
          sort_keys[idx+i*num_elements_ext] = co_to_idx(vector_td<unsigned int,D>(co_stride), matrix_size) + batch_no*num_elements_mat;
        }
        else{
          sort_keys[idx+i*num_elements_ext] = idx; // Could be anything, weight is zero
        }
        
        // Determine weight
        //
        
        REAL weight = (non_zero) ? REAL(1) : REAL(0);
        
        if( non_zero ){
          for( unsigned int dim=0; dim<D; dim++ ){	  
            if( stride.vec[dim] == 0 ){
              weight *= (REAL(1.0)-(co_disp.vec[dim]-co_stride.vec[dim])); }
            else{
              weight *= (REAL(1.0)-(co_stride.vec[dim]-co_disp.vec[dim])); }
          }
        }
        
        // Write out the sort values/weights
        //

        sort_values_weights[idx+i*num_elements_ext] = weight;
      }
    }
  };

  template<class T, unsigned int D> void 
  cuLinearResampleOperator<T,D>::write_sort_arrays( void* sk )
  {
/* DPCT_ORIG     thrust::device_vector<unsigned int> &sort_keys = *(thrust::device_vector<unsigned int>*)(sk);*/
    dpct::device_vector<unsigned int>& sort_keys = *(dpct::device_vector<unsigned int>*)(sk);
    typename uint64d<D>::Type matrix_size = from_std_vector<size_t,D>(this->offsets_->get_dimensions());
    int surplus = this->offsets_->get_number_of_dimensions()-D;
    unsigned int extended_dim = (surplus == 1) ? 1 : this->offsets_->get_size(D);

/* DPCT_ORIG     dim3 blockDim, gridDim;*/
    dpct::dim3 blockDim, gridDim;
    setup_grid( prod(matrix_size)*extended_dim, &blockDim, &gridDim );

/* DPCT_ORIG     write_sort_arrays_kernel<typename realType<T>::Type,D><<< gridDim, blockDim >>>
      ( vector_td<unsigned int,D>(matrix_size), extended_dim, this->offsets_->get_data_ptr(),
        raw_pointer_cast(&(sort_keys[0])),
        raw_pointer_cast(&(this->indices_)[0]),
        raw_pointer_cast(&(this->weights_)[0]) );*/
    /*
    DPCT1049:108: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    /*
    DPCT1129:107: The type "vector_td<unsigned int, D>" is used in the SYCL kernel, but it is not device copyable. The
    sycl::is_device_copyable specialization has been added for this type. Please review the code.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto vector_td_unsigned_int_D_matrix_size_ct0 = vector_td<unsigned int, D>(matrix_size);
            auto this_offsets__get_data_ptr_ct2 = this->offsets_->get_data_ptr();
            auto raw_pointer_cast_sort_keys_ct3 = dpct::get_raw_pointer(&(sort_keys[0]));
            auto raw_pointer_cast_this_indices__ct4 = raw_pointer_cast(&(this->indices_)[0]);
            auto raw_pointer_cast_this_weights__ct5 = raw_pointer_cast(&(this->weights_)[0]);

            cgh.parallel_for<dpct_kernel_name<class write_sort_arrays_kernel_29af00, typename realType<T>::Type,
                                              dpct_kernel_scalar<D>>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    write_sort_arrays_kernel<typename realType<T>::Type, D>(
                        vector_td_unsigned_int_D_matrix_size_ct0, extended_dim, this_offsets__get_data_ptr_ct2,
                        raw_pointer_cast_sort_keys_ct3, raw_pointer_cast_this_indices__ct4,
                        raw_pointer_cast_this_weights__ct5);
                });
        });
    }

    CHECK_FOR_CUDA_ERROR();
  };
  
  // This macro is a workaround for Cudas missing support for pure virtual functions.
  // It defines mult_M and mult_MH and intended to be shared among all classes derived from cuResampleOperator.
  //
  // 'cu' is automatically appendex to the macro argument (a workaround for the workaround).
  //

  SYCL_EXTERNAL DECLARE_CU_RESAMPLE_OPERATOR_SUPPORT(LinearResampleOperator)

      //
      // Instantiation
      //

      template class EXPORTGPUREG cuLinearResampleOperator<float, 1>;
  template class EXPORTGPUREG cuLinearResampleOperator<float_complext,1>;

  template class EXPORTGPUREG cuLinearResampleOperator<float,2>;
  template class EXPORTGPUREG cuLinearResampleOperator<float_complext,2>;

  template class EXPORTGPUREG cuLinearResampleOperator<float,3>;
  template class EXPORTGPUREG cuLinearResampleOperator<float_complext,3>;

  template class EXPORTGPUREG cuLinearResampleOperator<float,4>;
  template class EXPORTGPUREG cuLinearResampleOperator<float_complext,4>;

  template class EXPORTGPUREG cuLinearResampleOperator<double,1>;
  template class EXPORTGPUREG cuLinearResampleOperator<double_complext,1>;

  template class EXPORTGPUREG cuLinearResampleOperator<double,2>;
  template class EXPORTGPUREG cuLinearResampleOperator<double_complext,2>;

  template class EXPORTGPUREG cuLinearResampleOperator<double,3>;
  template class EXPORTGPUREG cuLinearResampleOperator<double_complext,3>;

  template class EXPORTGPUREG cuLinearResampleOperator<double,4>;
  template class EXPORTGPUREG cuLinearResampleOperator<double_complext,4>;
}
