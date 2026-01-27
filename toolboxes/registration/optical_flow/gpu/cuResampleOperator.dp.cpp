#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuResampleOperator.h"
#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/host_vector.h>*/
/* DPCT_ORIG #include <thrust/generate.h>*/
/* DPCT_ORIG #include <thrust/pair.h>*/
/* DPCT_ORIG #include <thrust/sort.h> */
/* DPCT_ORIG #include <thrust/binary_search.h>*/
/* DPCT_ORIG #include <thrust/iterator/counting_iterator.h>*/

namespace Gadgetron{

  template<class T, unsigned int D> void 
  cuResampleOperator<T,D>::mult_MH_preprocess()
  {
    this->preprocessed_ = false;
  
    // Check if a displacement field has been provided
    //
  
    if( !this->offsets_.get() ){
      throw cuda_error("cuResampleOperator::mult_MH_preprocess(): displacement field not set.");
    }

    // Make a device vector wrap of the displacement field
    //

    std::vector<size_t> _dims_disp = this->offsets_->get_dimensions(); _dims_disp.pop_back(); 
    unsigned int num_elements_disp = D;
    while(!_dims_disp.empty()){
      num_elements_disp *= _dims_disp.back();
      _dims_disp.pop_back();
    }

/* DPCT_ORIG     thrust::device_vector<REAL> displacements*/
    dpct::device_vector<REAL> displacements
        /* DPCT_ORIG       ( thrust::device_pointer_cast<REAL>(this->offsets_->get_data_ptr()), */
        (dpct::get_device_pointer<REAL>(this->offsets_->get_data_ptr()),
         /* DPCT_ORIG         thrust::device_pointer_cast<REAL>(this->offsets_->get_data_ptr()+num_elements_disp) );*/
         dpct::get_device_pointer<REAL>(this->offsets_->get_data_ptr() + num_elements_disp));

    // Make sort keys/values array from the deformation field
    //

    unsigned int num_elements_sort = num_elements_disp/D;

/* DPCT_ORIG     this->lower_bounds_ = thrust::device_vector<unsigned int>(num_elements_sort);*/
    this->lower_bounds_ = dpct::device_vector<unsigned int>(num_elements_sort);
/* DPCT_ORIG     this->upper_bounds_ = thrust::device_vector<unsigned int>(num_elements_sort);*/
    this->upper_bounds_ = dpct::device_vector<unsigned int>(num_elements_sort);

/* DPCT_ORIG     this->indices_ = thrust::device_vector<unsigned int>(get_num_neighbors()*num_elements_sort);*/
    this->indices_ = dpct::device_vector<unsigned int>(get_num_neighbors() * num_elements_sort);
/* DPCT_ORIG     this->weights_ = thrust::device_vector<REAL>(get_num_neighbors()*num_elements_sort);*/
    this->weights_ = dpct::device_vector<REAL>(get_num_neighbors() * num_elements_sort);

/* DPCT_ORIG     thrust::device_vector<unsigned int> sort_keys = thrust::device_vector<unsigned int>*/
    dpct::device_vector<unsigned int> sort_keys =
        dpct::device_vector<unsigned int>(get_num_neighbors() * num_elements_sort);

    // Fill arrays
    //

    write_sort_arrays((void*)(&sort_keys));
    
    // Make copy of sort_keys before the sort modifies it
    //

/* DPCT_ORIG     thrust::device_vector<unsigned int> sort_keys_copy(sort_keys);*/
    dpct::device_vector<unsigned int> sort_keys_copy(sort_keys);

    // Sort (twice since we have two value arrays)
    //

/* DPCT_ORIG     thrust::sort_by_key(sort_keys.begin(), sort_keys.end(), this->indices_.begin() );*/
    dpct::sort(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), sort_keys.begin(),
               sort_keys.end(), this->indices_.begin());
/* DPCT_ORIG     thrust::sort_by_key(sort_keys_copy.begin(), sort_keys_copy.end(), this->weights_.begin() );*/
    dpct::sort(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), sort_keys_copy.begin(),
               sort_keys_copy.end(), this->weights_.begin());

    // Find start/end indices (buckets) in the two values arrays
    //

/* DPCT_ORIG     thrust::counting_iterator<unsigned int> search_begin(0);*/
    oneapi::dpl::counting_iterator<unsigned int> search_begin(0);

/* DPCT_ORIG     thrust::lower_bound( sort_keys.begin(), sort_keys.end(),
                 search_begin, search_begin + num_elements_sort, this->lower_bounds_.begin() );*/
    oneapi::dpl::lower_bound(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), sort_keys.begin(),
                             sort_keys.end(), search_begin, search_begin + num_elements_sort,
                             this->lower_bounds_.begin());

/* DPCT_ORIG     thrust::upper_bound( sort_keys.begin(), sort_keys.end(),
                 search_begin, search_begin + num_elements_sort, this->upper_bounds_.begin() );*/
    oneapi::dpl::upper_bound(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), sort_keys.begin(),
                             sort_keys.end(), search_begin, search_begin + num_elements_sort,
                             this->upper_bounds_.begin());

    this->preprocessed_ = true;
  }

  template class EXPORTGPUREG cuResampleOperator<float,1>;
  template class EXPORTGPUREG cuResampleOperator<float_complext,1>;

  template class EXPORTGPUREG cuResampleOperator<float,2>;
  template class EXPORTGPUREG cuResampleOperator<float_complext,2>;

  template class EXPORTGPUREG cuResampleOperator<float,3>;
  template class EXPORTGPUREG cuResampleOperator<float_complext,3>;

  template class EXPORTGPUREG cuResampleOperator<float,4>;
  template class EXPORTGPUREG cuResampleOperator<float_complext,4>;

  template class EXPORTGPUREG cuResampleOperator<double,1>;
  template class EXPORTGPUREG cuResampleOperator<double_complext,1>;

  template class EXPORTGPUREG cuResampleOperator<double,2>;
  template class EXPORTGPUREG cuResampleOperator<double_complext,2>;

  template class EXPORTGPUREG cuResampleOperator<double,3>;
  template class EXPORTGPUREG cuResampleOperator<double_complext,3>;

  template class EXPORTGPUREG cuResampleOperator<double,4>;
  template class EXPORTGPUREG cuResampleOperator<double_complext,4>;
}
