#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "resampleOperator.h"
#include "cuNDArray_math.h"
#include "gpureg_export.h"
#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/device_vector.h>*/

namespace Gadgetron{

  template <class T, unsigned int D>
  class EXPORTGPUREG cuResampleOperator : public resampleOperator< cuNDArray<typename realType<T>::Type>, cuNDArray<T> >
  {    
  public:

    typedef typename realType<T>::Type REAL;
    
    cuResampleOperator() : resampleOperator< cuNDArray<REAL>, cuNDArray<T> >() {}
    virtual ~cuResampleOperator() {}
  
    virtual void reset()
    {
/* DPCT_ORIG       lower_bounds_ = thrust::device_vector<unsigned int>();*/
      lower_bounds_ = dpct::device_vector<unsigned int>();
/* DPCT_ORIG       upper_bounds_ = thrust::device_vector<unsigned int>();*/
      upper_bounds_ = dpct::device_vector<unsigned int>();
/* DPCT_ORIG       indices_ = thrust::device_vector<unsigned int>();*/
      indices_ = dpct::device_vector<unsigned int>();
/* DPCT_ORIG       weights_ = thrust::device_vector<REAL>();*/
      weights_ = dpct::device_vector<REAL>();
      resampleOperator< cuNDArray<typename realType<T>::Type>, cuNDArray<T> >::reset();
    }
    
    virtual void mult_MH_preprocess();
  
  protected:
    virtual unsigned int get_num_neighbors() = 0;
    virtual void write_sort_arrays( void* sort_keys ) = 0;
    
  protected:
/* DPCT_ORIG     thrust::device_vector<unsigned int> lower_bounds_;*/
    dpct::device_vector<unsigned int> lower_bounds_;
/* DPCT_ORIG     thrust::device_vector<unsigned int> upper_bounds_;*/
    dpct::device_vector<unsigned int> upper_bounds_;
/* DPCT_ORIG     thrust::device_vector<unsigned int> indices_;*/
    dpct::device_vector<unsigned int> indices_;
/* DPCT_ORIG     thrust::device_vector<REAL> weights_;*/
    dpct::device_vector<REAL> weights_;
  };
}
