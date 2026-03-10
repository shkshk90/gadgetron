#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "resampleOperator.h"
#include "cuNDArray_math.h"
#include "gpureg_export.h"
#include <dpct/dpl_utils.hpp>

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
      lower_bounds_ = dpct::device_vector<unsigned int>();
      upper_bounds_ = dpct::device_vector<unsigned int>();
      indices_ = dpct::device_vector<unsigned int>();
      weights_ = dpct::device_vector<REAL>();
      resampleOperator< cuNDArray<typename realType<T>::Type>, cuNDArray<T> >::reset();
    }
    
    virtual void mult_MH_preprocess();
  
  protected:
    virtual unsigned int get_num_neighbors() = 0;
    virtual void write_sort_arrays( void *pSort_keys ) = 0;
    
  protected:
    dpct::device_vector<unsigned int> lower_bounds_;
    dpct::device_vector<unsigned int> upper_bounds_;
    dpct::device_vector<unsigned int> indices_;
    dpct::device_vector<REAL> weights_;
  };
}
