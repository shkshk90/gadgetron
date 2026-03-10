#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "vector_td.h"
#include "cuNDArray.h"
#include <dpct/dpl_utils.hpp>

namespace Gadgetron {

template<class T, unsigned int D> void vector_fill(cuNDArray< vector_td<T,D> >* data,  vector_td<T,D> val);
template<class T, unsigned int D> void test_abs(cuNDArray< vector_td<T,D> >* data);
template <class T, unsigned int D> dpct::device_vector<T> test_norm(cuNDArray<vector_td<T, D>>* data);
template <class T, unsigned int D> dpct::device_vector<T> test_min(cuNDArray<vector_td<T, D>>* data);

template <class T, unsigned int D> dpct::device_vector<T> test_max(cuNDArray<vector_td<T, D>>* data);

template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > test_amax(cuNDArray< vector_td<T,D> >* data1, cuNDArray< vector_td<T,D> >* data2);
template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > test_amin(cuNDArray< vector_td<T,D> >* data1, cuNDArray< vector_td<T,D> >* data2);
template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > test_amin2(cuNDArray< vector_td<T,D> >* data, T val);
template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > test_amax2(cuNDArray< vector_td<T,D> >* data, T val);
}
