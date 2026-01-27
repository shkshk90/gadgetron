#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_operators.h"
#include "complext.h"
#include <functional>

#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/functional.h>*/
/* DPCT_ORIG #include <thrust/transform.h>*/
/* DPCT_ORIG #include <thrust/iterator/constant_iterator.h>*/
/* DPCT_ORIG #include <thrust/iterator/counting_iterator.h>*/
/* DPCT_ORIG #include <thrust/iterator/permutation_iterator.h>*/
#include <complex>

using namespace Gadgetron;
  // Private utility to verify array dimensions. 
  // It "replaces" NDArray::dimensions_equal() to support batch mode.
  // There is an identical function for all array instances (currently hoNDArray, cuNDArray, hoCuNDAraay)
  // !!! Remember to fix any bugs in all versions !!!
  //
  template<class T,class S> static bool compatible_dimensions( const cuNDArray<T> &x, const cuNDArray<S> &y )
  {
    return ((x.get_number_of_elements()%y.get_number_of_elements())==0);
  }

  template <typename T>
  /* DPCT_ORIG   class cuNDA_modulus : public thrust::unary_function<T,T>
    {*/
  /*
  DPCT1044:184: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::unary_function in the class definition.
  */
  class cuNDA_modulus {
  public:
    cuNDA_modulus(int x):mod(x) {};
/* DPCT_ORIG     __host__ __device__ T operator()(const T &y) const {return y%mod;}*/
    T operator()(const T& y) const { return y % mod; }

  private:
    const int mod;
  };

  //
  // This transform support batch mode when the number of elements in x is a multiple of the number of elements in y
  //
  template<class T,class S,class F>  
  static void equals_transform(cuNDArray<T> &x, const cuNDArray<S> &y){
    if (x.dimensions_equal(y)){
/* DPCT_ORIG       thrust::transform(x.begin(), x.end(), y.begin(), x.begin(), F());*/
      std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(),
                     y.begin(), x.begin(), F());
    } else if (compatible_dimensions(x,y))
      {
        if (y.get_number_of_elements() < x.get_number_of_elements()) {
/* DPCT_ORIG           typedef thrust::transform_iterator<cuNDA_modulus<int>, thrust::counting_iterator<int>, int>
 * transform_it;*/
          typedef oneapi::dpl::transform_iterator<cuNDA_modulus<int>, oneapi::dpl::counting_iterator<int>, int>
              transform_it;
/* DPCT_ORIG           transform_it indices = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                                 cuNDA_modulus<int>(y.get_number_of_elements()));*/
          transform_it indices = oneapi::dpl::make_transform_iterator(dpct::make_counting_iterator(0),
                                                                      cuNDA_modulus<int>(y.get_number_of_elements()));
/* DPCT_ORIG           thrust::permutation_iterator<thrust::device_ptr<S>, transform_it> p =
   thrust::make_permutation_iterator( y.begin(), indices);*/
          oneapi::dpl::permutation_iterator<dpct::device_pointer<S>, transform_it> p =
              oneapi::dpl::make_permutation_iterator(y.begin(), indices);
/* DPCT_ORIG           thrust::transform(x.begin(), x.end(), p, x.begin(), F());*/
          std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), p,
                         x.begin(), F());
        } else {
/* DPCT_ORIG           thrust::transform(x.begin(),x.end(),y.begin(),x.begin(),F());*/
          std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(),
                         y.begin(), x.begin(), F());
        }

      } else {
      throw std::runtime_error("The provided cuNDArrays have incompatible dimensions for Gadgetron::operator {+=,-=,*=,/=}");
    }
  }

  template <typename T>
  /* DPCT_ORIG   struct cuNDA_plus : public thrust::binary_function<complext<T>, T, complext<T> >
    {*/
  /*
  DPCT1044:185: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_plus {
/* DPCT_ORIG     __device__ complext<T> operator()(const complext<T> &x, const T &y) const {return x+y;}*/
    complext<T> operator()(const complext<T>& x, const T& y) const { return x + y; }
  };

  template <typename T>
  /* DPCT_ORIG   struct cuNDA_minus : public thrust::binary_function<complext<T>, T, complext<T> >
    {*/
  /*
  DPCT1044:186: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_minus {
/* DPCT_ORIG     __device__ complext<T> operator()(const complext<T> &x, const T &y) const {return x-y;}*/
    complext<T> operator()(const complext<T>& x, const T& y) const { return x - y; }
  };

  template <typename T>
  /* DPCT_ORIG   struct cuNDA_multiply : public thrust::binary_function<complext<T>, T, complext<T> >
    {*/
  /*
  DPCT1044:187: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_multiply {
/* DPCT_ORIG     __device__ complext<T> operator()(const complext<T> &x, const T &y) const {return x*y;}*/
    complext<T> operator()(const complext<T>& x, const T& y) const { return x * y; }
  };

  template <typename T>
  /* DPCT_ORIG   struct cuNDA_divide : public thrust::binary_function<complext<T>, T, complext<T> >
    {*/
  /*
  DPCT1044:188: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_divide {
/* DPCT_ORIG     __device__ complext<T> operator()(const complext<T> &x, const T &y) const {return x/y;}*/
    complext<T> operator()(const complext<T>& x, const T& y) const { return x / y; }
  };

  template<class T, class> cuNDArray<T> & Gadgetron::operator+= (cuNDArray<T> &x, const  cuNDArray<T> &y){
/* DPCT_ORIG     equals_transform< T,T,thrust::plus<T> >(x,y);*/
    equals_transform<T, T, std::plus<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T > & Gadgetron::operator+= (cuNDArray<T> &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), thrust::plus<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::plus<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator+= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_plus<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator+= (cuNDArray<complext<T> > &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), cuNDA_plus<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_plus<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator-= (cuNDArray<T> & x , const cuNDArray<T> & y){
/* DPCT_ORIG     equals_transform< T,T,thrust::minus<T> >(x,y);*/
    equals_transform<T, T, std::minus<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator-= (cuNDArray<T> &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), thrust::minus<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::minus<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator-= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_minus<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator-= (cuNDArray<complext<T> > &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), cuNDA_minus<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_minus<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator*= (cuNDArray<T> &x , const cuNDArray<T> &y){
/* DPCT_ORIG     equals_transform< T,T,thrust::multiplies<T> >(x,y);*/
    equals_transform<T, T, std::multiplies<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T>& Gadgetron::operator*= (cuNDArray<T> &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), thrust::multiplies<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::multiplies<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator*= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_multiply<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator*= (cuNDArray<complext<T> > &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), cuNDA_multiply<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_multiply<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator/= (cuNDArray<T> &x , const cuNDArray<T> &y){
/* DPCT_ORIG     equals_transform< T,T,thrust::divides<T> >(x,y);*/
    equals_transform<T, T, std::divides<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator/= (cuNDArray<T> &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), thrust::divides<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::divides<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator/= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_divide<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator/= (cuNDArray<complext<T> > &x , T y){
/* DPCT_ORIG     thrust::constant_iterator<T> iter(y);*/
    dpct::constant_iterator<T> iter(y);
/* DPCT_ORIG     thrust::transform(x.begin(), x.end(), iter, x.begin(), cuNDA_divide<T>());*/
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_divide<T>());
    return x;
  }


  cuNDArray<bool>& Gadgetron::operator&= (cuNDArray<bool> &x , cuNDArray<bool> &y){

/* DPCT_ORIG     equals_transform< bool,bool,thrust::logical_and<bool> >(x,y);*/
    equals_transform<bool, bool, std::logical_and<bool>>(x, y);
    return x;
  }
  cuNDArray<bool>& Gadgetron::operator|= (cuNDArray<bool> &x , cuNDArray<bool> &y){

/* DPCT_ORIG     equals_transform< bool,bool,thrust::logical_or<bool> >(x,y);*/
    equals_transform<bool, bool, std::logical_or<bool>>(x, y);
    return x;
  }

  //
  // Instantiation
  //
  template cuNDArray<float>& Gadgetron::operator+=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator+=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator-=(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator-=(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator*=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator*=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator/=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator/=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<float>(cuNDArray<complext<float>>& x,const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<float>(cuNDArray<complext<float>>& x, float y);

  template cuNDArray<double>& Gadgetron::operator+=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator+=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator-=(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator-=(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator*=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator*=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator/=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator/=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<double>(cuNDArray<complext<double>>& x,const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<double>(cuNDArray<complext<double>>& x, double y);


  template cuNDArray<complext<float>>& Gadgetron::operator+=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);

template cuNDArray<complext<double>>& Gadgetron::operator+=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);



  cuNDArray<bool>& Gadgetron::operator&=(cuNDArray<bool>& x, cuNDArray<bool>& y);
  cuNDArray<bool>& Gadgetron::operator|=(cuNDArray<bool>& x, cuNDArray<bool>& y);
  
